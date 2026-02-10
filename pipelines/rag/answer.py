import json
import logging
import os
from typing import List, Optional, Dict, Any
import time

from utils.mlflow_handler import MLflowHandler 
from utils.mlflow_schema import RunType 
from utils.metadata import get_git_commit, get_index_hash, PROMPT_VERSION, GUARDRAIL_VERSION

from google import genai

from pipelines.postprocess.truncate import apply_strict_truncation, reconstruct_final_answer
from pipelines.retrieval.search import Retriever
from pipelines.postprocess.checks import HallucinationChecker 
from pipelines.postprocess.align import Attributor, split_into_sentences
from pipelines.retrieval.hydrate import attach_text
from scripts.compute_dataset_hash import compute_dataset_hash
from utils.logging import log_event, setup_logger
from pipelines.postprocess.confidence import ConfidenceScorer
from pipelines.postprocess.refusal import check_refusal


class LLM:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite", 
                contents=prompt,
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                },
            )
            return response.text.strip()
        except Exception as e:
            return ""


def log_rag_run(query, answer, citations, dataset_hash, metrics):
    """
    Logs the RAG query execution to MLflow via MLflowHandler.
    Now adheres to Strict Schema with real lineage data.
    """
    tags = {
        "dataset_hash": dataset_hash,
        "query": query,
        "index_hash": get_index_hash(),
        "prompt_version": PROMPT_VERSION,
        "guardrail_version": GUARDRAIL_VERSION,
        "git_commit": get_git_commit(),
        "run_type": RunType.GUARDRAIL.value # Explicitly set for tag validation
    }
    
    # We use RunType.GUARDRAIL as the primary type for RAG answers 
    # because it includes refusal/confidence checks.
    with MLflowHandler.start_run(
        run_name="rag_query", 
        run_type=RunType.GUARDRAIL, 
        tags=tags, 
        nested=True
    ):
        MLflowHandler.log_metrics(RunType.GUARDRAIL, metrics)


def _construct_refusal(query, evidence, reason, dataset_hash, prior_metrics=None):
    """
    Constructs the refusal response and logs it to MLflow.
    """
    metrics = {
        "retrieval_latency": 0.0,
        "llm_latency": 0.0,
        "refusal_triggered": 1.0,
        "num_total_sentences": 0, 
        "confidence_score": 0.0, 
        "alignment_score": 0.0,
        "recall_score": 0.0,
        "num_supported_sentences": 0
    }
    
    if prior_metrics:
        metrics.update(prior_metrics)
        metrics["refusal_triggered"] = 1.0

    # Ensure refusal reason is available for inspection (logged as param if needed, but not metric)
    # Note: MLflowHandler tags handle the schema, we can add ad-hoc params if we want,
    # but for now we stick to the required tags.
    
    log_rag_run(query, "REFUSAL", [], dataset_hash, metrics)

    return {
        "query": query,
        "answer": "I cannot answer this reliably with the available evidence (Refusal Triggered).",
        "evidence": evidence,
        "citations": [],
        "metrics": metrics
    }


def format_evidence(evidence: List[dict]) -> str:
    blocks = []
    for i, e in enumerate(evidence):
        blocks.append(f"[{i+1}] {e['text']}")
    return "\n\n".join(blocks)


def adapt_for_rag(results, query):
    return {
        "query": query,
        "results": [
            {
                "paper_id": r["paper_id"],
                "chunk_id": r["chunk_id"],
                "section": r["chunk_id"].split("::")[2],
                "order": int(r["chunk_id"].split("::chunk::")[1]),
                "text": None,
            }
            for r in results
        ],
    }


def answer(
    query: str, 
    top_k: int = 8, 
    k_min: int = 1, 
    mode: str = "strict", 
    retriever = None, 
    eval_mode: bool = False,
    relevant_papers: Optional[List[str]] = None, 
    confidence_threshold: float = 0.0
):
    logger = setup_logger(name="rag_answer", log_dir="./logs", level=logging.INFO)
    
    if retriever is None:
        retriever = Retriever(top_k=top_k)
    
    attributor = Attributor(retriever.model)
    checker = HallucinationChecker() 
    current_dataset_hash = compute_dataset_hash()
        
    # --- 1. RETRIEVE & HYDRATE ---
    t0_retrieval = time.time()
    raw = retriever.search(query)
    retrieved_ids = [r["paper_id"] for r in raw.get("results", [])]
    
    retrieved = adapt_for_rag(raw["results"], query)
    hydrated = attach_text(retrieved)
    evidence = hydrated["results"]
    t1_retrieval = time.time()
    retrieval_latency = t1_retrieval - t0_retrieval
    
    base_metrics = {"retrieval_latency": retrieval_latency}
    
    should_refuse, reason = check_refusal(
        retrieved_chunks=evidence,
        alignment_details=[], 
        confidence_score=0.0,
        confidence_threshold=confidence_threshold,
        min_distinct_papers=k_min
    )
    
    if should_refuse and not evidence:
         return _construct_refusal(query, evidence, reason, current_dataset_hash, base_metrics)

    # --- PROMPT PREP ---
    evidence_text = format_evidence(evidence)
    system_prompt = f"""
    You are a scholarly assistant.
    
    Rules:
    - Use ONLY the evidence provided below.
    - Every sentence MUST include a citation in the format [index].
    - Example: "The sky is blue [1]."
    - If the evidence is insufficient, say: "UNSUPPORTED"
    
    Evidence:
    {evidence_text}
    """

    MAX_RETRIES = 1
    attempt = 0
    current_prompt = f"{system_prompt}\n\nQuestion:\n{query}"
    
    llm = LLM()
    t0_llm = time.time()
    
    while attempt < MAX_RETRIES:
        log_event(logger=logger, level=logging.INFO, message=f"Generation Attempt {attempt + 1}")
        
        # --- 2. GENERATE ---
        response = llm.generate(current_prompt)
        
        current_errors = []
        
        metrics = base_metrics.copy()
        metrics.update({
            "llm_latency": 0.0, 
            "refusal_triggered": 0.0, 
            "num_total_sentences": 0, 
            "confidence_score": 0.0,
            "num_supported_sentences": 0
        })

        # --- 3. SYNTAX CHECK ---
        syntax_result = checker.run_checks(response, evidence)
                
        if not syntax_result["verification_passed"]:
             current_errors.extend(syntax_result["errors"])
        else:
            # --- 4. ATTRIBUTION (ALIGN) ---
            sentences = split_into_sentences(response)
            attr_result = attributor.verify(sentences, evidence, threshold=0.2)
            
            # --- 5. TRUNCATE ---
            truncated_details = apply_strict_truncation(attr_result["details"])
            
            metrics["num_total_sentences"] = len(truncated_details)
            metrics["num_supported_sentences"] = len([d for d in truncated_details if d["verification_status"] == "supported"])
            
            # --- 6. CONFIDENCE SCORING ---
            scorer = ConfidenceScorer()    
            conf_metrics = scorer.calculate(
                alignment_details = attr_result["details"],
                retrieved_ids = retrieved_ids, 
                relevant_papers=relevant_papers if relevant_papers is not None else [], 
                k = top_k
            )
            # Filter metrics to only those allowed by schema (or rely on update overriding)
            # The schema validator will catch if scorer returns something wild.
            metrics.update(conf_metrics)

            # --- 7. REFUSAL CHECK (FINAL AUTHORITY) ---
            should_refuse, reason = check_refusal(
                retrieved_chunks=evidence,
                alignment_details=attr_result["details"], 
                confidence_score=metrics.get("confidence_score", 0.0),
                confidence_threshold=confidence_threshold,
                citation_precision=1.0, 
                min_distinct_papers=k_min
            )

            if should_refuse:
                return _construct_refusal(query, evidence, reason, current_dataset_hash, metrics)

        if not current_errors:
            # --- SUCCESS ---
            metrics["llm_latency"] = time.time() - t0_llm
            
            final_response = reconstruct_final_answer(truncated_details)
            
            real_citations = []
            for idx in syntax_result["cited_indices"]:
                if 1 <= idx <= len(evidence):
                    e = evidence[idx - 1] 
                    real_citations.append(f"{e['paper_id']}:{e['section']}:{e['chunk_id']}")

            if mode == "synthesis" and not final_response.strip().lower().startswith("synthesis"):
                final_response = "SYNTHESIS: " + final_response
            
            log_rag_run(query, final_response, real_citations, current_dataset_hash, metrics)
            
            return {
                "query": query,
                "answer": final_response,
                "citations": real_citations, 
                'metrics': metrics
            }
        
        error_msg = "; ".join(current_errors)
        log_event(logger=logger, level=logging.WARNING, message=f"Attempt {attempt + 1} failed: {error_msg}")
        current_prompt += f"\n\nPREVIOUS RESPONSE REJECTED. REASON: {error_msg}. \nREWRITE CORRECTLY USING [index]."
        attempt += 1

    return _construct_refusal(query, evidence, "Max Retries Failed", current_dataset_hash, metrics)

if __name__ == "__main__":
    with open(r"pipelines\evaluation\data\eval_queries.json", encoding="utf-8") as f:
        input = json.load(f)
    
    query = input[2]["query"]
    relevant_papers = input[2]["relevant_papers"]
    print("--- EVAL MODE ---")
    results = answer(
        query, 
        eval_mode=True, 
        relevant_papers=relevant_papers,
        confidence_threshold=0.8 
    )
    print(json.dumps(results, indent=2))