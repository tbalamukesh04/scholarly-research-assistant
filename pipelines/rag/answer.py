import json
import logging
import os
from typing import List, Optional, Dict, Any
import time

from utils.mlflow_handler import MLflowHandler 
from utils.mlflow_schema import RunType, ALLOWED_METRICS
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
                    "temperature": 0.0, 
                    "max_output_tokens": 1024,
                },
            )
            return response.text.strip()
        except Exception as e:
            return ""


def log_rag_run(query, answer, citations, dataset_hash, metrics):
    tags = {
        "dataset_hash": dataset_hash,
        "query": query,
        "index_hash": get_index_hash(),
        "prompt_version": PROMPT_VERSION,
        "guardrail_version": GUARDRAIL_VERSION,
        "git_commit": get_git_commit(),
        "run_type": RunType.GUARDRAIL.value 
    }
    
    # FILTER METRICS TO COMPLY WITH SCHEMA
    allowed_keys = ALLOWED_METRICS[RunType.GUARDRAIL]
    filtered_metrics = {k: v for k, v in metrics.items() if k in allowed_keys}
    
    with MLflowHandler.start_run(
        run_name="rag_query", 
        run_type=RunType.GUARDRAIL, 
        tags=tags, 
        nested=True
    ) as run:
        # Log only the allowed numerical metrics
        MLflowHandler.log_metrics(RunType.GUARDRAIL, filtered_metrics)
        
        # Log refusal reason as a param if present
        if metrics.get("refusal_reason"):
             MLflowHandler.log_params({"refusal_reason": metrics["refusal_reason"]})

        if hasattr(run, "info"):
            return run.info.run_id
        return None


def _construct_refusal(query, evidence, reason, dataset_hash, prior_metrics=None):
    metrics = {
        "retrieval_latency": 0.0,
        "llm_latency": 0.0,
        "refusal_triggered": 1.0,
        "num_total_sentences": 0, 
        "confidence_score": 0.0, 
        "alignment_score": 0.0,
        "recall_score": 0.0,
        "num_supported_sentences": 0,
        "retrieved_chunks": len(evidence) if evidence else 0,
        "refusal_reason": reason
    }
    
    if prior_metrics:
        metrics.update(prior_metrics)
        metrics["refusal_triggered"] = 1.0
        metrics["refusal_reason"] = reason
    
    run_id = log_rag_run(query, "REFUSAL", [], dataset_hash, metrics)

    return {
        "query": query,
        "answer": None,
        "answer_sentences": [],
        "citations": [],
        "metrics": metrics,
        "run_id": run_id,
        "index_hash": get_index_hash()
    }


def format_evidence(evidence: List[dict]) -> str:
    blocks = []
    for i, e in enumerate(evidence):
        text = e.get('text', '') or ""
        blocks.append(f"[{i+1}] {text}")
    return "\n\n".join(blocks)


def adapt_for_rag(results, query):
    adapted = []
    for r in results:
        chunk_id = r.get("chunk_id", "")
        parts = chunk_id.split("::")
        section = parts[2] if len(parts) > 2 else "unknown"
        
        order = 0
        if "::chunk::" in chunk_id:
            try:
                order = int(chunk_id.split("::chunk::")[1])
            except (IndexError, ValueError):
                order = 0
                
        adapted.append({
            "paper_id": r.get("paper_id", "unknown"),
            "chunk_id": chunk_id,
            "section": section,
            "order": order,
            "text": None, 
        })
        
    return {
        "query": query,
        "results": adapted
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
    current_index_hash = get_index_hash()
        
    t0_retrieval = time.time()
    raw = retriever.search(query)
    retrieved_ids = [r["paper_id"] for r in raw.get("results", [])]
    
    retrieved = adapt_for_rag(raw.get("results", []), query)
    hydrated = attach_text(retrieved)
    evidence = hydrated["results"]
    t1_retrieval = time.time()
    retrieval_latency = t1_retrieval - t0_retrieval
    
    base_metrics = {"retrieval_latency": retrieval_latency}
    
    metrics = base_metrics.copy()
    metrics.update({
        "llm_latency": 0.0, 
        "refusal_triggered": 0.0, 
        "num_total_sentences": 0, 
        "confidence_score": 0.0,
        "num_supported_sentences": 0,
        "retrieved_chunks": len(evidence)
    })

    should_refuse, reason = check_refusal(
        retrieved_chunks=evidence,
        alignment_details=[], 
        confidence_score=0.0,
        confidence_threshold=confidence_threshold,
        min_distinct_papers=k_min
    )
    
    if should_refuse and not evidence:
         return _construct_refusal(query, evidence, reason, current_dataset_hash, base_metrics)

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

    MAX_RETRIES = 3 
    attempt = 0
    current_prompt = f"{system_prompt}\n\nQuestion:\n{query}"
    
    llm = LLM()
    t0_llm = time.time()
    
    while attempt < MAX_RETRIES:
        log_event(logger=logger, level=logging.INFO, message=f"Generation Attempt {attempt + 1}")
        
        response = llm.generate(current_prompt)
        
        if not response:
            log_event(logger=logger, level=logging.WARNING, message=f"Attempt {attempt + 1} failed: Empty Response")
            attempt += 1
            time.sleep(1)
            continue

        current_errors = []
        
        metrics = base_metrics.copy()
        metrics.update({
            "llm_latency": 0.0, 
            "refusal_triggered": 0.0, 
            "num_total_sentences": 0, 
            "confidence_score": 0.0,
            "num_supported_sentences": 0,
            "retrieved_chunks": len(evidence)
        })

        syntax_result = checker.run_checks(response, evidence)
                
        if not syntax_result["verification_passed"]:
             current_errors.extend(syntax_result["errors"])
        else:
            sentences = split_into_sentences(response)
            attr_result = attributor.verify(sentences, evidence, threshold=0.2)
            
            truncated_details = apply_strict_truncation(attr_result["details"])
            
            metrics["num_total_sentences"] = len(truncated_details)
            metrics["num_supported_sentences"] = len([d for d in truncated_details if d["verification_status"] == "supported"])
            
            scorer = ConfidenceScorer()    
            conf_metrics = scorer.calculate(
                alignment_details = attr_result["details"],
                retrieved_ids = retrieved_ids, 
                relevant_papers=relevant_papers if relevant_papers is not None else [], 
                k = top_k
            )
            metrics.update(conf_metrics)

            should_refuse, reason = check_refusal(
                retrieved_chunks=evidence,
                alignment_details=attr_result["details"], 
                confidence_score=metrics.get("confidence_score", 0.0),
                confidence_threshold=confidence_threshold,
                citation_precision=metrics.get("citation_precision", 1.0),
                min_distinct_papers=k_min
            )
            
            if should_refuse:
                 metrics["refusal_reason"] = reason

            if should_refuse:
                return _construct_refusal(query, evidence, reason, current_dataset_hash, metrics)

        if not current_errors:
            metrics["llm_latency"] = time.time() - t0_llm
            final_response_text = reconstruct_final_answer(truncated_details)
            
            final_sentences = []
            used_citations_map = {} 
            next_citation_id = 1
            
            for det in truncated_details:
                c_indices = []
                if det["verification_status"] == "supported" and det["supported_by_chunk_index"] is not None:
                    idx = det["supported_by_chunk_index"]
                    if 0 <= idx < len(evidence):
                        if idx not in used_citations_map:
                             chunk = evidence[idx]
                             safe_text = chunk.get("text") or ""
                             
                             used_citations_map[idx] = {
                                 "citation_id": next_citation_id,
                                 "paper_id": chunk["paper_id"],
                                 "section": chunk["section"],
                                 "text": safe_text,
                                 "score": float(det["max_score"])
                             }
                             next_citation_id += 1
                        else:
                            if float(det["max_score"]) > used_citations_map[idx]["score"]:
                                 used_citations_map[idx]["score"] = float(det["max_score"])
                        
                        c_indices.append(used_citations_map[idx]["citation_id"])
                
                final_sentences.append({
                    "text": det["sentence"],
                    "verification_status": det["verification_status"],
                    "citation_indices": c_indices
                })
            
            final_citations = sorted(used_citations_map.values(), key=lambda x: x["citation_id"])
            
            if mode == "synthesis" and not final_response_text.strip().lower().startswith("synthesis"):
                final_response_text = "SYNTHESIS: " + final_response_text
            
            audit_citations = [f"{c['paper_id']}:{c['section']}:{c['citation_id']}" for c in final_citations]
            run_id = log_rag_run(query, final_response_text, audit_citations, current_dataset_hash, metrics)
            
            return {
                "query": query,
                "answer": final_response_text,
                "answer_sentences": final_sentences,
                "citations": final_citations, 
                'metrics': metrics,
                "run_id": run_id,
                "index_hash": current_index_hash
            }
        
        error_msg = "; ".join(current_errors)
        log_event(logger=logger, level=logging.WARNING, message=f"Attempt {attempt + 1} failed: {error_msg}")
        current_prompt += f"\n\nPREVIOUS RESPONSE REJECTED. REASON: {error_msg}. \nREWRITE CORRECTLY USING [index]."
        attempt += 1

    return _construct_refusal(query, evidence, "Max Retries Failed", current_dataset_hash, metrics)