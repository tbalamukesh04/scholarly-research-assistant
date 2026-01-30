import sys
import os
import re # Added for regex cleaning
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import json
import logging
import time
from typing import List, Optional

import mlflow
from openai import OpenAI

from pipelines.postprocess.truncate import apply_strict_truncation, reconstruct_final_answer
from pipelines.retrieval.search import Retriever
from pipelines.postprocess.checks import HallucinationChecker 
from pipelines.postprocess.align import Attributor, split_into_sentences
from pipelines.retrieval.hydrate import attach_text
from scripts.compute_dataset_hash import compute_dataset_hash
from utils.logging import log_event, setup_logger
from pipelines.postprocess.confidence import ConfidenceScorer
from pipelines.postprocess.refusal import check_refusal

# --- CONFIGURATION ---
LOCAL_MODEL_NAME = "qwen2.5:3b" 
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

class LLM:
    def __init__(self):
        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        )

    def generate(self, system_prompt: str, query: str, evidence_text: str) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Question: {query}

Evidence Sources:
{evidence_text}

Instructions:
1. Write a clear, detailed paragraph answering the question.
2. CITATION RULE: You must cite the [Source ID] at the end of every sentence.
   - CORRECT: "The model improves accuracy [1]."
   - WRONG: "The model improves accuracy (Chen et al., 2024)."
3. Do not mention author names or years. Only use the numbers [1], [2], etc.

Answer:"""}
            ]

            response = self.client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=messages,
                temperature=0.1, 
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"LLM Generation failed: {e}")
            return ""

def clean_text_for_rag(text: str) -> str:
    """
    Aggressively removes academic citations and existing brackets from the raw text
    so the 3B model doesn't get confused and mimic them.
    """
    # 1. Remove existing bracketed numbers e.g. [12], [4] to avoid confusion with our IDs
    text = re.sub(r'\[\d+\]', '', text)
    
    # 2. Remove typical (Author, Year) patterns to stop style contagion
    # Matches (Name et al., 2024) or (Name, 2024)
    text = re.sub(r'\([a-zA-Z\s\.,]+?\d{4}\)', '', text)
    
    # 3. Clean extra whitespace created by removals
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_evidence(evidence: List[dict]) -> str:
    blocks = []
    for i, e in enumerate(evidence):
        # Clean the text before presenting it
        clean_body = clean_text_for_rag(e['text'])
        # Add explicit Source ID tag
        blocks.append(f"Source ID [{i+1}]: {clean_body}")
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

def log_rag_run(query, answer, citations, dataset_hash, metrics):
    with mlflow.start_run(run_name="rag_query", nested=True):
        mlflow.log_param("query", query)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("model", LOCAL_MODEL_NAME)
        
        if metrics.get("refusal_reason"):
             mlflow.log_param("refusal_reason", metrics["refusal_reason"])

        mlflow.log_metric("num_citations", len(citations))
        mlflow.log_metric("total_sentences", metrics.get("total_sentences", 0))
        mlflow.log_metric("unaligned_sentences", metrics.get("unaligned_sentences", 0))
        mlflow.log_metric("truncated", int(metrics.get("truncated", False)))
        mlflow.log_metric("refused", int(metrics.get("refused", False)))
        mlflow.log_metric("retrieval_latency", metrics.get("retrieval_latency", 0.0))
        
        if "confidence_score" in metrics:
            mlflow.log_metric("confidence_score", metrics["confidence_score"])

def _construct_refusal(query, evidence, reason, dataset_hash, prior_metrics=None):
    metrics = {
        "retrieval_latency": 0.0, "llm_latency": 0.0,
        "retrieved_chunks": len(evidence) if evidence else 0,
        "refused": True, "refusal_reason": reason, 
        "total_sentences": 0, "unaligned_sentences": 0,
        "confidence_score": 0.0, "alignment_score": 0.0, "recall_score": 0.0
    }
    if prior_metrics:
        metrics.update(prior_metrics)
        metrics["refused"] = True
        metrics["refusal_reason"] = reason

    log_rag_run(query, "REFUSAL", [], dataset_hash, metrics)
    return {
        "query": query,
        "answer": "I cannot answer this reliably with the available evidence (Refusal Triggered).",
        "evidence": evidence, "citations": [], "metrics": metrics
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
        
    # --- 1. RETRIEVE ---
    t0_retrieval = time.time()
    raw = retriever.search(query)
    retrieved_ids = [r["paper_id"] for r in raw.get("results", [])]
    
    retrieved = adapt_for_rag(raw["results"], query)
    hydrated = attach_text(retrieved)
    evidence = hydrated["results"]
    t1_retrieval = time.time()
    retrieval_latency = t1_retrieval - t0_retrieval
    
    base_metrics = {"retrieval_latency": retrieval_latency}
    
    # Check 1: Empty Retrieval (Your requested feature)
    if not evidence:
         return _construct_refusal(query, [], "No evidence found (Empty Retrieval)", current_dataset_hash, base_metrics)

    should_refuse, reason = check_refusal(
        retrieved_chunks=evidence, alignment_details=[], confidence_score=0.0,
        confidence_threshold=confidence_threshold, min_distinct_papers=k_min
    )
    
    # --- PROMPT PREP ---
    evidence_text = format_evidence(evidence)
    
    system_prompt = """You are a helpful research assistant. 
Your goal is to answer the question using ONLY the provided evidence.
You must not use outside knowledge."""

    MAX_RETRIES = 2
    attempt = 0
    llm = LLM()
    t0_llm = time.time()
    
    while attempt < MAX_RETRIES:
        log_event(logger=logger, level=logging.INFO, message=f"Generation Attempt {attempt + 1}")
        
        # --- 2. GENERATE ---
        response = llm.generate(system_prompt, query, evidence_text)
        
        # DEBUG: See if regex cleaning worked
        print(f"\n--- DEBUG RAW LLM RESPONSE (Attempt {attempt+1}) ---\n{response}\n------------------------------------------------\n")
        
        current_errors = []
        metrics = base_metrics.copy()
        metrics.update({
            "llm_latency": 0.0, "retrieved_chunks": len(evidence), 
            "refused": False, "truncated": False, "attempts": attempt + 1, 
            "total_sentences": 0, "unaligned_sentences": 0, "confidence_score": 0.0
        })

        if len(response) < 20:
            current_errors.append("Response too short")

        if not current_errors:
            # --- 3. SYNTAX CHECK ---
            syntax_result = checker.run_checks(response, evidence)
            
            if not syntax_result["cited_indices"] and len(evidence) > 0:
                current_errors.append("No [index] citations found. (Do not use Author-Year)")
            
            if not syntax_result["verification_passed"]:
                 current_errors.extend(syntax_result["errors"])
            else:
                # --- 4. ATTRIBUTION (ALIGN) ---
                sentences = split_into_sentences(response)
                
                # Low threshold for 3B model
                attr_result = attributor.verify(sentences, evidence, threshold=0.15)
                
                # --- 5. TRUNCATE ---
                truncated_details = apply_strict_truncation(attr_result["details"])
                
                metrics["total_sentences"] = len(truncated_details)
                metrics["unaligned_sentences"] = len(attr_result["details"]) - len(truncated_details)
                metrics["truncated"] = len(attr_result["details"]) > len(truncated_details)
                
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
                    confidence_score=metrics["confidence_score"],
                    confidence_threshold=confidence_threshold,
                    citation_precision=1.0, 
                    min_distinct_papers=k_min
                )

                if should_refuse:
                    return _construct_refusal(query, evidence, reason, current_dataset_hash, metrics)

        if not current_errors:
            metrics["llm_latency"] = time.time() - t0_llm
            metrics["safety_check"] = "passed" 
            
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
        system_prompt += f"\n\nERROR IN PREVIOUS ATTEMPT: {error_msg}. USE ONLY [1], [2] FORMAT. NO AUTHOR NAMES."
        attempt += 1

    return _construct_refusal(query, evidence, "Max Retries Failed: Compliance", current_dataset_hash, metrics)

if __name__ == "__main__":
    try:
        with open(r"pipelines/evaluation/data/eval_queries.json", encoding="utf-8") as f:
            input_data = json.load(f)
        query = input_data[2]["query"]
        relevant_papers = input_data[2]["relevant_papers"]
    except:
        print("Warning: Eval data not found.")
        query = "What is the impact of transformers on NLP?"
        relevant_papers = []

    print("--- EVAL MODE ---")
    results = answer(
        query, 
        eval_mode=True, 
        relevant_papers=relevant_papers, 
        confidence_threshold=0.5 
    )
    print(json.dumps(results, indent=2))