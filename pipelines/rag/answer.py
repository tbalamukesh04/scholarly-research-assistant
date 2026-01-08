import json
import logging
import os
from typing import List
import time

import mlflow
from google import genai

from pipelines.retrieval.search import Retriever
from pipelines.postprocess.checks import HallucinationChecker 
from pipelines.postprocess.align import Attributor, split_into_sentences
from pipelines.retrieval.hydrate import attach_text
from scripts.compute_dataset_hash import compute_dataset_hash
from utils.logging import log_event, setup_logger


class LLM:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
            },
        )
        return response.text.strip()


def log_rag_run(query, answer, citations, dataset_hash, metrics):
    with mlflow.start_run(run_name="rag_query", nested=True):
        mlflow.log_param("query", query)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_metric("num_citations", len(citations))
        mlflow.log_metric("total_sentences", metrics.get("total_sentences", 0))
        mlflow.log_metric("unaligned_sentences", metrics.get("unaligned_sentences", 0))
        mlflow.log_metric("refused", int(metrics.get("refused", False)))
        mlflow.log_metric("retrieval_latency", metrics.get("retrieval_latency", 0.0))

def format_evidence(evidence: List[dict]) -> str:
    """
    Formats evidence with simple integer IDs.
    Example: [1] Text content...
    """
    blocks = []
    for i, e in enumerate(evidence):
        # 1-based indexing for the LLM
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
    query: str, top_k: int = 8, k_min: int = 1, mode: str = "strict", retriever=None
):
    logger = setup_logger(name="rag_answer", log_dir="./logs", level=logging.INFO)
    
    # --- INITIALIZATION ---
    if retriever is None:
        retriever = Retriever(top_k=top_k)
    
    # Reuse the embedding model from retriever for the attributor
    attributor = Attributor(retriever.model)
    checker = HallucinationChecker() 
    current_dataset_hash = compute_dataset_hash()
    
    # --- RETRIEVAL ---
    t0_retrieval = time.time()
    raw = retriever.search(query)
    print("RAW: \n", raw)
    retrieved = adapt_for_rag(raw["results"], query)
    hydrated = attach_text(retrieved)
    evidence = hydrated["results"]
    t1_retrieval = time.time()
    retrieval_latency = t1_retrieval - t0_retrieval

    if len(evidence) < k_min:
        print("Evidence ", len(evidence))
        return _construct_refusal(query, evidence, retrieval_latency, "Insufficient Information")

    # --- PROMPT PREP ---
    evidence_text = format_evidence(evidence)
    print("Evidence Length: ", len(evidence_text))
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

    # --- GENERATION LOOP (MAX 3) ---
    MAX_RETRIES = 1
    attempt = 0
    current_prompt = f"{system_prompt}\n\nQuestion:\n{query}"
    
    llm = LLM()
    t0_llm = time.time()
    
    while attempt < MAX_RETRIES:
        logger.info(f"Generation Attempt {attempt + 1}")
        
        response = llm.generate(current_prompt)
        
        # Check for Refusals
        if "UNSUPPORTED" in response.lower() and len(response) < 100:
            print("response: ", response)
            return _construct_refusal(query, evidence, retrieval_latency, "Model Refused Content")
             
        current_errors = []
        sentences = []
        
        # 1. Syntax Check (Format & Bounds)
        syntax_result = checker.run_checks(response, evidence)
                
        if not syntax_result["verification_passed"]:
             current_errors.extend(syntax_result["errors"])
        else:
            # 2. Attribution Check (Semantic Similarity)
            # Only run if syntax is valid to avoid parsing garbage
            sentences = split_into_sentences(response)
            attr_result = attributor.verify(sentences, evidence, threshold=0.35)
            
            if not attr_result["attribution_passed"]:
                current_errors.extend(attr_result["failures"])

        # Pass or Retry?
        if not current_errors:
            # --- SUCCESS ---
            t1_llm = time.time()
            
            # Map integer indices back to real IDs
            real_citations = []
            for idx in syntax_result["cited_indices"]:
                e = evidence[idx - 1] 
                real_citations.append(f"{e['paper_id']}:{e['section']}:{e['chunk_id']}")

            if mode == "synthesis" and not response.strip().lower().startswith("synthesis"):
                response = "SYNTHESIS: " + response
                
            metrics = {
                "retrieval_latency": retrieval_latency, 
                "llm_latency": t1_llm - t0_llm,
                "retrieved_chunks": len(evidence),
                "refused": False,
                "attempts": attempt + 1,
                "safety_check": "passed", 
                "total_sentences": len(sentences), 
                "unaligned_sentences": 0
            }
            
            log_rag_run(query, response, real_citations, current_dataset_hash, metrics)
            
            return {
                "query": query,
                "answer": response,
                "citations": real_citations, 
                'metrics': metrics
            }
        
        # --- FAILURE: PREPARE RETRY ---
        error_msg = "; ".join(current_errors)
        logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
        current_prompt += f"\n\nPREVIOUS RESPONSE REJECTED. REASON: {error_msg}. \nREWRITE CORRECTLY USING [index]."
        attempt += 1

    # --- FINAL REFUSAL ---
    logger.error("Max retries reached. Refusing answer.")
    return _construct_refusal(query, evidence, retrieval_latency, "Max Retries Failed")


def _construct_refusal(query, evidence, ret_latency, reason):
    return {
        "query": query,
        "answer": "I cannot answer this reliably with the available evidence (Verification Failed).",
        "evidence": evidence,
        "citations": [],
        "metrics": {
            "retrieval_latency": ret_latency,
            "llm_latency": 0.0,
            "retrieved_chunks": len(evidence),
            "refused": True,
            "refusal_reason": reason, 
            "total_sentences": 0, 
            "unaligned_sentences": 0
        }
    }

if __name__ == "__main__":
    query = """Which programming language is consistently mentioned as the primary
               tool for implementing models, functions, or data synthesis pipelines
               in various technical frameworks and research studies?"""
    results = answer(query, 10, 3)
    print(json.dumps(results, indent=2))