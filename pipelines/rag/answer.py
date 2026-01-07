import json
import logging
import os
from typing import List
import time

import mlflow
from google import genai

from pipelines.retrieval.search import Retriever
from pipelines.postprocess.checks import HallucinationChecker 
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


def log_rag_run(query, answer, citations, dataset_hash):
    with mlflow.start_run(run_name="rag_query", nested=True):
        mlflow.log_param("query", query)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_metric("num_citations", len(citations))


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
    if retriever is None:
        retriever = Retriever(top_k=top_k)
        
    checker = HallucinationChecker() 
    current_dataset_hash = compute_dataset_hash()
    
    # --- RETRIEVAL ---
    t0_retrieval = time.time()
    raw = retriever.search(query)
    retrieved = adapt_for_rag(raw["results"], query)
    hydrated = attach_text(retrieved)
    evidence = hydrated["results"]
    t1_retrieval = time.time()
    retrieval_latency = t1_retrieval - t0_retrieval

    if len(evidence) < k_min:
        return _construct_refusal(query, evidence, retrieval_latency, "Insufficient Information")

    # --- PROMPT PREP ---
    evidence_text = format_evidence(evidence)
    
    # Corrected: Explicitly defining the format in the prompt
    system_prompt = f"""
    You are a scholarly assistant.
    
    Rules:
    - Use ONLY the evidence provided below.
    - Every sentence MUST include a citation in the format [index].
    - Example: "The sky is blue [1]."
    - If the evidence is insufficient, say: "I cannot answer this reliably."
    
    Evidence:
    {evidence_text}
    """

    # --- GENERATION LOOP (MAX 3) ---
    MAX_RETRIES = 3
    attempt = 0
    current_prompt = f"{system_prompt}\n\nQuestion:\n{query}"
    
    llm = LLM()
    t0_llm = time.time()
    
    while attempt < MAX_RETRIES:
        logger.info(f"Generation Attempt {attempt + 1}")
        
        response = llm.generate(current_prompt)
        
        if "cannot answer" in response.lower() and len(response) < 100:
             return _construct_refusal(query, evidence, retrieval_latency, "Model Refused Content")

        check_result = checker.run_checks(response, evidence)
        
        if check_result["verification_passed"]:
            t1_llm = time.time()
            
            # Map integer indices back to real IDs for the final JSON
            real_citations = []
            for idx in check_result["cited_indices"]:
                # Adjust for 0-based list vs 1-based prompt
                e = evidence[idx - 1] 
                real_citations.append(f"{e['paper_id']}:{e['section']}:{e['chunk_id']}")

            if mode == "synthesis" and not response.strip().lower().startswith("synthesis"):
                response = "SYNTHESIS: " + response
                
            log_rag_run(query, response, real_citations, current_dataset_hash)
            
            return {
                "query": query,
                "answer": response,
                "citations": real_citations, 
                'metrics':{
                    "retrieval_latency": retrieval_latency, 
                    "llm_latency": t1_llm - t0_llm, 
                    "retrieved_chunks": len(evidence),
                    "refused": False,
                    "attempts": attempt + 1
                }
            }
        
        errors = "; ".join(check_result["errors"])
        logger.warning(f"Attempt {attempt + 1} failed checks: {errors}")
        # Corrected: Explicitly instructing the format for the retry
        current_prompt += f"\n\nPREVIOUS RESPONSE WAS REJECTED. REASON: {errors}. \nREWRITE THE ANSWER CORRECTLY USING [index]."
        attempt += 1

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
            "refusal_reason": reason
        }
    }

if __name__ == "__main__":
    query = "What is the role of the Transformer architecture and the attention mechanism in modern foundation models?"
    results = answer(query, 10, 3)
    print(json.dumps(results, indent=2))