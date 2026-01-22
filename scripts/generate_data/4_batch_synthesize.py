import json
import time
import os
import random
import sys
from typing import List
from google import genai

# Ensure we can import from pipelines
sys.path.append(os.getcwd())

from pipelines.retrieval.search import Retriever
from pipelines.postprocess.checks import HallucinationChecker
from pipelines.postprocess.align import Attributor, split_into_sentences
from pipelines.postprocess.truncate import apply_strict_truncation, reconstruct_final_answer
from pipelines.postprocess.confidence import ConfidenceScorer
from pipelines.postprocess.refusal import check_refusal

class BatchLLM:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate_batch(self, batch_data: List[dict]):
        prompt_blocks = []
        for idx, item in enumerate(batch_data):
            evidence_text = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(item['evidence'])])
            block = f"""
            --- CASE {idx+1} ---
            QUERY: {item['query']}
            CONTEXT:
            {evidence_text}
            """
            prompt_blocks.append(block)
        
        full_prompt = f"""
        You are a dataset generator. 
        For each CASE below, write a high-quality answer using ONLY the provided CONTEXT.
        
        Rules:
        1. Use [index] citations for every claim.
        2. If the context does not answer the query, write "REFUSE".
        3. Output a JSON list of strings. The list must have exactly {len(batch_data)} elements corresponding to the cases.
        
        {''.join(prompt_blocks)}
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=full_prompt,
                config={"response_mime_type": "application/json"}
            )
            parsed = json.loads(response.text)
            
            # --- ROBUSTNESS FIX: Handle Dict wrapping ---
            if isinstance(parsed, dict):
                # If LLM returns {"answers": [...]}, grab the first list value
                for key, value in parsed.items():
                    if isinstance(value, list):
                        parsed = value
                        break
            
            # Ensure it is now a list
            if not isinstance(parsed, list):
                print("LLM Error: JSON is not a list.")
                return ["ERROR"] * len(batch_data)
                
            return parsed
            
        except Exception as e:
            print(f"LLM Batch Error: {e}")
            return ["ERROR"] * len(batch_data)

def process_and_label(query, raw_answer, evidence, attributor, checker, scorer):
    if raw_answer == "ERROR":
        return None
    
    # 1. Handle Explicit Refusals
    if raw_answer == "REFUSE":
        return {
            "query": query,
            "response": "I cannot answer this based on the available evidence.",
            "label": "refusal", 
            "metrics": {"refused": True, "refusal_reason": "LLM Self-Refusal"}
        }

    # 2. Syntax Check
    syntax = checker.run_checks(raw_answer, evidence)
    if not syntax["verification_passed"]:
        return {
            "query": query,
            "response": raw_answer,
            "label": "rejected_syntax",
            "errors": syntax["errors"]
        }

    # 3. Attribution
    sentences = split_into_sentences(raw_answer)
    attr_result = attributor.verify(sentences, evidence, threshold=0.2)
    
    # 4. Truncation
    truncated_details = apply_strict_truncation(attr_result["details"])
    final_answer = reconstruct_final_answer(truncated_details)

    # 5. Confidence Scoring
    retrieved_ids = [e['paper_id'] for e in evidence]
    conf_metrics = scorer.calculate(
        alignment_details=attr_result["details"],
        retrieved_ids=retrieved_ids,
        k=len(evidence),
        relevant_papers=[]
    )
    
    # 6. Final Refusal Check
    should_refuse, reason = check_refusal(
        retrieved_chunks=evidence,
        alignment_details=attr_result["details"],
        confidence_score=conf_metrics["confidence_score"],
        confidence_threshold=0.5, 
        min_distinct_papers=1
    )

    if should_refuse:
        return {
            "query": query,
            "response": "I cannot answer this based on the available evidence.",
            "label": "refusal", 
            "metrics": conf_metrics
        }
    
    # 7. SUCCESS
    return {
        "query": query,
        "response": final_answer,
        "label": "accepted",
        "evidence_used": evidence,
        "metrics": conf_metrics
    }

def main():
    try:
        with open("data_validated.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Run File 3 first!")
        return
    
    print("Initializing pipeline components...")
    retriever_instance = Retriever(top_k=1) 
    attributor = Attributor(retriever_instance.model) 
    checker = HallucinationChecker()
    scorer = ConfidenceScorer()
    llm_batcher = BatchLLM()

    BATCH_SIZE = 10
    MAX_DAILY_CALLS = 5 
    OUTPUT_FILE = "dataset_training_master.jsonl"
    
    print(f"Starting Batch Synthesis. Target: {BATCH_SIZE * MAX_DAILY_CALLS} items...")
    random.shuffle(data)
    
    total_saved = 0
    
    for i in range(MAX_DAILY_CALLS):
        batch_slice = data[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        if not batch_slice:
            break
            
        print(f"--- Batch {i+1} / {MAX_DAILY_CALLS} ---")
        
        # 1. Generate
        print("  > Calling Gemini...")
        raw_answers = llm_batcher.generate_batch(batch_slice)
        
        batch_results = []
        
        # 2. Process
        print("  > Verifying answers...")
        for j, item in enumerate(batch_slice):
            # Guard against index errors if LLM returns short list
            if j < len(raw_answers):
                processed = process_and_label(
                    item['query'], 
                    raw_answers[j], 
                    item['evidence'], 
                    attributor, 
                    checker, 
                    scorer
                )
                if processed:
                    status = processed['label'].upper()
                    print(f"    [{j+1}] {status}: {item['query'][:50]}...")
                    batch_results.append(processed)
        
        # 3. INCREMENTAL SAVE (Safety)
        if batch_results:
            with open(OUTPUT_FILE, "a") as f:
                for entry in batch_results:
                    f.write(json.dumps(entry) + "\n")
            total_saved += len(batch_results)
            print(f"  > Saved {len(batch_results)} items. Total session: {total_saved}")
        
        time.sleep(2) 

    print(f"Job Complete. Total added this session: {total_saved}")

if __name__ == "__main__":
    main()