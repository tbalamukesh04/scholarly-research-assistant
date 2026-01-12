import json
import os
import glob
import uuid
import random

# --- Configuration ---
INPUT_DIR = "data/processed/chunks"
OUTPUT_FILE = "data/finetuning/refusal_qa.jsonl"

OOD_QUESTIONS = [
    "Who won the World Cup in 2022?", "Explain the plot of the movie 'Inception'.",
    "What is the capital of Australia?", "How do I bake a chocolate cake?",
    "What is the current stock price of Apple?", "Write a Python script to scrape Google.",
    "Who is the president of the United States?", "What are the health benefits of green tea?",
    "Translate 'Hello' into Spanish.", "What is the distance between Earth and the Moon?"
]

REFUSAL_RESPONSES = [
    "I cannot answer this question based on the provided documents.",
    "The provided context does not contain information regarding this topic.",
    "There is no evidence in the retrieved text to support an answer to this question.",
    "The available documents do not address this query."
]

def generate_refusal_qa():
    print(f"Reading from {INPUT_DIR}...")
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if len(files) < 1: # Adjusted to allow running even with 1 file for testing
        print("Warning: Need at least 2 processed papers for ideal mismatched refusals.")
    
    papers = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                paper_id = data.get("paper_id", os.path.basename(file_path).replace(".json", ""))
                
                # FIX: Flatten all chunks from all sections into one list
                all_chunks = []
                sections = data.get("sections", [])
                for sec in sections:
                    chunks = sec.get("chunks", [])
                    for c in chunks:
                        if len(c.get("text", "")) > 200:
                            all_chunks.append(c.get("text", ""))
                
                if all_chunks:
                    papers.append({
                        "id": paper_id,
                        "chunk": random.choice(all_chunks)
                    })
            except:
                continue

    examples = []
    
    # 1. Generate OOD Refusals
    for paper in papers:
        for _ in range(2):
            question = random.choice(OOD_QUESTIONS)
            response = random.choice(REFUSAL_RESPONSES)
            
            example = {
                "id": str(uuid.uuid4()),
                "type": "refusal",
                "source_refs": [],
                "instruction": question,
                "context": f"[Document {paper['id']}]: {paper['chunk']}",
                "response": response
            }
            examples.append(example)

    # 2. Generate Mismatched Refusals (Only if we have >1 paper)
    if len(papers) > 1:
        for target_paper in papers:
            specific_q = f"What are the specific results presented in document {target_paper['id']}?"
            other_papers = [p for p in papers if p['id'] != target_paper['id']]
            
            distractor = random.choice(other_papers)
            response = f"The provided context contains information from document {distractor['id']}, but the question asks about document {target_paper['id']}. Therefore, I cannot answer."
            
            example = {
                "id": str(uuid.uuid4()),
                "type": "refusal",
                "source_refs": [distractor['id']],
                "instruction": specific_q,
                "context": f"[Document {distractor['id']}]: {distractor['chunk']}",
                "response": response
            }
            examples.append(example)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for ex in examples:
            f_out.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} refusal examples.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_refusal_qa()