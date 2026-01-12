import json
import os
import glob
import uuid
import re
import random

# --- Configuration ---
INPUT_DIR = "data/processed/chunks"
OUTPUT_FILE = "data/finetuning/partial_qa.jsonl"

BROAD_INSTRUCTIONS = {
    "introduction": [
        "Provide a comprehensive overview of the introduction.",
        "Explain the background and motivation in detail.",
        "What is the full context established in this paper?"
    ],
    "method": [
        "Detail the complete methodology used in this work.",
        "Explain the experimental setup and model architecture fully.",
        "Step-by-step explanation of the proposed approach."
    ],
    "result": [
        "List all the experimental results and findings.",
        "Provide a detailed breakdown of the performance metrics.",
        "What are the full outcomes reported in the results section?"
    ],
    "discussion": [
        "Discuss all the implications and limitations mentioned.",
        "Provide the full analysis from the discussion section."
    ]
}

def split_sentences(text):
    if not text: return []
    text = re.sub(r'\s+', ' ', text).strip()
    return re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)

def generate_partial_qa():
    print(f"Reading from {INPUT_DIR}...")
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not files:
        print("No input files found.")
        return

    examples = []
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except:
                continue

        paper_id = data.get("paper_id", os.path.basename(file_path).replace(".json", ""))
        sections = data.get("sections", [])
        
        if isinstance(sections, list):
            for section in sections:
                # FIX: Correct key for section name
                header = section.get("section", "").lower()
                
                # FIX: Aggregate text from chunks
                chunks = section.get("chunks", [])
                content = " ".join([c.get("text", "") for c in chunks])
                
                if not content: continue

                section_type = None
                for key in BROAD_INSTRUCTIONS.keys():
                    if key in header:
                        section_type = key
                        break
                
                if not section_type: continue

                sentences = split_sentences(content)
                if len(sentences) < 4: continue

                cut_idx = random.randint(int(len(sentences)*0.3), int(len(sentences)*0.7))
                cut_idx = max(2, cut_idx)
                
                truncated_text = " ".join(sentences[:cut_idx])
                instruction = random.choice(BROAD_INSTRUCTIONS[section_type])

                example = {
                    "id": str(uuid.uuid4()),
                    "type": "partial",
                    "source_refs": [paper_id],
                    "instruction": instruction,
                    "context": f"[Document {paper_id}]: {truncated_text}", 
                    "response": truncated_text
                }
                examples.append(example)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for ex in examples:
            f_out.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} partial/truncated examples.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_partial_qa()