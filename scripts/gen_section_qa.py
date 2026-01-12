import json
import os
import glob
import uuid
import re
import random

# --- Configuration ---
INPUT_DIR = "data/processed/chunks"
OUTPUT_FILE = "data/finetuning/section_qa.jsonl"

# --- Extraction Rules & Filters ---
FORBIDDEN_PHRASES = [
    "this paper", "we propose", "our method", "in this work", 
    "we show", "the rest of the paper", "section", "figure", 
    "table", "dataset available", "code available", "appendix",
    "as shown", "author", "http", "doi", "license", "arxiv", "©"
]

SECTION_MAP = {
    "abstract": ["What is the main contribution of this paper?", "Summarize the abstract."],
    "introduction": ["What problem does this paper address?", "What is the motivation behind this research?"],
    "method": ["How does this paper approach the problem?", "Describe the methodology used."],
    "result": ["What results or findings does this paper present?", "What performance metrics are discussed?"],
    "conclusion": ["What are the key insights from this paper?", "Summarize the conclusions."]
}

def split_sentences(text):
    return re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)

def has_numeric_density(text):
    return bool(re.search(r'[\d%=<>]', text))

def is_factual(text):
    text_lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text_lower:
            return False
    return True

def is_valid_length(text):
    return 80 <= len(text) <= 1200

def check_coherence(instruction, response):
    inst_lower = instruction.lower()
    if "result" in inst_lower:
        if not has_numeric_density(response):
            return False
    return True

def generate_validated_qa():
    print(f"Reading from {INPUT_DIR}...")
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not files:
        print("No input files found.")
        return

    total_candidates = 0
    total_valid = 0
    examples = []
    seen_hashes = set()

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
                # FIX: Access 'section' key for header
                header = section.get("section", "").lower()
                
                # FIX: Iterate through chunks to get text
                chunks = section.get("chunks", [])
                full_section_text = " ".join([c.get("text", "") for c in chunks])
                
                if not full_section_text: continue

                # Match Templates
                matched_templates = []
                for key, templates in SECTION_MAP.items():
                    if key in header:
                        matched_templates = templates
                        break
                
                if not matched_templates: continue

                # Extract Candidates
                sentences = split_sentences(full_section_text)

                for sent in sentences:
                    sent = sent.strip()
                    total_candidates += 1

                    if not is_valid_length(sent): continue
                    if not is_factual(sent): continue
                    if not has_numeric_density(sent): continue
                    
                    sent_hash = hash(sent.lower())
                    if sent_hash in seen_hashes: continue
                    seen_hashes.add(sent_hash)

                    instruction = random.choice(matched_templates)
                    if not check_coherence(instruction, sent): continue

                    total_valid += 1
                    
                    example = {
                        "id": str(uuid.uuid4()),
                        "type": "grounded",
                        "source_refs": [paper_id],
                        "instruction": instruction,
                        "context": f"[Document {paper_id}]: {full_section_text}",
                        "response": sent
                    }
                    examples.append(example)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for ex in examples:
            f_out.write(json.dumps(ex) + "\n")

    rejection_rate = 0
    if total_candidates > 0:
        rejection_rate = ((total_candidates - total_valid) / total_candidates) * 100

    print("-" * 30)
    print(f"Total Candidates Generated: {total_candidates}")
    print(f"Total Valid Examples Kept:  {total_valid}")
    print(f"Rejection Rate:             {rejection_rate:.2f}%")
    print("-" * 30)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_validated_qa()