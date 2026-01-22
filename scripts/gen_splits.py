import json
import glob
import os
import random

INPUT_DIR = "data/finetuning"
OUTPUT_DIR = "data/finetuning/splits"

SEED = 42
SPLIT_RATIOS = (0.80, 0.15, 0.05)

def load_jsonl(filename):
    data = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
    return data
    
def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def merge_and_shuffle():
    print("----MERGING DATA----")
    
    grounded_data = load_jsonl(os.path.join(INPUT_DIR, "section_qa.jsonl"))
    partial_data = load_jsonl(os.path.join(INPUT_DIR, "partial_qa.jsonl"))
    refusal_data = load_jsonl(os.path.join(INPUT_DIR, "refusal_qa.jsonl"))
    
    print(f"Loaded {len(grounded_data)} Grounded QA examples")
    print(f"Loaded {len(partial_data)} Partial QA examples")
    print(f"Loaded {len(refusal_data)} Refusal QA examples")
    
    all_examples = grounded_data + partial_data + refusal_data
    if not all_examples:
        print("CRITICAL: No examples found. Check input paths and previous steps")
        return
        
    random.seed(SEED)
    random.shuffle(all_examples)
    
    total = len(all_examples)
    train_end = int(total * SPLIT_RATIOS[0])
    val_end = int(total * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
    
    train_set = all_examples[:train_end]
    val_set = all_examples[train_end: val_end]
    test_set = all_examples[val_end:]
    
    save_jsonl(train_set, os.path.join(OUTPUT_DIR, 'train.jsonl'))
    save_jsonl(val_set, os.path.join(OUTPUT_DIR, 'val.jsonl'))
    save_jsonl(test_set, os.path.join(OUTPUT_DIR, 'test.jsonl'))
    
    print("\n ---Final Statistics---")
    print(f"Total Examples: {total}")
    print(f"Train: {len(train_set)} ({len(train_set)/total*100:.1f}%)")
    print(f"Val: {len(val_set)} ({len(val_set)/total*100:.1f}%)")
    print(f"Test: {len(test_set)} ({len(test_set)/total*100:.1f}%)")
    
    print(f"\nSaved to {OUTPUT_DIR}/")
    
if __name__ == "__main__":
    merge_and_shuffle()