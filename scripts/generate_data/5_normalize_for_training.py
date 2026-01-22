import json
import os
import random
import glob

INPUT_GENERATED = "dataset_training_master.jsonl"
OUTPUT_DIR = "data/finetuning/final_splits"
TARGET_REFUSAL_RATIO = 0.2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def normalize_manual_entry(entry):
    return {
        "instruction": entry.get("instruction", ""), 
        "input": entry.get("context", ""),
        "output": entry.get("response", "")
    }
    
def normalize_generated_entry(entry):
    evidence_list = entry.get("evidence_used", [])
    formatted_context = ""
    for i, ev in enumerate(evidence_list):
        text = ev.get("text", "").strip()
        formatted_context += f"[{i+1}] {text}\n\n"
    
    return {
        "instruction": entry.get("query", ""),
        "input": formatted_context.strip(), 
        "output": entry.get("response", "")
    }
    
def main():
    ensure_dir(OUTPUT_DIR)
    print(f"Loading generated data from {INPUT_GENERATED}...")
    generated_accepted = []
    generated_refusals = []
    
    try:
        with open(INPUT_GENERATED, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                norm = normalize_generated_entry(data)
                label = data.get("label", "unknown")
                
                if label == "accepted":
                    generated_accepted.append(norm)
                elif label == "refusal":
                    generated_refusals.append(norm)
                    
        
    except FileNotFoundError:
        print(f"Warning: {INPUT_GENERATED} not found.")
        
    print(f"  > Raw Counts: {len(generated_accepted)} Accepted ,{len(generated_refusals)} Rejected.")
    
    manual_data = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        filename = f"{split}.jsonl"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    manual_data[split].append(normalize_manual_entry(json.loads(line)))
    
    target_refusal_count = int(len(generated_accepted)*0.5)
    
    random.seed(42)
    random.shuffle(generated_refusals)
    keep_refusals = generated_refusals[:target_refusal_count]
    
    print(f"  > Balancing Downsampling Refusals from {len(generated_refusals)} -> {len(keep_refusals)}")
    
    random.shuffle(generated_accepted)
    split_acc = int(len(generated_accepted) * 0.9) 
    split_ref = int(len(keep_refusals) * 0.8)
    
    final_train = manual_data["train"][:]
    final_train.extend(generated_accepted[:split_acc])
    final_train.extend(keep_refusals[:split_ref])
    
    final_val = manual_data["val"][:]
    final_val.extend(generated_accepted[split_acc:])
    final_val.extend(keep_refusals[split_ref:])
    
    final_test = manual_data["test"][:]
    
    datasets = {"train.jsonl": final_train, "val.jsonl": final_val, "test.jsonl": final_test}
    
    for fname, data in datasets.items():
        out_path = os.path.join(OUTPUT_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
            
            print(f"Saved {fname}: {len(data)} examples")

if __name__ == "__main__":
    main()