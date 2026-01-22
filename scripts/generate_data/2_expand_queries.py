import json
import itertools

def expand_templates():
    with open("data_templates.json", "r") as f:
        templates = json.load(f)
        
    sections = [
        "Abstract", "Introduction", "Related Work", 
        "Methodology", "Experiments", "Results", 
        "Discussion", "Conclusion"
    ]
    
    topics = [
        "latency", "accuracy", "transformer architecture", 
        "loss function", "hyperparameters", "training data", 
        "bias", "limitations", "ablation study", "state-of-the-art", 
        "GPU usage", "preprocessing"
    ]
    
    expanded_queries = []
    
    print("Expanding templates...")
    for temp in templates:
        for sec in sections:
            for top in topics:
                query = temp.replace("{section}", sec).replace("{topic}", top)
                expanded_queries.append(query)
                
    output_file = "data_candidates.json"
    with open(output_file, "w") as f:
        json.dump(expanded_queries, f, indent=2)
        
    print(f"Generated {len(expanded_queries)} candidate queries. Saved to {output_file}.")
    
if __name__ == "__main__":
    expand_templates()