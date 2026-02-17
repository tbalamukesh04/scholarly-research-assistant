import json
import sys
import re
from pathlib import Path

# Add root to path
sys.path.append('.')

from pipelines.retrieval.search import Retriever
from pipelines.postprocess.confidence import ConfidenceScorer

def diagnose():
    q_path = Path('pipelines/evaluation/data/eval_queries.json')
    if not q_path.exists():
        print(f'? Queries not found at {q_path}')
        return

    with open(q_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print('Initializing Retriever...')
    try:
        retriever = Retriever(top_k=3)
        scorer = ConfidenceScorer()
    except Exception as e:
        print(f'? Failed to init: {e}')
        return

    # Check the first query
    item = queries[0]
    q = item['query']
    relevant = item['relevant_papers']
    
    print(f'\n--- DIAGNOSIS FOR QUERY: {q[:40]}... ---')
    print(f'Expected Relevant IDs (from JSON): {relevant}')
    
    results = retriever.search(q)
    retrieved_ids = [r['paper_id'] for r in results['results']]
    
    print(f'Retrieved IDs (from Index):      {retrieved_ids}')
    
    print('\n--- NORMALIZATION CHECK ---')
    for r_id in retrieved_ids:
        print(f'Retrieved Raw: "{r_id}" -> Norm: "{scorer._normalize_id(r_id)}"')
        
    for rel_id in relevant:
        print(f'Expected  Raw: "{rel_id}" -> Norm: "{scorer._normalize_id(rel_id)}"')

if __name__ == '__main__':
    diagnose()
