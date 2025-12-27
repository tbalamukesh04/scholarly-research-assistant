import json
import logging
import os
from typing import List

from google import genai
from pipelines.retrieval.search import Retriever
from utils.logging import setup_logger, log_event

class LLM:
    def __init__(self):
        self.client = genai.Client(api_key = os.environ["GEMINI_API_KEY"])
        
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model = "gemini-2.5-flash-lite",
            contents = prompt, 
            config={
                "temperature": 0.2,
                "max_output_tokens": 512,
            }
        )
        return response.text.strip()
        
def format_evidence(evidence: List[dict]) -> str:
    blocks = []
    for e in evidence:
        blocks.append(
            f"[{e['paper_id']}:{e['section']}:{e['chunk_id']}]\n{e['text']}"
        )
    return f"\n\n".join(blocks)
    
def answer(query: str, top_k: int=8, k_min: int=3):
    logger = setup_logger(
        name = "rag_answer", 
        log_dir = "./logs", 
        level = logging.INFO
    )
    retriever = Retriever(top_k=top_k)
    retrieved = retriever.search(query)
    
    from pipelines.retrieval.hydrate import attach_text
    hydrated = attach_text(retrieved)
    
    evidence = hydrated["results"]
    
    if len(evidence) < k_min:
        log_event(
            logger = logger, 
            level = logging.INFO, 
            message = "Refusal: Insufficient Information",
            retrieved = len(evidence)
        )
        
        return {
            "query": query, 
            "answer": "I cannot answer this with the information that I have at the moment.", 
            "evidence": len(evidence),
            "citations": []
        }
        
    prompt = f'''
    You are a scholarly assistant.
    Answer the question with ONLY using the evidence below.
    Every claim must be cited.
    If the evidence is insufficient or contradictory, say so.
    Question:
    {query}
    
    Evidence:
        {format_evidence(evidence)}
    '''
    llm = LLM()
    response = llm.generate(prompt)
    if "does not explicitly" in response.lower() or "does not list" in response.lower():
        return {
            "query": query,
            "answer": "I cannot answer this reliably with the available evidence.",
            "citations": []
        }
    
    return {
        "query": query, 
        "answer": response, 
        "citations": [
            f"{e['paper_id']}:{e['section']}:{e['chunk_id']}"
            for e in evidence
        ]
    }
    
if __name__ == "__main__":
    query = "What are the different types of reasoning questions included in the EgoMAN dataset? "
    results = answer(query, 20, 3)
    print(json.dumps(results, indent=2))