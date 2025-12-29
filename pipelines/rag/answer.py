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
        '''
        Generates a response to the given prompt using the Gemini API.
        Args:
            prompt (str): The prompt to generate a response for.
        Returns:
            str: The generated response.
        '''
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
    '''
    Formats the evidence into a readable string.
    Args:
        evidence (List[dict]): The evidence to format.
    Returns:
        str: The formatted evidence.
    '''
    blocks = []
    for e in evidence:
        blocks.append(
            f"[{e['paper_id']}:{e['section']}:{e['chunk_id']}]\n{e['text']}"
        )
    return f"\n\n".join(blocks)
    
def answer(query: str, top_k: int=8, k_min: int=3, mode: str = "strict", retriever=None):
    '''
    Answers the given query using the RAG pipeline.
    Args:
        query (str): The query to answer.
        top_k (int): The number of responses to retrieve.
        k_min (int): The minimum number of responses required to answer the query.
        mode (str): The mode of the answer. Can be "strict" or "loose".
    Returns:
        dict: The answer and evidence.
    '''
    logger = setup_logger(
        name = "rag_answer", 
        log_dir = "./logs", 
        level = logging.INFO
    )
    if retriever is None:
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
    
    if mode == "strict":
        instruction = (
            "Answer the question with ONLY using the evidence below."
            "If the evidence is insufficient or contradictory, say so."
            "Every Claim Must be cited."
            
        )
    elif mode == "loose":
        instruction = (
            "Answer using the evidence below. "
            "You may synthesize across sources, but you MUST label the answer as SYNTHESIS. "
            "Every claim must be cited."
        )
    
        
    prompt = f'''
    You are a scholarly assistant.
    Instruction:
    {instruction}
    
    Question:
    {query}
    
    Evidence:
        {format_evidence(evidence)}
    '''
    llm = LLM()
    response = llm.generate(prompt)
    if mode == "synthesis" and not response.strip().lower().startswith("synthesis"):
        response = "SYNTHESIS: " + response                                                 
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
    query = "How does Pooling by Multihead Attention (PMA) differ from EOS and mean pooling for code embeddings?"
    results = answer(query, 20, 3)
    print(json.dumps(results, indent=2))