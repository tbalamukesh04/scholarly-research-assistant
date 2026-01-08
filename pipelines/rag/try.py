from pipelines.rag.answer import answer
import json
import time
from pipelines.retrieval.search import Retriever

with open("pipelines/evaluation/data/eval_queries.json", "r") as f:
    input = json.load(f)

retriever = Retriever()
accepted = []
for i in range(1, 10):
    query = input[i]["query"]
    response = answer(query=query, retriever=retriever)
    if not response["metrics"]["refused"]:
        accepted.append(input[i]["id"])
        
    time.sleep(3)

print(accepted)