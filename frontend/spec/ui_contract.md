# UI/Backend Interface Contract v1.0

## Overview
This document defines the strict JSON schema for the `/query` endpoint. The frontend treats this contract as the source of truth for rendering the Answer Surface, Evidence Deck, and System Status.

## Endpoint Definition
**POST** `/query`
**Content-Type:** `application/json`

### Request Schema
```json
{
  "query": "string",
  "top_k": 10,
  "mode": "strict", 
  "eval_mode": false
}```

### Root Object
```json
{
  "query": "string",
  "answer": "string | null", 
  "answer_sentences": [
    {
      "text": "string",
      "verification_status": "supported | unsupported",
      "citation_indices": [1, 2] 
    }
  ],
  "citations": [
    {
      "citation_id": 1, 
      "paper_id": "string",
      "section": "string",
      "text": "string (snippet)",
      "score": 0.0  // Alignment Confidence
    }
  ],
  "metrics": {
    "refused": boolean,
    "refusal_reason": "string | null",
    "confidence_score": 0.0, // Global Score (0.0 - 1.0)
    "total_latency": 0.0,
    "retrieval_latency": 0.0,
    "llm_latency": 0.0,
    "retrieved_chunks": 0,
    "truncated": false,
    "dropped_sentences": 0
  },
  "dataset_hash": "string",
  "index_hash": "string | null",
  "run_id": "string | null"
}```