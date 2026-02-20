# UI Layout & Information Hierarchy

## 1. Primary Container
**Layout Strategy:** Single-column vertical flow with a dedicated evidence sidebar (desktop) or collapsible bottom drawer (mobile).

### Zone A: Control Center (Top)
* **Input Field:** Multi-line textarea for complex queries.
* **Action:** Primary "Submit" button.
* **Configuration:** * Mode Selector: Toggle between "Strict" (High Precision) and "Exploratory" (High Recall). [Future Scope]

### Zone B: System Status (Dynamic Injection)
* **Operational State:** Spinner/Progress bar during processing.
* **Refusal Banner:** * *Condition:* Visible only when `refused=true`.
    * *Content:* "Answer Refused: [Refusal Reason]".
    * *Style:* Distinct warning color (Amber/Red).
* **Confidence Badge:** * *Content:* Global Confidence Score (0.0 - 1.0).
    * *Style:* Color-coded (Green > 0.8, Yellow > 0.5, Red < 0.5).

### Zone C: Answer Surface (Central Panel)
* **Content:** The generated answer text.
* **Granularity:** * Text is segmented by sentence.
    * *Visual State:* Each sentence carries a visual indicator of its `verification_status` (Supported vs. Unsupported).
* **Interactivity:** * Hovering a sentence highlights the corresponding citation in Zone D.

### Zone D: Evidence Deck (Sidebar/Drawer)
* **Header:** "References & Grounding".
* **List Items (Citations):**
    * **Identifier:** `[1]`, `[2]`, etc. matches text markers.
    * **Metadata:** Paper ID, Section Name.
    * **Snippet:** The specific chunk text used for grounding.
    * **Score:** Alignment confidence per citation.
* **State:** * Items dim/highlight based on Zone C interaction.

### Zone E: Diagnostics Console (Footer/Toggle)
* **Visibility:** Collapsed by default. Expandable via "Debug" link.
* **Telemetry:**
    * Total Latency
    * Retrieval Latency
    * LLM Latency
    * Dataset Hash (Version Control)

## 2. Data Binding & Schema Requirements

| UI Zone | Component | Backend Source Field (JSON Path) | Requirement Status |
| :--- | :--- | :--- | :--- |
| **Zone B** | Refusal Banner | `response.metrics.refused` (bool) | Existing |
| | Refusal Reason | `response.metrics.refusal_reason` (str) | Existing |
| | Confidence Score | `response.metrics.confidence_score` (float) | Existing |
| **Zone C** | Answer Text | `response.answer` (str) | Existing |
| | Sentence Verification | `response.answer_sentences[i].verification_status` | **MISSING** (Needs Schema Update) |
| | Sentence Text | `response.answer_sentences[i].text` | **MISSING** (Needs Schema Update) |
| | Sentence Citations | `response.answer_sentences[i].citations` | **MISSING** (Needs Schema Update) |
| **Zone D** | Citation ID | `response.citations[j].citation_id` | Existing (as `chunk_id`) |
| | Paper ID | `response.citations[j].paper_id` | Existing |
| | Section | `response.citations[j].section` | Existing |
| | Snippet (Text) | `response.citations[j].text` | **MISSING** (Needs Schema Update) |
| | Alignment Score | `response.citations[j].score` | Existing (Placeholder 0.0) |
| **Zone E** | Total Latency | `response.metrics.total_latency` | Existing |
| | Retrieval Latency | `response.metrics.retrieval_latency` | Existing |
| | LLM Latency | `response.metrics.llm_latency` | Existing |
| | Dataset Hash | `response.dataset_hash` | Existing |