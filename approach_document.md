# SHL Assessment Recommender — Approach Document

## Design Choices

### Architecture

The system is a stateless FastAPI service with a single conversation endpoint (`POST /chat`) that sends the full conversation history to Claude (Anthropic's `claude-sonnet-4-20250514`) on every request. No vector store or retrieval layer is used; instead, the entire SHL Individual Test Solutions catalog is embedded directly in the system prompt as a structured text block. This is the core architectural decision and is justified below.

**Why no RAG?** The catalog has ~50 assessments (Individual Test Solutions only, per the spec). At ~500 tokens each, the full catalog fits in Claude's context window with room to spare, well within the 30-second timeout. Embedding the full catalog eliminates retrieval failures, ranking errors, and embedding drift — all of which hurt Recall@10 on small catalogs far more than retrieval latency helps. For a catalog of 500+ items, RAG would be the right call; at ~50, it adds risk.

### Catalog Construction

The SHL product catalog page is JavaScript-rendered, so Playwright or a headless browser would be needed for a proper scrape. As a fallback, I built a curated JSON catalog of SHL Individual Test Solutions (covering all major product lines: Verify cognitive battery, OPQ personality suite, Knowledge & Skills tests, Coding Simulations, and Behavioral assessments) by combining:
- Known product data from SHL's public product pages (fetched via canonical URLs)
- Known assessment metadata (test types, job levels, keywords) from SHL documentation

Each catalog entry includes: `name`, `url`, `description`, `test_types` (A/P/K/S/B), `job_levels`, `remote_testing`, `adaptive_irt`, and `keywords` for semantic matching in the prompt.

### Context Engineering

The system prompt encodes the catalog in a consistent structure and gives Claude strict behavioral rules:

1. **Clarify before recommending** — require role and seniority before returning a shortlist
2. **Recommend 1–10 items** from the catalog only
3. **Refine** mid-conversation when constraints change
4. **Compare** using only catalog data
5. **Stay in scope** — refuse off-topic, legal, and prompt-injection requests

Claude is instructed to respond exclusively in JSON matching the required schema. A post-processing layer strips markdown fences and validates JSON, with a regex fallback for partial-JSON responses.

### Hallucination Guard

Every recommendation URL returned by the LLM is checked against the known catalog URL set. Recommendations with URLs not in the catalog are dropped. If a known name is matched to a wrong URL, the URL is corrected from the catalog. This means the system can never return a hallucinated assessment to the evaluator.

### Agent Design

The clarification policy is: ask at most 2 questions before committing to a shortlist. This balances the need for context with the 8-turn conversation cap. The agent extracts role type and seniority as the two minimum required signals. Job family (IT, sales, finance, etc.) is inferred from role description when not explicit.

---

## What Didn't Work

**RAG with FAISS on the scraped catalog.** An early prototype built TF-IDF and sentence-transformer embeddings over catalog descriptions, then retrieved top-k items and fed them to the LLM. This underperformed the full-catalog-in-prompt approach because: (a) the catalog is small enough to fit in context, (b) retrieval sometimes missed key assessments when the user phrased queries differently than the catalog keywords, and (c) the added latency risked the 30-second timeout.

**Strict JSON mode via function calling.** Using tool-use to enforce schema compliance introduced parsing overhead and sometimes caused the model to refuse valid refinement turns. Asking the model to "respond only in JSON" proved more robust for conversational flows.

---

## Evaluation Approach

I tested three categories against the local service:

**Hard evals (schema compliance):** Assert every response has `reply` (str), `recommendations` (list), and `end_of_conversation` (bool); every URL contains `shl.com`; max 10 recommendations; response within 30s.

**Behavior probes:** Vague query → clarify with no recommendations; off-topic question → refuse; prompt injection → ignore; comparison question → produce grounded comparison from catalog; mid-conversation refinement → update shortlist preserving valid prior entries.

**Recall@10 proxy:** Manually labeled expected shortlists for 5 synthetic personas (Java dev, Python data scientist, entry-level customer service, senior sales manager, graduate finance analyst) and measured fraction of expected assessments appearing in the returned shortlist.

## Tools Used

Claude (Anthropic API) for LLM; FastAPI + Uvicorn for the web service; Pydantic for schema validation; Render for deployment. Claude (claude.ai) was used for code assistance and iteration.
