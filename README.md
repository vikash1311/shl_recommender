# 🧠 SHL Assessment Recommender

> An AI-powered conversational recommender system for SHL Individual Test Solutions — built with FastAPI and Claude (claude-sonnet), deployed on Render via Docker.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Anthropic](https://img.shields.io/badge/Claude-Sonnet-orange?style=for-the-badge&logo=anthropic)](https://anthropic.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render)](https://render.com/)

---

## 📌 Overview

This project was built as part of the **SHL technical assessment**. It is a stateless conversational AI agent that recommends the right SHL assessments for a given role and seniority level. The agent clarifies requirements through conversation, then returns a structured JSON shortlist of up to 10 relevant assessments from the SHL Individual Test Solutions catalog.

**Key design decision:** Instead of RAG + vector retrieval, the entire ~50-item SHL catalog is embedded directly in the system prompt. At this catalog size, full-context injection eliminates retrieval failures and ranking errors that would otherwise hurt Recall@10. RAG would be the right call at 500+ items.

---

## ✨ Features

- **Conversational agent** — clarifies role type and seniority before committing to a shortlist (max 2 clarifying questions)
- **Claude-powered recommendations** — uses `claude-sonnet-4-20250514` with a structured system prompt encoding the full SHL catalog
- **Hallucination guard** — all returned URLs are validated against the known catalog; hallucinated assessments are dropped or URL-corrected automatically
- **Strict JSON output** — every response conforms to `{ reply, recommendations, end_of_conversation }` schema, validated by Pydantic
- **Stateless architecture** — full conversation history is sent on every request; no session storage required
- **Prompt injection resistance** — agent refuses off-topic, legal, and injection requests by design
- **Docker + Render ready** — one-command containerized deployment

---

## 🛠️ Tech Stack

| Layer            | Technology                                      |
|------------------|-------------------------------------------------|
| Web Framework    | FastAPI 0.115 + Uvicorn                         |
| LLM              | Anthropic Claude (`claude-sonnet-4-20250514`)   |
| Schema Validation| Pydantic v2                                     |
| HTTP Client      | httpx                                           |
| Config           | python-dotenv                                   |
| Containerization | Docker                                          |
| Deployment       | Render (Python web service)                     |

---

## 📁 Project Structure

```
shl_recommender/
├── main.py                 # FastAPI app — POST /chat endpoint, LLM orchestration
├── catalog.json            # Curated SHL Individual Test Solutions catalog (~50 items)
├── test_agent.py           # Evaluation suite (schema, behavior, Recall@10 probes)
├── approach_document.md    # Full design rationale and architecture decisions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
└── render.yaml             # Render deployment config (build + start + env vars)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/vikash1311/shl_recommender.git
cd shl_recommender

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## 🐳 Docker

```bash
# Build the image
docker build -t shl-recommender .

# Run the container
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key_here shl-recommender
```

---

## 📡 API Reference

### `POST /chat`

Send a conversation turn to the agent. Include the full message history on every request.

**Request body:**
```json
{
  "messages": [
    { "role": "user", "content": "I need assessments for a senior Java developer role." }
  ]
}
```

**Response schema:**
```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "description": "string",
      "test_types": ["A", "P", "K"],
      "remote_testing": true,
      "adaptive_irt": false
    }
  ],
  "end_of_conversation": false
}
```

**Behavior:**
- If `role` or `seniority` is missing, the agent asks up to 2 clarifying questions before returning recommendations
- Returns 1–10 assessments from the catalog only
- Refines shortlist mid-conversation when constraints change
- `end_of_conversation: true` when the agent has finished the interaction

---

## 🧪 Evaluation

Run the test suite against the local server:

```bash
python test_agent.py
```

Tests cover three categories:

**Schema compliance** — every response has correct types, URLs contain `shl.com`, max 10 recommendations, response within 30s.

**Behavior probes** — vague query → clarify; off-topic → refuse; prompt injection → ignore; comparison → catalog-grounded response; mid-conversation refinement → updated shortlist.

**Recall@10 proxy** — manually labeled expected shortlists for 5 synthetic personas (Java dev, Python data scientist, entry-level customer service, senior sales manager, graduate finance analyst).

---

## ☁️ Deployment on Render

The `render.yaml` configures a Python web service on Render:

```yaml
services:
  - type: web
    name: shl-recommender
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false
```

To deploy:

1. Fork this repository
2. Connect it to your [Render](https://render.com/) account
3. Create a new **Web Service** and point it to the repo
4. Add `ANTHROPIC_API_KEY` as an environment variable under **Environment**
5. Render will auto-deploy on every push to `main`

---

## 🧩 Architecture Notes

See [`approach_document.md`](./approach_document.md) for the full design rationale, including:

- Why full-catalog-in-prompt was chosen over RAG/FAISS
- Catalog construction methodology (SHL public product pages + known metadata)
- Context engineering and behavioral rules baked into the system prompt
- What was tried and didn't work (TF-IDF embeddings, strict function calling)
- Evaluation methodology and Recall@10 results

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Vikash Gautam**
- GitHub: [@vikash1311](https://github.com/vikash1311)
- Portfolio: [vikash-gautam.netlify.app](https://vikash-gautam.netlify.app/)
