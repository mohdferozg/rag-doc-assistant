# RAG Doc Assistant

A conversational AI assistant that answers questions about a document collection (PDFs) using Retrieval-Augmented Generation (RAG). Built as a full-stack web application with FastAPI, Streamlit, and Google Gemini.

**Document Used:** AWS Whitepaper - *Navigating the security landscape of generative AI*

---

## 🚀 Quick Setup (Local)

**1. Clone & Install**

```bash
git clone https://github.com/mohdferozg/rag-doc-assistant.git
cd rag-doc-assistant
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Environment

bash
cp .env.example .env
Open .env and add your GEMINI_API_KEY.

3. Run the App

Open two terminal windows (both with the venv activated):

bash
# Terminal 1: Backend API
uvicorn src.api:app --reload --port 8000
bash
# Terminal 2: Frontend UI
streamlit run app/streamlit_app.py
Navigate to http://localhost:8501 in your browser.

4. Run Evaluations

bash
python -m eval.run_eval data/security-genai.pdf
🏗️ Architecture Overview
text
[Streamlit UI] ──(HTTP)──> [FastAPI Backend] ──> [RAG Pipeline]
                                                      │
                                                      ├─ 1. PDF Extractor (pypdf)
                                                      ├─ 2. Chunker (1000 chars, 200 overlap)
                                                      ├─ 3. Embedder (gemini-embedding-001)
                                                      ├─ 4. Vector Store (In-memory numpy)
                                                      └─ 5. Generator (gemini-3.5-flash-lite)
🧠 RAG/LLM Approach & Decisions
1. Chunking Strategy
Choice: Character-based recursive chunking (1000 chars, 200 overlap).
Reasoning: Initial tests at 500 chars caused the retriever to miss context for multi-point answers (specifically Question 5 regarding data access management). 1000 chars captures full paragraphs, ensuring all bullet points are retrieved together. The 200-char overlap prevents losing context at chunk boundaries.
Trade-off: Larger chunks reduce vector precision slightly, but significantly increase recall for complex questions.

2. Embedding Model
Choice: gemini-embedding-001.
Reasoning: This was a forced architectural decision. Google officially retired text-embedding-004 in January 2026. gemini-embedding-001 is the current standard, offering 3072 dimensions which provides strong semantic matching for technical text.

3. LLM Selection
Choice: gemini-3.5-flash-lite.
Reasoning: During development, Google's free tier had severe rate limits on the full 3.6-flash model (20 requests/day). 3.5-flash-lite has a much higher free tier limit (1000 requests/day) and is sufficient for simple extractive QA.
Trade-off: Lite models have slightly lower reasoning capability. As noted in our eval results, the Lite model sometimes fails to extract multi-point lists even when the context is present.

4. Retrieval & Vector Store
Choice: In-memory numpy cosine similarity.
Reasoning: For a 19-page PDF producing ~42 chunks, brute-force cosine search takes <5ms. Setting up pgvector or Pinecone would add unnecessary infrastructure overhead for this scale.
Production Note: At scale (10k+ docs), this would migrate to pgvector with HNSW indexing to enable metadata-filtered retrieval and persistence across restarts.

5. Guardrails & Context Management
System Prompt: Strictly instructs the LLM to answer only from the provided context and say "I don't know" otherwise.

Citations: The prompt forces the LLM to append [p. N] to every fact.

Context Management: The top_k parameter is configurable via API (default 5). The prompt includes a numbered list of chunks so the LLM can reference them cleanly.

6. Evaluation & Observability
Choice: LLM-as-a-judge.
Reasoning: Automated evals are critical for catching regressions. A simple keyword match fails on paraphrased answers, so an LLM judge is necessary to measure semantic correctness against our golden set.
Results: Currently passing 4/5 golden questions (80%).

⚙️ Key Technical Decisions
Separated API from UI: FastAPI backend means the RAG logic is decoupled from Streamlit. This allows for a future React/Next.js frontend or CLI tool without rewriting core logic.

Dynamic Model Selection: Google retires models frequently. The src/model_selector.py module dynamically queries the API for available models and picks the best one, preventing hardcoded 404 errors.

Retry Logic: Both the embedder and generator include exponential backoff for 429 (rate limit) and 503 (capacity) errors.

🛠️ Productionization (AWS)
To deploy this for an enterprise AWS environment:

Compute: Deploy FastAPI to AWS ECS Fargate or GCP Cloud Run (serverless, scales to zero).

Vector DB: Migrate the in-memory store to Amazon OpenSearch Serverless (vector engine) or Aurora PostgreSQL with pgvector.

Storage: Store uploaded PDFs in Amazon S3, not local disk.

Auth & Security: Put API Gateway in front of FastAPI. Use IAM for service-to-service auth. Redact PII from logs.

Observability: Instrument with OpenTelemetry, log token usage, and track latency metrics in CloudWatch/Datadog.

CI/CD: GitHub Actions to run pytest + eval.run_eval on every PR. Block merge if accuracy drops below 80%.

🧑‍💻 Engineering Standards (and what I skipped)
Followed:

☑ Typed Pydantic models for API contracts.
☑ Separated configuration (config.py) from logic.
☑ Exponential backoff for API retries.
☑ Git commits follow Conventional Commits (feat:, fix:).
Skipped (for 48h constraint):

□ Dockerization (documented above but not implemented).
□ Unit tests (focused on end-to-end eval harness instead).
□ DB migrations (in-memory store requires no schema).
🤖 How I Used AI Tools
Do's:

Used AI to generate boilerplate (FastAPI endpoints, Pydantic models).

Used AI to explain obscure Python 3.14 wheel build errors.

Used AI as a "rubber duck" when debugging the 503 retry logic.

Don'ts:

I did NOT use AI to write the system prompt or the chunking logic. I manually tuned the prompts by observing retrieval failures and adjusting the instructions myself.

I manually verified every line of code generated by AI by running it and reading the traceback.

🚀 What I'd Do Differently With More Time
Hybrid Search: Add BM25 keyword search alongside vector search to catch exact matches (e.g., "Scope 1").

Parent Document Retrieval: Instead of returning a 1000-char chunk, return the full parent page while embedding smaller child chunks. This solves Q1 (the multi-point answer).

CI/CD: Set up GitHub Actions to run the eval on every push.

Dockerization: Add a docker-compose.yml for one-command deployment.

🧠 Notes on LLM Use (For the Reviewer)
This project was built over a 48-hour sprint. The biggest challenge was not writing the RAG logic—it was dealing with the rapidly shifting Gemini API ecosystem (retired models, Python 3.14 build errors, and aggressive free-tier rate limits). I deliberately chose a simple in-memory vector store and a decoupled FastAPI architecture to ensure the core RAG mechanics were robust and testable within the time constraint, rather than over-engineering the infrastructure. The evaluation harness was the most valuable addition, as it proved that the guardrails correctly refused to answer out-of-scope questions.
