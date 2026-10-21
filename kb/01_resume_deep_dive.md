# Resume Deep Dive — Suguna P J

## How to pitch yourself in 60 seconds
I am a full-stack engineer with close to two years of experience, mostly Python (Django, FastAPI) and React. At Operforce AI I built business modules on a multi-module enterprise platform: CRM, HR, inventory, sales and manufacturing. Before that I built full-stack apps with FastAPI and PostgreSQL at Tellme DigiInfotech. On the AI side I built a RAG assistant on top of NABARD microfinance reports and an ML classifier for E. coli promoter detection, and I use AI coding tools daily. I now want to move from building apps that call AI to building the AI systems themselves: retrieval, evaluation, and LLM pipelines in production.

## Timeline questions to prepare for
- Tellme DigiInfotech: Dec 2024 to Aug 2025 (Database Intern).
- Redsage Global: Aug 2025 to Oct 2025 (about three months). Be ready to explain the short stint calmly and in one or two sentences, without blaming anyone. Focus on what you shipped there and why the next role fit better.
- Gap Oct 2025 to Jan 2026. Have a truthful one-line answer ready, such as upskilling, projects, or job search, and name one concrete thing you built or learned.
- Operforce AI: Jan 2026 to Jul 2026. Why you left: state it neutrally and forward-looking.
- Verve Studio: Jul 2026 to present. Interviewers may ask whether you will keep freelancing. Answer clearly: you want a full-time AI engineering role, and freelancing showed you own work end to end.

## Operforce AI — likely questions
- Walk me through the architecture of the platform. Answer: React frontend talking to Django REST Framework APIs, PostgreSQL behind it, modules sharing a design system and a role-based menu.
- How did you prevent duplicate submissions in CRM visit tracking? Answer at two layers: frontend disables the submit button and tracks an in-flight flag; backend uses an idempotency key or a unique constraint and returns the existing record on a repeat. Mention that only the backend guard is truly reliable.
- How did you implement payroll and penalty rules for leave? Talk about modelling rules as data instead of hardcoded if-statements, validating in serializers, and writing tests for edge cases like half days and overlapping leave.
- Dependent dropdowns and prefill: explain fetching child options when the parent changes, cancelling stale requests, and prefilling edit forms from the API.
- How did you use Claude Code, Codex and Copilot? Give a concrete example: refactoring a large component, generating test cases, or debugging. Also say how you verify AI-generated code: read the diff, run tests, check edge cases.

## Bharat Microfinance Report Platform (NABARD) — the most important project for an AI role
Expect deep questions here. Know every layer.
- What was the source data? PDF reports with tables and narrative text.
- Ingestion: how did you extract text and tables from PDFs? Mention tools such as pdfplumber or PyMuPDF and that tables need special handling (convert to rows or markdown before chunking).
- Chunking: what chunk size and overlap, and why? A defensible answer is 300 to 800 tokens with 10 to 20 percent overlap, split on headings first.
- Embeddings and storage: which embedding model and which vector store? If you used pgvector in PostgreSQL, explain why: one database for app data and vectors, metadata filters by region and year in plain SQL.
- Retrieval: top-k, metadata filters (state, district, year), and whether you used hybrid search or reranking.
- Generation: the prompt template, how you forced citations, and what happens when the answer is not in the context.
- Evaluation: how did you know it worked? If you did not formally evaluate, say so honestly and describe how you would: a golden question set, recall@k for retrieval, faithfulness for answers.
- Be precise about your role and scope. If it was an academic or prototype project, say so. Interviewers respect honesty and punish inflated claims.

## Business Geo-Spread Application
- Leaflet map with filters by category, state, district and pin code.
- Likely questions: how do you render thousands of markers fast (clustering, server-side aggregation by region), how did you index the PostgreSQL tables for those filters (composite indexes), and would you use PostGIS for radius queries.

## E. coli Detection Using DNA Promoter Classification
- Dataset: the classic UCI promoter gene sequences dataset (106 sequences of 57 nucleotides, labelled promoter or non-promoter), if that is what you used. Confirm your own numbers.
- Preprocessing: one-hot encoding of A, C, G, T per position gives 57 x 4 = 228 features.
- Model: MLP classifier. Be ready to explain hidden layers, activation (ReLU), loss (cross-entropy), optimizer (Adam), and why a small dataset needs cross-validation and regularization.
- Why accuracy alone is not enough: report precision, recall, F1 and a confusion matrix.
- Served via Django: model loaded once at startup, prediction endpoint validates input sequence length and characters.

## Publication — Epigenetic Biomarker Discovery
- Know the one-paragraph summary: problem, data, methods, findings, limitation.
- Likely question: how did you handle high-dimensional data with few samples? Answer: feature selection, regularization, cross-validation, and caution about overfitting.

## Verve Studio
- Shows ownership: requirements, design, build, SEO, deployment.
- Connect it to AI engineering: you talk to non-technical clients, translate fuzzy needs into specs, and ship. That is exactly what applied AI work needs.
