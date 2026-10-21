# Agents, Prompting and LLMOps

## Prompt engineering basics
- Be explicit about the task, the audience and the output format.
- Give examples (few-shot) when format matters.
- Separate instructions from data using clear delimiters or XML-style tags.
- Ask for structured output (JSON with a schema) when code will parse the response; validate it with Pydantic and retry on failure.
- Chain-of-thought or letting the model reason before answering improves multi-step problems.
- Version your prompts like code and test them against a fixed evaluation set.

## Tool use and function calling
You describe tools with a name, description and JSON schema for parameters. The model decides to call a tool and returns structured arguments; your code executes the tool and sends the result back; the model continues. This is how LLMs query databases, call APIs, or run calculations reliably.

## Agents
An agent is an LLM in a loop: it plans, calls tools, observes results and decides the next step until the task is done. The ReAct pattern alternates reasoning and acting. Key engineering concerns: a maximum step limit, timeouts, clear tool descriptions, idempotent tools, logging every step, and human approval for risky actions such as payments or deletes.
Frameworks you may be asked about: LangChain, LangGraph (graph-based stateful agents), LlamaIndex (data and RAG focus), CrewAI, and vendor SDKs such as the Claude Agent SDK and OpenAI Agents SDK. Interviewers like hearing that you understand the plain loop underneath the framework.

## Model Context Protocol (MCP)
MCP is an open protocol for connecting LLM applications to tools and data sources through a standard client-server interface, so one integration works across many AI apps.

## Agentic RAG
Instead of a single retrieve-then-generate step, the model decides when to search, what to search for, and whether the results are good enough, possibly searching several times. Better for complex questions, but slower and harder to evaluate.

## Security: prompt injection
Prompt injection is when untrusted content (a web page, a document, a user message) contains instructions that hijack the model. Defences: treat retrieved content as data, never as instructions; least-privilege tools; require confirmation for sensitive actions; filter outputs; keep secrets out of prompts; enforce permissions in code, not in the prompt.

## Serving an LLM app with FastAPI
- Async endpoints so I/O waits (LLM API calls, database) do not block workers.
- Stream tokens to the client with Server-Sent Events for a responsive UI.
- Pydantic models for request and response validation.
- Retries with exponential backoff for rate limits (HTTP 429) and timeouts.
- Background jobs (Celery, RQ, or a queue) for document ingestion.
- Cache frequent answers and embeddings.

## LLMOps and observability
- Log prompts, retrieved chunks, model, tokens, latency and cost for every request (tools: Langfuse, LangSmith, Arize Phoenix, OpenTelemetry).
- Track quality with offline evals in CI and online signals such as thumbs up/down.
- Guardrails: input and output validation, PII redaction, topic filters.
- Cost control: smaller models where possible, prompt caching, shorter context, batching.
- Latency: streaming, parallel retrieval, caching, smaller rerank sets.

## System design: "Design a document Q&A assistant for a company"
1. Clarify: users, document types, volume, freshness, permissions, latency target, languages.
2. Ingestion: connectors, parsing (PDF, tables, OCR), chunking, embeddings, vector store with metadata and access-control lists; incremental updates by content hash.
3. Query: auth, query rewriting with chat history, hybrid retrieval with permission filters, rerank, prompt with citations, streaming response.
4. Evaluation: golden set, retrieval and faithfulness metrics, regression tests on every prompt or model change.
5. Operations: observability, cost and latency budgets, fallback model, feedback loop.
Talk through trade-offs at each step; that matters more than naming tools.
