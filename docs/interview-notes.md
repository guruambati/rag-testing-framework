# Interview Notes — RAG Testing Framework

## What I Built

A complete RAG pipeline built from scratch plus a metrics layer that evaluates
retrieval and generation quality. Built to learn how RAG systems work internally
and develop the judgment needed to test them professionally.

The pipeline goes: raw text → DocumentIngester → TextChunker → EmbeddingEngine
→ InMemoryVectorStore. The metrics layer evaluates context precision, context recall,
faithfulness, hallucination score, answer relevance, and citation validation.

No paid API key required — uses deterministic hash-based embeddings so every test
is repeatable in CI without network access.

## How I Would Explain It in an Interview

> "RAG systems have two failure surfaces: the retriever and the generator.
> If the retriever brings back the wrong chunks, the LLM can't answer correctly
> no matter how good it is. If the generator ignores the context and hallucinates,
> even perfect retrieval won't help.
>
> I built a testing framework that evaluates both surfaces with six metrics:
> context precision and recall for retrieval quality, faithfulness and hallucination
> score for generation quality, answer relevance for topic alignment, and citation
> validation for grounding.
>
> The pipeline itself is built from scratch — ingestion, chunking with three
> strategies, embedding, and cosine similarity search. This was intentional: I wanted
> to understand what's happening inside before testing it."

## What QA Problem It Solves

1. **Retriever regression** — chunking strategy or embedding model change breaks recall
2. **Silent hallucination** — LLM starts inventing facts not in retrieved context
3. **Citation drift** — model cites sources that don't support its claims
4. **Threshold alerting** — quality metrics drop below acceptable levels without notice
5. **CI integration** — catch retrieval/generation regressions on every push

## Key Design Decisions

**Why build from scratch instead of using LangChain?**
LangChain abstracts away the components I needed to test. Building from scratch meant
I understood exactly what to assert at each stage. In an interview: "I can also
integrate these metrics into a LangChain-based RAG pipeline — the metrics are
framework-agnostic."

**Why hash-based embeddings?**
Real embeddings (sentence-transformers) require downloading models and GPU/internet
access. Hash embeddings are deterministic and fast, making tests reliable in CI. The
EmbeddingEngine has a `use_real_model=True` flag that swaps in real embeddings when
available.

**Why separate metrics modules?**
Each metric can be used independently or composed into a full evaluation run. This
mirrors how RAGAS and DeepEval are structured — metrics are composable building blocks.

## What I Would Add Next

1. **RAGAS integration** — run the same test cases through the real RAGAS library
   and compare scores to validate my implementations
2. **Async retrieval** — test concurrent query handling
3. **Chunking ablation tests** — compare fixed vs sentence vs paragraph strategies
   against the same golden Q&A pairs to find the best strategy per document type
4. **LLM-as-judge faithfulness** — replace lexical overlap with an LLM scoring call
   for production-grade faithfulness evaluation
5. **Embedding drift detection** — alert when the embedding model changes and cosine
   distances shift significantly
