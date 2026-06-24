# Resume Bullets — RAG Testing Framework

## Option A — LLM Evaluation Engineer focus

- Built a RAG evaluation framework from scratch (Python, pytest) implementing six
  quality metrics — context precision, context recall, faithfulness, hallucination
  score, answer relevance, and citation validation — with 60+ tests and GitHub
  Actions CI integration

## Option B — AI QA Engineer focus

- Developed and tested a complete Retrieval-Augmented Generation pipeline covering
  document ingestion, three chunking strategies, hash-based embeddings, and cosine
  similarity retrieval; implemented RAG evaluation metrics aligned with RAGAS
  methodology without external API dependencies

## Option C — GenAI QA / AI SDET focus

- Designed a reusable RAG metrics library (faithfulness, hallucination detection,
  citation validation) built on token-overlap scoring; structured as composable
  pytest fixtures so each metric can gate CI pipelines independently

## Notes

- Strong talking point: "The metrics are framework-agnostic — they work with any
  RAG pipeline because they evaluate the inputs/outputs, not the framework itself"
- Follow-up answer when asked about RAGAS: "My implementations mirror the RAGAS
  metric definitions. The next step is to add a ragas_integration.py that runs
  the same test cases through the actual RAGAS library for comparison"
