"""conftest.py — shared fixtures for RAG tests."""

import pytest
from src.pipeline.rag_system import RAGSystem

PYTHON_DOC = """
Python is a high-level, general-purpose programming language.
Its design philosophy emphasizes code readability with significant indentation.
Python is dynamically typed and garbage-collected.
It supports object-oriented, procedural, and functional programming paradigms.
Python was created by Guido van Rossum and first released in 1991.
It is widely used in web development, data science, machine learning, and automation.
"""

CLOUD_DOC = """
Cloud computing delivers on-demand computer system resources over the internet.
AWS, Azure, and Google Cloud Platform are the three largest cloud providers.
Services are divided into IaaS, PaaS, and SaaS categories.
Kubernetes is widely used to orchestrate containerized workloads in the cloud.
Serverless computing allows developers to run code without managing servers.
"""

AI_DOC = """
Artificial intelligence is intelligence demonstrated by machines.
Machine learning is a subset of AI that enables systems to learn from data.
Deep learning uses neural networks with many layers to model complex patterns.
Large language models such as GPT and Claude are trained on vast text corpora.
Retrieval-Augmented Generation combines LLMs with document retrieval.
"""


@pytest.fixture
def loaded_rag() -> RAGSystem:
    rag = RAGSystem(chunk_size=300, overlap=30, strategy="sentence")
    rag.add_text(PYTHON_DOC, source="python.txt")
    rag.add_text(CLOUD_DOC,  source="cloud.txt")
    rag.add_text(AI_DOC,     source="ai.txt")
    return rag


@pytest.fixture
def empty_rag() -> RAGSystem:
    return RAGSystem()
