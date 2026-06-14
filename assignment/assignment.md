# MINTRI Lab Assignment 2 — Design and Implement a RAG System

## Briefing

**Retrieval-Augmented Generation (RAG)** combines Information Retrieval (IR) and text generation. Instead of relying solely on a language model's internal knowledge, a RAG system retrieves relevant documents from a corpus and uses them to generate grounded, context-aware answers.

This directly connects to:

- **Information Retrieval (IR):** retrieving relevant information
- **Text Mining:** processing, analyzing, and extracting insights from text

---

## What to Do (Groups of 2–3 Students)

- Choose a use case and a RAG platform/model
- Build a working RAG prototype
- Gather and prepare a document corpus
- Implement text mining features over retrieved results:
  - Classification
  - Clustering
  - Summarization
  - Keyword extraction
  - Topic analysis
- Evaluate the system's performance

---

## Example Use Cases

- Supporting students learning Information Retrieval and Text Mining
- Exploring scientific papers or course materials
- Searching and querying documents on a laptop hard disk
- Querying a company knowledge base
- Any other relevant and interesting domain (subject to approval)

---

## Tools and Frameworks

### LLMs

- OpenAI models
- Open-source models (e.g., LLaMA, Mistral)

### RAG Frameworks

- LangChain
- LlamaIndex
- Haystack

### Vector Databases

- Chroma
- FAISS
- Weaviate
- Pinecone

### Other Resources

- FCT IAEDU: https://iaedu.pt/pt
- Gemini for Students: https://gemini.google/students/

---

## Goal

Build a system that not only retrieves information but also understands and uses it effectively.

---

## What We Are Looking For

- Well-justified design choices
- Clear understanding of IR and RAG principles
- Ability to analyze strengths and limitations
- A working prototype

---

# Deliverables

1. GitHub repository
2. Running prototype
3. Report
4. Presentation

---

# Assessment Criteria

| Criterion | Description | Weight |
|------------|------------|---------|
| Problem Definition & Use Case | Clarity, relevance, and justification of the chosen application | 15% |
| RAG Design & Implementation | Quality of system design and working prototype | 25% |
| Corpus & Data Preparation | Appropriateness, quality, and preprocessing of the dataset | 15% |
| Text Mining Features | Use of text mining techniques | 15% |
| Evaluation & Validation | Evaluation methodology and performance analysis | 15% |
| Presentation & Report | Clarity and justification of design decisions | 15% |

---

# Sample Report Structure

## 1. Introduction & Problem Definition

- Clearly describe the use case
- Explain why the problem matters
- Define target users and needs
- State the research/engineering goal

## 2. System Overview (High-Level Architecture)

- RAG pipeline diagram
- Corpus
- Retriever
- Generator
- Text mining layer
- Design justification

## 3. Corpus & Data Preparation

- Dataset size, type, source
- Cleaning
- Chunking strategy
- Justification and trade-offs

## 4. Retrieval Component

- Lexical / Dense / Hybrid retrieval
- Embedding model
- Vector database
- Strengths and limitations

## 5. Generation Component

- LLM used
- Prompt design
- Context injection
- Hallucination mitigation

## 6. Text Mining Features

Examples:

- Summarization
- Keyword extraction
- Topic modelling
- Clustering

## 7. Evaluation & Validation

- Retrieval quality
- Answer quality
- User usefulness
- Test queries
- Success/failure analysis

## 8. Results & Discussion

- What worked
- What did not work
- Limitations

## 9. Future Improvements

- Better retrieval
- Improved prompts
- Better corpus
- Better evaluation

## 10. Conclusion

- Key contributions
- Lessons learned
