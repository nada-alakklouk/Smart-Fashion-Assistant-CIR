# 🤖 Smart Fashion Assistant: Conversational Information Retrieval (CIR) System

An intelligent, context-aware conversational fashion assistant powered by **Retrieval-Augmented Generation (RAG)**. The system leverages semantic search and memory-guided dialogue tracking to deliver personalized clothing and styling recommendations grounded in established fashion principles.

---

## 📌 Table of Contents
- [Introduction](#-introduction)
- [The Problem This Project Solves](#-the-problem-this-project-solves)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [File Structure](#-file-structure)
- [Methodology](#-methodology)
  - [1. Data Collection and Processing](#1-data-collection-and-processing)
  - [2. Information Retrieval (Semantic Search)](#2-information-retrieval-semantic-search)
  - [3. Conversational Pipeline & Context Tracking](#3-building-conversational-pipelines-and-context-tracking)
  - [4. Response Generation](#4-response-generation)
- [Technologies & Models Used](#-technologies--models-used)
- [Future Enhancements](#-future-enhancements)
- [References & Resources](#-references--resources)

---

## 📖 Introduction

Traditional search engines and basic chatbots treat queries in isolation, ignoring the flow of human dialogue. **Conversational Information Retrieval (CIR)** bridges this gap by merging semantic information retrieval with natural multi-turn conversations. 

This project implements a **Smart Fashion Assistant** using the **RAG (Retrieval-Augmented Generation)** framework. Built upon a curated knowledge base of styling guidelines (covering body shapes, skin tones, seasons, and occasions), the assistant processes user inputs, maintains a dialogue state, retrieves the most semantically relevant styling documents, and utilizes the advanced **Qwen2.5-7B-Instruct** LLM to generate highly personalized, factually grounded fashion advice.

---

## ⚠️ The Problem This Project Solves

1. **Multi-Factor Complexity:** Selecting the perfect outfit is a highly personalized task that depends on a combination of overlapping factors (e.g., body shape, skin tone, occasion, and season). Most general assistants focus on only one variable at a time.
2. **Context Blindness in Chatbots:** Many existing virtual fashion assistants lack conversational memory. They cannot track preferences expressed over multi-turn dialogues (e.g., remembering a user's body shape mentioned three turns ago when they ask for a color suggestion).
3. **LLM Hallucinations:** Out-of-the-box Large Language Models (LLMs) often suffer from "hallucinations," fabricating fashion rules or suggesting mismatched styling choices based on generic training patterns. 
   * **Our Solution:** By grounding the **Qwen2.5-7B-Instruct** model with relevant context retrieved dynamically from our custom **Chroma DB** vector database, we eliminate hallucinations and guarantee reliable, domain-expert recommendations.

---

## ⚙️ System Architecture & Workflow

The pipeline is organized into an end-to-end workflow consisting of five primary components: **User Interface (Streamlit)**, **Conversational Query Refactorer**, **Vector Database (Chroma DB)**, **Retrieval Module**, and the **Generator (Qwen2.5-7B-Instruct)**.

```text
  [ User Query ] ──> [ Query Reformulation (Chat History) ]
                               │
                               ▼
                    [ Embedding Generation ]
                               │
                               ▼
                  [ Chroma Vector Database ]
                               │ (Semantic Search - Cosine Distance)
                               ▼
                   [ Retrieve Top-2 Chunks ]
                               │
                               ▼
               [ Prompt Formatting (format-doc) ]
                               │
                               ▼
                  [ Qwen2.5-7B-Instruct ] ──> [ Natural Language Response ]
