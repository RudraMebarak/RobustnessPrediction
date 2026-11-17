# RobustnessPrediction: Predicting KGW Watermark Survival Under Adversarial Attacks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-green)](https://huggingface.co/)

> **Final Year Major Project — Indian Institute of Information Technology (IIIT), Nagpur**  
> *An end-to-end framework to analyze, benchmark, and predict the survivability and degradation of LLM watermarks across diverse linguistic domains, generation models, and adversarial perturbations.*

---

## 📑 Table of Contents
1. [Executive Overview](#-executive-overview)
2. [Research Motivation & Core Questions](#-research-motivation--core-questions)
3. [System & Hardware Requirements](#-system--hardware-requirements)
4. [End-to-End Pipeline Architecture](#-end-to-end-pipeline-architecture)
5. [Watermarking Protocol (KGW Formulation)](#-watermarking-protocol-kgw-formulation)
6. [Multi-Model & Multi-Domain Dataset Generation](#-multi-model--multi-domain-dataset-generation)
7. [Comprehensive Feature Engineering (42 Features)](#-comprehensive-feature-engineering-42-features)
8. [Adversarial Attack Taxonomy & Perturbation Engine](#-adversarial-attack-taxonomy--perturbation-engine)
9. [Machine Learning Models & Robustness Prediction](#-machine-learning-models--robustness-prediction)
10. [Empirical Results & Key Findings](#-empirical-results--key-findings)
11. [Interactive Web UI](#-interactive-web-ui)
12. [Repository & Codebase Structure](#-repository--codebase-structure)
13. [Installation & Quickstart Guide](#-installation--quickstart-guide)
14. [Council of Five Verification Sign-Off](#-council-of-five-verification-sign-off)

---

## 🔬 Executive Overview

As Large Language Models (LLMs) proliferate across academia, journalism, and programming, detecting machine-generated text is vital for mitigating plagiarism, misinformation, and copyright infringement. **Statistical watermarking**—particularly the soft watermarking scheme proposed by **Kirchenbauer et al. (KGW, 2023)**—embeds an imperceptible signal directly into token sampling distributions during text generation.

However, watermarks remain susceptible to adversarial post-processing attacks such as lexical substitution, syntactic perturbation, copy-paste dilution, and neural paraphrasing. 

**RobustnessPrediction** delivers an end-to-end empirical study and predictive framework:
- **Corpus Generation**: 1,200 controlled texts (600 watermarked, 600 unwatermarked controls) across **3 open-weight LLMs** (Qwen2.5-1.5B, Llama-3.2-1B, SmolLM3-3B) and **4 writing domains** (Academic, Creative, Conversational, Instructional) plus dedicated Code benchmarks.
- **Linguistic Characterization**: 42 rigorously engineered stylometric, syntactic, information-theoretic, and readability features extracted per sample.
- **Adversarial Benchmark**: 8 attack algorithms applied across configurable intensity ladders (low, medium, high, full), generating **10,800 attacked documents**.
- **Dual Evaluation**: Pre- and post-attack watermark strength ($z$-score, $p$-value, green-token ratio) paired with semantic preservation (ModernBERT cosine similarity).
- **Predictive Modeling**: Supervised ML models (XGBoost, Random Forest, Multi-Layer Perceptrons) that predict watermark degradation ($\Delta z$, relative $z$-drop %) based purely on pre-attack linguistic features.

---

## 🎯 Research Motivation & Core Questions

While prior work evaluates whether a watermark survives attacks in aggregate, practitioners cannot easily answer:
1. *Why does a watermark survive in one document but disappear in another under the exact same attack?*
2. *Can the intrinsic linguistic attributes of a document (e.g., lexical entropy, dependency tree depth, syntax density) predict its resilience prior to attack exposure?*
3. *What is the Pareto frontier between watermark degradation and semantic fidelity across diverse attack families?*

This project answers these questions by treating watermark robustness as a **supervised regression and classification problem** conditioned on document stylometry.

---

## 💻 System & Hardware Requirements

Generating watermarked text from 1B–3B parameter transformer models, executing neural paraphrasing pipelines, and performing dense dependency parsing require specific hardware considerations:

### Hardware Specifications
- **System Memory (RAM)**:
  - **Minimum**: **8 GB RAM** (Required for feature extraction pipelines, spaCy syntactic parsing, and tabular ML training).
  - **Recommended**: **16 GB RAM** (Recommended for multi-threaded batch feature extraction and memory-mapped dataset manipulation).
- **Graphics Processing Unit (GPU)**:
  - **Recommended GPU**: **NVIDIA T4 (16 GB VRAM, Turing Architecture)** or equivalent enterprise/consumer GPU (RTX 3060/4060/A10G).
  - **GPU Acceleration**: Hugging Face `transformers` causal generation (Qwen, Llama, SmolLM) and Seq2Seq paraphrasing (`humarin/chatgpt_paraphraser_on_T5_base`) run with 16-bit precision (`torch.float16`) on CUDA, achieving ~5–10x speedup over CPU execution.
- **CPU Fallback Execution**:
  - The entire test, detection, feature extraction, and ML training suite fully supports standard x86_64 multi-core CPUs (`torch.float32`, `CUDA_VISIBLE_DEVICES=""`).
- **Disk Storage**:
  - **At least 20 GB** free SSD space to store raw text corpora, multi-level attack outputs, ModernBERT semantic embeddings, checkpoints, and SQLite feature caches.

### Software Prerequisites
- **Operating System**: Windows 10/11 64-bit or Linux (Ubuntu 20.04/22.04 LTS).
- **Python Runtime**: Python 3.10 or 3.11 recommended.
- **Key Libraries**:
  - `torch>=2.0.0`
  - `transformers>=4.36.0`, `accelerate>=0.25.0`
  - `spacy>=3.7.0` (with `en_core_web_sm`)
  - `nltk>=3.8.0`, `textstat`, `wordfreq`
  - `xgboost>=2.0.0`, `scikit-learn>=1.3.0`, `scipy`
  - `pandas>=2.0.0`, `numpy<2.0.0` or compatible
