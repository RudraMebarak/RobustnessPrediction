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


---

## 🔄 End-to-End Pipeline Architecture

The experimental pipeline links text synthesis, linguistic characterization, adversarial disruption, and supervised prediction into a unified data flow:

```mermaid
flowchart TD
    subgraph DataGen ["1. Controlled Generation"]
        P["Prompt Bank\n(150 prompts × 4 domains)"] --> LLM["3 Open-Weight LLMs\n(Qwen2.5-1.5B, Llama-3.2-1B, SmolLM3-3B)"]
        LLM -->|KGW Processor (γ=0.25, δ=2.0)| WText["600 Watermarked Texts\n(200 per Model)"]
        LLM -->|Unwatermarked Sampling| UWText["600 Donor Control Texts\n(200 per Model)"]
    end

    subgraph FeatureEng ["2. Linguistic Characterization"]
        WText --> NLP["Multi-Tier NLP Pipeline\n(spaCy, textstat, wordfreq, NLTK)"]
        NLP --> FTable["42 Linguistic Features\n(Syntactic, Lexical, Readability, Entropy)"]
        WText --> InitScore["Initial KGW Detector\n(initial_z_score, green_fraction)"]
    end

    subgraph Adversarial ["3. Attack Execution Engine"]
        WText --> AttEngine["8 Adversarial Attack Modules\n(Copy-Paste, HELM Transforms, Synonym, Humarin)"]
        UWText -.->|Donor Spans| AttEngine
        AttEngine --> AttTexts["10,800 Attacked Documents\n(18 configurations per document)"]
    end

    subgraph Evaluation ["4. Post-Attack Detection & Semantics"]
        AttTexts --> PostDet["KGW Evaluator\n(baseline_z_score, detection status)"]
        AttTexts & WText --> ModernBERT["Alibaba ModernBERT-Base\n(Cosine Semantic Similarity)"]
        PostDet & ModernBERT & FTable & InitScore --> MasterCSV["Master Robustness Dataset\n(10,800 rows × 82 features/metrics)"]
    end

    subgraph MLModeling ["5. Predictive Machine Learning"]
        MasterCSV --> GroupSplit["GroupKFold Split (k=5)\n(Grouped strictly on doc_id)"]
        GroupSplit --> Regressors["Regression Suite\n(XGBoost, Random Forest, MLP)"]
        Regressors --> PredDrop["Predict Degradation:\nAbsolute Drop (Δz) & Relative Drop (%)"]
        Regressors --> WebApp["Interactive Web App (ui/app.py)"]
    end
```

---

## 🔒 Watermarking Protocol (KGW Formulation)

The watermark generation and detection system utilizes the soft-watermarking algorithm introduced by **Kirchenbauer et al. (ICML 2023)**.

### Mathematical Formulation

1. **Vocabulary Partitioning**:
   At generation step $t$, given the prefix context $x_{t-h}, \dots, x_{t-1}$ (where $h$ is the context history window), a pseudo-random seed is derived:
   $$\text{seed}_t = \text{Hash}(x_{t-1}, \dots, x_{t-h})$$
   In our pipeline, the seeding scheme is configured as `"selfhash"`. Using this seed, the model vocabulary $V$ (size $|V|$) is partitioned into:
   - **Green List ($G$)**: Privileged tokens of size $|G| = \gamma |V|$
   - **Red List ($R$)**: Non-privileged tokens of size $|R| = (1 - \gamma)|V|$
   where $\gamma \in (0, 1)$ is the green list proportion parameter (configured to $\gamma = 0.25$).

2. **Logit Perturbation**:
   The unwatermarked next-token logit vector $l_t \in \mathbb{R}^{|V|}$ produced by the language model is adjusted before softmax normalization:
   $$l_t'(k) = \begin{cases} 
   l_t(k) + \delta & \text{if } k \in G \\
   l_t(k) & \text{if } k \in R 
   \end{cases}$$
   where $\delta > 0$ is the watermark bias strength (configured to $\delta = 2.0$). Sampling is performed over the softmax of $l_t'$ with temperature $T = 0.7$ and nucleus threshold $p = 0.9$.

3. **Detection Hypothesis Testing**:
   For any text candidate sequence $S$ containing $T$ scored tokens, the detector counts how many tokens fall into their respective green list:
   $$|S_G| = \sum_{t=1}^T \mathbb{I}[s_t \in G(s_{t-1})]$$
   Under the null hypothesis $H_0$ (text was generated without the watermark), $|S_G| \sim \text{Binomial}(T, \gamma)$. The standard one-tailed $z$-score statistic is computed as:
   $$z = \frac{|S_G| - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}$$
   The corresponding one-sided $p$-value is:
   $$p = \frac{1}{2} \text{erfc}\left(\frac{z}{\sqrt{2}}\right)$$

4. **Decision Boundary ($z = 4.0$)**:
   The hypothesis $H_0$ is rejected (watermark detected) if and only if:
   $$z > 4.0$$
   A threshold of $z = 4.0$ provides an exceptionally stringent false positive rate (FPR):
   $$p_{\text{FPR}} = \frac{1}{2}\text{erfc}(4.0/\sqrt{2}) \approx 3.167 \times 10^{-5} \quad (\approx 1 \text{ in } 31,574)$$
   This conservative threshold prevents accidental false attribution on human-written or unwatermarked text.
