<!-- ===================================================================== -->
<!-- COVER PAGE (SPECIMEN-A, WITHOUT COPYRIGHT)                            -->
<!-- ===================================================================== -->

<div align="center">

# Predicting the Robustness of LLM Watermarks Against Textual Attacks Using Linguistic Features and Machine Learning

<br>

*Project report submitted to*  
*Indian Institute of Information Technology, Nagpur, in partial fulfillment of the requirements for the award of the degree of*

<br>

### **Bachelor of Technology**  
### **In**  
### **Computer Science and Engineering**

<br>

*by*

| Sr. No. | Name of the Student | Enrollment No. |
| :---: | :--- | :--- |
| 1 | [Author 1 Name] | [Enrollment No.] |
| 2 | [Author 2 Name (if applicable)] | [Enrollment No.] |
| 3 | [Author 3 Name (if applicable)] | [Enrollment No.] |

<br>

*Under the guidance of*  
**[Name of the Supervisor]**

<br>

**Indian Institute of Information Technology, Nagpur 441108 (India)**  
**2026**

</div>

---

<!-- ===================================================================== -->
<!-- INSIDE COVER PAGE (SPECIMEN-A, WITH COPYRIGHT)                        -->
<!-- ===================================================================== -->

<div align="center">

# Predicting the Robustness of LLM Watermarks Against Textual Attacks Using Linguistic Features and Machine Learning

<br>

*Project report submitted to*  
*Indian Institute of Information Technology, Nagpur, in partial fulfillment of the requirements for the award of the degree of*

<br>

### **Bachelor of Technology**  
### **In**  
### **Computer Science and Engineering**

<br>

*by*

| Sr. No. | Name of the Student | Enrollment No. |
| :---: | :--- | :--- |
| 1 | [Author 1 Name] | [Enrollment No.] |
| 2 | [Author 2 Name (if applicable)] | [Enrollment No.] |
| 3 | [Author 3 Name (if applicable)] | [Enrollment No.] |

<br>

*Under the guidance of*  
**[Name of the Supervisor]**

<br>

**Indian Institute of Information Technology, Nagpur 441108 (India)**  
**2026**

© Indian Institute of Information Technology, Nagpur (IIIT) 2026

</div>

---

<!-- ===================================================================== -->
<!-- DECLARATION-I (SPECIMEN-B)                                            -->
<!-- ===================================================================== -->

<div align="center">

### Department of Computer Science and Engineering  
#### Indian Institute of Information Technology, Nagpur

## Declaration-I

</div>

I/We, **[Student Name(s)]**, hereby declare that this project work titled **“Predicting the Robustness of LLM Watermarks Against Textual Attacks Using Linguistic Features and Machine Learning”** is carried out by me/us in the Department of Computer Science and Engineering of Indian Institute of Information Technology, Nagpur. The work is original and has not been submitted earlier whole or in part for the award of any degree/diploma at this or any other Institution/University.

| Sr. No. | Enrollment No. | Names | Signature |
| :---: | :--- | :--- | :---: |
| 1 | [Enrollment No.] | [Author 1 Name] | |
| 2 | [Enrollment No.] | [Author 2 Name] | |
| 3 | [Enrollment No.] | [Author 3 Name] | |

**Date:** [DD/MM/YYYY]

---

<!-- ===================================================================== -->
<!-- DECLARATION-II (SPECIMEN-C)                                           -->
<!-- ===================================================================== -->

<div align="center">

## Declaration-II

</div>

I/We, **[Student Name(s)]**, Enrollment No. **([Enrollment No.(s)])**, understand that plagiarism is defined as any one or the combination of the following:

1. Uncredited verbatim copying of individual sentences, paragraphs or illustrations (such as graphs, diagrams, etc.) from any source, published or unpublished, including the internet.
2. Uncredited improper paraphrasing of pages or paragraphs (changing a few words or phrases, or rearranging the original sentence order).
3. Credited verbatim copying of a major portion of a paper (or thesis chapter) without clear delineation of who did or wrote what. *(Source: IEEE, the institute, Dec. 2004)*

I have made sure that all the ideas, expressions, graphs, diagrams, etc. that are not a result of my own work, are properly credited. Long phrases or sentences that had to be used verbatim from published literature have been clearly identified using quotation marks.

I affirm that no portion of my work can be considered as plagiarism and I take full responsibility if such complaint occurs. I understand fully well the guide of the thesis may not be in a position to check for possibility of such incidences of plagiarism in this body of work.

| Sr. No. | Names | Signature |
| :---: | :--- | :---: |
| 1 | [Author 1 Name] | |
| 2 | [Author 2 Name] | |
| 3 | [Author 3 Name] | |

**Department Name:** Computer Science and Engineering  
**Indian Institute of Information Technology, Nagpur**  
**Date:** [DD/MM/YYYY]

---

<!-- ===================================================================== -->
<!-- THESIS CERTIFICATE (SPECIMEN-D)                                       -->
<!-- ===================================================================== -->

<div align="center">

## Certificate

</div>

This is to certify that the project titled **“Predicting the Robustness of LLM Watermarks Against Textual Attacks Using Linguistic Features and Machine Learning”**, submitted by **[Student Name(s)]** in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Computer Science and Engineering**, IIIT Nagpur. The work is comprehensive, complete and fit for final evaluation.

**Date:** [DD/MM/YYYY]

<br>

| | |
| :--- | :--- |
| **[Name of the Supervisor]** | **[Name of the Head]** |
| Designation, Dept. of CSE, IIIT Nagpur | Head, Dept. of CSE, IIIT Nagpur |

---

## Acknowledgments

We would like to express our sincere gratitude to our project supervisor, **[Name of the Supervisor]**, Department of Computer Science and Engineering, Indian Institute of Information Technology, Nagpur, for their continuous guidance, insightful technical feedback, and constant encouragement throughout the execution of this major project. Their expertise in machine learning and natural language processing shaped both our experimental methodology and analytical rigor.

We also extend our appreciation to **[Name of the Head]**, Head of the Department of Computer Science and Engineering, and the faculty and technical staff of IIIT Nagpur for providing the academic environment and computing infrastructure required to generate, attack, and evaluate the watermark dataset. Finally, we thank our fellow students and families for their unwavering support during the course of this research.

---

## Abstract

Watermarking has emerged as a foundational mechanism for attributing and verifying text generated by Large Language Models (LLMs). Statistical watermarking schemes—most notably the Kirchenbauer–Geiping–Wen (KGW) algorithm—embed an imperceptible statistical signal during autoregressive decoding by partitioning the vocabulary at each step into pseudorandom "green" and "red" token lists seeded by preceding context tokens, and adding a positive logit bias ($\delta$) to green-list tokens. During verification, a detector evaluates the proportion of green tokens via a one-sided hypothesis test, summarized by a detection Z-score ($Z$). However, in practical deployment, watermarked texts are frequently subjected to benign human editing or adversarial watermark-removal attacks, including neural paraphrasing, synonym substitution, copy-paste dilution, and character-level perturbations. Empirically assessing the vulnerability of every generated document requires running computationally expensive attack models and re-detection pipelines across numerous perturbation settings.

This project investigates how texts varying across entropy, writing domain, generator architecture, and linguistic structure react to watermark removal attacks, and establishes a machine learning framework to predict *a priori* how much a watermarked document's Z-score will drop under perturbation. Because no public dataset paired multi-domain watermarked texts with fine-grained linguistic profiles and multi-intensity attack trajectories, a comprehensive benchmark dataset was constructed from scratch. A total of 600 watermarked documents were generated across three open-weight instruction-tuned LLMs (`meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, and `HuggingFaceTB/SmolLM3-3B`) and four writing domains (Academic, Conversational, Creative, and Instructional) using KGW `selfhash` watermarking ($\gamma = 0.25, \delta = 2.0, Z_{\text{threshold}} = 4.0$). Each document was subsequently attacked under 18 distinct configurations spanning 8 attack families (`copy_paste`, `humarin`, `synonym_wordnet`, `helm_typo`, `helm_misspelling`, `helm_lowercase`, `helm_contraction`, and `helm_expansion`), producing 10,800 attacked samples evaluated for post-attack Z-score, watermark survival, and semantic similarity (`Alibaba-NLP/gte-modernbert-base`).

From each original watermarked text, 42 quantitative linguistic features were extracted across six categories: length and surface structure, lexical diversity and word frequency, part-of-speech and discourse markers, syntactic complexity, readability indices, and n-gram entropy/punctuation. Two complementary regression frameworks were trained using document-grouped 80/20 train–test splits (`GroupShuffleSplit`) and 5-fold `GroupKFold` cross-validation across XGBoost, Random Forest, and Neural Network (`MLPRegressor`) architectures: (1) six **general (attack-averaged) models** predicting equal-family-weighted absolute and relative Z-score drops per document, and (2) six **attack-conditioned models** predicting exact absolute and relative Z-score drops given a specific attack family and intensity. On held-out test documents, the general absolute drop models achieved $R^2 = 0.9466$ ($\text{MAE} = 0.2521$, Spearman $\rho = 0.9739$) using initial Z-score alone and $\text{MAE} = 0.2352$ ($R^2 = 0.9458$) when combined with the 42 linguistic features, while linguistic features alone achieved $R^2 = 0.7263$ ($\text{MAE} = 0.5618, \rho = 0.8511$) without detector score access. In the attack-conditioned regime, combining initial Z-score, linguistic features, and attack metadata achieved $R^2 = 0.9639$ ($\text{MAE} = 0.6671, \rho = 0.9701$) for absolute Z-drop and $R^2 = 0.9478$ ($\text{MAE} = 4.9470\%, \rho = 0.9632$) for relative Z-drop percentage. Finally, all 12 trained models were deployed in a lightweight web interface that extracts the 42 linguistic features in real time and predicts both general and attack-specific watermark degradation.

---

## Table of Contents

| Section | Title | Page |
| :--- | :--- | :---: |
| — | Abstract | i |
| — | Table of Contents | ii |
| — | List of Figures | iii |
| — | List of Tables | iv |
| — | List of Symbols, Abbreviations and Nomenclature | v |
| **1** | **Introduction** | **1** |
| 1.1 | Background and Motivation | 1 |
| 1.2 | Problem Statement | 2 |
| 1.3 | Aim and Scope of the Work | 2 |
| 1.4 | Significant Contributions | 3 |
| 1.5 | Organization of the Thesis | 3 |
| **2** | **Literature Review** | **5** |
| 2.1 | Evolution of Text Watermarking and Provenance Attribution | 5 |
| 2.2 | Statistical Green-List Watermarking (KGW Framework) | 5 |
| 2.3 | Taxonomy of Watermark Removal and Perturbation Attacks | 6 |
| 2.4 | Stylometry, Readability, and Linguistic Determinants of Robustness | 7 |
| 2.5 | Summary of Literature Review and Research Gap | 7 |
| **3** | **Dataset Construction and Experimental Setup** | **8** |
| 3.1 | Multi-Domain Prompt Curation | 8 |
| 3.2 | Watermarked Text Generation Across Three LLMs | 8 |
| 3.3 | Watermark Removal Attack Suite (8 Families, 18 Configurations) | 10 |
| 3.4 | Baseline Watermark Detection and Semantic Similarity Pipeline | 11 |
| **4** | **Methodology: Feature Extraction, Predictive Modeling, and UI Design** | **12** |
| 4.1 | Extraction of 42 Linguistic Features | 12 |
| 4.2 | Mathematical Formulation of Robustness Targets | 14 |
| 4.3 | General (Attack-Averaged) Regression Framework | 15 |
| 4.4 | Attack-Conditioned Regression Framework | 15 |
| 4.5 | Grouped Validation Protocol and Candidate Algorithms | 16 |
| 4.6 | Real-Time Web Application Architecture | 17 |
| **5** | **Results and Discussions** | **18** |
| 5.1 | Empirical Analysis of Attack Effectiveness and Watermark Survival | 18 |
| 5.2 | Semantic Preservation vs. Watermark Degradation Trade-Off | 20 |
| 5.3 | Role of Initial Z-Score and Linguistic Feature Correlations | 21 |
| 5.4 | Evaluation of General (Attack-Averaged) Models | 22 |
| 5.5 | Evaluation of Attack-Conditioned Models | 23 |
| 5.6 | Feature Importance Analysis and Key Findings | 25 |
| **6** | **Summary and Conclusions** | **26** |
| 6.1 | Summary of the Investigation | 26 |
| 6.2 | Major Conclusions | 26 |
| 6.3 | Scope for Future Work | 27 |
| — | **References** | **28** |
| **Appendix I** | **Descriptive Statistics of the 42 Linguistic Features (N = 600)** | **29** |
| **Appendix II** | **Complete Per-Configuration Empirical Attack Benchmark (18 Settings)** | **31** |

---

## List of Figures

| Figure No. | Caption | Page |
| :---: | :--- | :---: |
| Figure 3.1 | Boxplots of Initial KGW Z-Score ($Z_{\text{init}}$) Across Generator LLMs and Writing Domains ($N = 600$) | 9 |
| Figure 3.2 | Distributions of Document Word Count and Bigram Entropy Across Writing Domains ($N = 600$) | 10 |
| Figure 4.1 | Top 12 Linguistic Features Correlated with Equal-Family Absolute and Relative Z-Score Drop | 14 |
| Figure 4.2 | Heatmap of Spearman Correlations Between Linguistic Features and Relative Z-Drop Across Attack Families | 15 |
| Figure 5.1 | Mean Relative Watermark Z-Score Drop (%) Across All 18 Attack Configurations ($\pm 1\text{ SD}$) | 19 |
| Figure 5.2 | Watermark Survival Rate (%) Heatmap Across Attack Families and Intensity Levels ($Z_{\text{attacked}} \ge 4.0$) | 19 |
| Figure 5.3 | Semantic Preservation vs. Mean Relative Z-Score Drop Across Attack Families | 20 |
| Figure 5.4 | Sensitivity of Relative Z-Score Drop to Nominal Modification Rate (10%, 30%, 50%) | 21 |
| Figure 5.5 | Initial Z-Score ($Z_{\text{init}}$) vs. Equal-Family Absolute Drop and Relative Drop Across Generator Models | 22 |
| Figure 5.6 | 5-Fold GroupKFold Cross-Validation MAE Across Candidate Regressors for All 12 Experiments | 23 |
| Figure 5.7 | Held-Out Test $R^2$ Comparison Across Feature Regimes for General and Attack-Conditioned Models | 24 |
| Figure 5.8 | Held-Out Test Predicted vs. Actual Parity Plots for General Absolute Drop and Attack-Conditioned Relative Drop | 24 |
| Figure 5.9 | Top 15 Linguistic Features Ranked by Normalized Gini Importance in the Random Forest Model | 25 |

---

## List of Tables

| Table No. | Title | Page |
| :---: | :--- | :---: |
| Table 3.1 | Corpus Composition Across Generator Language Models and Writing Domains | 8 |
| Table 3.2 | KGW Watermarking and Detection Hyperparameters | 9 |
| Table 3.3 | Pre-Attack Watermark and Linguistic Summary by Generator Model and Domain | 9 |
| Table 3.4 | Specification of the 8 Attack Families and 18 Perturbation Configurations | 11 |
| Table 4.1 | Complete Taxonomy of the 42 Extracted Linguistic Features | 13 |
| Table 4.2 | Specification of the 12 Trained Predictive Models (6 General + 6 Attack-Conditioned) | 16 |
| Table 4.3 | Hyperparameter Configurations of the Three Candidate Regression Pipelines | 17 |
| Table 5.1 | Empirical Impact of Each Attack Family Across 10,800 Attacked Samples | 18 |
| Table 5.2 | 5-Fold GroupKFold Cross-Validation MAE Comparison Across Candidate Algorithms | 22 |
| Table 5.3 | Held-Out Test Set Performance of the Six General (Attack-Averaged) Models | 23 |
| Table 5.4 | Held-Out Test Set Performance of the Six Attack-Conditioned Models | 24 |
| Table I.1 | Descriptive Statistics of All 42 Original-Text Linguistic Features ($N = 600$) | 29 |
| Table II.1 | Empirical Benchmark Metrics Across All 18 Attack Configurations ($N = 10,800$) | 31 |

---

## List of Symbols, Abbreviations and Nomenclature

| Symbol / Abbreviation | Description |
| :--- | :--- |
| LLM | Large Language Model |
| KGW | Kirchenbauer–Geiping–Wen Statistical Watermarking Algorithm |
| $V$, $\text{card}(V)$ | Tokenizer vocabulary and its cardinality |
| $\gamma$ | Target fraction of vocabulary assigned to the green list ($\gamma = 0.25$) |
| $\delta$ | Positive logit bias added to green-list tokens during decoding ($\delta = 2.0$) |
| $T$ | Total number of tokens scored by the watermark detector |
| $n_G$ | Observed count of green-list tokens in a scored passage |
| $f_G$ | Observed green token fraction ($n_G / T$) |
| $Z_{\text{init}}$ | Initial pre-attack watermark detection Z-score |
| $Z_{\text{attacked}}$ | Post-attack watermark detection Z-score (`baseline_z_score`) |
| $Z_{\text{threshold}}$ | Detection Z-score threshold for positive watermark identification ($4.0$) |
| $\Delta Z$ | Absolute Z-score drop ($Z_{\text{init}} - Z_{\text{attacked}}$) |
| $\Delta Z_{\%}$ | Relative Z-score drop percentage ($\frac{Z_{\text{init}} - Z_{\text{attacked}}}{Z_{\text{init}}} \times 100$) |
| $\overline{\Delta Z}_{\text{eq}}$ | Equal-family-weighted document-level absolute Z-score drop |
| $\overline{\Delta Z}_{\%,\text{eq}}$ | Equal-family-weighted document-level relative Z-score drop (%) |
| TTR | Type-Token Ratio ($V_{\text{doc}} / N_{\text{words}}$) |
| MATTR | Moving-Average Type-Token Ratio (50-word sliding window) |
| POS | Part-of-Speech |
| CV | Cross-Validation |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| $R^2$ | Coefficient of Determination |
| $\rho$ | Spearman's Rank Correlation Coefficient |
| RF | Random Forest Regressor |
| XGBoost | Extreme Gradient Boosting Regressor |
| MLP | Multi-Layer Perceptron Regressor (Neural Network) |

---

<div align="center">

# CHAPTER 1

## Introduction

</div>

### 1.1 Background and Motivation

Recent advances in transformer-based Large Language Models (LLMs) have enabled automated text generation systems to produce fluent, contextually appropriate prose across diverse domains including scholarly writing, software documentation, creative storytelling, and conversational assistance. While these capabilities offer substantial productivity benefits, the inability to reliably distinguish machine-generated text from human-authored prose raises critical challenges regarding academic integrity, copyright attribution, automated disinformation, and training-data contamination.

Post-hoc passive classifiers that attempt to detect machine-generated text purely from surface stylometry often suffer from high false-positive rates and poor generalization across newer generator architectures. Statistical *watermarking* overcomes this limitation by actively embedding a cryptographically keyed, statistically verifiable signal into the token sampling process during generation [1, 2]. In the widely adopted Kirchenbauer–Geiping–Wen (KGW) scheme [1], a pseudorandom hash function seeded by preceding context tokens partitions the model's vocabulary at each step into a "green list" of size $\gamma |V|$ and a "red list" of size $(1 - \gamma)|V|$. Adding a moderate positive bias $\delta$ to the logits of green-list tokens increases their sampling probability without noticeably degrading text fluency. Any party holding the hash parameters and tokenizer can subsequently verify a passage by counting green tokens and computing a one-sided normal Z-statistic:

$$Z = \frac{n_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}$$

When $Z$ exceeds a rigorous threshold (typically $Z \ge 4.0$, corresponding to a false-positive probability $p < 3.2 \times 10^{-5}$), the text is flagged as watermarked.

### 1.2 Problem Statement

Although statistical watermarks are highly reliable on unmodified LLM outputs, real-world texts rarely remain static after generation. A user or adversary may edit sentences, substitute synonyms, paste watermarked paragraphs into longer human documents, introduce typographical or spelling variations, normalize casing, or pass the text through neural paraphrasing models [3, 4]. Each such transformation disrupts a fraction of the token sequences and local context windows required by the watermark detector, causing the detection Z-score to drop from its initial value $Z_{\text{init}}$ to a degraded post-attack value $Z_{\text{attacked}}$.

Crucially, different texts do not react uniformly to watermark removal attacks. Two documents generated under identical watermarking hyperparameters ($\gamma = 0.25, \delta = 2.0$) can experience vastly different Z-score reductions depending on their length, vocabulary diversity, n-gram entropy, syntactic structure, and writing domain. Currently, determining how resilient a watermarked passage is requires executing computationally intensive attack pipelines (including neural paraphrasers and embedding similarity models) and re-running the detector across every candidate perturbation. There is a clear scientific and practical need to investigate:
1. How intrinsic linguistic properties of a watermarked text—such as entropy, lexical richness, readability, and part-of-speech distribution—interact with different watermark removal attacks; and
2. Whether machine learning regression models can accurately predict both the **absolute Z-score drop ($\Delta Z$)** and **relative Z-score drop ($\Delta Z_{\%}$)** directly from pre-attack text features and initial watermark strength.

### 1.3 Aim and Scope of the Work

The primary aim of this project is to build a complete, self-contained empirical dataset and machine learning framework for predicting LLM watermark robustness against textual attacks, and to expose the trained models through a clean, real-time web interface.

The scope of the investigation covers four integrated phases:
1. **Custom Benchmark Dataset Generation:** Designing 600 domain-balanced prompts across four writing domains (Academic, Conversational, Creative, Instructional) and generating 600 KGW-watermarked documents using three open-weight instruction-tuned language models (`Llama-3.2-1B-Instruct`, `Qwen2.5-1.5B-Instruct`, and `SmolLM3-3B`).
2. **Multi-Family Attack Execution and Baseline Re-Detection:** Subjecting every watermarked document to 18 attack configurations across 8 attack families (`copy_paste`, `humarin`, `synonym_wordnet`, `helm_typo`, `helm_misspelling`, `helm_lowercase`, `helm_contraction`, `helm_expansion`), yielding 10,800 attacked texts scored for post-attack Z-score, watermark survival, and semantic cosine similarity (`Alibaba-NLP/gte-modernbert-base`).
3. **Linguistic Feature Engineering and Predictive Modeling:** Extracting 42 interpretable linguistic features from each original text and training 12 regression models—6 **general (attack-averaged)** models and 6 **attack-conditioned** models—evaluated with strict document-level group splitting (`GroupShuffleSplit` and 5-fold `GroupKFold`).
4. **Interactive Deployment:** Developing a zero-dependency web application (`ui/app.py` and `ui/index.html`) that computes the 42 linguistic features from user-supplied text and returns predictions across all 12 models in real time.

### 1.4 Significant Contributions

The major contributions of this work are enumerated below:
- **A 10,800-Sample Multi-Model Watermark Robustness Dataset:** We construct and release a balanced corpus pairing 600 watermarked source texts (3 LLMs $\times$ 4 domains) with 18 attack configurations, complete KGW detection statistics, and neural semantic similarity measurements.
- **A 42-Feature Pre-Attack Linguistic Profiler:** We design an automated feature extraction pipeline (`src/feature_extraction.py`) capturing surface length, lexical diversity (TTR, MATTR, Yule's K), word frequency (`wordfreq`), POS tags, dependency tree depth, passive voice, five readability indices, and bigram/trigram entropy.
- **Dual General and Attack-Conditioned Regression Suites:** We systematically compare XGBoost, Random Forest, and Neural Network regressors across three input regimes (*Initial Z only*, *Features only*, and *Initial Z + Features*). Our models achieve $R^2 = 0.9466$ ($\text{MAE} = 0.2352$ with features) on general absolute Z-drop, $R^2 = 0.7263$ using linguistic features alone without initial Z-score, $R^2 = 0.9639$ on attack-conditioned absolute Z-drop, and $R^2 = 0.9478$ ($\rho = 0.9632$) on attack-conditioned relative Z-drop.
- **End-to-End Real-Time Prediction UI:** We provide an intuitive web tool allowing practitioners to paste any watermarked passage, enter an optional initial Z-score and target attack, and inspect predicted Z-score drops alongside all 42 extracted linguistic features.

### 1.5 Organization of the Thesis

This project report is organized into six chapters as follows:
- **Chapter 1 (Introduction)** introduces the background of statistical LLM watermarking, defines the robustness prediction problem, outlines the project scope and contributions, and summarizes the report structure.
- **Chapter 2 (Literature Review)** surveys prior research on LLM watermarking algorithms, text perturbation and paraphrasing attacks, and linguistic determinants of watermark survival.
- **Chapter 3 (Dataset Construction and Experimental Setup)** describes the 600-prompt multi-domain design, generation across three LLMs with KGW `selfhash`, the 18-configuration attack suite, and the detection and semantic similarity pipelines.
- **Chapter 4 (Methodology: Feature Extraction, Predictive Modeling, and UI Design)** details the mathematical definitions of the 42 linguistic features, target aggregation, the 12 regression experiments, grouped cross-validation, and the web application architecture.
- **Chapter 5 (Results and Discussions)** presents a comprehensive empirical evaluation of attack severity, survival heatmaps, semantic trade-offs, feature correlations, and held-out test performance across all 12 models.
- **Chapter 6 (Summary and Conclusions)** synthesizes the findings of the study, presents enumerated conclusions, and outlines directions for future work.

---

<div align="center">

# CHAPTER 2

## Literature Review

</div>

### 2.1 Evolution of Text Watermarking and Provenance Attribution

Provenance verification for machine-generated text has historically followed two paradigms: *passive post-hoc detection* and *active watermarking*. Passive detectors train binary classifiers on surface features, perplexity curves, or neural representations to separate human and machine prose [4, 5]. However, as instruction-tuned LLMs closely match human token distributions across diverse domains, passive detectors experience severe performance degradation and elevated false-positive rates on non-native or formulaic human writing [4, 5]. Active watermarking resolves this fundamental limitation by modifying the generation process itself so that the resulting text carries a mathematically verifiable statistical signature [1, 2].

### 2.2 Statistical Green-List Watermarking (KGW Framework)

Kirchenbauer, Geiping, Wen, Katz, Miers, and Goldstein (2023) [1] formalized the soft red-list/green-list watermarking algorithm for autoregressive language models. Given a token prefix $(w_1, w_2, \dots, w_{t-1})$, a cryptographic hash function $H$ computes a seed from a local context window of preceding tokens. That seed drives a pseudorandom number generator that partitions the vocabulary $V$ into a green list $G_t$ of size $\gamma |V|$ and a red list $R_t$ of size $(1 - \gamma)|V|$. Before applying the softmax function to the model's raw logit vector $\ell_t \in \mathbb{R}^{|V|}$, a constant positive bias $\delta > 0$ is added to all tokens in $G_t$:

$$\tilde{\ell}_{t, v} = \ell_{t, v} + \delta \cdot \mathbb{I}[v \in G_t]$$

In high-entropy decoding steps where several candidate tokens have comparable probabilities, the bias $\delta$ substantially increases the likelihood of sampling a green token; in low-entropy steps where a single token dominates (such as fixed idioms, syntactic function words, or deterministic technical terms), the original token is preserved, maintaining text quality [1].

In a follow-up investigation on watermark reliability, Kirchenbauer et al. (2024) [2] analyzed context-window hashing schemes and highlighted the advantages of `selfhash`, in which the hash key incorporates the candidate token ID alongside preceding context tokens. Because `selfhash` depends only on a narrow local token neighborhood, edits to one region of a document only disrupt green-list scoring in the immediate vicinity of the edited tokens, leaving unedited spans verifiable.

### 2.3 Taxonomy of Watermark Removal and Perturbation Attacks

A central theme in recent watermarking literature is evaluating robustness against post-generation text transformations:
1. **Neural Paraphrasing Attacks:** Krishna et al. (2023) [3] introduced discourse-level paraphrasing models (such as DIPPER) and showed that rewriting sentences while preserving semantic content alters token sequences sufficiently to reduce watermark Z-scores. Sadasivan et al. (2023) [4] similarly analyzed T5-based paraphrasers (such as `humarin/chatgpt_paraphraser_on_T5_base`) and demonstrated that aggressive beam-search or sampling rewrites pose the strongest threat to n-gram-seeded watermarks.
2. **Dilution and Copy-Paste Mixing:** Kirchenbauer et al. [2] studied "copy-paste" attacks where watermarked passages are embedded within unwatermarked human text (or vice versa). Because unwatermarked tokens have an expected green fraction of $\gamma$, inserting unwatermarked spans dilutes the overall green-token ratio toward $\gamma$ in proportion to the inserted fraction.
3. **Lexical Synonym Substitution:** Word-level attacks replace content words with synonyms drawn from lexical databases such as WordNet [7]. Each replaced word changes both its own token(s) and the context hash for immediately following tokens, causing localized Z-score erosion while maintaining high semantic fidelity.
4. **Character-Level and Surface Perturbations (HELM Suite):** Liang et al. (2023) [5] standardized robustness perturbations in the Holistic Evaluation of Language Models (HELM) benchmark, including random character typos (insertions, deletions, substitutions, transpositions), common dictionary misspellings, casing normalization (`lowercase`), and contraction/expansion transformations. Under subword tokenizers (BPE/tiktoken), random internal character typos split a single word into multiple unfamiliar subword tokens, breaking local `selfhash` windows far more severely than rare dictionary misspellings or contractions.

### 2.4 Stylometry, Readability, and Linguistic Determinants of Robustness

Prior watermark studies note qualitatively that generation entropy and passage length influence watermark strength [1, 2], but do not systematically model how a document's broader linguistic profile governs its vulnerability to attack. In quantitative linguistics and stylometry, text structure is characterized through complementary feature families:
- **Lexical Diversity:** Classical Type-Token Ratio (TTR) is sensitive to document length; Covington and McFall (2010) [6] introduced the Moving-Average Type-Token Ratio (MATTR) over a fixed sliding window to measure local lexical richness independently of total word count, alongside frequency-stratified metrics such as hapax legomena, dislegomena, and Yule's characteristic $K$.
- **Word Frequency and Technicality:** Corpus-derived Zipf frequency scales (`wordfreq`) quantify how common or rare a document's vocabulary is, separating everyday conversational prose from specialized academic terminology.
- **Syntactic Structure and Readability:** Dependency parse tree depth, clause length, passive voice ratio, and classical readability formulas—Flesch Reading Ease, Flesch–Kincaid Grade Level, Gunning Fog Index, and SMOG Index—capture structural complexity and syllable density.
- **Information-Theoretic N-Gram Entropy:** Shannon entropy over word bigrams and trigrams measures surface sequence unpredictability, which directly reflects the diversity of token transitions available during watermarked generation.

### 2.5 Summary of Literature Review and Research Gap

While the existing literature thoroughly characterizes KGW watermark generation [1, 2] and demonstrates empirically that various attacks degrade detection Z-scores [2, 3, 4], robustness evaluation remains reactive: to know how much a text's watermark will drop under attack, one must actually execute the attacks and re-score the text. Furthermore, no prior work systematically links a comprehensive 42-feature linguistic profile of the original watermarked text—across multiple LLMs and domains—to predictive machine learning models of both general (attack-averaged) and attack-conditioned watermark degradation. Addressing this gap is the core focus of the present investigation.

---

<div align="center">

# CHAPTER 3

## Dataset Construction and Experimental Setup

</div>

### 3.1 Multi-Domain Prompt Curation

To investigate how domain-specific writing styles affect watermark embedding and post-attack survival, 600 distinct generation prompts were curated and organized into four balanced CSV prompt banks in `src/prompts/` (150 prompts per domain):
1. **Academic Writing (`academic_writing_prompts_150.csv`):** Research-style analytical prompts covering computer science, biology, economics, and engineering topics characterized by formal register, dense terminology, and complex syntax.
2. **Conversational Writing (`conversational_writing_prompts_150.csv`):** Informal dialogue, personal advice, and everyday discussion prompts featuring higher pronoun usage, shorter clauses, and accessible vocabulary.
3. **Creative Writing (`creative_writing_prompts_150.csv`):** Narrative fiction, descriptive scenes, and imaginative storytelling prompts exhibiting rich imagery, varied sentence lengths, and high lexical diversity.
4. **Instructional Writing (`instructional_writing_prompts_150.csv`):** Step-by-step procedural guides, tutorials, and how-to explanations characterized by imperative verbs, structured paragraphs, and direct second-person address.

*Table 3.1: Corpus Composition Across Generator Language Models and Writing Domains*

| Generator Language Model | Academic | Conversational | Creative | Instructional | Total Original Docs | Total Attacked Samples (×18) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `meta-llama/Llama-3.2-1B-Instruct` | 50 | 50 | 50 | 50 | 200 | 3,600 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 50 | 50 | 50 | 50 | 200 | 3,600 |
| `HuggingFaceTB/SmolLM3-3B` | 50 | 50 | 50 | 50 | 200 | 3,600 |
| **Total Across Corpus** | **150** | **150** | **150** | **150** | **600** | **10,800** |

### 3.2 Watermarked Text Generation Across Three LLMs

As shown in Table 3.1, the 600 prompts were partitioned evenly across three open-weight instruction-tuned causal language models spanning 1B to 3B parameters (`Data_Llama`, `Data_qwen`, and `Data_SMOL`), yielding 200 watermarked documents per model (50 per domain). Each document was generated using the Hugging Face `transformers` pipeline integrated with `WatermarkLogitsProcessor` from the KGW reference implementation (`src/generation/generate_documents.py`). Prompt tokens were stripped prior to saving so that only watermarked model completions were stored and analyzed.

*Table 3.2: KGW Watermarking and Detection Hyperparameters*

| Hyperparameter | Setting | Role in Generation and Detection |
| :--- | :---: | :--- |
| Green-list fraction ($\gamma$) | `0.25` | 25% of tokenizer vocabulary placed in the green list at each step |
| Logit bias ($\delta$) | `2.0` | Additive logit boost applied to green-list tokens before softmax |
| Seeding scheme | `selfhash` | Context + candidate token pseudorandom hash seeding |
| Green token selection | `True` | Deterministic green-list partitioning enabled |
| Detection Z-threshold ($Z_{\text{threshold}}$) | `4.0` | Hypothesis test threshold ($p \approx 3.17 \times 10^{-5}$) for survival |
| Decoding temperature / top-$p$ | `0.7` / `0.9` | Nucleus sampling parameters during generation |
| Maximum new tokens | `600` | Upper bound on generated completion length |

Table 3.3 and Figures 3.1–3.2 summarize the pre-attack watermark and linguistic characteristics across the three generator models and four domains. Across all 600 original documents, the mean initial green token fraction was **0.468** (nearly double the null expectation $\gamma = 0.25$), and the mean initial Z-score ($Z_{\text{init}}$) was **14.79** ($\text{SD} = 5.52$, range $[2.82, 40.54]$).

*Table 3.3: Pre-Attack Watermark and Linguistic Summary by Generator Model and Domain ($N = 600$)*

| Group | Category | Docs | Mean $Z_{\text{init}}$ | SD $Z_{\text{init}}$ | Mean Green Fraction | Mean Word Count | Mean Flesch Reading Ease | Mean MATTR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Generator** | `Llama-3.2-1B-Instruct` | 200 | 10.405 | 3.591 | 0.432 | 510.7 | 40.32 | 0.751 |
| **Generator** | `SmolLM3-3B` | 200 | 14.465 | 3.748 | 0.455 | 706.5 | 40.56 | 0.808 |
| **Generator** | `Qwen2.5-1.5B-Instruct` | 200 | 19.490 | 4.889 | 0.517 | 701.7 | 42.47 | 0.818 |
| **Domain** | Academic | 150 | 11.985 | 4.652 | 0.427 | 648.7 | 12.39 | 0.801 |
| **Domain** | Instructional | 150 | 14.195 | 5.214 | 0.471 | 560.7 | 42.55 | 0.778 |
| **Domain** | Conversational | 150 | 16.209 | 5.489 | 0.489 | 634.1 | 58.35 | 0.799 |
| **Domain** | Creative | 150 | 16.758 | 5.474 | 0.485 | 715.0 | 51.18 | 0.791 |

![Figure 3.1: Boxplots of Initial KGW Z-Score Across Generator LLMs and Writing Domains](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig3_1_initial_z_by_model_domain.png)

*Figure 3.1: Boxplots of Initial KGW Z-Score ($Z_{\text{init}}$) Across (a) Generator Language Models and (b) Writing Domains ($N = 600$).*

As Figure 3.1 illustrates, `Qwen2.5-1.5B-Instruct` achieves the highest initial watermark Z-scores (mean $19.49$), followed by `SmolLM3-3B` ($14.47$) and `Llama-3.2-1B-Instruct` ($10.41$). Across domains, Academic texts exhibit the lowest mean $Z_{\text{init}}$ ($11.99$) and the lowest Flesch Reading Ease ($12.39$, reflecting dense technical prose where constrained terminology leaves fewer high-entropy token choices for the $\delta = 2.0$ green-list bias), whereas Conversational ($16.21$) and Creative ($16.76$) writing permit stronger initial watermark embedding. Figure 3.2 further highlights the variation in document length and bigram entropy across the four writing domains.

![Figure 3.2: Distributions of Document Word Count and Bigram Entropy Across Writing Domains](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig3_2_length_and_entropy_distributions.png)

*Figure 3.2: Distributions of (a) Document Word Count and (b) Bigram Entropy Across Writing Domains ($N = 600$).*

### 3.3 Watermark Removal Attack Suite (8 Families, 18 Configurations)

Implemented in `src/attacks/` and orchestrated via `src/run_attacks_sequentially.sh`, each of the 600 watermarked documents was subjected to **8 distinct attack families** across **18 total configurations** (5 multi-intensity families tested at `low`, `medium`, and `high`, plus 3 whole-text deterministic families tested at `full`), generating $600 \times 18 = 10,800$ attacked documents:

1. **Copy-Paste Dilution (`copy_paste` — `src/attacks/dilution_cp.py`):** Inserts unwatermarked human donor text into the watermarked passage using whitespace tokenization (`num_insertions = 1`), targeting modified token fractions of `0.10` (`low`), `0.30` (`medium`), and `0.50` (`high`).
2. **Neural T5 Paraphrasing (`humarin` — `src/attacks/paraphrase_humarin.py`):** Rewrites the document in sentence-aware 500-token chunks using `humarin/chatgpt_paraphraser_on_T5_base` under three decoding presets: conservative beam search (`low`: 5 beams, repetition penalty $1.2$), diverse beam search (`medium`: 5 beams, diversity penalty $1.0$, repetition penalty $2.0$), and nucleus sampling (`high`: `do_sample=True`, temperature $1.0$, top-$p = 0.92$).
3. **WordNet Synonym Substitution (`synonym_wordnet` — `src/attacks/attackmode.py`):** Replaces eligible content words with WordNet synonyms at word-replacement probabilities of `0.10` (`low`), `0.30` (`medium`), and `0.50` (`high`).
4. **HELM Random Character Typos (`helm_typo` — `src/attacks/helm.py`):** Independently selects any alphabetic word of length $\ge 4$ with probability `0.10` (`low`), `0.30` (`medium`), or `0.50` (`high`) and applies one random internal character edit (deletion, insertion, substitution, or adjacent transposition while preserving the first and last characters).
5. **HELM Dictionary Misspelling (`helm_misspelling` — `src/attacks/helm.py`):** Checks alphabetic words against a curated 32-word dictionary of common human misspellings (e.g., *because* $\rightarrow$ *becuase*, *receive* $\rightarrow$ *recieve*) and substitutes matches with probability `0.10` (`low`), `0.30` (`medium`), or `0.50` (`high`).
6. **HELM Lowercasing (`helm_lowercase` — `src/attacks/helm.py`):** Converts all uppercase characters in the document to lowercase (`full`), altering subword token IDs for capitalized words.
7. **HELM Contraction (`helm_contraction` — `src/attacks/helm.py`):** Replaces multi-word auxiliary phrases with standard English contractions (e.g., *do not* $\rightarrow$ *don't*, *will not* $\rightarrow$ *won't*) across the full text (`full`).
8. **HELM Expansion (`helm_expansion` — `src/attacks/helm.py`):** Expands English contractions into their full multi-word forms across the entire text (`full`).

*Table 3.4: Specification of the 8 Attack Families and 18 Perturbation Configurations*

| Attack Family | Intensities | Samples | Perturbation Mechanism & Intensity Parameterization |
| :--- | :--- | :---: | :--- |
| `copy_paste` | `low`, `medium`, `high` | 1,800 | Unwatermarked donor insertion; target modified fraction $0.10, 0.30, 0.50$ |
| `humarin` | `low`, `medium`, `high` | 1,800 | T5 neural paraphrasing (`humarin/chatgpt_paraphraser_on_T5_base`); beam vs. sampling presets |
| `synonym_wordnet` | `low`, `medium`, `high` | 1,800 | WordNet synonym replacement; word modification probability $p \in \{0.10, 0.30, 0.50\}$ |
| `helm_typo` | `low`, `medium`, `high` | 1,800 | Random internal char delete/insert/replace/transpose on words $\ge 4$ chars; $p \in \{0.10, 0.30, 0.50\}$ |
| `helm_misspelling` | `low`, `medium`, `high` | 1,800 | Lookup replacement from 32-word common misspelling lexicon; $p \in \{0.10, 0.30, 0.50\}$ |
| `helm_lowercase` | `full` | 600 | Whole-document case folding (`text.lower()`) |
| `helm_contraction` | `full` | 600 | Whole-document phrase contraction mapping |
| `helm_expansion` | `full` | 600 | Whole-document contraction expansion mapping |

### 3.4 Baseline Watermark Detection and Semantic Similarity Pipeline

Following attack execution, every one of the 10,800 attacked files was processed through two evaluation stages (`baseline_detect_attacked.py` and `src/update_and_combine_features.py`):
1. **Post-Attack KGW Re-Detection:** Each attacked text was tokenized and scored using the generator-matched `WatermarkDetector` ($\gamma = 0.25, \delta = 2.0$, `selfhash`, $Z_{\text{threshold}} = 4.0$), recording `baseline_num_tokens_scored`, `baseline_num_green_tokens`, `baseline_green_fraction`, `baseline_z_score` ($Z_{\text{attacked}}$), `baseline_p_value`, and `watermark_survived` ($\mathbb{I}[Z_{\text{attacked}} \ge 4.0]$).
2. **Neural Semantic Similarity:** Cosine similarity (`semantic_similarity_cosine`) between the sentence embeddings of the original watermarked text and the attacked text was computed using `Alibaba-NLP/gte-modernbert-base` to quantify meaning preservation.

All metadata, detection outputs, and original-text linguistic features were consolidated into `src/merged_robustness_dataset.csv` ($10,800 \text{ rows} \times 82 \text{ columns}$).

---

<div align="center">

# CHAPTER 4

## Methodology: Feature Extraction, Predictive Modeling, and UI Design

</div>

### 4.1 Extraction of 42 Linguistic Features

To characterize the intrinsic linguistic structure of each watermarked document prior to any attack, `src/feature_extraction.py` extracts **42 numerical features** from each of the 600 original texts using `spaCy` (`en_core_web_sm`), `textstat`, and `wordfreq`. Crucially, all 42 features are computed strictly on the pre-attack watermarked text (`src/original_features_combined.csv`), guaranteeing that no post-attack information leaks into the predictors. The complete taxonomy is presented in Table 4.1.

*Table 4.1: Complete Taxonomy of the 42 Extracted Linguistic Features*

| Category | Feature Name | Mathematical / Operational Definition |
| :--- | :--- | :--- |
| **1. Length & Surface Structure (8)** | `character_count` | Total character length of the text |
| | `word_count` | Number of alphabetic word tokens ($N$) |
| | `sentence_count` | Number of sentences segmented by `spaCy` ($S$) |
| | `average_word_length` | Mean characters per alphabetic word |
| | `average_sentence_length` | Mean words per sentence ($N / S$) |
| | `max_sentence_length` | Maximum word count of any single sentence |
| | `paragraph_count` | Count of non-empty newline-delimited paragraphs |
| | `paragraph_length_variance` | Variance of word counts across paragraphs |
| **2. Lexical Diversity & Frequency (10)** | `vocabulary_size` | Number of unique lowercased word types $V_w$ |
| | `type_token_ratio` | $\text{TTR} = V_w / N$ |
| | `mattr` | Moving-Average TTR across sliding windows of $L = 50$ words |
| | `hapax_legomena_ratio` | Fraction of tokens occurring exactly once ($\text{count}(w) = 1$) |
| | `dislegomena_ratio` | Fraction of tokens occurring exactly twice ($\text{count}(w) = 2$) |
| | `lexical_density` | Fraction of content words (`NOUN`, `VERB`, `ADJ`, `ADV`) |
| | `stopword_ratio` | Fraction of tokens in `spaCy` English `STOP_WORDS` |
| | `rare_word_ratio` | Fraction of words with Zipf frequency $< 3.5$ (`wordfreq`) |
| | `average_word_frequency` | Mean Zipf frequency across all alphabetic words |
| | `vocabulary_richness` | Yule's characteristic $K = 10^4 \times \frac{\sum i^2 V_i - N}{N^2}$ |
| **3. POS & Discourse Ratios (10)** | `noun_ratio` | Proportion of `NOUN` + `PROPN` tokens |
| | `verb_ratio` | Proportion of `VERB` + `AUX` tokens |
| | `adjective_ratio` | Proportion of `ADJ` tokens |
| | `adverb_ratio` | Proportion of `ADV` tokens |
| | `pronoun_ratio` | Proportion of `PRON` tokens |
| | `first_person_pronoun_ratio` | Proportion of `{i, me, my, mine, myself, we, us, our, ours, ourselves}` |
| | `second_person_pronoun_ratio` | Proportion of `{you, your, yours, yourself, yourselves}` |
| | `conjunction_ratio` | Proportion of coordinating + subordinating conjunctions (`CCONJ`, `SCONJ`) |
| | `named_entity_density` | Named entities recognized by `spaCy` divided by word count |
| | `numeric_expression_ratio` | Numeric tokens (`like_num` or `POS == NUM`) divided by word count |
| **4. Syntactic Complexity (3)** | `dependency_tree_depth` | Mean root-to-token path length in the dependency parse tree |
| | `average_clause_length` | Word count divided by estimated finite clause count |
| | `passive_voice_ratio` | Fraction of sentences containing `nsubjpass`, `auxpass`, or `csubjpass` |
| **5. Readability Indices (5)** | `flesch_reading_ease` | $206.835 - 1.015(N/S) - 84.6(\text{syllables}/N)$ |
| | `flesch_kincaid_grade` | $0.39(N/S) + 11.8(\text{syllables}/N) - 15.59$ |
| | `gunning_fog_index` | $0.4 \left[(N/S) + 100(\text{complex words}/N)\right]$ |
| | `smog_index` | $1.0430\sqrt{\text{polysyllables} \times (30/S)} + 3.1291$ |
| | `average_syllables_per_word` | Mean syllable count per word (`textstat` + `cmudict`) |
| **6. Entropy, Repetition & Punctuation (6)** | `bigram_entropy` | Shannon entropy $H = -\sum p(b)\log_2 p(b)$ over word bigrams |
| | `trigram_entropy` | Shannon entropy $H = -\sum p(t)\log_2 p(t)$ over word trigrams |
| | `repetition_ratio` | Weighted repetition of $1$-, $2$-, and $3$-grams ($0.5 R_1 + 0.3 R_2 + 0.2 R_3$) |
| | `punctuation_density` | Punctuation characters divided by total character count |
| | `comma_ratio` | Commas per sentence |
| | `technical_vocabulary_ratio` | Fraction of words with length $\ge 8$ and Zipf frequency $< 4.0$ |

### 4.2 Mathematical Formulation of Robustness Targets

For each attacked observation $(d, a, i)$—representing document $d \in \{1, \dots, 600\}$, attack family $a \in \mathcal{A}$ ($|\mathcal{A}| = 8$), and intensity $i \in \mathcal{I}_a$—two target variables quantify watermark loss:

$$\Delta Z_{d,a,i} = Z_{\text{init}, d} - Z_{\text{attacked}, d,a,i}$$

$$\Delta Z_{\%, d,a,i} = \left(\frac{Z_{\text{init}, d} - Z_{\text{attacked}, d,a,i}}{Z_{\text{init}, d}}\right) \times 100$$

For the **general (document-level)** models (`src/model_maker.py`), five attack families have $|\mathcal{I}_a| = 3$ intensities while three deterministic HELM families have $|\mathcal{I}_a| = 1$. To give every attack family equal weight ($1/8$), drops are first averaged across intensities within each family and then averaged across the 8 families:

$$\overline{\Delta Z}_{\text{eq}, d} = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \left( \frac{1}{|\mathcal{I}_a|} \sum_{i \in \mathcal{I}_a} \Delta Z_{d,a,i} \right), \qquad \overline{\Delta Z}_{\%,\text{eq}, d} = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \left( \frac{1}{|\mathcal{I}_a|} \sum_{i \in \mathcal{I}_a} \Delta Z_{\%, d,a,i} \right)$$

Figures 4.1 and 4.2 illustrate the bivariate Spearman rank correlations between the extracted linguistic features and the robustness targets across the dataset.

![Figure 4.1: Top 12 Linguistic Features Correlated with Equal-Family Absolute and Relative Z-Score Drop](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig4_1_top_feature_correlations.png)

*Figure 4.1: Top 12 Linguistic Features Correlated with (a) Equal-Family Absolute Z-Drop and (b) Equal-Family Relative Z-Drop (%) Across 600 Documents.*

![Figure 4.2: Heatmap of Spearman Correlations Between Linguistic Features and Relative Z-Drop Across Attack Families](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig4_2_attack_feature_heatmap.png)

*Figure 4.2: Heatmap of Spearman Correlations Between Key Linguistic Features and Relative Z-Drop (%) Across the 8 Attack Families.*

### 4.3 General (Attack-Averaged) Regression Framework

Implemented in `src/model_maker.py`, the general framework trains **six document-level models** ($N = 600$ rows, one per original document) to predict $\overline{\Delta Z}_{\text{eq}, d}$ (`equal_family_absolute_drop`) and $\overline{\Delta Z}_{\%,\text{eq}, d}$ (`equal_family_relative_drop`) under three feature regimes:
1. **Initial Z Only (`1` feature):** Uses only `initial_z_score` ($Z_{\text{init}}$), testing how much overall degradation is determined purely by pre-attack watermark strength.
2. **Features Only (`42` features):** Uses only the 42 linguistic features, enabling prediction even when the detector's initial Z-score is unavailable.
3. **Initial Z + Features (`43` features):** Combines `initial_z_score` with all 42 linguistic features.

### 4.4 Attack-Conditioned Regression Framework

Implemented in `src/attack_condtitioned_models.py`, the attack-conditioned framework trains **six sample-level models** ($N = 10,800$ rows) to predict $\Delta Z_{d,a,i}$ (`z_score_drop`) and $\Delta Z_{\%, d,a,i}$ (`relative_z_drop_pct`) for a specific attack family and intensity. Each pipeline wraps a `ColumnTransformer` that applies median imputation (and standard scaling for neural networks) to the numerical features and `OneHotEncoder(handle_unknown="ignore")` to the categorical attack inputs (`attack_name` and `intensity`). Table 4.2 lists all 12 trained models.

*Table 4.2: Specification of the 12 Trained Predictive Models (6 General + 6 Attack-Conditioned)*

| Suite | Experiment Identifier | Target Variable | Numeric Features | Categorical Features | Saved Model Artifact |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **General** | `absolute_z_initial_only` | `equal_family_absolute_drop` | 1 ($Z_{\text{init}}$) | 0 | `absolute_z_initial_only.joblib` |
| **General** | `absolute_z_features_only` | `equal_family_absolute_drop` | 42 (Linguistic) | 0 | `absolute_z_features_only.joblib` |
| **General** | `absolute_z_initial_plus_features` | `equal_family_absolute_drop` | 43 ($Z_{\text{init}}$ + Ling.) | 0 | `absolute_z_initial_plus_features.joblib` |
| **General** | `relative_z_initial_only` | `equal_family_relative_drop` | 1 ($Z_{\text{init}}$) | 0 | `relative_z_initial_only.joblib` |
| **General** | `relative_z_features_only` | `equal_family_relative_drop` | 42 (Linguistic) | 0 | `relative_z_features_only.joblib` |
| **General** | `relative_z_initial_plus_features` | `equal_family_relative_drop` | 43 ($Z_{\text{init}}$ + Ling.) | 0 | `relative_z_initial_plus_features.joblib` |
| **Attack-Cond.** | `absolute_initial_plus_attack` | `z_score_drop` | 1 ($Z_{\text{init}}$) | 2 (`attack`, `intensity`) | `absolute_initial_plus_attack.joblib` |
| **Attack-Cond.** | `absolute_features_attack` | `z_score_drop` | 42 (Linguistic) | 2 (`attack`, `intensity`) | `absolute_features_attack.joblib` |
| **Attack-Cond.** | `absolute_initial_features_attack` | `z_score_drop` | 43 ($Z_{\text{init}}$ + Ling.) | 2 (`attack`, `intensity`) | `absolute_initial_features_attack.joblib` |
| **Attack-Cond.** | `relative_initial_plus_attack` | `relative_z_drop_pct` | 1 ($Z_{\text{init}}$) | 2 (`attack`, `intensity`) | `relative_initial_plus_attack.joblib` |
| **Attack-Cond.** | `relative_features_attack` | `relative_z_drop_pct` | 42 (Linguistic) | 2 (`attack`, `intensity`) | `relative_features_attack.joblib` |
| **Attack-Cond.** | `relative_initial_features_attack` | `relative_z_drop_pct` | 43 ($Z_{\text{init}}$ + Ling.) | 2 (`attack`, `intensity`) | `relative_initial_features_attack.joblib` |

### 4.5 Grouped Validation Protocol and Candidate Algorithms

Because the 18 attacked rows of each document share identical original-text features, standard random row splitting would place the same document in both training and test sets, causing severe data leakage. Both `src/model_maker.py` and `src/attack_condtitioned_models.py` enforce **strict document-grouped splitting** (`GroupShuffleSplit`, `test_size = 0.20`, `random_state = 42`) keyed on `doc_id`:
- **Training Split:** 480 unique documents (8,640 attacked samples).
- **Held-Out Test Split:** 120 unseen documents (2,160 attacked samples).

Within the 480-document training split, three candidate regression families—**XGBoost** [8], **Random Forest** [9], and **Multi-Layer Perceptron (Neural Network)**—are evaluated using **5-fold `GroupKFold` cross-validation** grouped by `doc_id`. The candidate achieving the lowest cross-validated **Mean Absolute Error (MAE)** is selected, refitted on the entire 480-document training set, serialized with `joblib`, and evaluated once on the 120-document held-out test set using MAE, RMSE, $R^2$, and Spearman $\rho$. Table 4.3 summarizes the candidate architectures.

*Table 4.3: Hyperparameter Configurations of the Three Candidate Regression Pipelines*

| Candidate Model | Preprocessing Steps | Key Hyperparameters |
| :--- | :--- | :--- |
| **XGBoost (`XGBRegressor`)** | `SimpleImputer(strategy="median")` (+ `OneHotEncoder` for attack models) | `n_estimators=350`, `max_depth=3`, `learning_rate=0.04`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=3`, `reg_lambda=2.0`, `random_state=42` |
| **Random Forest (`RandomForestRegressor`)** | `SimpleImputer(strategy="median")` (+ `OneHotEncoder` for attack models) | `n_estimators=500`, `max_depth=None`, `min_samples_split=4`, `min_samples_leaf=2`, `max_features=0.8`, `random_state=42` |
| **Neural Network (`MLPRegressor`)** | `SimpleImputer(strategy="median")` + `StandardScaler()` (+ `OneHotEncoder`) | `hidden_layer_sizes=(64, 32)`, `activation="relu"`, `solver="adam"`, `alpha=0.01`, `learning_rate_init=0.001`, `early_stopping=True`, `validation_fraction=0.15`, `random_state=42` |

### 4.6 Real-Time Web Application Architecture

To provide a clean, accessible user interface for all 12 models, a web application was built in `ui/app.py` and `ui/index.html`:
- **Backend (`ui/app.py`):** Uses Python's standard `ThreadingHTTPServer` to preload `spaCy` (`en_core_web_sm`) and all 12 `.joblib` pipelines along with their `.json` test metadata. It exposes `GET /` (serving the single-page UI), `GET /attacks` (returning the 8 valid attack families and their supported intensities), and `POST /predict`.
- **Frontend (`ui/index.html`):** Provides a responsive interface where the user pastes watermarked text, optionally enters the initial Z-score ($Z_{\text{init}}$), and optionally selects a target attack family and intensity. Upon clicking **Predict**, the UI displays two side-by-side cards (**Absolute Z-drop** and **Relative Z-drop %**) for the 6 general models—including the estimated post-attack Z-score ($Z_{\text{init}} - \widehat{\Delta Z}$) and held-out test MAE/$R^2$—followed by a matching section for the 6 attack-conditioned models and an expandable table of all 42 extracted linguistic features.

---

<div align="center">

# CHAPTER 5

## Results and Discussions

</div>

### 5.1 Empirical Analysis of Attack Effectiveness and Watermark Survival

Across all 10,800 attacked samples in the benchmark dataset, the mean absolute Z-score drop is **4.701** ($\text{SD} = 5.434$), the mean relative Z-score drop is **31.904%** ($\text{SD} = 32.843\%$), and the overall watermark survival rate ($Z_{\text{attacked}} \ge 4.0$) is **80.29%**. Table 5.1 aggregates the empirical performance of each of the 8 attack families, while Figures 5.1 and 5.2 illustrate the fine-grained behavior across all 18 attack configurations.

*Table 5.1: Empirical Impact of Each Attack Family Across 10,800 Attacked Samples*

| Attack Family | Configurations | Samples ($n$) | Mean $\Delta Z$ | Mean $\Delta Z_{\%}$ | Watermark Survival Rate (%) | Mean Cosine Similarity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `humarin` (T5 Paraphraser) | `low`, `medium`, `high` | 1,800 | 13.642 | 92.209% | 2.944% | 0.842 |
| `helm_typo` | `low`, `medium`, `high` | 1,800 | 5.600 | 37.921% | 90.556% | 0.955 |
| `copy_paste` | `low`, `medium`, `high` | 1,800 | 4.457 | 30.447% | 94.000% | 0.954 |
| `synonym_wordnet` | `low`, `medium`, `high` | 1,800 | 3.166 | 21.660% | 96.500% | 0.958 |
| `helm_lowercase` | `full` | 600 | 3.038 | 20.933% | 96.667% | 0.984 |
| `helm_misspelling` | `low`, `medium`, `high` | 1,800 | 0.219 | 1.511% | 99.278% | 0.999 |
| `helm_expansion` | `full` | 600 | 0.164 | 1.071% | 99.333% | 1.000 |
| `helm_contraction` | `full` | 600 | 0.157 | 1.033% | 99.333% | 1.000 |

![Figure 5.1: Mean Relative Watermark Z-Score Drop Across All 18 Attack Configurations](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_1_relative_drop_by_config.png)

*Figure 5.1: Mean Relative Watermark Z-Score Drop (%) Across All 18 Attack Configurations (Error Bars Show $\pm 1\text{ SD}$).*

![Figure 5.2: Watermark Survival Rate Heatmap Across Attack Families and Intensity Levels](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_2_survival_heatmap.png)

*Figure 5.2: Watermark Survival Rate (%) Heatmap Across Attack Families and Intensity Levels ($Z_{\text{attacked}} \ge 4.0$).*

Several important empirical patterns emerge from Table 5.1 and Figures 5.1–5.2:
1. **Dominance of Neural Paraphrasing (`humarin`):** Whole-sentence neural rewriting via `humarin` reduces the watermark Z-score by **89.88%** (`low`), **91.80%** (`medium`), and **94.94%** (`high`), driving watermark survival down to **5.50%**, **2.50%**, and **0.83%**, respectively. Because paraphrasing restructures clause order and replaces token sequences throughout the passage, almost all local `selfhash` windows are invalidated.
2. **Contrast Between `helm_typo` and `helm_misspelling`:** Even though both attacks operate at word probabilities $p \in \{0.10, 0.30, 0.50\}$, `helm_typo` causes a **37.92%** mean relative Z-drop (`15.96%` at `low`, `40.07%` at `medium`, `57.73%` at `high`, where survival drops to `74.33%`), whereas `helm_misspelling` causes only a **1.51%** mean relative Z-drop (`99.28%` survival). This occurs because `helm_misspelling` only alters words matching a fixed 32-word dictionary, whereas `helm_typo` perturbs any word of $\ge 4$ letters and splits single words into multiple out-of-context subword tokens.
3. **Proportional Scaling of `copy_paste` Dilution:** Under `copy_paste`, target donor insertion fractions of `10%`, `30%`, and `50%` produce mean relative Z-score drops of **10.17%**, **30.84%**, and **50.32%**, closely tracking the theoretical dilution curve while preserving a **84.33%** survival rate even at high intensity due to the strong initial Z-scores ($Z_{\text{init}} \approx 14.79$).

### 5.2 Semantic Preservation vs. Watermark Degradation Trade-Off

An effective watermark attack from an adversary's perspective must degrade $Z_{\text{attacked}}$ without destroying the document's meaning or readability. Figure 5.3 plots the Pareto frontier between mean cosine semantic similarity (`Alibaba-NLP/gte-modernbert-base`) and mean relative Z-score drop across all 8 attack families, and Figure 5.4 compares the sensitivity of the four probability/fraction-parameterized attack families (`copy_paste`, `helm_typo`, `synonym_wordnet`, and `helm_misspelling`) against the nominal perturbation rate ($10\%, 30\%, 50\%$).

![Figure 5.3: Semantic Preservation vs. Mean Relative Z-Score Drop Across Attack Families](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_3_similarity_vs_drop.png)

*Figure 5.3: Semantic Preservation (Cosine Similarity) vs. Mean Relative Z-Score Drop (%) Across Attack Families.*

![Figure 5.4: Sensitivity of Relative Z-Score Drop to Nominal Modification Rate](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_4_modification_rate.png)

*Figure 5.4: Sensitivity of Mean Relative Z-Score Drop (%) to Nominal Modification Rate (10%, 30%, 50%).*

As shown in Figure 5.3, `humarin` achieves watermark removal only at the cost of the largest semantic drift (cosine similarity dropping to `0.833–0.851`). By contrast, `copy_paste` and `helm_typo` follow a steeper degradation trajectory per unit of embedding similarity change, and Figure 5.4 shows that `helm_typo` lies *above* the $1:1$ reference line ($57.73\%$ Z-drop at $50\%$ word typo rate) because a single character typo corrupts both the modified word's subword tokens and the context hash of subsequent tokens.

### 5.3 Role of Initial Z-Score and Linguistic Feature Correlations

Figure 5.5 examines how the pre-attack watermark Z-score ($Z_{\text{init}}$) relates to the document-level equal-family absolute drop ($\overline{\Delta Z}_{\text{eq}}$) and relative drop ($\overline{\Delta Z}_{\%,\text{eq}}$) across all 600 documents.

![Figure 5.5: Initial Z-Score vs. Equal-Family Absolute Drop and Relative Drop Across Generator Models](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_5_initial_z_vs_drop.png)

*Figure 5.5: Initial Z-Score ($Z_{\text{init}}$) vs. (a) Equal-Family Absolute Z-Drop and (b) Equal-Family Relative Z-Drop (%) Across Generator Models.*

Figure 5.5 reveals a fundamental distinction between absolute and relative robustness:
- **Absolute Z-drop scales linearly with $Z_{\text{init}}$ ($r = 0.97$):** Because each attack family removes an approximately constant proportion of the excess green tokens in a passage, documents starting with a higher initial Z-score lose proportionally more raw Z-score points.
- **Relative Z-drop is nearly orthogonal to $Z_{\text{init}}$ ($r = 0.08$):** Dividing by $Z_{\text{init}}$ cancels out initial watermark magnitude, leaving per-document variation driven by linguistic structure—such as syllable density (`average_syllables_per_word`), word length, readability (`flesch_reading_ease`), vocabulary size, and n-gram entropy (`bigram_entropy`, `trigram_entropy`), as seen earlier in Figure 4.1(b).

### 5.4 Evaluation of General (Attack-Averaged) Models

Table 5.2 and Figure 5.6(a) report the 5-fold `GroupKFold` cross-validation MAE across all three candidate algorithms (XGBoost, Random Forest, and Neural Network), while Table 5.3 summarizes the held-out test performance on the 120 unseen documents.

*Table 5.2: 5-Fold GroupKFold Cross-Validation MAE Comparison Across Candidate Algorithms (Mean $\pm$ SD)*

| Suite | Experiment | XGBoost CV MAE | Random Forest CV MAE | Neural Network CV MAE | Selected Winner |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **General** | `absolute_z_initial_only` | $0.3090 \pm 0.0071$ | $0.3228 \pm 0.0056$ | **$0.2927 \pm 0.0171$** | NeuralNetwork |
| **General** | `absolute_z_features_only` | **$0.5791 \pm 0.0298$** | $0.5831 \pm 0.0343$ | $0.7222 \pm 0.1002$ | XGBoost |
| **General** | `absolute_z_initial_plus_features` | **$0.2444 \pm 0.0127$** | $0.2515 \pm 0.0075$ | $0.5796 \pm 0.0861$ | XGBoost |
| **General** | `relative_z_initial_only` | $2.0544 \pm 0.0727$ | $2.1772 \pm 0.0698$ | **$1.9975 \pm 0.1030$** | NeuralNetwork |
| **General** | `relative_z_features_only` | $1.7232 \pm 0.0594$ | **$1.6444 \pm 0.0861$** | $2.9322 \pm 0.5707$ | RandomForest |
| **General** | `relative_z_initial_plus_features` | $1.6841 \pm 0.0332$ | **$1.6043 \pm 0.0602$** | $3.3344 \pm 0.4223$ | RandomForest |
| **Attack-Cond.** | `absolute_initial_plus_attack` | $0.8057 \pm 0.0170$ | $0.8502 \pm 0.0171$ | **$0.7710 \pm 0.0182$** | NeuralNetwork |
| **Attack-Cond.** | `absolute_features_attack` | $1.0231 \pm 0.0374$ | **$0.9930 \pm 0.0426$** | $1.4058 \pm 0.1673$ | RandomForest |
| **Attack-Cond.** | `absolute_initial_features_attack` | $0.6959 \pm 0.0186$ | **$0.6848 \pm 0.0109$** | $0.9838 \pm 0.2700$ | RandomForest |
| **Attack-Cond.** | `relative_initial_plus_attack` | $5.6540 \pm 0.1307$ | $6.1516 \pm 0.0989$ | **$5.5375 \pm 0.0977$** | NeuralNetwork |
| **Attack-Cond.** | `relative_features_attack` | $5.0534 \pm 0.1446$ | **$4.9810 \pm 0.1525$** | $6.3944 \pm 0.9717$ | RandomForest |
| **Attack-Cond.** | `relative_initial_features_attack` | $5.0309 \pm 0.1223$ | **$4.9505 \pm 0.1500$** | $6.1910 \pm 0.6276$ | RandomForest |

![Figure 5.6: 5-Fold GroupKFold Cross-Validation MAE Across Candidate Regressors for All 12 Experiments](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_6_cv_mae_comparison.png)

*Figure 5.6: 5-Fold GroupKFold Cross-Validation MAE Across Candidate Regressors for (a) General and (b) Attack-Conditioned Experiments.*

*Table 5.3: Held-Out Test Set Performance of the Six General (Attack-Averaged) Models ($N_{\text{test}} = 120$ Documents)*

| Experiment | Target | Selected Model | Features | Test MAE | Test RMSE | Test $R^2$ | Spearman $\rho$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `absolute_z_initial_only` | Absolute Drop | NeuralNetwork | 1 | 0.2521 | **0.3237** | **0.9466** | **0.9739** |
| `absolute_z_initial_plus_features` | Absolute Drop | XGBoost | 43 | **0.2352** | 0.3263 | 0.9458 | 0.9725 |
| `absolute_z_features_only` | Absolute Drop | XGBoost | 42 | 0.5618 | 0.7331 | 0.7263 | 0.8511 |
| `relative_z_initial_only` | Relative Drop (%) | NeuralNetwork | 1 | 1.8161 | 2.3250 | 0.1104 | 0.2833 |
| `relative_z_initial_plus_features` | Relative Drop (%) | RandomForest | 43 | **1.5882** | **2.1913** | **0.2098** | **0.4924** |
| `relative_z_features_only` | Relative Drop (%) | RandomForest | 42 | 1.6008 | 2.2393 | 0.1748 | 0.4865 |

On the general held-out test set (Table 5.3):
- Combining `initial_z_score` with the 42 linguistic features (`absolute_z_initial_plus_features`, XGBoost) achieves the lowest absolute drop error (**$\text{MAE} = 0.2352$**, $R^2 = 0.9458$, $\rho = 0.9725$).
- When initial Z-score is omitted (`absolute_z_features_only`), XGBoost trained solely on the 42 linguistic features still explains **72.63% of the variance ($R^2 = 0.7263, \rho = 0.8511$)**, because document length, vocabulary size, and n-gram entropy strongly predict watermark capacity and expected loss.
- For equal-family relative drop, adding the 42 linguistic features nearly doubles the rank correlation from $\rho = 0.2833$ (`relative_z_initial_only`) to **$\rho = 0.4924$** (`relative_z_initial_plus_features`, $\text{MAE} = 1.5882\%$).

### 5.5 Evaluation of Attack-Conditioned Models

Table 5.4 and Figures 5.7–5.8 present the held-out test set evaluation for the six attack-conditioned models across all 2,160 test samples (120 unseen documents $\times$ 18 attack configurations).

*Table 5.4: Held-Out Test Set Performance of the Six Attack-Conditioned Models ($N_{\text{test}} = 2,160$ Samples)*

| Experiment | Target | Selected Model | Numeric Features | Categorical Features | Test MAE | Test RMSE | Test $R^2$ | Spearman $\rho$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `absolute_initial_plus_attack` | Absolute Drop | NeuralNetwork | 1 | 2 | 0.7421 | 1.1477 | 0.9544 | 0.9634 |
| `absolute_initial_features_attack` | Absolute Drop | RandomForest | 43 | 2 | **0.6671** | **1.0214** | **0.9639** | **0.9701** |
| `absolute_features_attack` | Absolute Drop | RandomForest | 42 | 2 | 0.9372 | 1.5277 | 0.9192 | 0.9627 |
| `relative_initial_plus_attack` | Relative Drop (%) | NeuralNetwork | 1 | 2 | 5.4871 | 8.1783 | 0.9394 | 0.9546 |
| `relative_initial_features_attack` | Relative Drop (%) | RandomForest | 43 | 2 | **4.9470** | **7.5898** | **0.9478** | **0.9632** |
| `relative_features_attack` | Relative Drop (%) | RandomForest | 42 | 2 | 4.9765 | 7.6476 | 0.9470 | 0.9630 |

![Figure 5.7: Held-Out Test R2 Comparison Across Feature Regimes for General and Attack-Conditioned Models](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_7_r2_comparison.png)

*Figure 5.7: Held-Out Test $R^2$ Comparison Across Feature Regimes for (a) General and (b) Attack-Conditioned Models.*

![Figure 5.8: Held-Out Test Predicted vs. Actual Parity Plots](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_8_pred_vs_actual.png)

*Figure 5.8: Held-Out Test Predicted vs. Actual Parity Plots for (a) General Absolute Drop (`absolute_z_initial_plus_features`) and (b) Attack-Conditioned Relative Drop (`relative_initial_features_attack`).*

As shown in Table 5.4 and Figures 5.7–5.8, conditioning on the attack family and intensity produces major gains across every metric:
- **Attack-Conditioned Absolute Drop:** The full model (`absolute_initial_features_attack`, Random Forest) achieves **$R^2 = 0.9639$**, **$\text{MAE} = 0.6671$**, **$\text{RMSE} = 1.0214$**, and **Spearman $\rho = 0.9701$**, outperforming both the Z-only model ($\text{MAE} = 0.7421$) and the features-only model ($\text{MAE} = 0.9372, R^2 = 0.9192$).
- **Attack-Conditioned Relative Drop (%):** While attack-averaged relative drop had $R^2 = 0.2098$ due to averaging across wildly different attacks, conditioning on `(attack_name, intensity)` raises held-out test $R^2$ to **0.9478** ($\text{MAE} = 4.9470\%, \rho = 0.9632$). Even without initial Z-score (`relative_features_attack`), the 42 linguistic features + attack metadata achieve **$R^2 = 0.9470$** ($\text{MAE} = 4.9765\%, \rho = 0.9630$).

### 5.6 Feature Importance Analysis and Key Findings

To understand which linguistic features drive document-level sensitivity inside the tree-based models, Figure 5.9 plots the top 15 linguistic features ranked by their normalized Gini importance share in the trained `relative_features_attack` Random Forest regressor.

![Figure 5.9: Top 15 Linguistic Features Ranked by Normalized Gini Importance in the Random Forest Model](file:///home/rudru/majorproject/RobustnessPrediction/report_figures/fig5_9_feature_importance.png)

*Figure 5.9: Top 15 Linguistic Features Ranked by Normalized Gini Importance Share in the Attack-Conditioned Random Forest Model.*

The feature importance profile in Figure 5.9 shows that robustness refinements are distributed across multiple linguistic dimensions:
1. **Surface Length and Punctuation Density (`character_count` 4.74%, `punctuation_density` 4.61%, `word_count` 2.84%):** Longer passages with higher punctuation density provide more frequent syntactic boundaries and larger token sample sizes $T$.
2. **Lexical Specificity and Entities (`named_entity_density` 3.63%, `rare_word_ratio` 3.15%, `vocabulary_size` 2.94%):** Named entities and rare words affect how synonym replacement (`synonym_wordnet`), casing (`helm_lowercase`), and neural paraphrasing (`humarin`) alter subword tokenization.
3. **Morphosyntactic Structure and Entropy (`second_person_pronoun_ratio` 3.14%, `adjective_ratio` 3.14%, `dependency_tree_depth` 3.10%, `verb_ratio` 3.02%, `bigram_entropy` 2.75%, `trigram_entropy` 2.59%):** Syntactic tree depth, POS ratios, and n-gram entropies capture both domain style and the local unpredictability of token transitions.

---

<div align="center">

# CHAPTER 6

## Summary and Conclusions

</div>

### 6.1 Summary of the Investigation

This B.Tech project investigated how watermarked texts generated across multiple LLMs and writing domains react to watermark removal attacks, and demonstrated that watermark Z-score degradation can be accurately predicted using pre-attack linguistic features and machine learning. Specifically:
1. A balanced corpus of **600 KGW-watermarked documents** ($\gamma = 0.25, \delta = 2.0$, `selfhash`) was generated across **3 open-weight LLMs** (`Llama-3.2-1B-Instruct`, `Qwen2.5-1.5B-Instruct`, `SmolLM3-3B`) and **4 writing domains** (Academic, Conversational, Creative, Instructional).
2. All 600 documents were subjected to **18 attack configurations across 8 attack families**, producing **10,800 attacked texts** evaluated for post-attack KGW Z-score, watermark survival ($Z_{\text{attacked}} \ge 4.0$), and neural semantic similarity (`Alibaba-NLP/gte-modernbert-base`).
3. An automated pipeline (`src/feature_extraction.py`) extracted **42 linguistic features** from each original text, and **12 regression pipelines** (6 general attack-averaged models and 6 attack-conditioned models) were trained and validated under strict document-grouped splitting (`GroupShuffleSplit` and 5-fold `GroupKFold`).
4. All 12 trained models were integrated into a lightweight, zero-dependency **web interface** (`ui/app.py`, `ui/index.html`) that performs live feature extraction and multi-model prediction in 1–2 seconds.

### 6.2 Major Conclusions

1. **Watermark survival varies sharply by attack mechanism:** Across 10,800 attacked samples, neural paraphrasing (`humarin`) is the most destructive threat (mean relative Z-drop $92.21\%$, survival rate $2.94\%$), followed by random character typos (`helm_typo`, $37.92\%$ drop) and copy-paste dilution (`copy_paste`, $30.45\%$ drop), whereas closed-lexicon misspellings (`helm_misspelling`, $1.51\%$ drop) and contractions/expansions ($1.03\%–1.07\%$ drop) leave $>99\%$ of watermarks intact.
2. **General absolute Z-score drop is predicted with high accuracy ($R^2 = 0.9466$):** On unseen test documents, `absolute_z_initial_plus_features` (XGBoost) achieves $\text{MAE} = 0.2352$ and Spearman $\rho = 0.9725$, while `absolute_z_initial_only` achieves $R^2 = 0.9466$ ($\rho = 0.9739$).
3. **Linguistic features enable strong prediction even without initial Z-score:** In the absence of detector access, the 42 linguistic features alone achieve $R^2 = 0.7263$ ($\rho = 0.8511$) on general absolute drop and $R^2 = 0.9470$ ($\rho = 0.9630$) on attack-conditioned relative drop.
4. **Attack conditioning unlocks highly accurate relative drop prediction ($R^2 = 0.9478$):** Incorporating attack family and intensity alongside initial Z-score and the 42 linguistic features (`relative_initial_features_attack`, Random Forest) achieves $\text{MAE} = 4.9470\%$, $R^2 = 0.9478$, and $\rho = 0.9632$ across 2,160 held-out test samples.
5. **Combining initial Z-score with the 42 linguistic features consistently yields the best model:** In every general and attack-conditioned experiment, the combined feature regime (`Initial Z + Features`) achieves the lowest held-out test MAE.

### 6.3 Scope for Future Work

1. **Extension to Additional Watermarking Algorithms:** Evaluating the 42-feature predictive framework on distortion-free exponential minimum sampling watermarks and semantic-level watermarks.
2. **Integrated Live Detector Scoring in the Web UI:** Coupling the web application directly with local tokenizer and KGW detector instances so that $Z_{\text{init}}$ can be computed automatically alongside linguistic feature extraction.
3. **Adaptive Watermark Parameter Tuning:** Using the predicted fragility score of a draft passage to dynamically adjust $\delta$ or $\gamma$ during generation for high-risk domains.

---

## References

[1] Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., and Goldstein, T., “A Watermark for Large Language Models,” *Proceedings of the 40th International Conference on Machine Learning (ICML)*, Vol. 202, pp. 17061–17084, Jul. 2023.

[2] Kirchenbauer, J., Geiping, J., Wen, Y., Shu, M., Saifullah, K., Kong, K., Fernando, K., Saha, A., Goldblum, M., and Goldstein, T., “On the Reliability of Watermarks for Large Language Models,” *Proceedings of the Twelfth International Conference on Learning Representations (ICLR)*, May 2024.

[3] Krishna, K., Song, Y., Karpinska, M., Wieting, J., and Iyyer, M., “Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval is an Effective Defense,” *Advances in Neural Information Processing Systems (NeurIPS)*, Vol. 36, pp. 27469–27500, Dec. 2023.

[4] Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang, W., and Feizi, S., “Can AI-Generated Text be Reliably Detected?,” *arXiv preprint arXiv:2303.11156*, Mar. 2023.

[5] Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., Narayanan, D., Wu, Y., Kumar, A., et al., “Holistic Evaluation of Language Models,” *Transactions on Machine Learning Research (TMLR)*, 2023.

[6] Covington, M. A., and McFall, J. D., “Cutting the Gordian Knot: The Moving-Average Type–Token Ratio (MATTR),” *Journal of Quantitative Linguistics*, Vol. 17, No. 2, pp. 94–100, May 2010.

[7] Miller, G. A., “WordNet: A Lexical Database for English,” *Communications of the ACM*, Vol. 38, No. 11, pp. 39–41, Nov. 1995.

[8] Chen, T., and Guestrin, C., “XGBoost: A Scalable Tree Boosting System,” *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, Aug. 2016.

[9] Breiman, L., “Random Forests,” *Machine Learning*, Vol. 45, No. 1, pp. 5–32, Oct. 2001.

---

<div align="center">

# APPENDIX I

## Descriptive Statistics of the 42 Linguistic Features ($N = 600$)

</div>

*Table I.1: Descriptive Statistics of All 42 Original-Text Linguistic Features Across the 600 Watermarked Documents*

| # | Feature Name | Mean | Std. Dev. | Min | Median (50%) | Max |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `character_count` | 4295.102 | 1287.981 | 292.000 | 4690.000 | 6588.000 |
| 2 | `word_count` | 639.618 | 189.027 | 47.000 | 693.500 | 917.000 |
| 3 | `sentence_count` | 40.802 | 16.756 | 4.000 | 40.000 | 148.000 |
| 4 | `average_word_length` | 5.262 | 0.758 | 3.306 | 5.266 | 8.833 |
| 5 | `average_sentence_length` | 17.635 | 8.257 | 3.953 | 16.706 | 113.500 |
| 6 | `max_sentence_length` | 49.363 | 48.971 | 15.000 | 40.000 | 666.000 |
| 7 | `vocabulary_size` | 279.500 | 123.786 | 17.000 | 247.000 | 578.000 |
| 8 | `type_token_ratio` | 0.443 | 0.140 | 0.022 | 0.424 | 0.809 |
| 9 | `mattr` | 0.792 | 0.087 | 0.221 | 0.791 | 0.928 |
| 10 | `hapax_legomena_ratio` | 0.296 | 0.146 | 0.000 | 0.268 | 0.723 |
| 11 | `dislegomena_ratio` | 0.068 | 0.021 | 0.000 | 0.067 | 0.143 |
| 12 | `lexical_density` | 0.596 | 0.076 | 0.421 | 0.597 | 0.786 |
| 13 | `stopword_ratio` | 0.448 | 0.089 | 0.229 | 0.447 | 0.705 |
| 14 | `rare_word_ratio` | 0.051 | 0.033 | 0.000 | 0.044 | 0.271 |
| 15 | `average_word_frequency` | 5.588 | 0.292 | 4.275 | 5.597 | 6.231 |
| 16 | `vocabulary_richness` | 132.876 | 90.836 | 27.055 | 130.257 | 1170.049 |
| 17 | `noun_ratio` | 0.312 | 0.083 | 0.111 | 0.313 | 0.645 |
| 18 | `verb_ratio` | 0.152 | 0.028 | 0.056 | 0.154 | 0.308 |
| 19 | `adjective_ratio` | 0.091 | 0.028 | 0.023 | 0.087 | 0.176 |
| 20 | `adverb_ratio` | 0.040 | 0.023 | 0.000 | 0.038 | 0.132 |
| 21 | `pronoun_ratio` | 0.079 | 0.053 | 0.000 | 0.067 | 0.333 |
| 22 | `dependency_tree_depth` | 3.159 | 1.593 | 1.260 | 3.044 | 30.911 |
| 23 | `average_clause_length` | 5.055 | 0.774 | 2.633 | 4.978 | 8.486 |
| 24 | `passive_voice_ratio` | 0.095 | 0.096 | 0.000 | 0.069 | 0.667 |
| 25 | `flesch_reading_ease` | 41.115 | 24.165 | -59.679 | 41.396 | 103.926 |
| 26 | `flesch_kincaid_grade` | 12.051 | 4.438 | 0.468 | 11.764 | 49.780 |
| 27 | `gunning_fog_index` | 14.793 | 4.971 | 2.744 | 14.294 | 52.845 |
| 28 | `smog_index` | 13.732 | 3.356 | 3.129 | 13.524 | 30.000 |
| 29 | `average_syllables_per_word` | 1.740 | 0.258 | 1.134 | 1.737 | 2.415 |
| 30 | `bigram_entropy` | 8.622 | 0.773 | 4.115 | 8.709 | 9.687 |
| 31 | `trigram_entropy` | 8.863 | 0.740 | 4.170 | 9.006 | 9.755 |
| 32 | `repetition_ratio` | 0.217 | 0.171 | 0.009 | 0.182 | 0.977 |
| 33 | `punctuation_density` | 0.042 | 0.020 | 0.003 | 0.038 | 0.138 |
| 34 | `comma_ratio` | 0.929 | 0.594 | 0.007 | 0.812 | 6.800 |
| 35 | `paragraph_count` | 19.718 | 10.314 | 1.000 | 18.000 | 93.000 |
| 36 | `paragraph_length_variance` | 1037.097 | 2241.218 | 0.000 | 466.713 | 32632.240 |
| 37 | `first_person_pronoun_ratio` | 0.017 | 0.025 | 0.000 | 0.004 | 0.154 |
| 38 | `second_person_pronoun_ratio` | 0.022 | 0.027 | 0.000 | 0.013 | 0.228 |
| 39 | `technical_vocabulary_ratio` | 0.070 | 0.039 | 0.000 | 0.066 | 0.278 |
| 40 | `named_entity_density` | 0.026 | 0.052 | 0.000 | 0.020 | 1.192 |
| 41 | `numeric_expression_ratio` | 0.022 | 0.024 | 0.000 | 0.016 | 0.209 |
| 42 | `conjunction_ratio` | 0.072 | 0.019 | 0.000 | 0.070 | 0.159 |

---

<div align="center">

# APPENDIX II

## Complete Per-Configuration Empirical Attack Benchmark (18 Settings)

</div>

*Table II.1: Empirical Benchmark Metrics Across All 18 Attack Configurations ($N = 600$ Documents per Configuration; $10,800$ Total)*

| Attack Family | Intensity | Docs ($n$) | Mean $\Delta Z$ | Mean $\Delta Z_{\%}$ | SD $\Delta Z_{\%}$ | Survival Rate (%) | Mean Semantic Similarity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `humarin` | `high` | 600 | 14.032 | 94.942% | 10.453% | 0.833% | 0.833 |
| `humarin` | `medium` | 600 | 13.591 | 91.801% | 12.249% | 2.500% | 0.842 |
| `humarin` | `low` | 600 | 13.303 | 89.884% | 13.917% | 5.500% | 0.851 |
| `helm_typo` | `high` | 600 | 8.516 | 57.728% | 7.669% | 74.333% | 0.925 |
| `helm_typo` | `medium` | 600 | 5.918 | 40.072% | 7.213% | 97.667% | 0.955 |
| `helm_typo` | `low` | 600 | 2.365 | 15.961% | 5.714% | 99.667% | 0.984 |
| `copy_paste` | `high` | 600 | 7.367 | 50.324% | 5.669% | 84.333% | 0.927 |
| `copy_paste` | `medium` | 600 | 4.513 | 30.844% | 5.158% | 98.000% | 0.954 |
| `copy_paste` | `low` | 600 | 1.491 | 10.173% | 3.749% | 99.667% | 0.982 |
| `synonym_wordnet` | `high` | 600 | 5.049 | 34.511% | 7.460% | 91.333% | 0.931 |
| `synonym_wordnet` | `medium` | 600 | 3.297 | 22.566% | 6.462% | 98.500% | 0.957 |
| `synonym_wordnet` | `low` | 600 | 1.153 | 7.901% | 4.659% | 99.667% | 0.986 |
| `helm_lowercase` | `full` | 600 | 3.038 | 20.933% | 10.973% | 96.667% | 0.984 |
| `helm_misspelling` | `high` | 600 | 0.358 | 2.471% | 2.559% | 99.167% | 0.998 |
| `helm_misspelling` | `medium` | 600 | 0.224 | 1.552% | 2.027% | 99.333% | 0.999 |
| `helm_misspelling` | `low` | 600 | 0.075 | 0.509% | 1.242% | 99.333% | 1.000 |
| `helm_expansion` | `full` | 600 | 0.164 | 1.071% | 2.838% | 99.333% | 1.000 |
| `helm_contraction` | `full` | 600 | 0.157 | 1.033% | 1.797% | 99.333% | 1.000 |
