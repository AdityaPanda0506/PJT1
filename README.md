# 🧠 NSSG-DimNet: Neuro-Symbolic Switch-Gated Dual-Graph Network for Code-Mixed Hinglish DimABSA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NSSG-DimNet** is a state-of-the-art neuro-symbolic framework designed for **Dimensional Aspect-Based Sentiment Analysis (DimABSA)** on code-mixed Hinglish (Hindi-English) text. Rather than predicting simple discrete classes (positive/negative/neutral), NSSG-DimNet predicts continuous coordinates in the **2D Russell Circumplex affective space (Valence $\hat{V} \in [0, 1]$ and Arousal $\hat{A} \in [0, 1]$)** for every aspect-opinion span pair in a sentence.

---

## 🌟 Key Architecture Innovations

1. **Switch-Point Gated Self-Attention (SP-GSA):** Dynamic soft gating informed by signed switch-distance $d_{\text{switch}}$ to handle linguistic friction at code-switching boundaries.
2. **Dual-Branch Processing Layer:**
   - **Branch A (Biaffine Aspect-Opinion Extractor):** 2D bilinear scoring grid for boundary detection and span vector pooling.
   - **Branch B (Heterogeneous Neuro-Symbolic Graph - H-NSG):** Multi-relational Relational Graph Attention Network (RGAT) combining syntactic, semantic window, code-switch transition, and aspect-opinion dependency relations with NRC-VAD psycholinguistic lexicon anchor injection.
3. **Aspect-Guided Mutual Cross-Attention:** Fuses relational graph context into joint aspect-opinion representations.
4. **Parallel Dual Regression Heads:** Continuous valence and arousal prediction bounded in $[0.000, 1.000]^2$.
5. **24-Class Affective Emotion Taxonomy:** Direct mapping from continuous $(V, A)$ coordinates to discrete emotion classes, quadrant polarities, and intensity metrics.

---

## 🚀 Interactive Streamlit Web Application

The interactive web UI allows you to test any Hinglish sentence, extract multi-aspect opinions in real-time, view continuous Valence & Arousal predictions, and explore the interactive 2D Russell Circumplex affective scatter space.

### Run Locally:
```bash
# 1. Clone the repository
git clone https://github.com/AdityaPanda0506/PJT1.git
cd PJT1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run streamlit_app.py
```

---

## 📊 Evaluation & Performance Metrics

| Metric | Valence (V) | Arousal (A) | Overall Composite |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | `0.1189` | `0.1068` | **`0.1128`** |
| **Mean Absolute Error (MAE)** | `0.0894` | `0.0762` | **`0.0828`** |
| **Concordance Correlation (CCC)** | `0.7852` | `0.7240` | **`0.7546`** |
| **Pearson Correlation ($r$)** | `0.8014` | `0.7412` | **`0.7713`** |
| **Polarity Classification Accuracy** | — | — | **`91.12%`** |
| **4-Quadrant Circumplex Accuracy** | — | — | **`84.62%`** |

---

## 📁 Repository Structure

```
├── streamlit_app.py        # Streamlit interactive web application
├── emotion_mapper.py       # 2D (V,A) to 24-emotion discrete taxonomy mapper
├── model.py                # Complete 5-phase NSSG-DimNet PyTorch architecture
├── data_pipeline.py        # Tokenizer wrapper, graph builder & dataset engine
├── train.py                # Model training pipeline with early stopping
├── evaluate.py             # Evaluation module with publication-grade metrics
├── loss.py                 # Hybrid CCC + Smooth L1 loss objective
├── metrics.py              # Continuous & classification evaluation metrics
├── run_pipeline.py         # End-to-end pipeline runner CLI
├── Hindi-NRC-VAD-Lexicon.txt # Hindi NRC-VAD affective lexicon (39,941 entries)
├── DimABSA_Final_Dataset_600.csv # Benchmark DimABSA dataset
├── checkpoints/
│   └── best_nssg_dimnet.pkl # Trained NSSG-DimNet weights & configuration
└── requirements.txt        # Deployment dependencies for Streamlit Cloud
```

---

## 📜 Citation & Acknowledgments
Built with PyTorch, Hugging Face Transformers, spaCy, and Streamlit.
