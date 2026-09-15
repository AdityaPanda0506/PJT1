"""
Streamlit Web App for NSSG-DimNet (Hinglish DimABSA)
===================================================
Features:
1. Automated Multi-Aspect & Multi-Opinion Extraction from code-mixed Hinglish sentences.
2. Fine-grained 2D Emotion Mapping (24 distinct discrete affective categories).
3. Interactive Russell Circumplex 2D (Valence vs. Arousal) Scatter Space with Plotly.
4. Direct checkpoint loading from 'checkpoints/best_nssg_dimnet.pkl'.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import pickle
from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from emotion_mapper import map_valence_arousal_to_emotion, EMOTION_TAXONOMY


# Page configuration
st.set_page_config(
    page_title="NSSG-DimNet: Code-Mixed Hinglish DimABSA",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .aspect-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_nssg_dimnet(pkl_path: str = "checkpoints/best_nssg_dimnet.pkl"):
    """Loads model package and creates inference engine from pickle checkpoint."""
    if not os.path.exists(pkl_path):
        return None, f"Checkpoint not found at '{pkl_path}'. Please train the model first."

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    from data_pipeline import HinglishTokenizerWrapper, FastGraphBuilder, MasterLexiconEngine, HinglishAspectOpinionExtractor
    from model import NSSGDimNet

    model_name = payload.get("model_name", "xlm-roberta-base")
    arch_cfg = payload.get("architecture_config", {})
    hidden_dim = arch_cfg.get("hidden_dim", 768)

    # Initialize tokenizer and graph builder
    tokenizer = HinglishTokenizerWrapper(model_name)
    graph_builder = FastGraphBuilder()

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NSSGDimNet(
        model_name=model_name,
        hidden_dim=hidden_dim,
        max_switch_dist=arch_cfg.get("max_switch_dist", 16),
        num_heads=arch_cfg.get("num_heads", 8),
        num_rgat_layers=arch_cfg.get("num_rgat_layers", 2),
        dropout=0.0
    ).to(device)

    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()

    lexicon_priors = payload.get("lexicon_priors", {})
    global_lang_map = payload.get("global_lang_map", {})

    lex_engine = MasterLexiconEngine()
    if lexicon_priors:
        lex_engine.priors.update(lexicon_priors)
    else:
        lexicon_priors = lex_engine.priors

    if not global_lang_map and os.path.exists("data/lang_map_cache.pkl"):
        with open("data/lang_map_cache.pkl", "rb") as lf:
            global_lang_map = pickle.load(lf)

    extractor = HinglishAspectOpinionExtractor(lex_engine)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "graph_builder": graph_builder,
        "extractor": extractor,
        "lex_engine": lex_engine,
        "lexicon_priors": lexicon_priors,
        "global_lang_map": global_lang_map,
        "payload": payload,
        "device": device
    }, None


def predict_single_tuple(
    pipeline_obj: dict,
    sentence: str,
    aspect: str,
    opinion: str,
    max_length: int = 128,
    max_switch_dist: int = 16
) -> Dict[str, Any]:
    """Predicts continuous Valence and Arousal scores for a single (sentence, aspect, opinion) tuple."""
    model = pipeline_obj["model"]
    tokenizer = pipeline_obj["tokenizer"]
    graph_builder = pipeline_obj["graph_builder"]
    lex_engine = pipeline_obj.get("lex_engine")
    lex_priors = pipeline_obj["lexicon_priors"]
    global_lang_map = pipeline_obj["global_lang_map"]
    device = pipeline_obj["device"]

    raw_words = sentence.split() or ["<unk>"]
    word_lang_tags = [global_lang_map.get(re.sub(r"[^\w]", "", w.lower()), "EN") for w in raw_words]

    encoding = tokenizer(
        sentence,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)

    subwords = tokenizer.convert_ids_to_tokens(input_ids)
    seq_len = int(attention_mask.sum().item())
    valid_subwords = subwords[:seq_len]

    subword_lang_tags = []
    lexicon_list = []
    w_idx = 0
    for sub_tok in valid_subwords:
        clean_sub = re.sub(r"[^\w]", "", sub_tok.lower())
        if lex_engine is not None:
            v_l, a_l, conf = lex_engine.lookup(clean_sub)
        else:
            v_l, a_l, conf = lex_priors.get(clean_sub, (0.5, 0.5, 0.0))
        lexicon_list.append([v_l, a_l, conf])

        subword_lang_tags.append(word_lang_tags[w_idx] if w_idx < len(word_lang_tags) else "EN")
        if not sub_tok.startswith("##") and not sub_tok.startswith("Ġ"):
            w_idx = min(len(word_lang_tags) - 1, w_idx + 1)

    lexicon_tensor = torch.zeros((1, max_length, 3), dtype=torch.float32)
    if lexicon_list:
        lexicon_tensor[0, :seq_len] = torch.tensor(lexicon_list, dtype=torch.float32)

    from data_pipeline import compute_signed_switch_distance, locate_subword_span
    d_switch_list = compute_signed_switch_distance(subword_lang_tags, max_dist=max_switch_dist)
    d_switch_indices = torch.zeros((1, max_length), dtype=torch.long)
    for i, d in enumerate(d_switch_list):
        if i < max_length:
            d_switch_indices[0, i] = d + max_switch_dist

    # Locate exact token spans using character offset mapping
    asp_s, asp_e = locate_subword_span(sentence, aspect, tokenizer, max_len=max_length)
    op_s, op_e = locate_subword_span(sentence, opinion, tokenizer, max_len=max_length)

    # Compute opinion prior vector directly
    op_v_list, op_a_list = [], []
    op_words = re.findall(r"\w+", opinion.lower())
    for w in op_words:
        if lex_engine is not None:
            v_w, a_w, conf = lex_engine.lookup(w)
            if conf > 0:
                op_v_list.append(v_w)
                op_a_list.append(a_w)
    op_v_mean = float(np.mean(op_v_list)) if op_v_list else 0.50
    op_a_mean = float(np.mean(op_a_list)) if op_a_list else 0.50
    op_lex_prior = torch.tensor([[op_v_mean, op_a_mean]], dtype=torch.float32).to(device)

    A_syn, A_sem, A_cs, A_ao = graph_builder.build_graphs(
        tokens=valid_subwords,
        lang_tags=subword_lang_tags,
        aspect_span=(asp_s, asp_e),
        opinion_span=(op_s, op_e)
    )

    adj_multi = torch.zeros((1, 4, max_length, max_length), dtype=torch.float32)
    adj_multi[0, 0, :seq_len, :seq_len] = torch.from_numpy(A_syn)
    adj_multi[0, 1, :seq_len, :seq_len] = torch.from_numpy(A_sem)
    adj_multi[0, 2, :seq_len, :seq_len] = torch.from_numpy(A_cs)
    adj_multi[0, 3, :seq_len, :seq_len] = torch.from_numpy(A_ao)

    batch = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0).to(device),
        "d_switch_indices": d_switch_indices.to(device),
        "lexicon_priors": lexicon_tensor.to(device),
        "adj_matrices": adj_multi.to(device),
        "aspect_span": torch.tensor([[asp_s, asp_e]], dtype=torch.long).to(device),
        "opinion_span": torch.tensor([[op_s, op_e]], dtype=torch.long).to(device),
        "op_lexicon_prior": op_lex_prior
    }

    with torch.no_grad():
        out = model(batch)
        v_val = float(out["valence_pred"][0].item())
        a_val = float(out["arousal_pred"][0].item())

    # Map to fine-grained emotion taxonomy
    emotion_info = map_valence_arousal_to_emotion(v_val, a_val)
    emotion_info["aspect"] = aspect
    emotion_info["opinion"] = opinion
    return emotion_info


def analyze_full_sentence(pipeline_obj: dict, sentence: str) -> List[Dict[str, Any]]:
    """Automatically extracts all aspect-opinion pairs and performs dimensional inference on all."""
    extractor = pipeline_obj["extractor"]
    pairs = extractor.extract_aspect_opinion_pairs(sentence)
    
    results = []
    for p in pairs:
        res = predict_single_tuple(pipeline_obj, sentence, p["aspect"], p["opinion"])
        results.append(res)
    return results


def build_circumplex_plot(aspect_results: List[Dict[str, Any]]) -> go.Figure:
    """Builds an interactive Plotly 2D Russell Circumplex Scatter Plot."""
    fig = go.Figure()

    # Background Quadrant Reference Rectangles
    # Q1: Top Right (Positive & Active)
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1.0, y1=1.0, fillcolor="rgba(34, 197, 94, 0.07)", line=dict(width=0))
    # Q2: Top Left (Negative & Active)
    fig.add_shape(type="rect", x0=0.0, y0=0.5, x1=0.5, y1=1.0, fillcolor="rgba(239, 68, 68, 0.07)", line=dict(width=0))
    # Q3: Bottom Left (Negative & Passive)
    fig.add_shape(type="rect", x0=0.0, y0=0.0, x1=0.5, y1=0.5, fillcolor="rgba(249, 115, 22, 0.07)", line=dict(width=0))
    # Q4: Bottom Right (Positive & Passive)
    fig.add_shape(type="rect", x0=0.5, y0=0.0, x1=1.0, y1=0.5, fillcolor="rgba(59, 130, 246, 0.07)", line=dict(width=0))

    # Center Crosshairs
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(148, 163, 184, 0.5)", line_width=1.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(148, 163, 184, 0.5)", line_width=1.5)

    # Quadrant Labels
    fig.add_annotation(x=0.85, y=0.92, text="<b>Q1: JOY / EXCITEMENT</b>", showarrow=False, font=dict(color="#22C55E", size=11))
    fig.add_annotation(x=0.15, y=0.92, text="<b>Q2: ANGER / FRUSTRATION</b>", showarrow=False, font=dict(color="#EF4444", size=11))
    fig.add_annotation(x=0.15, y=0.08, text="<b>Q3: SADNESS / DISAPPOINTMENT</b>", showarrow=False, font=dict(color="#F97316", size=11))
    fig.add_annotation(x=0.85, y=0.08, text="<b>Q4: SERENITY / CALM</b>", showarrow=False, font=dict(color="#3B82F6", size=11))

    # Reference Taxonomy faint points
    tax_v = [e["v"] for e in EMOTION_TAXONOMY]
    tax_a = [e["a"] for e in EMOTION_TAXONOMY]
    tax_labels = [f"{e['emoji']} {e['name']}" for e in EMOTION_TAXONOMY]
    
    fig.add_trace(go.Scatter(
        x=tax_v,
        y=tax_a,
        mode="markers+text",
        text=tax_labels,
        textposition="top center",
        textfont=dict(size=8, color="rgba(148, 163, 184, 0.6)"),
        marker=dict(size=5, color="rgba(148, 163, 184, 0.4)"),
        hoverinfo="text",
        name="Emotion Archetypes"
    ))

    # Active Aspect Points
    if aspect_results:
        for item in aspect_results:
            asp = item["aspect"]
            op = item["opinion"]
            v = item["valence"]
            a = item["arousal"]
            emo = item["emotion_name"]
            emoji = item["emoji"]
            color = item["color"]

            hover_html = (
                f"<b>Target Aspect:</b> {asp}<br>"
                f"<b>Opinion:</b> {op}<br>"
                f"<b>Emotion:</b> {emoji} {emo}<br>"
                f"<b>Valence:</b> {v:.4f}<br>"
                f"<b>Arousal:</b> {a:.4f}<br>"
                f"<b>Intensity:</b> {item['intensity_percent']}%"
            )

            fig.add_trace(go.Scatter(
                x=[v],
                y=[a],
                mode="markers+text",
                name=f"Aspect: {asp}",
                text=[f"<b>{emoji} {asp}</b>"],
                textposition="bottom center",
                textfont=dict(size=12, color=color),
                marker=dict(
                    size=16,
                    color=color,
                    line=dict(width=2, color="#FFFFFF"),
                    symbol="circle"
                ),
                hovertext=hover_html,
                hoverinfo="text"
            ))

    fig.update_layout(
        title="<b>2D Russell Circumplex Affective Space</b>",
        xaxis=dict(
            title="<b>Valence (V)</b> — Unpleasant (0.0) to Pleasant (1.0)",
            range=[0.0, 1.0],
            dtick=0.1,
            gridcolor="rgba(51, 65, 85, 0.5)",
            zeroline=False
        ),
        yaxis=dict(
            title="<b>Arousal (A)</b> — Passive / Calm (0.0) to Active / Excited (1.0)",
            range=[0.0, 1.0],
            dtick=0.1,
            gridcolor="rgba(51, 65, 85, 0.5)",
            zeroline=False
        ),
        height=550,
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        showlegend=False
    )
    return fig


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">🧠 NSSG-DimNet: Code-Mixed Hinglish DimABSA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Neuro-Symbolic Switch-Gated Dual-Graph Network for Automated Multi-Aspect & Fine-Grained Affective Intelligence</div>', unsafe_allow_html=True)

pipeline, err = load_nssg_dimnet()

if err:
    st.error(err)
    st.info("Run `python run_pipeline.py --epochs 8` to train the model and generate checkpoints.")
else:
    payload = pipeline["payload"]
    test_metrics = payload.get("test_metrics", {})
    param_stats = payload.get("parameter_summary", {})

    # Sidebar: Model Metrics & Info
    st.sidebar.header("📊 Model Metrics Summary")
    if test_metrics:
        st.sidebar.metric("Overall Avg RMSE", f"{test_metrics.get('rmse_avg', 0):.4f}")
        st.sidebar.metric("Overall Avg CCC", f"{test_metrics.get('ccc_avg', 0):.4f}")
        st.sidebar.metric("Polarity Accuracy", f"{test_metrics.get('acc_polarity', 0):.2f}%")
        st.sidebar.metric("4-Quadrant Accuracy", f"{test_metrics.get('acc_quadrant', 0):.2f}%")
        st.sidebar.caption(f"Valence RMSE: {test_metrics.get('rmse_v', 0):.4f} | Arousal RMSE: {test_metrics.get('rmse_a', 0):.4f}")
    if param_stats:
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Total Params: **{param_stats.get('total_params', 0):,}**")
        st.sidebar.caption(f"Trainable Params: **{param_stats.get('trainable_params', 0):,}**")

    # Sample sentences selector
    sample_options = [
        "khaana bahut swaad tha lekin service bilkul slow thi aur staff ka behaviour rude tha",
        "dwary ye adhikari apne power ka galat use kr rha hai mukhyamantri ji ispe action lijiye",
        "phone ki camera quality outstanding hai but battery backup bohot poor hai",
        "movie ki acting zabardast thi direction awesome tha par climax thoda boring laga",
        "Custom Input"
    ]
    
    st.markdown("### 📝 Input Sentence")
    selected_sample = st.selectbox("Choose a sample sentence or enter your own:", sample_options, index=0)
    
    if selected_sample == "Custom Input":
        default_sent = ""
    else:
        default_sent = selected_sample

    sentence_input = st.text_area("Hinglish Sentence (Multi-Aspect)", value=default_sent, height=80, placeholder="Enter any code-mixed Hinglish sentence...")

    mode = st.radio("Analysis Mode:", ["⚡ Auto-Detect All Aspects & Opinions (Multi-Aspect)", "🎯 Manual Single Aspect-Opinion"], horizontal=True)

    if mode == "🎯 Manual Single Aspect-Opinion":
        c1, c2 = st.columns(2)
        with c1:
            manual_asp = st.text_input("Target Aspect", value="service")
        with c2:
            manual_op = st.text_input("Opinion Phrase", value="bilkul slow thi")

    if st.button("🚀 Analyze Sentiment & Emotions", type="primary"):
        if not sentence_input.strip():
            st.warning("Please enter a sentence to analyze.")
        else:
            with st.spinner("Executing Neuro-Symbolic Dual-Graph Neural Inference..."):
                if mode.startswith("⚡"):
                    results = analyze_full_sentence(pipeline, sentence_input)
                else:
                    results = [predict_single_tuple(pipeline, sentence_input, manual_asp, manual_op)]

            st.markdown("---")
            st.markdown(f"### 🎯 Detected Aspects & Affective Dimensions ({len(results)} Aspect{'s' if len(results) > 1 else ''})")

            col_cards, col_plot = st.columns([1.1, 1.2])

            with col_cards:
                for idx, item in enumerate(results, 1):
                    asp = item["aspect"]
                    op = item["opinion"]
                    v = item["valence"]
                    a = item["arousal"]
                    emo = item["emotion_name"]
                    emoji = item["emoji"]
                    color = item["color"]
                    quad = item["quadrant"]
                    desc = item["description"]
                    intensity = item["intensity_percent"]

                    st.markdown(f"""
                    <div class="aspect-card" style="border-left-color: {color};">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:#F8FAFC;">#{idx} Aspect: <span style="color:#60A5FA;">{asp}</span></h4>
                            <span style="font-size:1.3rem;">{emoji}</span>
                        </div>
                        <p style="margin:4px 0 10px 0; color:#CBD5E1;"><b>Opinion:</b> <i>"{op}"</i></p>
                        <div style="margin-bottom:8px;">
                            <span class="metric-badge" style="background-color:{color}22; color:{color}; border:1px solid {color}44;">
                                Emotion: {emo}
                            </span>
                            <span class="metric-badge" style="background-color:#334155; color:#94A3B8;">
                                Intensity: {intensity}%
                            </span>
                        </div>
                        <div style="font-size:0.9rem; color:#94A3B8; margin-bottom:6px;">
                            <b>Valence (Pleasantness):</b> <span style="color:#F1F5F9; font-weight:600;">{v:.4f}</span> | 
                            <b>Arousal (Intensity):</b> <span style="color:#F1F5F9; font-weight:600;">{a:.4f}</span>
                        </div>
                        <div style="font-size:0.85rem; color:#64748B;">
                            <i>{desc}</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_plot:
                fig = build_circumplex_plot(results)
                st.plotly_chart(fig, use_container_width=True)

            # Summary Table
            st.markdown("### 📋 Multi-Aspect Summary Table")
            df_display = pd.DataFrame([
                {
                    "Aspect": r["aspect"],
                    "Opinion": r["opinion"],
                    "Valence (V)": f"{r['valence']:.4f}",
                    "Arousal (A)": f"{r['arousal']:.4f}",
                    "Emotion State": f"{r['emoji']} {r['emotion_name']}",
                    "Circumplex Quadrant": r["quadrant"],
                    "Intensity": f"{r['intensity_percent']}%"
                }
                for r in results
            ])
            st.dataframe(df_display, use_container_width=True)
