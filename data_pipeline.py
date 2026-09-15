"""
NSSG-DimNet Data Pipeline & Lexicon Engine (Phase 1)
===================================================
Incorporates:
1. Dataset: "DimABSA_Final_Dataset_600.csv" / "DimABSA_Final_600.csv" (Unpacked to 1,101 tuples).
2. Lexicons: "Hindi-NRC-VAD-Lexicon.txt" (19,971 entries mapping English & Hindi words to Valence & Arousal).
3. Language Annotations: "all.txt" (Comprehensive token-level language identification tags).
4. Stratified Continuous 70/15/15 train/val/test splits using 2D (V, A) quantile binning.
5. Signed Switch-Distance (d_switch) calculation & Multi-Relational Dual-Graph construction.
6. Pre-cached memory dataset for ultra-fast, high-throughput training and evaluation.
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import math
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import spacy



# ---------------------------------------------------------------------------
# 1. Dataset Loading & Multi-Aspect Unpacking
# ---------------------------------------------------------------------------

def parse_semicolon_list(raw_val: Any) -> List[str]:
    """Parses '{item1; item2}' or 'item1; item2' into a clean python list."""
    if pd.isna(raw_val):
        return []
    s = str(raw_val).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    if not s.strip():
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def load_and_flatten_dataset(csv_path: str = "DimABSA_Final_Dataset_600.csv") -> pd.DataFrame:
    """
    Loads dataset and unpacks multi-aspect semicolon records into flat rows:
    (sentence_id, sentence, code_switch, aspect, opinion, valence, arousal).
    """
    if not os.path.exists(csv_path):
        alt = "DimABSA_Final_600.csv" if "Dataset" in csv_path else "DimABSA_Final_Dataset_600.csv"
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"Cannot find dataset at {csv_path} or {alt}")

    df_raw = pd.read_csv(csv_path)
    records = []

    for _, row in df_raw.iterrows():
        s_id = row.get("sentence_id", len(records) + 1)
        sentence = str(row["sentence"]).strip()
        code_switch = str(row.get("code_switch", "")).strip()

        aspects = parse_semicolon_list(row["all_aspects"])
        opinions = parse_semicolon_list(row["all_opinions"])
        valences = parse_semicolon_list(row["valence_scores"])
        arousals = parse_semicolon_list(row["arousal_scores"])

        n = min(len(aspects), len(opinions), len(valences), len(arousals))
        for i in range(n):
            try:
                v = float(valences[i])
                a = float(arousals[i])
                v = max(0.0, min(1.0, v))
                a = max(0.0, min(1.0, a))
                records.append({
                    "sentence_id": s_id,
                    "sentence": sentence,
                    "code_switch": code_switch,
                    "aspect": aspects[i],
                    "opinion": opinions[i],
                    "valence": v,
                    "arousal": a
                })
            except ValueError:
                continue

    flattened = pd.DataFrame(records)
    print(f"[Data Pipeline] Unpacked dataset: {len(df_raw)} raw rows -> {len(flattened)} aspect-opinion tuples.", flush=True)
    return flattened


# ---------------------------------------------------------------------------
# 2. Master NRC-VAD Lexicon & Language Tag Integration
# ---------------------------------------------------------------------------

class MasterLexiconEngine:
    """
    Loads and integrates:
    1. 'Hindi-NRC-VAD-Lexicon.txt' (19,971 entries with English & Hindi words, Valence, Arousal).
    2. Empirical affective priors from the DimABSA dataset.
    Provides fast token lookup returning (Valence, Arousal, is_anchor_confidence).
    """
    def __init__(
        self,
        hindi_nrc_path: str = "Hindi-NRC-VAD-Lexicon.txt",
        df_dataset: Optional[pd.DataFrame] = None
    ):
        self.priors: Dict[str, Tuple[float, float, float]] = {}
        self.global_v_mean = 0.500
        self.global_a_mean = 0.500

        # 1. Load Hindi-NRC-VAD Lexicon
        self._load_hindi_nrc(hindi_nrc_path)

        # 2. Integrate empirical dataset opinion priors
        if df_dataset is not None and len(df_dataset) > 0:
            self._integrate_dataset_priors(df_dataset)

    def _load_hindi_nrc(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"[Lexicon Engine] Notice: {file_path} not found at default path, continuing.", flush=True)
            return

        try:
            df = pd.read_csv(file_path, sep="\t", on_bad_lines="skip", dtype=str)
            df = df.dropna(subset=["Valence", "Arousal"])
            
            eng_words = df["English Word"].fillna("").astype(str).str.strip().str.lower().tolist()
            hi_words = df["Hindi Word"].fillna("").astype(str).str.strip().str.lower().tolist()
            valences = pd.to_numeric(df["Valence"], errors="coerce").fillna(0.5).tolist()
            arousals = pd.to_numeric(df["Arousal"], errors="coerce").fillna(0.5).tolist()

            count = 0
            for eng, hi, v, a in zip(eng_words, hi_words, valences, arousals):
                if eng and eng != "nan":
                    self.priors[eng] = (float(v), float(a), 1.0)
                    count += 1
                if hi and hi != "nan":
                    self.priors[hi] = (float(v), float(a), 1.0)
                    count += 1

            # Add Romanized Hinglish sentiment vocabulary
            hinglish_priors = {
                # High Valence, High Arousal (Positive & Excited / Delicious / Great)
                "swaad": (0.92, 0.70, 1.0), "swad": (0.92, 0.70, 1.0), "swadisht": (0.94, 0.72, 1.0), "swadist": (0.92, 0.70, 1.0),
                "tasty": (0.95, 0.75, 1.0), "delicious": (0.94, 0.70, 1.0), "yummy": (0.92, 0.68, 1.0),
                "badhiya": (0.88, 0.65, 1.0), "badiya": (0.88, 0.65, 1.0), "mast": (0.90, 0.78, 1.0),
                "zabardast": (0.92, 0.85, 1.0), "jabardast": (0.92, 0.85, 1.0), "shandar": (0.94, 0.75, 1.0),
                "shaandar": (0.94, 0.75, 1.0), "lajawab": (0.95, 0.78, 1.0), "lajawaab": (0.95, 0.78, 1.0),
                "kamaal": (0.93, 0.80, 1.0), "kamal": (0.93, 0.80, 1.0), "gazab": (0.92, 0.82, 1.0),
                "superb": (0.92, 0.75, 1.0), "awesome": (0.93, 0.80, 1.0), "excellent": (0.94, 0.75, 1.0),
                "acha": (0.82, 0.55, 1.0), "achha": (0.82, 0.55, 1.0), "achi": (0.82, 0.55, 1.0), "achhi": (0.82, 0.55, 1.0), "acchi": (0.82, 0.55, 1.0),
                "good": (0.80, 0.50, 1.0), "great": (0.88, 0.65, 1.0), "helpful": (0.82, 0.40, 1.0), "friendly": (0.85, 0.45, 1.0),
                "cooperative": (0.80, 0.40, 1.0), "polite": (0.82, 0.35, 1.0), "clean": (0.78, 0.45, 1.0), "saaf": (0.78, 0.45, 1.0),
                "sasta": (0.75, 0.40, 1.0), "affordable": (0.75, 0.35, 1.0), "reasonable": (0.78, 0.35, 1.0), "peaceful": (0.80, 0.20, 1.0),
                "shant": (0.80, 0.20, 1.0), "khush": (0.88, 0.70, 1.0), "pasand": (0.84, 0.50, 1.0),

                # Low Valence (Negative / Bad / Slow / Rude / Expensive)
                "slow": (0.35, 0.20, 1.0), "dheema": (0.35, 0.20, 1.0), "dheemi": (0.35, 0.20, 1.0), "dhimi": (0.35, 0.20, 1.0), "dhima": (0.35, 0.20, 1.0),
                "late": (0.30, 0.40, 1.0), "deri": (0.30, 0.40, 1.0), "rude": (0.08, 0.85, 1.0), "badtameez": (0.08, 0.85, 1.0),
                "arrogant": (0.12, 0.75, 1.0), "ghatiya": (0.10, 0.70, 1.0), "ganda": (0.12, 0.65, 1.0), "gandi": (0.12, 0.65, 1.0),
                "bura": (0.15, 0.60, 1.0), "buri": (0.15, 0.60, 1.0), "bekaar": (0.12, 0.65, 1.0), "bekar": (0.12, 0.65, 1.0),
                "bakwas": (0.10, 0.75, 1.0), "kharab": (0.12, 0.70, 1.0), "worst": (0.05, 0.85, 1.0), "terrible": (0.08, 0.80, 1.0),
                "pathetic": (0.08, 0.75, 1.0), "horrible": (0.08, 0.85, 1.0), "mehnga": (0.28, 0.58, 1.0), "mehenga": (0.28, 0.58, 1.0),
                "expensive": (0.30, 0.55, 1.0), "noisy": (0.32, 0.75, 1.0), "shor": (0.32, 0.75, 1.0), "crowded": (0.38, 0.65, 1.0),
                "bheed": (0.38, 0.65, 1.0), "unhygienic": (0.10, 0.65, 1.0), "dirty": (0.12, 0.65, 1.0), "cold": (0.35, 0.30, 1.0),
                "thanda": (0.35, 0.30, 1.0), "kachha": (0.25, 0.45, 1.0), "raw": (0.25, 0.45, 1.0)
            }
            self.priors.update(hinglish_priors)

            print(f"[Lexicon Engine] Successfully loaded {count} entries from '{file_path}' + {len(hinglish_priors)} Hinglish vocabulary priors.", flush=True)
        except Exception as e:
            print(f"[Lexicon Engine] Error reading {file_path}: {e}", flush=True)

    def _integrate_dataset_priors(self, df: pd.DataFrame):
        self.global_v_mean = float(df["valence"].mean())
        self.global_a_mean = float(df["arousal"].mean())

        word_v: Dict[str, List[float]] = {}
        word_a: Dict[str, List[float]] = {}

        for _, row in df.iterrows():
            v = float(row["valence"])
            a = float(row["arousal"])
            ctx = f"{row['opinion']} {row['aspect']}"
            words = re.findall(r"\w+", ctx.lower())
            for w in words:
                if len(w) > 1:
                    word_v.setdefault(w, []).append(v)
                    word_a.setdefault(w, []).append(a)

        added = 0
        for w, v_list in word_v.items():
            if w not in self.priors:
                mean_v = float(np.mean(v_list))
                mean_a = float(np.mean(word_a[w]))
                conf = min(1.0, len(v_list) / 2.0)
                self.priors[w] = (mean_v, mean_a, conf)
                added += 1

        print(f"[Lexicon Engine] Total active lexicon entries (NRC + Dataset): {len(self.priors)} tokens.", flush=True)

    def lookup(self, token: str) -> Tuple[float, float, float]:
        clean = re.sub(r"[^\w]", "", token.lower())
        if clean in self.priors:
            return self.priors[clean]
        return self.global_v_mean, self.global_a_mean, 0.0


def load_all_txt_language_map(file_path: str = "all.txt", cache_path: str = "data/lang_map_cache.pkl") -> Dict[str, str]:
    """
    Parses 'all.txt' language tags with fast pickle caching.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                lang_map = pickle.load(f)
            print(f"[Language Pipeline] Loaded {len(lang_map)} cached language alignments from '{cache_path}'.", flush=True)
            return lang_map
        except Exception:
            pass

    lang_map = {}
    if not os.path.exists(file_path):
        return lang_map

    try:
        df_all = pd.read_csv(file_path, sep="\t", header=None, names=["word", "lang"], on_bad_lines="skip", dtype=str)
        df_all = df_all.dropna()
        lang_map = dict(zip(df_all["word"].str.lower(), df_all["lang"].str.upper()))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(lang_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Language Pipeline] Successfully parsed and cached {len(lang_map)} language alignments from '{file_path}'.", flush=True)
    except Exception as e:
        print(f"[Language Pipeline] Warning: Could not parse {file_path}: {e}", flush=True)

    return lang_map


# ---------------------------------------------------------------------------
# 3. Stratified Continuous 2D Quantile Split
# ---------------------------------------------------------------------------

def stratified_continuous_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_bins: int = 5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    df_copy = df.copy()

    v_bins = pd.qcut(df_copy["valence"], q=num_bins, labels=False, duplicates="drop")
    a_bins = pd.qcut(df_copy["arousal"], q=num_bins, labels=False, duplicates="drop")
    df_copy["strata_id"] = v_bins.astype(str) + "_" + a_bins.astype(str)

    train_rows, val_rows, test_rows = [], [], []

    for _, group in df_copy.groupby("strata_id"):
        shuffled = group.sample(frac=1.0, random_state=seed)
        n = len(shuffled)
        n_tr = int(round(n * train_ratio))
        n_va = int(round(n * val_ratio))
        if n >= 3 and n_tr == 0:
            n_tr = 1
        if n >= 3 and n_va == 0:
            n_va = 1

        train_rows.append(shuffled.iloc[:n_tr])
        val_rows.append(shuffled.iloc[n_tr:n_tr + n_va])
        test_rows.append(shuffled.iloc[n_tr + n_va:])

    df_train = pd.concat(train_rows, ignore_index=True).drop(columns=["strata_id"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_val = pd.concat(val_rows, ignore_index=True).drop(columns=["strata_id"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_test = pd.concat(test_rows, ignore_index=True).drop(columns=["strata_id"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[Data Pipeline] Stratified Splits: Train={len(df_train)} (70%), Val={len(df_val)} (15%), Test={len(df_test)} (15%)", flush=True)
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# 4. Code-Switch Distance & Fast Graph Adjacency Generation
# ---------------------------------------------------------------------------

def parse_code_switch_tags(cs_string: str) -> Dict[str, str]:
    tag_map = {}
    if not cs_string or pd.isna(cs_string):
        return tag_map
    for tok in str(cs_string).strip().split():
        if ":" in tok:
            w, lang = tok.rsplit(":", 1)
            tag_map[w.strip().lower()] = lang.strip().upper()
    return tag_map


def compute_signed_switch_distance(lang_tags: List[str], max_dist: int = 16) -> List[int]:
    n = len(lang_tags)
    if n == 0:
        return []
    switch_indices = [
        i for i in range(1, n)
        if lang_tags[i] != lang_tags[i - 1] and lang_tags[i] in ["EN", "HI"] and lang_tags[i - 1] in ["EN", "HI"]
    ]
    if not switch_indices:
        return [0] * n

    d_switch = []
    for i in range(n):
        closest = min(switch_indices, key=lambda s: abs(i - s))
        raw = i - closest
        clamped = max(-max_dist, min(max_dist, raw))
        d_switch.append(clamped)
    return d_switch


class FastGraphBuilder:
    """Optimized Graph Builder for syntactic, semantic, code-switch, and aspect-opinion graphs."""
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])
        except Exception:
            self.nlp = spacy.blank("en")

    def build_graphs(
        self,
        tokens: List[str],
        lang_tags: List[str],
        aspect_span: Tuple[int, int],
        opinion_span: Tuple[int, int],
        window_size: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(tokens)
        A_syn = np.eye(n, dtype=np.float32)
        A_sem = np.eye(n, dtype=np.float32)
        A_cs = np.zeros((n, n), dtype=np.float32)
        A_ao = np.zeros((n, n), dtype=np.float32)

        # Fast Syntactic dependency graph
        if n > 0:
            doc = self.nlp(" ".join(tokens))
            for token in doc:
                h_idx, d_idx = token.head.i, token.i
                if h_idx < n and d_idx < n:
                    A_syn[h_idx, d_idx] = 1.0
                    A_syn[d_idx, h_idx] = 1.0

        # Semantic sliding window
        for i in range(n):
            for j in range(max(0, i - window_size), min(n, i + window_size + 1)):
                dist = abs(i - j)
                if dist > 0:
                    A_sem[i, j] = 1.0 / (dist + 0.5)

        # Code-Switch transition bridges
        for i in range(1, n):
            if lang_tags[i] != lang_tags[i - 1] and lang_tags[i] in ["EN", "HI"] and lang_tags[i - 1] in ["EN", "HI"]:
                A_cs[i - 1, i] = 1.0
                A_cs[i, i - 1] = 1.0

        # Aspect-Opinion links
        a_s, a_e = aspect_span
        o_s, o_e = opinion_span
        for i in range(a_s, min(n, a_e + 1)):
            for j in range(o_s, min(n, o_e + 1)):
                A_ao[i, j] = 1.0
                A_ao[j, i] = 1.0

        return A_syn, A_sem, A_cs, A_ao


# Alias for backward compatibility
GraphBuilder = FastGraphBuilder



# ---------------------------------------------------------------------------
# 5. Pre-Cached DimABSA PyTorch Dataset
# ---------------------------------------------------------------------------

class DimABSADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        lexicon_engine: MasterLexiconEngine,
        global_lang_map: Dict[str, str],
        graph_builder: FastGraphBuilder,
        max_length: int = 128,
        max_switch_dist: int = 16
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.lexicon_engine = lexicon_engine
        self.global_lang_map = global_lang_map
        self.graph_builder = graph_builder
        self.max_length = max_length
        self.max_switch_dist = max_switch_dist

        # Pre-cache all tensor objects for lightning-fast training
        self.samples = [self._process_row(i) for i in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.samples)

def locate_subword_span(sentence: str, phrase: str, tokenizer, max_len: int = 128) -> Tuple[int, int]:
    """
    Locates exact subword token boundary indices (start_idx, end_idx) using character-level offset alignment.
    """
    phrase_clean = str(phrase).strip().lower()
    sent_lower = str(sentence).lower()
    char_start = sent_lower.find(phrase_clean)
    if char_start == -1:
        words = phrase_clean.split()
        if words:
            char_start = sent_lower.find(words[0])
            char_end = char_start + len(words[0]) if char_start != -1 else 0
        else:
            return 0, 0
    else:
        char_end = char_start + len(phrase_clean)

    if char_start == -1:
        return 0, 0

    try:
        raw_tok = getattr(tokenizer, "tokenizer", tokenizer)
        encoding = raw_tok(sentence, return_offsets_mapping=True, truncation=True, max_length=max_len)
        offsets = encoding.get("offset_mapping", [])
    except Exception:
        return 0, 0

    start_token = None
    end_token = None

    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if start_token is None and e > char_start:
            start_token = i
        if start_token is not None and s < char_end:
            end_token = i

    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = start_token

    return min(start_token, max_len - 1), min(end_token, max_len - 1)


class DimABSADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        lexicon_engine: MasterLexiconEngine,
        global_lang_map: Dict[str, str],
        graph_builder: FastGraphBuilder,
        max_length: int = 128,
        max_switch_dist: int = 16
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.lexicon_engine = lexicon_engine
        self.global_lang_map = global_lang_map
        self.graph_builder = graph_builder
        self.max_length = max_length
        self.max_switch_dist = max_switch_dist

        # Pre-cache all tensor objects for lightning-fast training
        self.samples = [self._process_row(i) for i in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.samples)

    def _process_row(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sentence = str(row["sentence"])
        aspect = str(row["aspect"])
        opinion = str(row["opinion"])
        v_target = float(row["valence"])
        a_target = float(row["arousal"])
        
        row_cs_map = parse_code_switch_tags(row.get("code_switch", ""))

        raw_words = sentence.split() or ["<unk>"]
        word_lang_tags = []
        for w in raw_words:
            clean_w = re.sub(r"[^\w]", "", w.lower())
            tag = row_cs_map.get(clean_w, self.global_lang_map.get(clean_w, "EN"))
            word_lang_tags.append(tag)

        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        subwords = self.tokenizer.convert_ids_to_tokens(input_ids)
        seq_len = int(attention_mask.sum().item())
        valid_subwords = subwords[:seq_len]

        subword_lang_tags = []
        lexicon_priors = []

        w_idx = 0
        for sub_tok in valid_subwords:
            clean_sub = re.sub(r"[^\w]", "", sub_tok.lower())
            v_lex, a_lex, is_anchor = self.lexicon_engine.lookup(clean_sub)
            lexicon_priors.append([v_lex, a_lex, is_anchor])

            subword_lang_tags.append(word_lang_tags[w_idx] if w_idx < len(word_lang_tags) else "EN")
            if not sub_tok.startswith("##") and not sub_tok.startswith("Ġ"):
                w_idx = min(len(word_lang_tags) - 1, w_idx + 1)

        lexicon_tensor = torch.zeros((self.max_length, 3), dtype=torch.float32)
        if lexicon_priors:
            lexicon_tensor[:seq_len] = torch.tensor(lexicon_priors, dtype=torch.float32)

        d_switch_list = compute_signed_switch_distance(subword_lang_tags, max_dist=self.max_switch_dist)
        d_switch_indices = torch.zeros(self.max_length, dtype=torch.long)
        for i, d in enumerate(d_switch_list):
            if i < self.max_length:
                d_switch_indices[i] = d + self.max_switch_dist

        asp_start, asp_end = locate_subword_span(sentence, aspect, self.tokenizer, self.max_length)
        op_start, op_end = locate_subword_span(sentence, opinion, self.tokenizer, self.max_length)

        # Compute Opinion Lexicon Prior vector [V_lex_op, A_lex_op]
        op_v_list, op_a_list = [], []
        for i in range(op_start, min(seq_len, op_end + 1)):
            if lexicon_priors and i < len(lexicon_priors):
                v_l, a_l, conf = lexicon_priors[i]
                if conf > 0:
                    op_v_list.append(v_l)
                    op_a_list.append(a_l)
        
        op_v_mean = float(np.mean(op_v_list)) if op_v_list else 0.50
        op_a_mean = float(np.mean(op_a_list)) if op_a_list else 0.50
        op_lex_tensor = torch.tensor([op_v_mean, op_a_mean], dtype=torch.float32)

        A_syn, A_sem, A_cs, A_ao = self.graph_builder.build_graphs(
            tokens=valid_subwords,
            lang_tags=subword_lang_tags,
            aspect_span=(asp_start, asp_end),
            opinion_span=(op_start, op_end)
        )

        adj_multi = torch.zeros((4, self.max_length, self.max_length), dtype=torch.float32)
        adj_multi[0, :seq_len, :seq_len] = torch.from_numpy(A_syn)
        adj_multi[1, :seq_len, :seq_len] = torch.from_numpy(A_sem)
        adj_multi[2, :seq_len, :seq_len] = torch.from_numpy(A_cs)
        adj_multi[3, :seq_len, :seq_len] = torch.from_numpy(A_ao)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "d_switch_indices": d_switch_indices,
            "lexicon_priors": lexicon_tensor,
            "op_lexicon_prior": op_lex_tensor,
            "adj_matrices": adj_multi,
            "aspect_span": torch.tensor([asp_start, asp_end], dtype=torch.long),
            "opinion_span": torch.tensor([op_start, op_end], dtype=torch.long),
            "valence_target": torch.tensor(v_target, dtype=torch.float32),
            "arousal_target": torch.tensor(a_target, dtype=torch.float32),
            "sentence": sentence,
            "aspect": aspect,
            "opinion": opinion
        }


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# 6. Tokenizer Wrapper & Master Builder
# ---------------------------------------------------------------------------

class HinglishTokenizerWrapper:
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.model_name = model_name
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"[Tokenizer] Initialized HuggingFace Tokenizer: {model_name}", flush=True)

    def __call__(self, text: str, max_length: int = 128, padding: str = "max_length", truncation: bool = True, return_tensors: Optional[str] = None):
        return self.tokenizer(text, max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids.tolist())


def build_data_loaders(
    csv_path: str = "DimABSA_Final_Dataset_600.csv",
    hindi_nrc_path: str = "Hindi-NRC-VAD-Lexicon.txt",
    all_txt_path: str = "all.txt",
    data_dir: str = "data",
    batch_size: int = 16,
    max_length: int = 128,
    max_switch_dist: int = 16,
    tokenizer_name: str = "xlm-roberta-base"
) -> Tuple[DataLoader, DataLoader, DataLoader, DimABSADataset, DimABSADataset, DimABSADataset]:
    os.makedirs(data_dir, exist_ok=True)

    # 1. Load and flatten DimABSA dataset
    flattened_df = load_and_flatten_dataset(csv_path)
    flattened_df.to_csv(os.path.join(data_dir, "flattened_dimabsa.csv"), index=False)

    # 2. Stratified Continuous 70/15/15 Split
    df_train, df_val, df_test = stratified_continuous_split(flattened_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    # 3. Master Lexicon Engine (Hindi-NRC-VAD + Dataset)
    lexicon_engine = MasterLexiconEngine(hindi_nrc_path=hindi_nrc_path, df_dataset=flattened_df)

    # 4. Master Language Tags (all.txt)
    global_lang_map = load_all_txt_language_map(file_path=all_txt_path)

    # 5. Infrastructure
    tokenizer = HinglishTokenizerWrapper(tokenizer_name)
    graph_builder = FastGraphBuilder()

    print("[Data Pipeline] Pre-caching Train / Val / Test tensors in memory...", flush=True)
    train_dataset = DimABSADataset(df_train, tokenizer, lexicon_engine, global_lang_map, graph_builder, max_length, max_switch_dist)
    val_dataset = DimABSADataset(df_val, tokenizer, lexicon_engine, global_lang_map, graph_builder, max_length, max_switch_dist)
    test_dataset = DimABSADataset(df_test, tokenizer, lexicon_engine, global_lang_map, graph_builder, max_length, max_switch_dist)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[Data Pipeline] Ready! Batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}", flush=True)
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# 7. Automated Multi-Aspect & Multi-Opinion Span Extractor
# ---------------------------------------------------------------------------

class HinglishAspectOpinionExtractor:
    """
    Syntactic-Lexical Span Extractor for Code-Mixed Hinglish.
    Extracts all (aspect, opinion) pairs present in a multi-aspect sentence.
    """
    def __init__(self, lexicon_engine: Optional[MasterLexiconEngine] = None):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["textcat"])
        except Exception:
            self.nlp = spacy.blank("en")
        self.lexicon_engine = lexicon_engine

        # Common Hinglish discourse connectives and stop particles
        self.discourse_markers = {"lekin", "par", "parantu", "kintu", "magar", "but", "however", "aur", "and", "or", "ya", "phir", "bhi", "toh"}

    def extract_aspect_opinion_pairs(self, sentence: str) -> List[Dict[str, str]]:
        """
        Extracts clean, non-redundant (aspect, opinion) pairs from multi-aspect sentences.
        """
        if not sentence or not sentence.strip():
            return []

        # Split clauses by discourse markers or punctuation
        delimiters = r"[,;.?!\n]|(?:\s+(?:lekin|par|parantu|kintu|magar|but|however|aur|and|or|ya|phir|bhi)\s+)"
        raw_clauses = re.split(delimiters, sentence, flags=re.IGNORECASE)
        clauses = [c.strip() for c in raw_clauses if c and len(c.strip()) > 3]

        if not clauses:
            clauses = [sentence.strip()]

        pairs = []
        stop_particles = {"hai", "tha", "thi", "the", "ka", "ki", "ke", "ko", "se", "me", "mein", "pe", "par", "ye", "yeh", "wo", "woh", "ispe", "unpe", "ji", "kr", "kar", "rha", "raha", "rahi", "rahe", "hona", "hua", "hui"}

        for clause in clauses:
            words = clause.split()
            if not words:
                continue

            doc = self.nlp(clause)
            
            # Find candidate aspect: prefer NOUN/PROPN or first content noun chunk
            best_aspect = None
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] or token.dep_ in ["nsubj", "dobj"]:
                    clean_t = re.sub(r"[^\w]", "", token.text.lower()).strip()
                    if clean_t not in stop_particles and len(clean_t) > 1:
                        best_aspect = token.text
                        break

            if not best_aspect:
                for chunk in doc.noun_chunks:
                    clean_chunk = re.sub(r"[^\w\s]", "", chunk.text.lower()).strip()
                    chunk_words = [w for w in clean_chunk.split() if w not in stop_particles and len(w) > 1]
                    if chunk_words:
                        best_aspect = chunk_words[0]
                        break

            if not best_aspect:
                for w in words:
                    clean_w = re.sub(r"[^\w]", "", w.lower())
                    if clean_w not in stop_particles and len(clean_w) > 1:
                        best_aspect = w
                        break

            if not best_aspect:
                best_aspect = words[0]

            # Construct opinion: all remaining words in the clause
            op_words = []
            asp_tokens = [w.lower() for w in best_aspect.split()]
            for w in words:
                clean_w = re.sub(r"[^\w]", "", w.lower())
                if clean_w not in asp_tokens:
                    op_words.append(w)

            opinion_str = " ".join(op_words).strip() if op_words else clause

            # Deduplicate by aspect name
            if not any(p["aspect"].lower() == best_aspect.lower() for p in pairs):
                pairs.append({
                    "aspect": best_aspect,
                    "opinion": opinion_str
                })


        return pairs



if __name__ == "__main__":
    train_ld, val_ld, test_ld, _, _, _ = build_data_loaders()
    print("Master Data Pipeline executed successfully.")

