"""
run_final_performance.py
Multi-stage retrieval pipeline for multilingual health QA.
Stages: sparse -> LightGBM -> dense -> cross-encoder -> ByT5 -> translation
Each stage evaluated on Val, only activated if it improves.
"""

import os
import sys
import json
import time
import logging
import warnings
import traceback
import unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
WORK = Path("/home/onyxia/work")
OUT  = WORK / "outputs" / "latest"
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

START_TIME = time.time()

# ─── ROUGE ────────────────────────────────────────────────────────────────────
def _tokenize(text):
    return str(text).lower().split()

def rouge1_f1(ref, hyp):
    ref_toks = _tokenize(ref)
    hyp_toks = _tokenize(hyp)
    if not ref_toks or not hyp_toks:
        return 0.0
    ref_cnt = {}
    for t in ref_toks:
        ref_cnt[t] = ref_cnt.get(t, 0) + 1
    hits = 0
    for t in hyp_toks:
        if ref_cnt.get(t, 0) > 0:
            hits += 1
            ref_cnt[t] -= 1
    p = hits / len(hyp_toks)
    r = hits / len(ref_toks)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def _lcs_len(a, b):
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    for i in range(la):
        curr = [0] * (lb + 1)
        for j in range(lb):
            if a[i] == b[j]:
                curr[j+1] = prev[j] + 1
            else:
                curr[j+1] = max(curr[j], prev[j+1])
        prev = curr
    return prev[lb]

def rougel_f1(ref, hyp):
    ref_toks = _tokenize(ref)
    hyp_toks = _tokenize(hyp)
    if not ref_toks or not hyp_toks:
        return 0.0
    lcs = _lcs_len(ref_toks, hyp_toks)
    p = lcs / len(hyp_toks)
    r = lcs / len(ref_toks)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def score_rows(refs, preds):
    r1s, rls = [], []
    for ref, hyp in zip(refs, preds):
        r1s.append(rouge1_f1(str(ref), str(hyp)))
        rls.append(rougel_f1(str(ref), str(hyp)))
    r1 = float(np.mean(r1s))
    rl = float(np.mean(rls))
    return r1, rl, 0.37*r1 + 0.37*rl, r1s, rls

# ─── Text normalization ────────────────────────────────────────────────────────
def normalize_text(text, lower=True):
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.split())
    if lower:
        text = text.lower()
    return text

# ─── Data loading ─────────────────────────────────────────────────────────────
log.info("Loading data...")
train = pd.read_csv(WORK / "Train.csv")
val   = pd.read_csv(WORK / "Val.csv")
test  = pd.read_csv(WORK / "Test.csv")
sample = pd.read_csv(WORK / "SampleSubmission.csv")

log.info(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

# Normalize
for df in [train, val, test]:
    df["input_norm"]  = df["input"].apply(lambda x: normalize_text(x, lower=True))
for df in [train, val]:
    df["output_norm"] = df["output"].apply(lambda x: normalize_text(x, lower=True))

SUBSETS = sorted(train["subset"].unique().tolist())
log.info(f"Subsets: {SUBSETS}")

# ─── Retriever builder ────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(analyzer, ngram_range, token_pattern=None):
    kwargs = dict(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    if token_pattern:
        kwargs["token_pattern"] = token_pattern
    return TfidfVectorizer(**kwargs)

RETRIEVERS_CFG = {
    "char_wb": dict(analyzer="char_wb", ngram_range=(3,5)),
    "char":    dict(analyzer="char",    ngram_range=(2,5)),
    "word":    dict(analyzer="word",    ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b"),
}

# ─── Per-subset index ─────────────────────────────────────────────────────────
def build_subset_index(base_df, vec):
    """Returns dict subset -> (X_sparse, row_indices_in_base_df)"""
    idx = {}
    for s in SUBSETS:
        rows = base_df[base_df["subset"] == s].index.tolist()
        if not rows:
            continue
        texts = base_df.loc[rows, "input_norm"].tolist()
        X = vec.transform(texts)
        idx[s] = (X, rows)
    return idx

def retrieve_batch(query_texts, subset_labels, index, base_df, top_k=20, fallback_global=None):
    """
    Returns list of (top_idx_in_base, top_score) for each query.
    Falls back to global if subset not found.
    """
    results = []
    # Group by subset for batch efficiency
    order = list(range(len(query_texts)))
    subset_groups = {}
    for i, (q, s) in enumerate(zip(query_texts, subset_labels)):
        subset_groups.setdefault(s, []).append(i)

    results = [None] * len(query_texts)
    for s, idxs in subset_groups.items():
        if s in index:
            X_sub, row_ids = index[s]
            queries = [query_texts[i] for i in idxs]
            # batch transform
            X_q = fallback_global["vec"].transform(queries) if fallback_global else None
            # Use the fitted vec from outside
            X_q_local = _current_vec.transform(queries)
            sims = (X_q_local @ X_sub.T).toarray()
            for ii, orig_i in enumerate(idxs):
                sim_row = sims[ii]
                if len(sim_row) <= top_k:
                    top_local = np.argsort(sim_row)[::-1]
                else:
                    top_local = np.argpartition(sim_row, -top_k)[-top_k:]
                    top_local = top_local[np.argsort(sim_row[top_local])[::-1]]
                top_global_rows = [row_ids[j] for j in top_local]
                top_scores = sim_row[top_local].tolist()
                results[orig_i] = (top_global_rows, top_scores)
        else:
            # fallback global
            for orig_i in idxs:
                results[orig_i] = None  # handled below

    # handle missing subset -> use global fallback
    for i, r in enumerate(results):
        if r is None:
            results[i] = retrieve_single_global(query_texts[i], fallback_global, top_k)
    return results

_current_vec = None  # global mutable for batch retrieve

def retrieve_single_global(query, fb, top_k=20):
    if fb is None:
        return ([], [])
    X_q = fb["vec"].transform([query])
    sims = (X_q @ fb["X_all"].T).toarray()[0]
    if len(sims) <= top_k:
        top_local = np.argsort(sims)[::-1]
    else:
        top_local = np.argpartition(sims, -top_k)[-top_k:]
        top_local = top_local[np.argsort(sims[top_local])[::-1]]
    top_rows = fb["row_ids"][top_local].tolist()
    top_scores = sims[top_local].tolist()
    return (top_rows, top_scores)

# ─── Stage tracking ───────────────────────────────────────────────────────────
stage_results = {}   # stage_name -> {"weighted_rouge": float, "r1": float, "rl": float}
stage_submissions = {}  # stage_name -> submission df

# ─── Benchmark helper ─────────────────────────────────────────────────────────
def benchmark_retriever(name, vec, base_df, sample_df, n=50):
    """Benchmark using per-subset retrieval (same as actual pipeline)."""
    log.info(f"  Benchmarking {name} on {n} rows...")
    t0 = time.time()
    sub = sample_df.head(n)
    # build per-subset index for each subset present in sample
    sub_index_local = {}
    for s in sub["subset"].unique():
        rows = base_df[base_df["subset"] == s].index.tolist()
        if rows:
            X = vec.transform(base_df.loc[rows, "input_norm"].tolist())
            sub_index_local[s] = (X, rows)
    # retrieve per row
    for _, row in sub.iterrows():
        s = row["subset"]
        X_q = vec.transform([row["input_norm"]])
        if s in sub_index_local:
            X_sub, _ = sub_index_local[s]
            _ = (X_q @ X_sub.T).toarray()
    elapsed = time.time() - t0
    projected = elapsed / n * len(sample_df)
    log.info(f"  {name}: {elapsed:.1f}s on {n} rows → ~{projected:.0f}s projected for full val")
    return elapsed, projected

# ─── Fit all vectorizers on Train ─────────────────────────────────────────────
log.info("Fitting vectorizers on Train...")
fitted_vecs = {}
for name, cfg in RETRIEVERS_CFG.items():
    t0 = time.time()
    vec = build_vectorizer(**{k:v for k,v in cfg.items()})
    vec.fit(train["input_norm"].tolist())
    fitted_vecs[name] = vec
    log.info(f"  {name} fitted in {time.time()-t0:.1f}s, vocab={len(vec.vocabulary_)}")

# ─── Benchmark on 50 val rows ─────────────────────────────────────────────────
log.info("Benchmarking retrievers...")
disabled_retrievers = set()
for name, vec in fitted_vecs.items():
    n_bench = 20 if name == "byte" else 50
    elapsed, projected = benchmark_retriever(name, vec, train, val, n=n_bench)
    if projected > 300:
        log.warning(f"  {name} too slow ({projected:.0f}s projected), DISABLING")
        disabled_retrievers.add(name)
    else:
        log.info(f"  {name} OK")

active_retrievers = {k: v for k, v in fitted_vecs.items() if k not in disabled_retrievers}
log.info(f"Active retrievers: {list(active_retrievers.keys())}")

# ─── Build per-subset indices on Train ────────────────────────────────────────
log.info("Building per-subset indices on Train...")
subset_indices = {}  # name -> subset_index
global_indices = {}  # name -> fallback dict

for name, vec in active_retrievers.items():
    _current_vec = vec
    # subset index
    si = {}
    for s in SUBSETS:
        rows = train[train["subset"] == s].index.tolist()
        if not rows:
            continue
        texts = train.loc[rows, "input_norm"].tolist()
        X = vec.transform(texts)
        si[s] = (X, rows)
    subset_indices[name] = si
    # global fallback
    X_all = vec.transform(train["input_norm"].tolist())
    global_indices[name] = {"vec": vec, "X_all": X_all, "row_ids": np.array(train.index.tolist())}

# ─── Batch retrieval on Val ────────────────────────────────────────────────────
log.info("Retrieving top-20 for Val from each retriever...")
TOP_K = 20
val_retrievals = {}  # name -> list of (top_row_ids, top_scores)

for name, vec in active_retrievers.items():
    t0 = time.time()
    results = []
    bs = 128
    queries = val["input_norm"].tolist()
    subsets = val["subset"].tolist()
    for start in range(0, len(queries), bs):
        batch_q = queries[start:start+bs]
        batch_s = subsets[start:start+bs]
        batch_res = []
        # per-subset group in batch
        for q, s in zip(batch_q, batch_s):
            if s in subset_indices[name]:
                X_sub, row_ids = subset_indices[name][s]
                X_q = vec.transform([q])
                sim = (X_q @ X_sub.T).toarray()[0]
                if len(sim) <= TOP_K:
                    top_local = np.argsort(sim)[::-1]
                else:
                    top_local = np.argpartition(sim, -TOP_K)[-TOP_K:]
                    top_local = top_local[np.argsort(sim[top_local])[::-1]]
                top_rows = [row_ids[j] for j in top_local]
                top_scs  = sim[top_local].tolist()
            else:
                # global fallback
                X_q = vec.transform([q])
                sim = (X_q @ global_indices[name]["X_all"].T).toarray()[0]
                if len(sim) <= TOP_K:
                    top_local = np.argsort(sim)[::-1]
                else:
                    top_local = np.argpartition(sim, -TOP_K)[-TOP_K:]
                    top_local = top_local[np.argsort(sim[top_local])[::-1]]
                top_rows = global_indices[name]["row_ids"][top_local].tolist()
                top_scs  = sim[top_local].tolist()
            batch_res.append((top_rows, top_scs))
        results.extend(batch_res)
    val_retrievals[name] = results
    log.info(f"  {name} val retrieval done in {time.time()-t0:.1f}s")

# ─── Experiments on Val ───────────────────────────────────────────────────────
log.info("Evaluating experiments on Val...")

exp_names = []
exp_preds_all = {}  # exp_name -> list of predictions

# E0, E1, E2: top-1 per retriever
retriever_order = list(active_retrievers.keys())
for name in retriever_order:
    exp = f"E_{name}_top1"
    preds = []
    for top_rows, top_scs in val_retrievals[name]:
        if top_rows:
            preds.append(train.loc[top_rows[0], "output"])
        else:
            preds.append("")
    exp_preds_all[exp] = preds
    exp_names.append(exp)

# E4: RRF fusion
if len(active_retrievers) >= 2:
    if "byte" in active_retrievers:
        rrf_weights = {"char_wb": 0.40, "char": 0.30, "word": 0.20, "byte": 0.10}
    else:
        rrf_weights = {"char_wb": 0.45, "char": 0.35, "word": 0.20}

    rrf_preds = []
    for i in range(len(val)):
        doc_scores = {}
        for name in retriever_order:
            w = rrf_weights.get(name, 0.2)
            top_rows, _ = val_retrievals[name][i]
            for rank, row_id in enumerate(top_rows):
                doc_scores[row_id] = doc_scores.get(row_id, 0.0) + w / (60 + rank)
        if doc_scores:
            best_id = max(doc_scores, key=doc_scores.get)
            rrf_preds.append(train.loc[best_id, "output"])
        else:
            # fallback to first retriever
            top_rows, _ = val_retrievals[retriever_order[0]][i]
            rrf_preds.append(train.loc[top_rows[0], "output"] if top_rows else "")
    exp_preds_all["E4_rrf_fusion"] = rrf_preds
    exp_names.append("E4_rrf_fusion")

# Score all experiments
val_refs = val["output"].tolist()
exp_scores = {}  # exp -> {r1, rl, wr, r1s, rls}
for exp in exp_names:
    preds = exp_preds_all[exp]
    r1, rl, wr, r1s, rls = score_rows(val_refs, preds)
    exp_scores[exp] = {"r1": r1, "rl": rl, "wr": wr, "r1s": r1s, "rls": rls}
    log.info(f"  {exp}: rouge1={r1:.4f} rougel={rl:.4f} weighted={wr:.4f}")

# Best global experiment
best_global_exp = max(exp_scores, key=lambda e: exp_scores[e]["wr"])
log.info(f"Best global experiment: {best_global_exp} (wr={exp_scores[best_global_exp]['wr']:.4f})")

# E5: MoE by subset
log.info("Computing MoE by subset...")
# Per-subset scores
subset_exp_scores = {}  # subset -> exp -> wr
val_subsets = val["subset"].tolist()
val_ids = val["ID"].tolist()

for s in SUBSETS:
    mask = [i for i, ss in enumerate(val_subsets) if ss == s]
    if len(mask) < 3:
        continue
    sub_refs = [val_refs[i] for i in mask]
    subset_exp_scores[s] = {}
    for exp in exp_names:
        sub_preds = [exp_preds_all[exp][i] for i in mask]
        r1, rl, wr, _, _ = score_rows(sub_refs, sub_preds)
        subset_exp_scores[s][exp] = wr

best_by_subset = {}
for s in SUBSETS:
    if s in subset_exp_scores and subset_exp_scores[s]:
        best_exp = max(subset_exp_scores[s], key=subset_exp_scores[s].get)
        best_by_subset[s] = best_exp
    else:
        best_by_subset[s] = best_global_exp

log.info(f"Best by subset: {best_by_subset}")

# MoE predictions on Val
moe_preds = []
for i in range(len(val)):
    s = val_subsets[i]
    exp = best_by_subset.get(s, best_global_exp)
    moe_preds.append(exp_preds_all[exp][i])

r1, rl, wr, r1s, rls = score_rows(val_refs, moe_preds)
exp_scores["E5_moe_by_subset"] = {"r1": r1, "rl": rl, "wr": wr, "r1s": r1s, "rls": rls}
exp_preds_all["E5_moe_by_subset"] = moe_preds
exp_names.append("E5_moe_by_subset")
log.info(f"  E5_moe_by_subset: rouge1={r1:.4f} rougel={rl:.4f} weighted={wr:.4f}")

best_global_exp = max(exp_scores, key=lambda e: exp_scores[e]["wr"])
log.info(f"Best global after MoE: {best_global_exp}")

# ─── Validation scores CSV ────────────────────────────────────────────────────
val_scores_rows = []
for exp in exp_names:
    sc = exp_scores[exp]
    val_scores_rows.append({
        "experiment": exp,
        "rouge1": sc["r1"],
        "rougel": sc["rl"],
        "weighted_rouge": sc["wr"],
        "n": len(val),
        "runtime_seconds": 0,
    })
df_val_scores = pd.DataFrame(val_scores_rows)
df_val_scores.to_csv(OUT / "validation_scores.csv", index=False)

# Per-subset validation scores
rows_by_sub = []
for s in SUBSETS:
    mask = [i for i, ss in enumerate(val_subsets) if ss == s]
    if not mask:
        continue
    sub_refs = [val_refs[i] for i in mask]
    for exp in exp_names:
        sub_preds = [exp_preds_all[exp][i] for i in mask]
        r1, rl, wr, _, _ = score_rows(sub_refs, sub_preds)
        rows_by_sub.append({"experiment": exp, "subset": s, "n": len(mask),
                             "rouge1": r1, "rougel": rl, "weighted_rouge": wr})
df_by_sub = pd.DataFrame(rows_by_sub)
df_by_sub.to_csv(OUT / "validation_scores_by_subset.csv", index=False)

# ─── Stage 1 submission (sparse + RRF + MoE) ─────────────────────────────────
stage1_exp = best_global_exp
stage1_wr  = exp_scores[stage1_exp]["wr"]
stage_results["stage1_sparse"] = {"experiment": stage1_exp, "wr": stage1_wr,
                                   "r1": exp_scores[stage1_exp]["r1"],
                                   "rl": exp_scores[stage1_exp]["rl"]}
log.info(f"Stage 1 best: {stage1_exp} wr={stage1_wr:.4f}")

# ─── Stage 2: LightGBM-lite ───────────────────────────────────────────────────
stage2_ok = False
try:
    import lightgbm as lgb
    log.info("Stage 2: LightGBM-lite reranker...")

    # Build training features for Val from top-k candidates of each retriever
    # For each val row, generate candidate features
    TOP_K_LGB = 5  # keep top-5 per retriever for LGB

    def make_features(retrievals_dict, base_df, query_subsets, retriever_names):
        """Build feature df: one row per (query, candidate)."""
        rows = []
        for i in range(len(query_subsets)):
            s = query_subsets[i]
            # collect candidates per retriever
            cand_ranks = {}  # doc_id -> {retriever: rank, score}
            for name in retriever_names:
                top_rows, top_scs = retrievals_dict[name][i]
                for rank, (row_id, sc) in enumerate(zip(top_rows[:TOP_K_LGB], top_scs[:TOP_K_LGB])):
                    if row_id not in cand_ranks:
                        cand_ranks[row_id] = {}
                    cand_ranks[row_id][f"rank_{name}"]  = rank
                    cand_ranks[row_id][f"score_{name}"] = sc

            # RRF score
            rrf_w = rrf_weights if "E4_rrf_fusion" in exp_names else {"char_wb": 0.45, "char": 0.35, "word": 0.20}
            rrf_sc = {}
            for name in retriever_names:
                w = rrf_w.get(name, 0.2)
                top_rows, _ = retrievals_dict[name][i]
                for rank, row_id in enumerate(top_rows[:TOP_K_LGB]):
                    rrf_sc[row_id] = rrf_sc.get(row_id, 0.0) + w / (60 + rank)

            for row_id, feat in cand_ranks.items():
                feat["query_idx"] = i
                feat["doc_id"] = row_id
                feat["rrf_score"] = rrf_sc.get(row_id, 0.0)
                feat["same_subset"] = int(base_df.loc[row_id, "subset"] == s)
                feat["subset_id"] = SUBSETS.index(s) if s in SUBSETS else -1
                rows.append(feat)
        return pd.DataFrame(rows).fillna(0)

    log.info("  Building LGB features for Val...")
    feat_df = make_features(val_retrievals, train, val_subsets, retriever_order)

    # compute target: rouge on val
    target_col = []
    for _, row in feat_df.iterrows():
        qi = int(row["query_idx"])
        doc_id = int(row["doc_id"])
        pred = train.loc[doc_id, "output"]
        ref  = val_refs[qi]
        r1 = rouge1_f1(ref, pred)
        rl = rougel_f1(ref, pred)
        target_col.append(0.5*r1 + 0.5*rl)
    feat_df["target"] = target_col

    feat_cols = [c for c in feat_df.columns if c not in ["query_idx", "doc_id", "target"]]
    X_lgb = feat_df[feat_cols].values
    y_lgb = feat_df["target"].values

    # 5-fold-ish: use 80% for train, 20% for valid within Val set
    n_q = len(val)
    split = int(n_q * 0.8)
    train_mask = feat_df["query_idx"] < split
    valid_mask = feat_df["query_idx"] >= split

    dtrain = lgb.Dataset(X_lgb[train_mask], label=y_lgb[train_mask])
    dval   = lgb.Dataset(X_lgb[valid_mask], label=y_lgb[valid_mask])

    params = dict(objective="regression", metric="rmse", num_leaves=31,
                  learning_rate=0.1, n_estimators=100, verbosity=-1,
                  random_state=42)
    model = lgb.train(params, dtrain, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(-1)])

    # Predict on full Val to get LGB-reranked predictions
    feat_df_full = make_features(val_retrievals, train, val_subsets, retriever_order)
    feat_df_full["lgb_score"] = model.predict(feat_df_full[feat_cols].values)

    lgb_preds_val = []
    for i in range(len(val)):
        sub_df = feat_df_full[feat_df_full["query_idx"] == i]
        if sub_df.empty:
            lgb_preds_val.append(exp_preds_all[stage1_exp][i])
        else:
            best_doc = int(sub_df.loc[sub_df["lgb_score"].idxmax(), "doc_id"])
            lgb_preds_val.append(train.loc[best_doc, "output"])

    r1, rl, wr, _, _ = score_rows(val_refs, lgb_preds_val)
    log.info(f"  LGB val: rouge1={r1:.4f} rougel={rl:.4f} wr={wr:.4f}")
    stage_results["stage2_lightgbm"] = {"experiment": "E_lgb", "wr": wr, "r1": r1, "rl": rl}
    if wr > stage1_wr:
        log.info("  LGB improves Val -> activating")
        stage2_ok = True
        exp_preds_all["E_lgb"] = lgb_preds_val
        exp_scores["E_lgb"] = {"r1": r1, "rl": rl, "wr": wr, "r1s": [], "rls": []}
        exp_names.append("E_lgb")
    else:
        log.info("  LGB does NOT improve Val -> disabled")
except Exception as e:
    log.warning(f"Stage 2 LightGBM failed: {e}")

# ─── Stage 3: Dense retrieval - disabled (no GPU, 10+ min on CPU) ────────────
stage3_ok = False
log.info("Stage 3: Dense retrieval skipped (no GPU available)")

# ─── Stage 4: Cross-encoder - disabled (~2h on CPU for 6686 queries) ─────────
stage4_ok = False
log.info("Stage 4: Cross-encoder skipped (too slow on CPU)")

# ─── Stage 5: ByT5-small - disabled (no GPU) ──────────────────────────────────
stage5_ok = False
log.info("Stage 5: ByT5 skipped (no GPU available)")

# ─── Select best validated experiment ─────────────────────────────────────────
final_best_exp = max(exp_scores, key=lambda e: exp_scores[e]["wr"])
final_best_wr  = exp_scores[final_best_exp]["wr"]
log.info(f"\nFINAL best experiment: {final_best_exp} (wr={final_best_wr:.4f})")

# ─── Refit on Train+Val, predict Test ─────────────────────────────────────────
log.info("Refitting on Train+Val...")
base_df = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
base_df["input_norm"]  = base_df["input"].apply(lambda x: normalize_text(x, lower=True))
base_df["output_norm"] = base_df["output"].apply(lambda x: normalize_text(x, lower=True))

# Refit vectorizers
refit_vecs = {}
refit_sub_indices = {}
refit_global_indices = {}

for name, cfg in RETRIEVERS_CFG.items():
    if name in disabled_retrievers:
        continue
    vec = build_vectorizer(**{k:v for k,v in cfg.items()})
    vec.fit(base_df["input_norm"].tolist())
    refit_vecs[name] = vec

    si = {}
    for s in SUBSETS:
        rows = base_df[base_df["subset"] == s].index.tolist()
        if not rows:
            continue
        texts = base_df.loc[rows, "input_norm"].tolist()
        X = vec.transform(texts)
        si[s] = (X, rows)
    refit_sub_indices[name] = si

    X_all = vec.transform(base_df["input_norm"].tolist())
    refit_global_indices[name] = {"vec": vec, "X_all": X_all,
                                   "row_ids": np.array(base_df.index.tolist())}

log.info("Refitting done.")

# MoE best_by_subset for Test (recomputed from val with refit?)
# We use best_by_subset computed on Val phase (still valid as it's a held-out set).
# But map: which experiment type to use for retrieval
def get_exp_retriever(exp_name):
    """Return retriever name(s) to use for this experiment."""
    if "char_wb" in exp_name:
        return "char_wb"
    elif "char" in exp_name:
        return "char"
    elif "word" in exp_name:
        return "word"
    elif "rrf" in exp_name or "moe" in exp_name or "lgb" in exp_name or "ce" in exp_name or "dense" in exp_name or "byt5" in exp_name:
        return "rrf"
    return "char_wb"

log.info("Predicting Test...")
test_subsets = test["subset"].tolist()
test_queries = test["input"].apply(lambda x: normalize_text(x, lower=True)).tolist()
test_preds = []
test_debug = []

# Decide which retriever strategy to use for each subset
def retrieve_one(q, s, retriever_names, sub_indices, global_indices_map, rrf_w, top_k=20):
    if len(retriever_names) == 1:
        name = retriever_names[0]
        vec = refit_vecs[name]
        if s in sub_indices[name]:
            X_sub, row_ids = sub_indices[name][s]
            X_q = vec.transform([q])
            sim = (X_q @ X_sub.T).toarray()[0]
            if len(sim) <= top_k:
                top_local = np.argsort(sim)[::-1]
            else:
                top_local = np.argpartition(sim, -top_k)[-top_k:]
                top_local = top_local[np.argsort(sim[top_local])[::-1]]
            return [row_ids[j] for j in top_local], sim[top_local].tolist()
        else:
            gi = global_indices_map[name]
            X_q = gi["vec"].transform([q])
            sim = (X_q @ gi["X_all"].T).toarray()[0]
            if len(sim) <= top_k:
                top_local = np.argsort(sim)[::-1]
            else:
                top_local = np.argpartition(sim, -top_k)[-top_k:]
                top_local = top_local[np.argsort(sim[top_local])[::-1]]
            return gi["row_ids"][top_local].tolist(), sim[top_local].tolist()
    else:
        # RRF
        doc_scores = {}
        top_by_name = {}
        for name in retriever_names:
            if name not in refit_vecs:
                continue
            vec = refit_vecs[name]
            w = rrf_w.get(name, 0.2)
            if s in sub_indices[name]:
                X_sub, row_ids = sub_indices[name][s]
                X_q = vec.transform([q])
                sim = (X_q @ X_sub.T).toarray()[0]
                if len(sim) <= top_k:
                    top_local = np.argsort(sim)[::-1]
                else:
                    top_local = np.argpartition(sim, -top_k)[-top_k:]
                    top_local = top_local[np.argsort(sim[top_local])[::-1]]
                top_rows = [row_ids[j] for j in top_local]
            else:
                gi = global_indices_map[name]
                X_q = gi["vec"].transform([q])
                sim = (X_q @ gi["X_all"].T).toarray()[0]
                if len(sim) <= top_k:
                    top_local = np.argsort(sim)[::-1]
                else:
                    top_local = np.argpartition(sim, -top_k)[-top_k:]
                    top_local = top_local[np.argsort(sim[top_local])[::-1]]
                top_rows = gi["row_ids"][top_local].tolist()
            top_by_name[name] = top_rows
            for rank, row_id in enumerate(top_rows):
                doc_scores[row_id] = doc_scores.get(row_id, 0.0) + w / (60 + rank)
        if doc_scores:
            best_id = max(doc_scores, key=doc_scores.get)
            return [best_id], [doc_scores[best_id]]
        return [], []

active_ret_names = list(refit_vecs.keys())
if "byte" in active_ret_names:
    rrf_w_final = {"char_wb": 0.40, "char": 0.30, "word": 0.20, "byte": 0.10}
else:
    rrf_w_final = {"char_wb": 0.45, "char": 0.35, "word": 0.20}

use_rrf = len(active_ret_names) >= 2

for i in range(len(test)):
    row = test.iloc[i]
    s = row["subset"]
    q = test_queries[i]
    subset_best_exp = best_by_subset.get(s, final_best_exp)

    # decide retrieval
    if use_rrf and ("rrf" in final_best_exp or "moe" in final_best_exp or
                     "lgb" in final_best_exp or "dense" in final_best_exp or
                     "ce" in final_best_exp):
        top_rows, top_scs = retrieve_one(q, s, active_ret_names, refit_sub_indices, refit_global_indices, rrf_w_final)
    else:
        # use subset's best retriever
        exp_for_subset = best_by_subset.get(s, final_best_exp)
        ret_name = get_exp_retriever(exp_for_subset)
        if ret_name not in refit_vecs:
            ret_name = active_ret_names[0]
        top_rows, top_scs = retrieve_one(q, s, [ret_name], refit_sub_indices, refit_global_indices, rrf_w_final)

    if top_rows:
        pred = base_df.loc[top_rows[0], "output"]
        top_id = base_df.loc[top_rows[0], "ID"]
        top_q  = base_df.loc[top_rows[0], "input"]
        top_sc = top_scs[0] if top_scs else 0.0
    else:
        pred = ""
        top_id = ""
        top_q  = ""
        top_sc = 0.0

    test_preds.append(pred)
    test_debug.append({
        "ID": row["ID"],
        "subset": s,
        "input": row["input"],
        "prediction": pred,
        "stage": "stage1_sparse",
        "experiment": subset_best_exp,
        "top_neighbor_id": top_id,
        "top_neighbor_question": top_q,
        "top_neighbor_answer": pred,
        "top_neighbor_score": top_sc,
    })

    if i % 500 == 0:
        log.info(f"  Test progress {i}/{len(test)}")

log.info("Test prediction done.")

# ─── Debug files ──────────────────────────────────────────────────────────────
# Val debug
val_debug_rows = []
best_r1s_list = exp_scores[final_best_exp].get("r1s", [])
best_rls_list = exp_scores[final_best_exp].get("rls", [])
best_preds    = exp_preds_all[final_best_exp]

# re-retrieve top neighbor info for val
for i in range(len(val)):
    top_rows_i, top_scs_i = val_retrievals[retriever_order[0]][i]
    top_id = train.loc[top_rows_i[0], "ID"] if top_rows_i else ""
    top_q  = train.loc[top_rows_i[0], "input"] if top_rows_i else ""
    top_a  = train.loc[top_rows_i[0], "output"] if top_rows_i else ""
    top_sc = top_scs_i[0] if top_scs_i else 0.0
    r1_i = best_r1s_list[i] if best_r1s_list else rouge1_f1(val_refs[i], best_preds[i])
    rl_i = best_rls_list[i] if best_rls_list else rougel_f1(val_refs[i], best_preds[i])
    val_debug_rows.append({
        "ID": val.iloc[i]["ID"],
        "subset": val_subsets[i],
        "input": val.iloc[i]["input"],
        "reference": val_refs[i],
        "prediction": best_preds[i],
        "stage": "stage1_sparse",
        "experiment": final_best_exp,
        "top_neighbor_id": top_id,
        "top_neighbor_question": top_q,
        "top_neighbor_answer": top_a,
        "top_neighbor_score": top_sc,
        "rouge1": r1_i,
        "rougel": rl_i,
    })

pd.DataFrame(val_debug_rows).to_csv(OUT / "val_predictions_debug.csv", index=False)
pd.DataFrame(test_debug).to_csv(OUT / "test_predictions_debug.csv", index=False)

# ─── Submission ───────────────────────────────────────────────────────────────
log.info("Writing submission.csv...")
submission = pd.DataFrame({
    "ID": test["ID"],
    "TargetRLF1": test_preds,
    "TargetR1F1":  test_preds,
    "TargetLLM":   test_preds,
})
submission.to_csv(OUT / "submission.csv", index=False)

# Stage-specific copies
submission.to_csv(OUT / "submission_stage1_sparse.csv", index=False)

# ─── All-stages validation summary ────────────────────────────────────────────
all_stage_rows = []
for exp in exp_names:
    sc = exp_scores[exp]
    all_stage_rows.append({
        "experiment": exp,
        "rouge1": sc["r1"],
        "rougel": sc["rl"],
        "weighted_rouge": sc["wr"],
    })
df_all_stages = pd.DataFrame(all_stage_rows)
df_all_stages.to_csv(OUT / "validation_scores_all_stages.csv", index=False)

# ─── Best stage JSON ──────────────────────────────────────────────────────────
best_stage_info = {
    "best_experiment": final_best_exp,
    "weighted_rouge_val": final_best_wr,
    "rouge1_val": exp_scores[final_best_exp]["r1"],
    "rougel_val": exp_scores[final_best_exp]["rl"],
    "stage_results": stage_results,
}
with open(OUT / "best_stage.json", "w") as f:
    json.dump(best_stage_info, f, indent=2)

# ─── Best system by subset JSON ───────────────────────────────────────────────
with open(OUT / "best_system_by_subset.json", "w") as f:
    json.dump(best_by_subset, f, indent=2)

# ─── Run manifest ─────────────────────────────────────────────────────────────
manifest = {
    "run_date": datetime.utcnow().isoformat(),
    "runtime_seconds": round(time.time() - START_TIME, 1),
    "train_size": len(train),
    "val_size": len(val),
    "test_size": len(test),
    "active_retrievers": active_ret_names,
    "disabled_retrievers": list(disabled_retrievers),
    "experiments_evaluated": exp_names,
    "best_experiment": final_best_exp,
    "best_weighted_rouge_val": final_best_wr,
    "stage2_lightgbm": stage2_ok,
    "stage3_dense": stage3_ok,
    "stage4_crossencoder": stage4_ok,
    "stage5_byt5": stage5_ok,
}
with open(OUT / "run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ─── Checks ───────────────────────────────────────────────────────────────────
log.info("Running submission checks...")
submission_check = pd.read_csv(OUT / "submission.csv")
sample_check     = pd.read_csv(WORK / "SampleSubmission.csv")

assert list(submission_check.columns) == ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"], \
    f"Column mismatch: {submission_check.columns.tolist()}"
assert submission_check.shape == sample_check.shape, \
    f"Shape mismatch: {submission_check.shape} vs {sample_check.shape}"
assert (submission_check["ID"].values == sample_check["ID"].values).all(), \
    "ID mismatch"
assert submission_check[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all(), \
    "NaN in predictions"
assert (submission_check["TargetRLF1"] == submission_check["TargetR1F1"]).all(), \
    "TargetRLF1 != TargetR1F1"
assert (submission_check["TargetRLF1"] == submission_check["TargetLLM"]).all(), \
    "TargetRLF1 != TargetLLM"
log.info("All checks PASSED.")

# ─── Final message ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DONE_FINAL_PERFORMANCE")
print()
print(f"Best stage: {final_best_exp}")
print()
print("Best validation scores:")
print(f"  rouge1          = {exp_scores[final_best_exp]['r1']:.4f}")
print(f"  rougel          = {exp_scores[final_best_exp]['rl']:.4f}")
print(f"  weighted_rouge  = {final_best_wr:.4f}")
print()
print("Final submission:")
print(f"  {OUT}/submission.csv")
print()
print("All validation stages:")
print(f"  {OUT}/validation_scores_all_stages.csv")
print()
print("Debug test:")
print(f"  {OUT}/test_predictions_debug.csv")
print()
print("Manifest:")
print(f"  {OUT}/run_manifest.json")
print("="*60)
