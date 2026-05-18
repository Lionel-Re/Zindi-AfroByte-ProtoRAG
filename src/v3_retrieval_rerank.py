"""
V3 Retrieval + Rerank
- char retriever (3,4) max_features=150000 réparé
- byte n-grams (3,5) max_features=200000
- RRF + MoE amélioré avec 3-4 retrievers
- LightGBM-lite reranker vectorisé (pas d'iterrows)
- Comparer tout vs baseline WR=0.30645
"""


import os, sys, json, time, logging, warnings, traceback, unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORK = Path("/home/onyxia/work")
OUT  = WORK / "outputs" / "latest"
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("V3 RETRIEVAL RERANK")
log.info("=" * 60)
START = time.time()

BASELINE_WR = 0.30644937186027954

# ── ROUGE ──────────────────────────────────────────────────────────────────────
def _tok(text, max_len=None):
    t = str(text).lower().split()
    return t[:max_len] if max_len else t

def rouge1_f1(ref, hyp, max_len=None):
    r, h = _tok(ref, max_len), _tok(hyp, max_len)
    if not r or not h: return 0.0
    cnt = {}
    for t in r: cnt[t] = cnt.get(t, 0) + 1
    hits = 0
    for t in h:
        if cnt.get(t, 0) > 0:
            hits += 1
            cnt[t] -= 1
    p, rv = hits/len(h), hits/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def _lcs(a, b):
    if not a or not b: return 0
    prev = [0]*(len(b)+1)
    for ai in a:
        cur = [0]*(len(b)+1)
        for j, bj in enumerate(b):
            cur[j+1] = prev[j]+1 if ai==bj else max(cur[j], prev[j+1])
        prev = cur
    return prev[len(b)]

def rougel_f1(ref, hyp, max_len=None):
    r, h = _tok(ref, max_len), _tok(hyp, max_len)
    if not r or not h: return 0.0
    lcs = _lcs(r, h)
    p, rv = lcs/len(h), lcs/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def score_rows(refs, preds, max_len=None):
    r1s = [rouge1_f1(str(a), str(b), max_len) for a,b in zip(refs, preds)]
    rls = [rougel_f1(str(a), str(b), max_len) for a,b in zip(refs, preds)]
    r1, rl = float(np.mean(r1s)), float(np.mean(rls))
    return r1, rl, 0.37*r1+0.37*rl, r1s, rls

# ── Normalize ─────────────────────────────────────────────────────────────────
def norm(t): return unicodedata.normalize("NFC", str(t)).lower()

# ── Load data ─────────────────────────────────────────────────────────────────
log.info("Loading data...")
train  = pd.read_csv(WORK / "Train.csv")
val    = pd.read_csv(WORK / "Val.csv")
test   = pd.read_csv(WORK / "Test.csv")
sample = pd.read_csv(WORK / "SampleSubmission.csv")
for df in [train, val, test]:
    df["input_norm"] = df["input"].apply(norm)
for df in [train, val]:
    df["output_norm"] = df["output"].apply(norm)

val_refs  = val["output"].tolist()
val_ids   = val["ID"].tolist()
val_subs  = val["subset"].tolist()
SUBSETS   = sorted(train["subset"].unique().tolist())
log.info(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

# ── Retriever configs ──────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

def make_byte_analyzer(n_min=3, n_max=5):
    def analyzer(s):
        b = s.encode("utf-8")
        out = []
        for n in range(n_min, n_max+1):
            for i in range(len(b)-n+1):
                out.append(b[i:i+n].hex())
        return out
    return analyzer

RETRIEVER_CFGS = {
    "char_wb": dict(analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True, dtype=np.float32),
    "char":    dict(analyzer="char",    ngram_range=(3,4), min_df=2, sublinear_tf=True, dtype=np.float32, max_features=150000),
    "word":    dict(analyzer="word",    ngram_range=(1,2), min_df=1, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b"),
    "byte":    dict(analyzer=make_byte_analyzer(3,5), min_df=2, sublinear_tf=True, dtype=np.float32, max_features=200000),
}

# ── Fit & Benchmark ────────────────────────────────────────────────────────────
log.info("Fitting vectorizers on Train...")
fitted_vecs = {}
for name, cfg in RETRIEVER_CFGS.items():
    t0 = time.time()
    vec = TfidfVectorizer(**cfg)
    vec.fit(train["input_norm"].tolist())
    fitted_vecs[name] = vec
    log.info(f"  {name} fit: {time.time()-t0:.1f}s vocab={len(vec.vocabulary_)}")

log.info("Benchmarking retrievers...")
disabled = set()
bench_rows  = {"char_wb": 50, "char": 50, "word": 50, "byte": 20}
bench_thresh_s = 60  # benchmark duration threshold in seconds

for name, vec in fitted_vecs.items():
    nb = bench_rows.get(name, 50)
    sub_bench = val.head(nb)
    # Build per-subset index for benchmark
    sub_idx = {}
    for s in sub_bench["subset"].unique():
        rows = train[train["subset"]==s].index.tolist()
        if rows:
            sub_idx[s] = vec.transform(train.loc[rows, "input_norm"])
    t0 = time.time()
    for _, row in sub_bench.iterrows():
        s = row["subset"]
        if s in sub_idx:
            X_q = vec.transform([row["input_norm"]])
            _ = (X_q @ sub_idx[s].T).toarray()
    elapsed = time.time() - t0
    projected = elapsed / nb * len(val)
    log.info(f"  {name}: bench={elapsed:.2f}s/{nb}rows → proj={projected:.0f}s")
    if elapsed > bench_thresh_s:
        log.warning(f"  {name} bench>{bench_thresh_s}s → DISABLED")
        disabled.add(name)
    else:
        log.info(f"  {name} OK (bench<{bench_thresh_s}s)")

active = {k: v for k, v in fitted_vecs.items() if k not in disabled}
log.info(f"Active retrievers: {list(active.keys())}")

# ── Build per-subset indices on Train ─────────────────────────────────────────
log.info("Building per-subset indices on Train...")
sub_indices = {}   # name -> {subset -> X_sparse}
sub_rowids  = {}   # name -> {subset -> [row_ids]}

for name, vec in active.items():
    sub_indices[name] = {}
    sub_rowids[name]  = {}
    for s in SUBSETS:
        rows = train[train["subset"]==s].index.tolist()
        if rows:
            sub_indices[name][s]  = vec.transform(train.loc[rows, "input_norm"])
            sub_rowids[name][s]   = rows
    X_all = vec.transform(train["input_norm"])
    sub_indices[name]["_global"]  = X_all
    sub_rowids[name]["_global"]   = train.index.tolist()

TOP_K = 20

def retrieve_one(q_norm, subset, name):
    vec = active[name]
    X_q = vec.transform([q_norm])
    if subset in sub_indices[name]:
        X_sub = sub_indices[name][subset]
        rids  = sub_rowids[name][subset]
    else:
        X_sub = sub_indices[name]["_global"]
        rids  = sub_rowids[name]["_global"]
    sim = (X_q @ X_sub.T).toarray()[0]
    if len(sim) <= TOP_K:
        top = np.argsort(sim)[::-1]
    else:
        top = np.argpartition(sim, -TOP_K)[-TOP_K:]
        top = top[np.argsort(sim[top])[::-1]]
    return [rids[j] for j in top], sim[top].tolist()

# ── Val retrieval ─────────────────────────────────────────────────────────────
log.info("Retrieving for Val...")
val_retrievals = {}  # name -> list of (rows, scores)
for name in active:
    t0 = time.time()
    res = []
    for q, s in zip(val["input_norm"], val["subset"]):
        res.append(retrieve_one(q, s, name))
    val_retrievals[name] = res
    log.info(f"  {name} val done in {time.time()-t0:.1f}s")

# ── RRF weights ────────────────────────────────────────────────────────────────
rnames = list(active.keys())
_w_map = {"char_wb": 0.40, "char": 0.30, "word": 0.20, "byte": 0.10}
total_w = sum(_w_map.get(n, 0.2) for n in rnames)
rrf_weights = {n: _w_map.get(n, 0.2)/total_w for n in rnames}
log.info(f"RRF weights: {rrf_weights}")

# ── Experiments ────────────────────────────────────────────────────────────────
log.info("Evaluating experiments on Val...")
exp_preds = {}
exp_scores = {}

# Top-1 per retriever
for name in rnames:
    exp = f"E_{name}_top1"
    preds = [train.loc[rows[0], "output"] if rows else "" for rows, _ in val_retrievals[name]]
    exp_preds[exp]  = preds
    r1, rl, wr, _, _ = score_rows(val_refs, preds)
    exp_scores[exp] = {"r1": r1, "rl": rl, "wr": wr}
    log.info(f"  {exp}: R1={r1:.4f} RL={rl:.4f} WR={wr:.4f}")

# RRF fusion
if len(rnames) >= 2:
    rrf_preds = []
    for i in range(len(val)):
        doc_sc = {}
        for name in rnames:
            w = rrf_weights[name]
            rows, _ = val_retrievals[name][i]
            for rank, rid in enumerate(rows):
                doc_sc[rid] = doc_sc.get(rid, 0.0) + w / (60+rank)
        if doc_sc:
            rrf_preds.append(train.loc[max(doc_sc, key=doc_sc.get), "output"])
        else:
            rrf_preds.append(train.loc[val_retrievals[rnames[0]][i][0][0], "output"])
    exp_preds["E_rrf"] = rrf_preds
    r1, rl, wr, _, _ = score_rows(val_refs, rrf_preds)
    exp_scores["E_rrf"] = {"r1": r1, "rl": rl, "wr": wr}
    log.info(f"  E_rrf: R1={r1:.4f} RL={rl:.4f} WR={wr:.4f}")

# MoE by subset
all_exps = list(exp_preds.keys())
best_global = max(exp_scores, key=lambda e: exp_scores[e]["wr"])
subset_best = {}
for s in SUBSETS:
    mask = [i for i, ss in enumerate(val_subs) if ss == s]
    if len(mask) < 3: subset_best[s] = best_global; continue
    srefs = [val_refs[i] for i in mask]
    best_e, best_wr = best_global, -1
    for e in all_exps:
        sp = [exp_preds[e][i] for i in mask]
        _, _, wr, _, _ = score_rows(srefs, sp)
        if wr > best_wr: best_wr = wr; best_e = e
    subset_best[s] = best_e
log.info(f"MoE best by subset: {subset_best}")

moe_preds = [exp_preds[subset_best.get(s, best_global)][i] for i, s in enumerate(val_subs)]
r1, rl, wr, r1s, rls = score_rows(val_refs, moe_preds)
exp_preds["E_moe"]  = moe_preds
exp_scores["E_moe"] = {"r1": r1, "rl": rl, "wr": wr, "r1s": r1s, "rls": rls}
log.info(f"  E_moe: R1={r1:.4f} RL={rl:.4f} WR={wr:.4f}")

best_sparse = max(exp_scores, key=lambda e: exp_scores[e]["wr"])
best_sparse_wr = exp_scores[best_sparse]["wr"]
log.info(f"Best sparse/RRF/MoE: {best_sparse} WR={best_sparse_wr:.5f}")

# ── Stage 2 LightGBM ──────────────────────────────────────────────────────────
lgb_wr = None
lgb_preds_val = None
lgb_ok = False

try:
    import lightgbm as lgb
    log.info("Stage 2 LightGBM-lite reranker...")
    TOP_K_LGB = 5

    def build_features(retrievals, query_norms, query_subs, rnames, rrf_w, cand_df=None):
        if cand_df is None: cand_df = train
        rows = []
        for i, (q_norm, s) in enumerate(zip(query_norms, query_subs)):
            q_len = len(q_norm.split())
            cands = {}
            for name in rnames:
                r_list, s_list = retrievals[name][i]
                for rank, (rid, sc) in enumerate(zip(r_list[:TOP_K_LGB], s_list[:TOP_K_LGB])):
                    if rid not in cands: cands[rid] = {}
                    cands[rid][f"rank_{name}"]  = rank
                    cands[rid][f"score_{name}"] = sc
            rrf_sc = {}
            for name in rnames:
                w = rrf_w.get(name, 0.2)
                r_list, _ = retrievals[name][i]
                for rank, rid in enumerate(r_list[:TOP_K_LGB]):
                    rrf_sc[rid] = rrf_sc.get(rid, 0.0) + w/(60+rank)
            for rid, feat in cands.items():
                feat["query_idx"]    = i
                feat["doc_id"]       = rid
                feat["rrf_score"]    = rrf_sc.get(rid, 0.0)
                feat["same_subset"]  = int(cand_df.loc[rid, "subset"] == s)
                feat["subset_id"]    = SUBSETS.index(s) if s in SUBSETS else -1
                feat["query_len"]    = q_len
                feat["answer_len"]   = len(str(cand_df.loc[rid, "output"]).split())
                rows.append(feat)
        df = pd.DataFrame(rows).fillna(0)
        return df

    log.info("  Building LGB features for Val (fast)...")
    t0 = time.time()
    feat_df = build_features(val_retrievals, val["input_norm"].tolist(), val_subs, rnames, rrf_weights)
    log.info(f"  Features built: {len(feat_df)} rows in {time.time()-t0:.1f}s")

    # Vectorized target computation (no iterrows)
    log.info("  Computing ROUGE targets (capped at 200 words)...")
    t0 = time.time()
    doc_ids = feat_df["doc_id"].astype(int).values
    q_idxs  = feat_df["query_idx"].astype(int).values
    train_outs = train["output"].values
    MAX_ROUGE = 200
    r1s_t = [rouge1_f1(val_refs[qi], train_outs[di], MAX_ROUGE) for qi, di in zip(q_idxs, doc_ids)]
    rls_t = [rougel_f1(val_refs[qi], train_outs[di], MAX_ROUGE) for qi, di in zip(q_idxs, doc_ids)]
    feat_df["target"] = [0.5*r1+0.5*rl for r1,rl in zip(r1s_t, rls_t)]
    log.info(f"  Targets computed in {time.time()-t0:.1f}s")

    feat_cols = [c for c in feat_df.columns if c not in ("query_idx", "doc_id", "target")]
    X = feat_df[feat_cols].values
    y = feat_df["target"].values

    # Train/valid split by query_idx
    n_q    = len(val)
    split  = int(n_q * 0.8)
    tr_m   = feat_df["query_idx"].values < split
    vl_m   = ~tr_m

    dtrain = lgb.Dataset(X[tr_m], label=y[tr_m])
    dvalid = lgb.Dataset(X[vl_m], label=y[vl_m])

    params = dict(
        objective="regression", metric="rmse",
        num_leaves=31, learning_rate=0.1,
        verbosity=-1, random_state=42, num_threads=4,
    )
    LGB_MODEL_PATH = "/tmp/v3_lgb_model.txt"
    import os as _os
    if _os.path.exists(LGB_MODEL_PATH):
        log.info(f"  Loading cached LGB model from {LGB_MODEL_PATH}")
        model = lgb.Booster(model_file=LGB_MODEL_PATH)
    else:
        log.info("  Training LGB model...")
        t0 = time.time()
        model = lgb.train(
            params, dtrain, num_boost_round=200, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(-1)],
        )
        log.info(f"  LGB trained in {time.time()-t0:.1f}s, best_iter={model.best_iteration}")
        model.save_model(LGB_MODEL_PATH)
        log.info(f"  LGB model saved to {LGB_MODEL_PATH}")

    feat_df["lgb_score"] = model.predict(X)
    lgb_preds_val = []
    for i in range(n_q):
        sub_df = feat_df[feat_df["query_idx"]==i]
        if sub_df.empty:
            lgb_preds_val.append(exp_preds[best_sparse][i])
        else:
            best_doc = int(sub_df.loc[sub_df["lgb_score"].idxmax(), "doc_id"])
            lgb_preds_val.append(train.loc[best_doc, "output"])

    r1, rl, wr, r1s_lgb, rls_lgb = score_rows(val_refs, lgb_preds_val)
    lgb_wr = wr
    log.info(f"  LGB val: R1={r1:.4f} RL={rl:.4f} WR={wr:.4f} (baseline={BASELINE_WR:.5f})")

    if wr > BASELINE_WR:
        lgb_ok = True
        exp_preds["E_lgb"]  = lgb_preds_val
        exp_scores["E_lgb"] = {"r1": r1, "rl": rl, "wr": wr, "r1s": r1s_lgb, "rls": rls_lgb}
        log.info(f"  LGB IMPROVES baseline! WR {BASELINE_WR:.5f} → {wr:.5f}")
    else:
        log.info(f"  LGB does NOT improve baseline ({wr:.5f} < {BASELINE_WR:.5f})")

except Exception as exc:
    log.warning(f"Stage 2 LightGBM failed: {exc}")
    log.warning(traceback.format_exc())

# ── Select best experiment ─────────────────────────────────────────────────────
all_exp_names = list(exp_scores.keys())
best_exp = max(exp_scores, key=lambda e: exp_scores[e]["wr"])
best_wr  = exp_scores[best_exp]["wr"]
log.info(f"\nBest experiment: {best_exp} WR={best_wr:.5f} (baseline={BASELINE_WR:.5f})")

# ── Test retrieval & prediction ────────────────────────────────────────────────
log.info("Refitting on Train+Val for Test...")
base_df = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
base_df["input_norm"]  = base_df["input"].apply(norm)
base_df["output_norm"] = base_df["output"].apply(norm)
test["input_norm"] = test["input"].apply(norm)

refit_vecs  = {}
refit_sub_i = {}
refit_sub_r = {}

for name, cfg in RETRIEVER_CFGS.items():
    if name in disabled: continue
    t0 = time.time()
    v2 = TfidfVectorizer(**cfg)
    v2.fit(base_df["input_norm"])
    refit_vecs[name] = v2
    si, sr = {}, {}
    for s in SUBSETS:
        rows = base_df[base_df["subset"]==s].index.tolist()
        if rows:
            si[s] = v2.transform(base_df.loc[rows, "input_norm"])
            sr[s] = rows
    si["_global"] = v2.transform(base_df["input_norm"])
    sr["_global"] = base_df.index.tolist()
    refit_sub_i[name] = si
    refit_sub_r[name] = sr
    log.info(f"  refit {name} in {time.time()-t0:.1f}s")

def retrieve_test(q_norm, subset, name):
    vec = refit_vecs[name]
    X_q = vec.transform([q_norm])
    if subset in refit_sub_i[name]:
        X_s = refit_sub_i[name][subset]
        rids = refit_sub_r[name][subset]
    else:
        X_s = refit_sub_i[name]["_global"]
        rids = refit_sub_r[name]["_global"]
    sim = (X_q @ X_s.T).toarray()[0]
    if len(sim) <= TOP_K:
        top = np.argsort(sim)[::-1]
    else:
        top = np.argpartition(sim, -TOP_K)[-TOP_K:]
        top = top[np.argsort(sim[top])[::-1]]
    return [rids[j] for j in top], sim[top].tolist()

log.info("Test retrieval...")
test_retrievals = {}
for name in refit_vecs:
    t0 = time.time()
    res = []
    for q, s in zip(test["input_norm"], test["subset"]):
        res.append(retrieve_test(q, s, name))
    test_retrievals[name] = res
    log.info(f"  {name} test done in {time.time()-t0:.1f}s")

test_rnames = list(refit_vecs.keys())

# Determine test prediction strategy based on best_exp
def predict_test_rrf(idx):
    doc_sc = {}
    for name in test_rnames:
        w = rrf_weights.get(name, 0.2)
        rows, _ = test_retrievals[name][idx]
        for rank, rid in enumerate(rows):
            doc_sc[rid] = doc_sc.get(rid, 0.0) + w/(60+rank)
    return base_df.loc[max(doc_sc, key=doc_sc.get), "output"] if doc_sc else ""

def predict_test_top1(idx, name):
    rows, _ = test_retrievals[name][idx]
    return base_df.loc[rows[0], "output"] if rows else ""

# MoE for test: use subset_best from val evaluation (mapped to retriever)
def predict_test_moe(idx):
    s = test["subset"].iloc[idx]
    e = subset_best.get(s, best_exp)
    # Map experiment to retrieval strategy
    if "rrf" in e or "moe" in e or "lgb" in e:
        return predict_test_rrf(idx)
    for name in test_rnames:
        if name in e:
            return predict_test_top1(idx, name)
    return predict_test_rrf(idx)

log.info("Generating Test predictions...")
test_subs = test["subset"].tolist()

if lgb_ok and best_exp == "E_lgb":
    # LGB reranker for test
    log.info("  Using LGB reranker for Test...")
    test_feat = build_features(test_retrievals, test["input_norm"].tolist(), test_subs, test_rnames, rrf_weights, cand_df=base_df)
    # Keep query_idx and doc_id before column alignment
    test_q_idx = test_feat["query_idx"].values.copy()
    test_doc_id = test_feat["doc_id"].values.copy()
    # Align feature columns for model
    for c in feat_cols:
        if c not in test_feat.columns: test_feat[c] = 0
    X_test_lgb = test_feat[feat_cols].values
    lgb_scores  = model.predict(X_test_lgb)
    test_preds = []
    for i in range(len(test)):
        mask = test_q_idx == i
        if not mask.any():
            test_preds.append(predict_test_rrf(i))
        else:
            best_doc = int(test_doc_id[mask][np.argmax(lgb_scores[mask])])
            test_preds.append(base_df.loc[best_doc, "output"])
elif "moe" in best_exp or best_wr > BASELINE_WR:
    log.info("  Using MoE strategy for Test...")
    test_preds = [predict_test_moe(i) for i in range(len(test))]
elif "rrf" in best_exp:
    log.info("  Using RRF for Test...")
    test_preds = [predict_test_rrf(i) for i in range(len(test))]
elif best_exp == "baseline":
    log.info("  Keeping baseline predictions for Test...")
    prev_sub = pd.read_csv(OUT / "submission_backup_after_byt5_failed.csv")
    test_preds = prev_sub["TargetRLF1"].tolist()
else:
    log.info(f"  Using top-1 from {best_exp} for Test...")
    retriever_name = [n for n in test_rnames if n in best_exp]
    rn = retriever_name[0] if retriever_name else test_rnames[0]
    test_preds = [predict_test_top1(i, rn) for i in range(len(test))]

# ── Save debug files ───────────────────────────────────────────────────────────
# Val debug - top neighbor info from best retriever
best_r_name = [n for n in rnames if f"E_{n}_top1" == best_sparse]
best_r_name = best_r_name[0] if best_r_name else rnames[0]

val_debug_rows = []
for i, (vid, s) in enumerate(zip(val_ids, val_subs)):
    rows0, scs0 = val_retrievals[rnames[0]][i]
    rrf_sc_i = {}
    for name in rnames:
        w = rrf_weights[name]
        rr, _ = val_retrievals[name][i]
        for rk, rid in enumerate(rr): rrf_sc_i[rid] = rrf_sc_i.get(rid,0.0)+w/(60+rk)
    best_rrf_doc = max(rrf_sc_i, key=rrf_sc_i.get) if rrf_sc_i else (rows0[0] if rows0 else None)
    best_doc_score = rrf_sc_i.get(best_rrf_doc, 0.0) if best_rrf_doc else 0.0
    pred = exp_preds[best_exp][i]
    ref  = val_refs[i]
    val_debug_rows.append({
        "ID": vid, "subset": s,
        "input": val["input"].iloc[i],
        "reference": ref,
        "prediction": pred,
        "experiment": best_exp,
        "top_neighbor_id": train.index[rows0[0]] if rows0 else "",
        "top_neighbor_score": scs0[0] if scs0 else 0.0,
        "rrf_score": best_doc_score,
        "rouge1": rouge1_f1(ref, pred),
        "rougel": rougel_f1(ref, pred),
    })
val_debug_df = pd.DataFrame(val_debug_rows)
val_debug_df.to_csv(OUT / "val_predictions_debug.csv", index=False)

# Test debug
test_debug_rows = []
for i, row in enumerate(test.itertuples()):
    rows0, scs0 = (test_retrievals[test_rnames[0]][i] if test_rnames else ([], []))
    test_debug_rows.append({
        "ID": row.ID, "subset": row.subset,
        "input": row.input,
        "prediction": test_preds[i],
        "stage": "v3_retrieval",
        "experiment": best_exp,
        "top_neighbor_id": "",
        "top_neighbor_question": "",
        "top_neighbor_answer": base_df.loc[rows0[0], "output"] if rows0 else "",
        "top_neighbor_score": scs0[0] if scs0 else 0.0,
    })
pd.DataFrame(test_debug_rows).to_csv(OUT / "test_predictions_debug.csv", index=False)

# ── Build submission ───────────────────────────────────────────────────────────
sub = sample[["ID"]].copy()
pred_map = dict(zip(test["ID"], test_preds))
sub["TargetRLF1"] = sub["ID"].map(pred_map)
sub["TargetR1F1"] = sub["TargetRLF1"]
sub["TargetLLM"]  = sub["TargetRLF1"]

# ── Final check ───────────────────────────────────────────────────────────────
assert list(sub.columns) == ["ID","TargetRLF1","TargetR1F1","TargetLLM"]
assert sub.shape == sample.shape, f"Shape mismatch {sub.shape} vs {sample.shape}"
assert (sub["ID"].values == sample["ID"].values).all(), "ID mismatch"
assert sub[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all(), "NaN found"
assert (sub["TargetRLF1"] == sub["TargetR1F1"]).all()
assert (sub["TargetRLF1"] == sub["TargetLLM"]).all()
log.info("Submission checks PASSED")

if best_wr > BASELINE_WR:
    sub.to_csv(OUT / "submission_stage2_lightgbm.csv" if lgb_ok else OUT / "submission_v3_retrieval.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    log.info(f"Updated submission.csv (WR {BASELINE_WR:.5f} → {best_wr:.5f})")
else:
    import shutil
    shutil.copy(OUT / "submission_backup_after_byt5_failed.csv", OUT / "submission.csv")
    log.info(f"WR not improved ({best_wr:.5f} <= {BASELINE_WR:.5f}). Kept baseline submission.")

# ── Update outputs ─────────────────────────────────────────────────────────────
# validation_scores_all_stages.csv
try:
    sc_df = pd.read_csv(OUT / "validation_scores_all_stages.csv")
    new_rows = []
    for e, sc in exp_scores.items():
        if not any(sc_df["experiment"]==e):
            new_rows.append({"experiment": e, "rouge1": sc["r1"], "rougel": sc["rl"], "weighted_rouge": sc["wr"]})
    if new_rows:
        sc_df = pd.concat([sc_df, pd.DataFrame(new_rows)], ignore_index=True)
    sc_df.to_csv(OUT / "validation_scores_all_stages.csv", index=False)
except Exception as e:
    log.warning(f"validation_scores update failed: {e}")

# best_stage.json
try:
    with open(OUT / "best_stage.json") as f: bs = json.load(f)
    if best_wr > bs.get("weighted_rouge_val", 0):
        bs["best_experiment"]  = best_exp
        bs["weighted_rouge_val"] = best_wr
        bs["rouge1_val"] = exp_scores[best_exp]["r1"]
        bs["rougel_val"] = exp_scores[best_exp]["rl"]
        bs.setdefault("stage_results", {})["v3"] = {"experiment": best_exp, "wr": best_wr}
        with open(OUT / "best_stage.json", "w") as f: json.dump(bs, f, indent=2)
        log.info(f"Updated best_stage.json")
except Exception as e:
    log.warning(f"best_stage.json update failed: {e}")

# run_manifest.json
try:
    with open(OUT / "run_manifest.json") as f: mf = json.load(f)
    mf["v3_retrieval"] = {
        "active_retrievers": list(active.keys()),
        "disabled_retrievers": list(disabled),
        "best_experiment": best_exp, "best_wr": best_wr,
        "lgb_ok": lgb_ok, "lgb_wr": lgb_wr,
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUT / "run_manifest.json", "w") as f: json.dump(mf, f, indent=2)
except Exception as e:
    log.warning(f"run_manifest update failed: {e}")

# v3 report
report = {
    "timestamp": datetime.now().isoformat(),
    "baseline_wr": BASELINE_WR,
    "active_retrievers": list(active.keys()),
    "disabled_retrievers": list(disabled),
    "experiments": {e: {"wr": sc["wr"], "r1": sc["r1"], "rl": sc["rl"]} for e, sc in exp_scores.items()},
    "best_experiment": best_exp,
    "best_wr": best_wr,
    "improved": best_wr > BASELINE_WR,
    "lgb_ok": lgb_ok,
    "lgb_wr": lgb_wr,
    "elapsed_s": time.time()-START,
}
with open(OUT / "v3_retrieval_rerank_report.json", "w") as f:
    json.dump(report, f, indent=2)

final_sub = pd.read_csv(OUT / "submission.csv")
assert list(final_sub.columns) == ["ID","TargetRLF1","TargetR1F1","TargetLLM"]
assert final_sub.shape == sample.shape
assert (final_sub["ID"].values == sample["ID"].values).all()
assert final_sub[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
assert (final_sub["TargetRLF1"] == final_sub["TargetR1F1"]).all()
assert (final_sub["TargetRLF1"] == final_sub["TargetLLM"]).all()
log.info("Final submission.csv checks PASSED")

log.info("\n" + "=" * 60)
log.info("DONE_V3")
log.info(f"Active retrievers: {list(active.keys())}")
log.info(f"Best experiment:   {best_exp}")
log.info(f"Best val WR:       {best_wr:.5f}")
log.info(f"Baseline WR:       {BASELINE_WR:.5f}")
log.info(f"Improved:          {best_wr > BASELINE_WR}")
log.info(f"LGB enabled:       {lgb_ok}")
log.info(f"LGB WR:            {lgb_wr}")
log.info(f"Final submission:  {OUT}/submission.csv")
log.info(f"V3 report:         {OUT}/v3_retrieval_rerank_report.json")
log.info("=" * 60)
