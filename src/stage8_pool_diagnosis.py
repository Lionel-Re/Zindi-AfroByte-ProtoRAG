"""
Stage 8 — Pool Recall Diagnosis (read-only, no model training, no submission update)
Baseline: Stage5 routed, Val WR=0.37563, Public=0.732376
Rebuilds Stage5 Val candidate pool, computes oracle@K, classifies errors, source analysis.
"""
import os, sys, json, re, time, pickle, logging, warnings, unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

os.environ["HF_HOME"]                = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"]     = "/tmp/hf_cache/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WORK      = Path("/home/onyxia/work")
OUT       = WORK / "outputs" / "latest"
CACHE_DIR = Path("/tmp/stage6_cache")
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("STAGE 8 — Pool Recall Diagnosis (read-only)")
log.info("=" * 60)
START = time.time()

BASELINE_WR_S5 = 0.37563
S4_SUBSETS     = {"Swa_Ken", "Lug_Uga"}   # Stage5 routing uses Stage4 for these
ORACLE_KS      = [1, 5, 10, 20, 50]
HIT_THRESHOLDS = [0.30, 0.40, 0.50]

# ── ROUGE ──────────────────────────────────────────────────────────────────────
def _tok(t, ml=None):
    t2 = str(t).lower().split(); return t2[:ml] if ml else t2

def rouge1_f1(ref, hyp, ml=200):
    r, h = _tok(ref, ml), _tok(hyp, ml)
    if not r or not h: return 0.0
    cnt = {}
    for t in r: cnt[t] = cnt.get(t, 0) + 1
    hits = 0
    for t in h:
        if cnt.get(t, 0) > 0: hits += 1; cnt[t] -= 1
    p, rv = hits/len(h), hits/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def _lcs(a, b):
    if not a or not b: return 0
    prev = [0]*(len(b)+1)
    for ai in a:
        cur = [0]*(len(b)+1)
        for j, bj in enumerate(b): cur[j+1] = prev[j]+1 if ai==bj else max(cur[j], prev[j+1])
        prev = cur
    return prev[len(b)]

def rougel_f1(ref, hyp, ml=200):
    r, h = _tok(ref, ml), _tok(hyp, ml)
    if not r or not h: return 0.0
    lcs = _lcs(r, h); p, rv = lcs/len(h), lcs/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def wr(r1, rl): return 0.37*r1 + 0.37*rl

def norm_text(t): return unicodedata.normalize("NFC", str(t)).lower()

# ── Data ───────────────────────────────────────────────────────────────────────
log.info("Loading data...")
train  = pd.read_csv(WORK/"Train.csv")
val    = pd.read_csv(WORK/"Val.csv")
for df in [train, val]: df["input_norm"] = df["input"].apply(norm_text)
for df in [train, val]: df["output_norm"] = df["output"].apply(norm_text)
val_refs = val["output"].tolist()
val_subs = val["subset"].tolist()
SUBSETS  = sorted(train["subset"].unique())
log.info(f"Train {len(train)} | Val {len(val)}")

# ── TASK 1 — Protect Stage5 submission ────────────────────────────────────────
sub_path = OUT / "submission.csv"
backup_path = OUT / "submission_backup_before_stage8_pool_diagnosis.csv"
import shutil
shutil.copy(str(sub_path), str(backup_path))
log.info(f"Backup: {backup_path.name}")

# Verify Stage5 routed
s5d = pd.read_csv(OUT / "stage5_val_predictions_debug.csv")
s4d = pd.read_csv(OUT / "crossencoder_val_predictions_debug.csv")
s5_preds = s5d["prediction"].tolist()
s4_preds = s4d["prediction"].tolist()
s5_routed = [s4_preds[i] if val_subs[i] in S4_SUBSETS else s5_preds[i]
             for i in range(len(val_subs))]

# Selected WR per query
selected_r1s = [rouge1_f1(val_refs[i], s5_routed[i]) for i in range(len(val))]
selected_rls = [rougel_f1(val_refs[i], s5_routed[i]) for i in range(len(val))]
selected_wrs = [wr(selected_r1s[i], selected_rls[i]) for i in range(len(val))]
global_sel_wr = float(np.mean(selected_wrs))
log.info(f"Stage5 routed Val WR verify: {global_sel_wr:.5f}")

# ── Sparse retrievers (exact Stage5 config) ────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

def make_byte_analyzer(n_min=3, n_max=5):
    def analyzer(s):
        b = s.encode("utf-8"); out = []
        for n in range(n_min, n_max+1):
            for i in range(len(b)-n+1): out.append(b[i:i+n].hex())
        return out
    return analyzer

CFGS = {
    "char_wb": dict(analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True, dtype=np.float32),
    "char":    dict(analyzer="char",    ngram_range=(3,4), min_df=2, sublinear_tf=True, dtype=np.float32, max_features=150000),
    "word":    dict(analyzer="word",    ngram_range=(1,2), min_df=1, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b"),
    "byte":    dict(analyzer=make_byte_analyzer(3,5), min_df=2, sublinear_tf=True, dtype=np.float32, max_features=200000),
}

log.info("Fitting sparse vectorizers on Train...")
vecs = {}
for name, cfg in CFGS.items():
    t0 = time.time(); v = TfidfVectorizer(**cfg); v.fit(train["input_norm"])
    log.info(f"  {name}: {time.time()-t0:.1f}s"); vecs[name] = v

log.info("Building per-subset sparse indices...")
sub_i, sub_r = {}, {}
for name, v in vecs.items():
    sub_i[name] = {s: v.transform(train[train["subset"]==s]["input_norm"])
                   for s in SUBSETS if len(train[train["subset"]==s]) > 0}
    sub_r[name] = {s: train[train["subset"]==s].index.tolist() for s in SUBSETS}

TOP_K_SPARSE = 20
rnames = list(vecs.keys())
_wmap  = {"char_wb": 0.40, "char": 0.30, "word": 0.20, "byte": 0.10}
tw     = sum(_wmap[n] for n in rnames)
rrf_w_sparse = {n: _wmap[n]/tw for n in rnames}

def retrieve_sparse(q, s, name, topk=TOP_K_SPARSE):
    v = vecs[name]; X_q = v.transform([q])
    if s in sub_i[name]: X_s, rids = sub_i[name][s], sub_r[name][s]
    else: X_s = v.transform(train["input_norm"]); rids = train.index.tolist()
    sim = (X_q @ X_s.T).toarray()[0]; tk = min(topk, len(sim))
    top = np.argpartition(sim, -tk)[-tk:] if len(sim) > tk else np.argsort(sim)[::-1]
    top = top[np.argsort(sim[top])[::-1]]
    return [rids[j] for j in top], sim[top].tolist()

log.info("Sparse retrieval for Val...")
t0 = time.time()
val_sparse_ret = {name: [] for name in vecs}
for q, s in zip(val["input_norm"], val_subs):
    for name in vecs: val_sparse_ret[name].append(retrieve_sparse(q, s, name))
log.info(f"  done in {time.time()-t0:.1f}s")

# ── Dense retrieval (Stage5: minilm + e5s) ─────────────────────────────────────
import faiss
S5_DENSE_CONFIGS = [
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 0.5),
    ("e5s",    "intfloat/multilingual-e5-small",                               0.5),
]
dense_model_keys = []
dense_sub_idx    = {}
dense_sub_rids   = {}
val_dense_ret    = {}

for model_key, _, rrf_w_d in S5_DENSE_CONFIGS:
    try:
        tr_embs  = np.load(str(CACHE_DIR / f"train_embs_{model_key}.npy")).astype(np.float32)
        val_embs = np.load(str(CACHE_DIR / f"val_embs_{model_key}.npy")).astype(np.float32)
        log.info(f"  Loaded cached embeddings {model_key} dim={tr_embs.shape[1]}")
        dim = tr_embs.shape[1]
        dense_sub_idx[model_key] = {}; dense_sub_rids[model_key] = {}
        for s in SUBSETS:
            mask = (train["subset"] == s).values
            if not mask.any(): continue
            idxs = train[train["subset"]==s].index.tolist()
            fi = faiss.IndexFlatIP(dim); fi.add(tr_embs[mask])
            dense_sub_idx[model_key][s]  = fi
            dense_sub_rids[model_key][s] = idxs
        global_fi   = faiss.IndexFlatIP(dim); global_fi.add(tr_embs)
        global_rids = train.index.tolist()

        val_dense_ret[model_key] = []
        for i, (q_emb, s) in enumerate(zip(val_embs, val_subs)):
            qv = q_emb.reshape(1, -1)
            if s in dense_sub_idx[model_key]:
                fi2 = dense_sub_idx[model_key][s]; rids2 = dense_sub_rids[model_key][s]
            else: fi2 = global_fi; rids2 = global_rids
            tk = min(20, fi2.ntotal); D, I = fi2.search(qv, tk)
            val_dense_ret[model_key].append(
                ([rids2[j] for j in I[0] if j>=0],
                 [float(D[0][k]) for k in range(len(I[0])) if I[0][k]>=0]))
        log.info(f"  Val dense retrieval done for {model_key}")
        dense_model_keys.append((model_key, rrf_w_d))
    except Exception as e:
        log.warning(f"Dense {model_key} failed: {e}")

log.info(f"Dense models: {[k for k,_ in dense_model_keys]}")

# ── Load CE scores ─────────────────────────────────────────────────────────────
val_ce = {}
ce_cache_path = CACHE_DIR / "val_ce_scores_stage6.pkl"
if ce_cache_path.exists():
    with open(ce_cache_path, "rb") as f: val_ce = pickle.load(f)
    log.info(f"Loaded Stage6 CE scores for {len(val_ce)} val queries")
else:
    log.warning("CE cache not found — CE scores will be unavailable")

# ── Pool builder with source tracking ─────────────────────────────────────────
def get_pool_with_sources(i, top_ks=20, top_kd=20):
    """Returns dict: rid -> {sparse_retrievers: set, dense_retrievers: set, rrf_sp, rrf_dn,
                              sp_<name>_r, sp_<name>_s, dn_<name>_r, dn_<name>_s}"""
    seen = {}
    for name in rnames:
        rows, scs = val_sparse_ret[name][i]
        w = rrf_w_sparse[name]
        for rank, (rid, sc) in enumerate(zip(rows[:top_ks], scs[:top_ks])):
            if rid not in seen: seen[rid] = {"sparse_rets": set(), "dense_rets": set()}
            seen[rid]["sparse_rets"].add(name)
            seen[rid][f"sp_{name}_r"] = rank
            seen[rid][f"sp_{name}_s"] = sc
            seen[rid]["rrf_sp"] = seen[rid].get("rrf_sp", 0.0) + w/(60+rank)
    for mk, rw in dense_model_keys:
        rows, scs = val_dense_ret[mk][i]
        for rank, (rid, sc) in enumerate(zip(rows[:top_kd], scs[:top_kd])):
            if rid not in seen: seen[rid] = {"sparse_rets": set(), "dense_rets": set()}
            seen[rid]["dense_rets"].add(mk)
            seen[rid][f"dn_{mk}_r"] = rank
            seen[rid][f"dn_{mk}_s"] = sc
            rw2 = rw / (sum(w2 for _, w2 in dense_model_keys) or 1.0)
            seen[rid]["rrf_dn"] = seen[rid].get("rrf_dn", 0.0) + rw2/(60+rank)
    return seen

# ── TASK 2 + 3 + 4 + 5 — Oracle analysis ──────────────────────────────────────
log.info("Computing oracle analysis for Val pool...")
t0 = time.time()

per_query_rows = []      # one row per val query
all_pool_rows  = []      # one row per (query, candidate) pair — used for source analysis

for i in range(len(val)):
    ref    = val_refs[i]
    sub    = val_subs[i]
    q_id   = val.iloc[i]["ID"]
    sel_wr = selected_wrs[i]
    sel_r1 = selected_r1s[i]
    sel_rl = selected_rls[i]

    pool = get_pool_with_sources(i)
    if not pool:
        per_query_rows.append({
            "ID": q_id, "subset": sub, "pool_size": 0,
            "selected_wr": sel_wr, "selected_r1": sel_r1, "selected_rl": sel_rl,
            **{f"oracle_wr_{k}": sel_wr for k in ORACLE_KS},
            "error_type": "A_solved_near" if sel_wr >= 0.50 else "C_recall_failure",
        })
        continue

    ce_q = val_ce.get(i, {})  # {rid -> ce_score}

    # Compute RRF for each candidate
    for rid in pool:
        rrf_total = pool[rid].get("rrf_sp", 0.0) + pool[rid].get("rrf_dn", 0.0)
        pool[rid]["rrf_total"] = rrf_total

    # Sort by RRF (descending)
    rrf_sorted = sorted(pool.keys(), key=lambda r: pool[r]["rrf_total"], reverse=True)
    # Sort by CE (descending, if available)
    ce_sorted  = sorted(pool.keys(), key=lambda r: ce_q.get(r, -999), reverse=True)

    # Compute WR for each candidate in pool
    cand_wrs = {}
    for rid in pool:
        cand_out = str(train.loc[rid, "output"])
        r1 = rouge1_f1(ref, cand_out)
        rl = rougel_f1(ref, cand_out)
        w  = wr(r1, rl)
        cand_wrs[rid] = (r1, rl, w)
        # Pool row for source analysis
        all_pool_rows.append({
            "query_idx": i, "query_id": q_id, "subset": sub,
            "rid": rid,
            "cand_r1": r1, "cand_rl": rl, "cand_wr": w,
            "cand_ans_len": len(cand_out.split()),
            "rrf_total": pool[rid]["rrf_total"],
            "ce_score": ce_q.get(rid, np.nan),
            "sparse_rets": "|".join(sorted(pool[rid]["sparse_rets"])),
            "dense_rets":  "|".join(sorted(pool[rid]["dense_rets"])),
            "n_sparse_sources": len(pool[rid]["sparse_rets"]),
            "n_dense_sources":  len(pool[rid]["dense_rets"]),
            "rrf_rank": rrf_sorted.index(rid),
            "ce_rank":  ce_sorted.index(rid) if rid in ce_sorted else -1,
            "is_same_subset": (train.loc[rid, "subset"] == sub),
        })

    # Oracle@K by RRF rank
    oracle_wr_at_k = {}
    for k in ORACLE_KS:
        top_k_rids = rrf_sorted[:k]
        if not top_k_rids:
            oracle_wr_at_k[k] = 0.0
        else:
            oracle_wr_at_k[k] = max(cand_wrs[r][2] for r in top_k_rids)

    # Oracle@K by CE rank
    oracle_wr_ce_at_k = {}
    for k in ORACLE_KS:
        top_k_rids = ce_sorted[:k] if ce_sorted else rrf_sorted[:k]
        if not top_k_rids:
            oracle_wr_ce_at_k[k] = 0.0
        else:
            oracle_wr_ce_at_k[k] = max(cand_wrs[r][2] for r in top_k_rids)

    # Best candidate info
    best_rid  = max(pool.keys(), key=lambda r: cand_wrs[r][2])
    best_wr_v = cand_wrs[best_rid][2]
    best_rank_rrf = rrf_sorted.index(best_rid)
    best_rank_ce  = ce_sorted.index(best_rid) if best_rid in ce_sorted else -1

    # Gap metrics
    gap5   = oracle_wr_at_k[5]  - sel_wr
    gap20  = oracle_wr_at_k[20] - sel_wr
    gap_full = best_wr_v - sel_wr   # oracle@all

    # Hit rates
    hit_flags = {}
    for k in ORACLE_KS:
        for thr in HIT_THRESHOLDS:
            hit_flags[f"hit_{k}_thr_{int(thr*100):02d}"] = int(oracle_wr_at_k[k] >= thr)

    # Error classification (Task 4)
    oracle50 = oracle_wr_at_k.get(50, best_wr_v)
    oracle20 = oracle_wr_at_k[20]
    oracle5  = oracle_wr_at_k[5]
    if sel_wr >= 0.50:
        etype = "A_solved"
    elif sel_wr >= 0.40:
        etype = "A_near_solved"
    elif sel_wr < 0.35 and oracle20 >= 0.45:
        etype = "B_reranking_failure"
    elif oracle50 < 0.30:
        etype = "C_recall_failure"
    elif oracle50 < 0.45:
        etype = "D_weak_pool"
    else:
        etype = "E_ambiguous_medium"

    row = {
        "ID": q_id, "subset": sub, "pool_size": len(pool),
        "selected_wr": sel_wr, "selected_r1": sel_r1, "selected_rl": sel_rl,
        "selected_ans_len": len(str(s5_routed[i]).split()),
        "best_cand_wr": best_wr_v, "best_cand_r1": cand_wrs[best_rid][0],
        "best_cand_rl": cand_wrs[best_rid][1],
        "best_cand_rank_rrf": best_rank_rrf, "best_cand_rank_ce": best_rank_ce,
        "best_cand_ans_len": len(str(train.loc[best_rid, "output"]).split()),
        "best_cand_sparse_rets": "|".join(sorted(pool[best_rid]["sparse_rets"])),
        "best_cand_dense_rets":  "|".join(sorted(pool[best_rid]["dense_rets"])),
        "best_cand_same_subset": (train.loc[best_rid, "subset"] == sub),
        "gap5": gap5, "gap20": gap20, "gap_full": gap_full,
        "error_type": etype,
    }
    for k in ORACLE_KS:
        row[f"oracle_wr_{k}"] = oracle_wr_at_k[k]
        row[f"oracle_wr_ce_{k}"] = oracle_wr_ce_at_k[k]
    row.update(hit_flags)
    per_query_rows.append(row)

    if i % 500 == 0: log.info(f"  {i}/{len(val)} queries processed")

log.info(f"Oracle analysis done in {time.time()-t0:.1f}s")

per_q  = pd.DataFrame(per_query_rows)
pool_df = pd.DataFrame(all_pool_rows)

# ── TASK 7 — Save outputs ──────────────────────────────────────────────────────
# stage8_pool_recall_summary.csv — global oracle table
summary_rows = []
for k in ORACLE_KS:
    r = {"k": k, "oracle_wr": float(per_q[f"oracle_wr_{k}"].mean()),
         "oracle_wr_ce": float(per_q[f"oracle_wr_ce_{k}"].mean()),
         "gap_vs_selected": float(per_q[f"oracle_wr_{k}"].mean()) - global_sel_wr}
    for thr in HIT_THRESHOLDS:
        r[f"hit_rate_thr_{int(thr*100):02d}"] = float(per_q[f"hit_{k}_thr_{int(thr*100):02d}"].mean())
    summary_rows.append(r)
summary_df = pd.DataFrame(summary_rows)
summary_df["selected_wr_global"] = global_sel_wr
summary_df.to_csv(OUT/"stage8_pool_recall_summary.csv", index=False)
log.info("Wrote stage8_pool_recall_summary.csv")

# stage8_oracle_by_subset.csv
oracle_by_sub_rows = []
for s in SUBSETS:
    mask = per_q["subset"] == s
    sub_df = per_q[mask]
    n = len(sub_df)
    r = {"subset": s, "n": n, "selected_wr": float(sub_df["selected_wr"].mean())}
    for k in ORACLE_KS:
        r[f"oracle_wr_{k}"] = float(sub_df[f"oracle_wr_{k}"].mean())
        r[f"gap_{k}"] = r[f"oracle_wr_{k}"] - r["selected_wr"]
    for thr in HIT_THRESHOLDS:
        for k in ORACLE_KS:
            r[f"hit_{k}_thr_{int(thr*100):02d}"] = float(sub_df[f"hit_{k}_thr_{int(thr*100):02d}"].mean())
    oracle_by_sub_rows.append(r)
oracle_by_sub = pd.DataFrame(oracle_by_sub_rows)
oracle_by_sub.to_csv(OUT/"stage8_oracle_by_subset.csv", index=False)
log.info("Wrote stage8_oracle_by_subset.csv")

# stage8_error_types_by_subset.csv
etypes = ["A_solved", "A_near_solved", "B_reranking_failure", "C_recall_failure",
          "D_weak_pool", "E_ambiguous_medium"]
err_rows = []
for s in SUBSETS + ["ALL"]:
    mask = per_q["subset"] == s if s != "ALL" else pd.Series([True]*len(per_q))
    sub_df = per_q[mask]; n = len(sub_df)
    r = {"subset": s, "n": n}
    for e in etypes:
        cnt = (sub_df["error_type"] == e).sum()
        r[f"{e}_count"] = cnt; r[f"{e}_pct"] = 100.0*cnt/n if n>0 else 0
    err_rows.append(r)
err_df = pd.DataFrame(err_rows)
err_df.to_csv(OUT/"stage8_error_types_by_subset.csv", index=False)
log.info("Wrote stage8_error_types_by_subset.csv")

# stage8_pool_recall_by_subset.csv — key metrics per subset
recall_rows = []
for s in SUBSETS:
    mask = per_q["subset"] == s
    sub_df = per_q[mask]; n = len(sub_df)
    r = {"subset": s, "n": n,
         "selected_wr": float(sub_df["selected_wr"].mean()),
         "pool_size_avg": float(sub_df["pool_size"].mean()),
         "oracle_wr_5":  float(sub_df["oracle_wr_5"].mean()),
         "oracle_wr_20": float(sub_df["oracle_wr_20"].mean()),
         "oracle_wr_50": float(sub_df[f"oracle_wr_{50}"].mean()) if 50 in ORACLE_KS else float(sub_df["best_cand_wr"].mean()),
         "best_cand_wr":float(sub_df["best_cand_wr"].mean()),
         "gap_5":  float(sub_df["oracle_wr_5"].mean())  - float(sub_df["selected_wr"].mean()),
         "gap_20": float(sub_df["oracle_wr_20"].mean()) - float(sub_df["selected_wr"].mean()),
         "best_rank_rrf_avg": float(sub_df["best_cand_rank_rrf"].mean()),
         "best_rank_ce_avg":  float(sub_df["best_cand_rank_ce"].mean()),
         "pct_best_rank_rrf_le5":  float((sub_df["best_cand_rank_rrf"] < 5).mean()),
         "pct_best_rank_ce_le5":   float((sub_df["best_cand_rank_ce"] < 5).mean()),
         "hit_5_030":  float(sub_df["hit_5_thr_30"].mean()),
         "hit_20_030": float(sub_df["hit_20_thr_30"].mean()),
         "hit_5_040":  float(sub_df["hit_5_thr_40"].mean()),
         "hit_20_040": float(sub_df["hit_20_thr_40"].mean()),
         "hit_5_050":  float(sub_df["hit_5_thr_50"].mean()),
         "hit_20_050": float(sub_df["hit_20_thr_50"].mean()),
    }
    recall_rows.append(r)
recall_df = pd.DataFrame(recall_rows)
recall_df.to_csv(OUT/"stage8_pool_recall_by_subset.csv", index=False)
log.info("Wrote stage8_pool_recall_by_subset.csv")

# stage8_candidate_source_analysis.csv
src_rows = []
for s in SUBSETS:
    pool_sub = pool_df[pool_df["subset"] == s]
    per_q_sub = per_q[per_q["subset"] == s]
    n_q = len(per_q_sub)
    # Find the oracle (best) candidate per query
    best_per_q = pool_sub.loc[pool_sub.groupby("query_idx")["cand_wr"].idxmax()]

    def src_cat(row):
        has_sp = len(row["sparse_rets"]) > 0 if row["sparse_rets"] else False
        has_dn = len(row["dense_rets"]) > 0 if row["dense_rets"] else False
        if has_sp and has_dn: return "both"
        if has_sp: return "sparse_only"
        if has_dn: return "dense_only"
        return "unknown"

    best_per_q = best_per_q.copy()
    best_per_q["src_cat"] = best_per_q.apply(src_cat, axis=1)

    total_best = len(best_per_q)
    r = {"subset": s, "n_queries": n_q,
         "pct_oracle_sparse_only": 100.0*(best_per_q["src_cat"]=="sparse_only").sum()/total_best if total_best>0 else 0,
         "pct_oracle_dense_only":  100.0*(best_per_q["src_cat"]=="dense_only").sum()/total_best  if total_best>0 else 0,
         "pct_oracle_both":        100.0*(best_per_q["src_cat"]=="both").sum()/total_best        if total_best>0 else 0,
         "avg_oracle_cand_wr":     float(best_per_q["cand_wr"].mean()),
         "avg_oracle_rrf_rank":    float(best_per_q["rrf_rank"].mean()),
         "avg_oracle_ce_rank":     float(best_per_q["ce_rank"].mean()),
         "pct_oracle_rrf_rank_le5":   float((best_per_q["rrf_rank"] < 5).mean()),
         "pct_oracle_ce_rank_le5":    float((best_per_q["ce_rank"] < 5).mean()),
         "pct_oracle_same_subset":    float(best_per_q["is_same_subset"].mean()),
         "avg_pool_size":          float(pool_sub.groupby("query_idx")["rid"].count().mean()),
    }
    src_rows.append(r)
src_df = pd.DataFrame(src_rows)
src_df.to_csv(OUT/"stage8_candidate_source_analysis.csv", index=False)
log.info("Wrote stage8_candidate_source_analysis.csv")

# stage8_hard_examples.csv — hardest queries (recall failures + reranking failures)
hard_types = {"B_reranking_failure", "C_recall_failure", "D_weak_pool"}
hard = per_q[per_q["error_type"].isin(hard_types)].copy()
hard = hard.sort_values("selected_wr")
hard["question"] = hard["ID"].map(dict(zip(val["ID"], val["input"])))
hard[["ID","subset","error_type","selected_wr","oracle_wr_5","oracle_wr_20",
      "best_cand_wr","best_cand_rank_rrf","best_cand_rank_ce","pool_size","question"
      ]].to_csv(OUT/"stage8_hard_examples.csv", index=False)
log.info("Wrote stage8_hard_examples.csv")

# ── TASK 6 — Decision report ───────────────────────────────────────────────────
log.info("Generating decision report...")

def classify_bottleneck(row):
    wr_sel  = row["selected_wr"]
    wr_or5  = row["oracle_wr_5"]
    wr_or20 = row["oracle_wr_20"]
    wr_or50 = row.get("oracle_wr_50", row["best_cand_wr"])
    hit30_50 = row.get("hit_50_thr_30", row.get("hit_20_thr_30", 0))
    hit40_50 = row.get("hit_50_thr_40", row.get("hit_20_thr_40", 0))

    headroom = wr_or50 - wr_sel
    if headroom < 0.03:
        return "already_strong"
    if wr_or50 < 0.30:
        return "recall"
    if wr_or50 < 0.40 and hit40_50 < 0.50:
        return "recall_weak_pool"
    if wr_or20 >= 0.45 and wr_sel < 0.35:
        return "reranking"
    if wr_or5 >= 0.40 and wr_sel < 0.35:
        return "reranking_ce_could_help"
    if wr_or20 > wr_or5 * 1.15 and wr_or5 < 0.35:
        return "fusion"
    return "mixed"

# Per subset bottleneck
bottleneck_by_sub = {}
for s in SUBSETS:
    r = oracle_by_sub[oracle_by_sub["subset"]==s].iloc[0]
    r2 = recall_df[recall_df["subset"]==s].iloc[0]
    row = {"oracle_wr_5": r["oracle_wr_5"], "oracle_wr_20": r["oracle_wr_20"],
           "oracle_wr_50": r2["oracle_wr_50"], "selected_wr": r["selected_wr"],
           "best_cand_wr": r2["best_cand_wr"],
           "hit_50_thr_30": r2.get("hit_20_030", 0), "hit_50_thr_40": r2.get("hit_20_040", 0)}
    bottleneck_by_sub[s] = classify_bottleneck(row)

# Build report
rep_lines = []
rep_lines += ["# Stage 8 — Pool Recall Diagnosis Report", "",
              f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}",
              f"Baseline: Stage5 routed | Val WR = {global_sel_wr:.5f} | Public = 0.732376", ""]

rep_lines += ["## 1. Global Oracle Table", ""]
rep_lines += ["| K | oracle_WR | oracle_WR(CE) | gap_vs_selected | hit@thr=0.30 | hit@thr=0.40 | hit@thr=0.50 |"]
rep_lines += ["|---|-----------|--------------|-----------------|--------------|--------------|--------------|"]
for _, row in summary_df.iterrows():
    k = int(row["k"])
    rep_lines.append(f"| {k} | {row['oracle_wr']:.4f} | {row['oracle_wr_ce']:.4f} | "
                     f"+{row['gap_vs_selected']:.4f} | {row['hit_rate_thr_30']:.3f} | "
                     f"{row['hit_rate_thr_40']:.3f} | {row['hit_rate_thr_50']:.3f} |")
rep_lines += ["", f"Selected (Stage5 routed): **{global_sel_wr:.5f}**", ""]

rep_lines += ["## 2. Per-Subset Oracle Table", ""]
rep_lines += ["| Subset | N | Selected | oracle@5 | oracle@20 | oracle@50 | gap@20 | bottleneck |"]
rep_lines += ["|--------|---|----------|----------|-----------|-----------|--------|------------|"]
for s in SUBSETS:
    r = oracle_by_sub[oracle_by_sub["subset"]==s].iloc[0]
    r2 = recall_df[recall_df["subset"]==s].iloc[0]
    rep_lines.append(f"| {s} | {r['n']} | {r['selected_wr']:.4f} | {r['oracle_wr_5']:.4f} | "
                     f"{r['oracle_wr_20']:.4f} | {r2['oracle_wr_50']:.4f} | "
                     f"+{r2['gap_20']:.4f} | {bottleneck_by_sub[s]} |")
rep_lines += [""]

rep_lines += ["## 3. Aka_Gha Diagnosis", ""]
s = "Aka_Gha"
r = oracle_by_sub[oracle_by_sub["subset"]==s].iloc[0]
r2 = recall_df[recall_df["subset"]==s].iloc[0]
rc = src_df[src_df["subset"]==s].iloc[0]
rep_lines += [
    f"- Selected WR: {r['selected_wr']:.4f}",
    f"- oracle@5: {r['oracle_wr_5']:.4f} (+{r['gap_5']:.4f})",
    f"- oracle@20: {r['oracle_wr_20']:.4f} (+{r2['gap_20']:.4f})",
    f"- oracle@50 (full pool): {r2['oracle_wr_50']:.4f}",
    f"- hit@20 (WR≥0.30): {r2['hit_20_030']:.3f}",
    f"- hit@20 (WR≥0.40): {r2['hit_20_040']:.3f}",
    f"- avg pool size: {r2['pool_size_avg']:.1f}",
    f"- avg best rank (RRF): {r2['best_rank_rrf_avg']:.1f}",
    f"- % oracle from sparse only: {rc['pct_oracle_sparse_only']:.1f}%",
    f"- % oracle from dense only:  {rc['pct_oracle_dense_only']:.1f}%",
    f"- % oracle same subset:      {rc['pct_oracle_same_subset']*100:.1f}%",
    f"- **Bottleneck: {bottleneck_by_sub[s]}**", ""]

err_aka = err_df[err_df["subset"]==s].iloc[0]
rep_lines += ["Error type breakdown:"]
for e in etypes:
    rep_lines.append(f"  - {e}: {err_aka[f'{e}_count']} ({err_aka[f'{e}_pct']:.1f}%)")
rep_lines += [""]

rep_lines += ["## 4. Amh_Eth Diagnosis", ""]
s = "Amh_Eth"
r = oracle_by_sub[oracle_by_sub["subset"]==s].iloc[0]
r2 = recall_df[recall_df["subset"]==s].iloc[0]
rc = src_df[src_df["subset"]==s].iloc[0]
rep_lines += [
    f"- Selected WR: {r['selected_wr']:.4f}",
    f"- oracle@5: {r['oracle_wr_5']:.4f} (+{r['gap_5']:.4f})",
    f"- oracle@20: {r['oracle_wr_20']:.4f} (+{r2['gap_20']:.4f})",
    f"- oracle@50 (full pool): {r2['oracle_wr_50']:.4f}",
    f"- hit@20 (WR≥0.30): {r2['hit_20_030']:.3f}",
    f"- hit@20 (WR≥0.40): {r2['hit_20_040']:.3f}",
    f"- avg pool size: {r2['pool_size_avg']:.1f}",
    f"- avg best rank (RRF): {r2['best_rank_rrf_avg']:.1f}",
    f"- % oracle from sparse only: {rc['pct_oracle_sparse_only']:.1f}%",
    f"- % oracle from dense only:  {rc['pct_oracle_dense_only']:.1f}%",
    f"- % oracle same subset:      {rc['pct_oracle_same_subset']*100:.1f}%",
    f"- **Bottleneck: {bottleneck_by_sub[s]}**", ""]

err_amh = err_df[err_df["subset"]==s].iloc[0]
rep_lines += ["Error type breakdown:"]
for e in etypes:
    rep_lines.append(f"  - {e}: {err_amh[f'{e}_count']} ({err_amh[f'{e}_pct']:.1f}%)")
rep_lines += [""]

rep_lines += ["## 5. Ranking vs Recall Conclusion", ""]
global_oracle50 = float(summary_df[summary_df["k"]==50]["oracle_wr"].values[0]) if 50 in ORACLE_KS else float(per_q["best_cand_wr"].mean())
global_oracle5  = float(summary_df[summary_df["k"]==5]["oracle_wr"].values[0])
global_oracle20 = float(summary_df[summary_df["k"]==20]["oracle_wr"].values[0])
headroom_5_to_sel  = global_oracle5  - global_sel_wr
headroom_20_to_sel = global_oracle20 - global_sel_wr
headroom_50_to_sel = global_oracle50 - global_sel_wr
rep_lines += [
    f"- Global selected WR:   {global_sel_wr:.4f}",
    f"- Global oracle@5:      {global_oracle5:.4f}  (headroom +{headroom_5_to_sel:.4f})",
    f"- Global oracle@20:     {global_oracle20:.4f} (headroom +{headroom_20_to_sel:.4f})",
    f"- Global oracle@50:     {global_oracle50:.4f} (headroom +{headroom_50_to_sel:.4f})",
    "",
    "**Interpretation:**",
    f"- Max theoretical gain with perfect reranker (same pool): +{headroom_50_to_sel:.4f}",
    f"- Gain achievable with oracle@5 (top-5 candidates): +{headroom_5_to_sel:.4f}",
    "",
    "If oracle@5 >> selected: **reranking failure** (good candidates exist but not selected).",
    "If oracle@50 is low: **recall failure** (good candidates not in pool at all).",
    ""]

rep_lines += ["## 6. Recommendations", ""]
recs = []
for s in SUBSETS:
    b = bottleneck_by_sub[s]
    r = oracle_by_sub[oracle_by_sub["subset"]==s].iloc[0]
    r2 = recall_df[recall_df["subset"]==s].iloc[0]
    gap = r2["gap_20"]
    if b in ("recall", "recall_weak_pool"):
        recs.append(f"- **{s}** (sel={r['selected_wr']:.3f}, oracle@50={r2['oracle_wr_50']:.3f}): "
                    f"RECALL bottleneck — better retrieval needed (BM25+, language-specific models)")
    elif b in ("reranking", "reranking_ce_could_help"):
        recs.append(f"- **{s}** (sel={r['selected_wr']:.3f}, oracle@5={r['oracle_wr_5']:.3f}): "
                    f"RERANKING bottleneck — stronger cross-encoder (BGE-reranker-v2-m3) could help")
    elif b == "fusion":
        recs.append(f"- **{s}** (sel={r['selected_wr']:.3f}): FUSION bottleneck — pool fusion or retrieval ordering issue")
    elif b == "already_strong":
        recs.append(f"- **{s}** (sel={r['selected_wr']:.3f}): already strong, limited headroom ({gap:.3f})")
    else:
        recs.append(f"- **{s}** (sel={r['selected_wr']:.3f}, oracle@20={r['oracle_wr_20']:.3f}): "
                    f"MIXED — consider both recall improvement and reranking (+{gap:.3f} headroom)")

rep_lines += recs
rep_lines += [""]

# Error type global summary
err_all = err_df[err_df["subset"]=="ALL"].iloc[0]
total_n = err_all["n"]
rep_lines += ["### Global error type breakdown:"]
for e in etypes:
    rep_lines.append(f"- {e}: {err_all[f'{e}_count']} / {total_n} ({err_all[f'{e}_pct']:.1f}%)")
rep_lines += [""]

report_text = "\n".join(rep_lines)
with open(OUT/"stage8_pool_recall_report.md", "w") as f: f.write(report_text)
log.info("Wrote stage8_pool_recall_report.md")

# ── TASK 8 — Final message ─────────────────────────────────────────────────────
total_secs = time.time() - START
log.info("")
log.info("DONE_STAGE8_POOL_DIAGNOSIS")
log.info(f"Stage5 public:")
log.info(f"0.732376")
log.info(f"Stage5 Val:")
log.info(f"{global_sel_wr:.5f}")
log.info(f"Global selected WR:")
log.info(f"{global_sel_wr:.5f}")
log.info(f"Global oracle@5:")
log.info(f"{global_oracle5:.5f}")
log.info(f"Global oracle@20:")
log.info(f"{global_oracle20:.5f}")
log.info(f"Global oracle@50:")
log.info(f"{global_oracle50:.5f}")

aka_sel  = float(oracle_by_sub[oracle_by_sub["subset"]=="Aka_Gha"]["selected_wr"].values[0])
aka_or20 = float(oracle_by_sub[oracle_by_sub["subset"]=="Aka_Gha"]["oracle_wr_20"].values[0])
amh_sel  = float(oracle_by_sub[oracle_by_sub["subset"]=="Amh_Eth"]["selected_wr"].values[0])
amh_or20 = float(oracle_by_sub[oracle_by_sub["subset"]=="Amh_Eth"]["oracle_wr_20"].values[0])

log.info(f"Aka_Gha bottleneck:")
log.info(f"{bottleneck_by_sub['Aka_Gha']} (sel={aka_sel:.4f} oracle@20={aka_or20:.4f})")
log.info(f"Amh_Eth bottleneck:")
log.info(f"{bottleneck_by_sub['Amh_Eth']} (sel={amh_sel:.4f} oracle@20={amh_or20:.4f})")

# Recommended next action
global_rr_pct = err_all["B_reranking_failure_pct"]
global_rc_pct = err_all["C_recall_failure_pct"]
global_wp_pct = err_all["D_weak_pool_pct"]
if global_rr_pct > global_rc_pct and global_rr_pct > 15:
    rec_action = f"stronger cross-encoder (BGE-reranker-v2-m3) — {global_rr_pct:.1f}% reranking failures"
elif global_rc_pct > 20 or bottleneck_by_sub.get("Aka_Gha","") in ("recall","recall_weak_pool"):
    rec_action = f"better retrieval for Aka_Gha/Amh_Eth — {global_rc_pct:.1f}% recall failures, oracle@50 still low"
else:
    rec_action = f"mixed: reranking ({global_rr_pct:.1f}%) + weak pool ({global_wp_pct:.1f}%) — stronger CE reranker first"

log.info(f"Recommended next action:")
log.info(rec_action)
log.info("Candidate for public submission:")
log.info("NO")
log.info(f"Total elapsed: {total_secs:.0f}s")
log.info("=" * 60)
