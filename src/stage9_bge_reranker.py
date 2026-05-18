"""
Stage 9 — BGE-reranker-v2-m3 for Lug_Uga and Swa_Ken (Stage5 pool)
Baseline: Stage5 routed, Val WR=0.37563, Public=0.732376
Stage8 diagnosis: Lug_Uga oracle@5=0.508 (+0.142), Swa_Ken oracle@5=0.569 (+0.100) — reranking bottleneck.
BGE replaces mMiniLMv2-L12 for these two subsets only. No new retrieval.
"""
import os, sys, json, re, time, pickle, logging, warnings, unicodedata, shutil, traceback
from pathlib import Path

os.environ["HF_HOME"]                = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"]     = "/tmp/hf_cache/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import faiss
warnings.filterwarnings("ignore")

WORK      = Path("/home/onyxia/work")
OUT       = WORK / "outputs" / "latest"
CACHE_DIR = Path("/tmp/stage6_cache")
S9_CACHE  = Path("/tmp/stage9_cache")
S9_CACHE.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("STAGE 9 — BGE Reranker for Lug_Uga and Swa_Ken")
log.info("=" * 60)
START = time.time()

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BASELINE_WR_S5   = 0.37563
THRESHOLD_UPDATE = 0.38263      # +0.007 over Stage5
S4_SUBSETS       = {"Swa_Ken", "Lug_Uga"}  # Stage5 routing used Stage4 for these
BGE_SUBSETS      = {"Lug_Uga", "Swa_Ken"}  # subsets to score with BGE
TOP_K_SPARSE     = 20
TOP_K_DENSE      = 20
MIN_GAIN         = 0.005        # per-subset routing threshold

# Candidate for public if:
CRIT_GLOBAL_WR   = 0.38263   # OR
CRIT_LUG_DELTA   = 0.06      # Lug_Uga + 0.006 global
CRIT_LUG_GLOBAL  = 0.006
CRIT_LUGSWA_BOTH = 0.007     # Lug+Swa together give +0.007

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

def score_rows(refs, preds):
    r1s = [rouge1_f1(str(a), str(b)) for a, b in zip(refs, preds)]
    rls = [rougel_f1(str(a), str(b)) for a, b in zip(refs, preds)]
    r1, rl = float(np.mean(r1s)), float(np.mean(rls))
    return r1, rl, 0.37*r1 + 0.37*rl, r1s, rls

def norm(t): return unicodedata.normalize("NFC", str(t)).lower()

def subset_wr(preds, refs, subs, s):
    mask = [i for i, ss in enumerate(subs) if ss == s]
    if not mask: return 0.0
    r1s = [rouge1_f1(refs[i], preds[i]) for i in mask]
    rls = [rougel_f1(refs[i], preds[i]) for i in mask]
    return 0.37*np.mean(r1s) + 0.37*np.mean(rls)

# ── Data ───────────────────────────────────────────────────────────────────────
log.info("Loading data...")
train  = pd.read_csv(WORK/"Train.csv")
val    = pd.read_csv(WORK/"Val.csv")
test   = pd.read_csv(WORK/"Test.csv")
sample = pd.read_csv(WORK/"SampleSubmission.csv")
for df in [train, val, test]: df["input_norm"] = df["input"].apply(norm)
for df in [train, val]:       df["output_norm"] = df["output"].apply(norm)
val_refs  = val["output"].tolist()
val_subs  = val["subset"].tolist()
test_subs = test["subset"].tolist()
SUBSETS   = sorted(train["subset"].unique())
log.info(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

# ── ACTION 0 — Protect Stage5 submission ───────────────────────────────────────
sub_path    = OUT / "submission.csv"
backup_path = OUT / "submission_backup_before_stage9_bge_reranker.csv"
shutil.copy(str(sub_path), str(backup_path))
log.info(f"Backup: {backup_path.name}")

s5d = pd.read_csv(OUT / "stage5_val_predictions_debug.csv")
s4d = pd.read_csv(OUT / "crossencoder_val_predictions_debug.csv")
s5_preds  = s5d["prediction"].tolist()
s4_preds  = s4d["prediction"].tolist()
s5_routed = [s4_preds[i] if val_subs[i] in S4_SUBSETS else s5_preds[i]
             for i in range(len(val_subs))]
r1b, rlb, wrb, _, _ = score_rows(val_refs, s5_routed)
log.info(f"Stage5 routed Val WR verify: {wrb:.5f}")

# ── Sparse retrievers (Stage5 config) ──────────────────────────────────────────
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

rnames = list(vecs.keys())
_wmap  = {"char_wb": 0.40, "char": 0.30, "word": 0.20, "byte": 0.10}
tw     = sum(_wmap[n] for n in rnames)
rrf_w_sparse = {n: _wmap[n]/tw for n in rnames}

def retrieve_sparse(q, s, name, topk=TOP_K_SPARSE, cv=None, si=None, sr=None, cdf=None):
    cv_ = cv or vecs; si_ = si or sub_i; sr_ = sr or sub_r; df_ = cdf or train
    v = cv_[name]; X_q = v.transform([q])
    if s in si_[name]: X_s, rids = si_[name][s], sr_[name][s]
    else: X_s = v.transform(df_["input_norm"]); rids = df_.index.tolist()
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

# ── Dense retrieval (Stage5: minilm + e5s from cache) ─────────────────────────
S5_DENSE_CONFIGS = [
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False, 0.5),
    ("e5s",    "intfloat/multilingual-e5-small",                               True,  0.5),
]
dense_model_keys = []
dense_sub_idx    = {}
dense_sub_rids   = {}
val_dense_ret    = {}

for model_key, _, _, rrf_w_d in S5_DENSE_CONFIGS:
    try:
        tr_embs  = np.load(str(CACHE_DIR / f"train_embs_{model_key}.npy")).astype(np.float32)
        val_embs = np.load(str(CACHE_DIR / f"val_embs_{model_key}.npy")).astype(np.float32)
        log.info(f"  Loaded cached {model_key} dim={tr_embs.shape[1]}")
        dim = tr_embs.shape[1]
        dense_sub_idx[model_key] = {}; dense_sub_rids[model_key] = {}
        for s in SUBSETS:
            mask = (train["subset"] == s).values
            if not mask.any(): continue
            idxs = train[train["subset"]==s].index.tolist()
            fi = faiss.IndexFlatIP(dim); fi.add(tr_embs[mask])
            dense_sub_idx[model_key][s] = fi; dense_sub_rids[model_key][s] = idxs
        gfi = faiss.IndexFlatIP(dim); gfi.add(tr_embs); grids = train.index.tolist()
        val_dense_ret[model_key] = []
        for i, (q_emb, s) in enumerate(zip(val_embs, val_subs)):
            qv = q_emb.reshape(1, -1)
            if s in dense_sub_idx[model_key]: fi2 = dense_sub_idx[model_key][s]; rids2 = dense_sub_rids[model_key][s]
            else: fi2 = gfi; rids2 = grids
            tk = min(TOP_K_DENSE, fi2.ntotal); D, I = fi2.search(qv, tk)
            val_dense_ret[model_key].append(
                ([rids2[j] for j in I[0] if j>=0],
                 [float(D[0][k]) for k in range(len(I[0])) if I[0][k]>=0]))
        log.info(f"  Val dense retrieval done for {model_key}")
        dense_model_keys.append((model_key, rrf_w_d))
    except Exception as e: log.warning(f"Dense {model_key}: {e}")

log.info(f"Dense models: {[k for k,_ in dense_model_keys]}")

# ── Load old CE scores ─────────────────────────────────────────────────────────
val_ce = {}
ce_path = CACHE_DIR / "val_ce_scores_stage6.pkl"
if ce_path.exists():
    with open(ce_path, "rb") as f: val_ce = pickle.load(f)
    log.info(f"Loaded old CE scores for {len(val_ce)} val queries")

# ── Pool builder ────────────────────────────────────────────────────────────────
def get_pool(i, sp_ret, dn_ret, top_ks=TOP_K_SPARSE, top_kd=TOP_K_DENSE):
    seen = {}
    for name in rnames:
        rows, scs = sp_ret[name][i]
        for rank, (rid, sc) in enumerate(zip(rows[:top_ks], scs[:top_ks])):
            if rid not in seen: seen[rid] = {}
            seen[rid][f"sp_{name}_r"] = rank; seen[rid][f"sp_{name}_s"] = sc
    for mk, _ in dense_model_keys:
        if mk not in dn_ret: continue
        rows, scs = dn_ret[mk][i]
        for rank, (rid, sc) in enumerate(zip(rows[:top_kd], scs[:top_kd])):
            if rid not in seen: seen[rid] = {}
            seen[rid][f"dn_{mk}_r"] = rank; seen[rid][f"dn_{mk}_s"] = sc
    return seen

# ── Stage9B — BGE-reranker-v2-m3 ───────────────────────────────────────────────
BGE_MODEL_USED = None
bge_tokenizer  = None
bge_model      = None

for bge_name in ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"]:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        log.info(f"Loading BGE model: {bge_name}...")
        t0 = time.time()
        bge_tokenizer = AutoTokenizer.from_pretrained(bge_name, cache_dir="/tmp/hf_cache")
        bge_model = AutoModelForSequenceClassification.from_pretrained(
            bge_name, cache_dir="/tmp/hf_cache").to(DEVICE)
        if DEVICE == "cuda":
            bge_model = bge_model.half()  # FP16 for ~8x speedup on T4
        bge_model.eval()
        test_enc = bge_tokenizer([["health query", "health answer"]], padding=True,
                                  truncation=True, max_length=64, return_tensors="pt")
        test_enc = {k: v.to(DEVICE) for k, v in test_enc.items()}
        with torch.no_grad(): _ = bge_model(**test_enc).logits
        BGE_MODEL_USED = bge_name
        prec = "fp16" if DEVICE == "cuda" else "fp32"
        log.info(f"  BGE loaded: {bge_name} [{prec}], {time.time()-t0:.1f}s")
        break
    except Exception as e:
        log.warning(f"  {bge_name} failed: {e}"); bge_tokenizer = None; bge_model = None

def bge_score_pairs(pairs, batch_size=32):
    """pairs: list of [query_text, doc_text]. Returns list of float scores."""
    if bge_model is None: return [0.0]*len(pairs)
    all_scores = []
    t_s = time.time()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start+batch_size]
        enc = bge_tokenizer(batch, padding=True, truncation=True,
                             max_length=512, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = bge_model(**enc)
        scores = out.logits.squeeze(-1).float().cpu().numpy()
        all_scores.extend(scores.tolist() if scores.ndim > 0 else [float(scores)])
        if (start // batch_size) % 100 == 0 and start > 0:
            elapsed = time.time()-t_s; rate = len(all_scores)/elapsed if elapsed > 0 else 0
            log.info(f"    BGE: {len(all_scores)}/{len(pairs)} ({rate:.0f} pairs/s)")
    return all_scores

# BGE scoring for Val: only Lug_Uga and Swa_Ken
val_bge_cache = S9_CACHE / "val_bge_scores_stage9.pkl"
val_bge = {}  # {query_idx: {rid: score}}

if val_bge_cache.exists():
    with open(val_bge_cache, "rb") as f: val_bge = pickle.load(f)
    log.info(f"Loaded cached BGE Val scores for {len(val_bge)} queries")
elif bge_model is not None:
    log.info(f"Collecting all BGE pairs for {BGE_SUBSETS}...")
    t0 = time.time()
    all_bge_meta = []   # (query_idx, rid)
    all_bge_pairs = []  # [query_text, doc_text]
    val_inputs = val["input"].tolist()
    for i, (q, s) in enumerate(zip(val_inputs, val_subs)):
        if s not in BGE_SUBSETS: continue
        pool = get_pool(i, val_sparse_ret, val_dense_ret)
        for rid in pool:
            all_bge_meta.append((i, rid))
            all_bge_pairs.append([q, str(train.loc[rid, "output"])])
    log.info(f"  {len(all_bge_pairs)} total pairs collected in {time.time()-t0:.1f}s")
    log.info(f"  Scoring with BGE (batch_size=64)...")
    t0 = time.time()
    all_scores = bge_score_pairs(all_bge_pairs, batch_size=64)
    elapsed = time.time()-t0
    log.info(f"  {len(all_scores)} pairs scored in {elapsed:.0f}s ({len(all_scores)/elapsed:.0f} pairs/s)")
    for (qi, rid), sc in zip(all_bge_meta, all_scores):
        if qi not in val_bge: val_bge[qi] = {}
        val_bge[qi][rid] = sc
    with open(val_bge_cache, "wb") as f: pickle.dump(val_bge, f, protocol=4)
    log.info(f"  BGE Val cached: {len(val_bge)} queries")
else:
    log.warning("BGE model not loaded — BGE features will be NaN")

# ── Stage9C — Direct BGE evaluation ────────────────────────────────────────────
log.info("Stage9C: Direct BGE evaluation at different depths...")
direct_results = {}

for k_depth in [5, 10, 20]:
    preds_direct = list(s5_routed)  # start from Stage5 routed
    for i, (s) in enumerate(val_subs):
        if s not in BGE_SUBSETS: continue
        if i not in val_bge: continue
        bge_q = val_bge[i]
        # Get RRF-top-K candidates
        pool = get_pool(i, val_sparse_ret, val_dense_ret)
        rrf_scores = {}
        for rid in pool:
            rrf_sp = sum(rrf_w_sparse[n]/(60+pool[rid].get(f"sp_{n}_r", TOP_K_SPARSE))
                         for n in rnames if f"sp_{n}_r" in pool[rid])
            rrf_dn = sum(0.5/(60+pool[rid].get(f"dn_{mk}_r", TOP_K_DENSE))
                         for mk, _ in dense_model_keys if f"dn_{mk}_r" in pool[rid])
            rrf_scores[rid] = rrf_sp + rrf_dn
        top_k_rids = sorted(pool.keys(), key=lambda r: rrf_scores.get(r,0), reverse=True)[:k_depth]
        if not top_k_rids: continue
        best_rid = max(top_k_rids, key=lambda r: bge_q.get(r, -999))
        preds_direct[i] = str(train.loc[best_rid, "output"])

    r1, rl, wr_v, _, _ = score_rows(val_refs, preds_direct)
    exp_name = f"9C_direct_BGE_top{k_depth}"
    direct_results[exp_name] = {"preds": preds_direct[:], "r1": r1, "rl": rl, "wr": wr_v}
    lug_wr = subset_wr(preds_direct, val_refs, val_subs, "Lug_Uga")
    swa_wr = subset_wr(preds_direct, val_refs, val_subs, "Swa_Ken")
    log.info(f"  {exp_name}: WR={wr_v:.5f} Lug={lug_wr:.4f} Swa={swa_wr:.4f} "
             f"(Δ={wr_v-BASELINE_WR_S5:+.5f})")

# ── Stage9D — Feature builder ───────────────────────────────────────────────────
def build_features_v9(sp_ret, dn_ret, q_inputs, q_subs, ce_scores, bge_scores,
                       cand_df, target_bge_subsets=BGE_SUBSETS, top_ks=TOP_K_SPARSE, top_kd=TOP_K_DENSE):
    rows = []
    for i, (q_in, s) in enumerate(zip(q_inputs, q_subs)):
        pool = get_pool(i, sp_ret, dn_ret, top_ks, top_kd)
        if not pool: continue
        q_words = norm(q_in).split(); q_len = len(q_words)
        q_bigrams = set(zip(q_words, q_words[1:]))

        # RRF
        rrf_sp, rrf_dn = {}, {}
        for name in rnames:
            w = rrf_w_sparse[name]; r_list, _ = sp_ret[name][i]
            for rank, rid in enumerate(r_list[:top_ks]):
                rrf_sp[rid] = rrf_sp.get(rid, 0.0) + w/(60+rank)
        for mk, rw in dense_model_keys:
            r_list, _ = dn_ret[mk][i]
            for rank, rid in enumerate(r_list[:top_kd]):
                rrf_dn[rid] = rrf_dn.get(rid, 0.0) + rw/(60+rank)

        # CE and BGE for this query
        ce_q  = ce_scores.get(i, {})
        bge_q = bge_scores.get(i, {})  # only populated for BGE_SUBSETS

        # Sort by RRF for rank
        all_rrf = {rid: rrf_sp.get(rid,0)+rrf_dn.get(rid,0) for rid in pool}
        rrf_sorted = sorted(pool.keys(), key=lambda r: all_rrf.get(r,0), reverse=True)
        rrf_rank_map = {rid: rank for rank, rid in enumerate(rrf_sorted)}

        # CE rank
        ce_sorted = sorted(pool.keys(), key=lambda r: ce_q.get(r,-999), reverse=True)
        ce_rank_map = {rid: rank for rank, rid in enumerate(ce_sorted)}

        # BGE rank (only if available)
        use_bge = s in target_bge_subsets and bool(bge_q)
        if use_bge:
            bge_sorted = sorted(pool.keys(), key=lambda r: bge_q.get(r,-999), reverse=True)
            bge_rank_map = {rid: rank for rank, rid in enumerate(bge_sorted)}
            bge_mean = float(np.mean([bge_q.get(r,-999) for r in pool if bge_q.get(r,-999)>-999] or [0]))
        else:
            bge_rank_map = {}; bge_mean = 0.0

        # Answer length stats
        ans_lens = [len(str(cand_df.loc[rid, "output"]).split()) for rid in pool]
        med_ans_len = float(np.median(ans_lens)) if ans_lens else 1.0

        for rid, info in pool.items():
            feat = {"query_idx": i, "doc_id": rid}
            # Sparse ranks/scores
            for name in rnames:
                feat[f"rank_{name}"]  = info.get(f"sp_{name}_r", top_ks)
                feat[f"score_{name}"] = info.get(f"sp_{name}_s", 0.0)
            # Dense ranks/scores
            for mk, _ in dense_model_keys:
                feat[f"rank_dense_{mk}"]  = info.get(f"dn_{mk}_r", top_kd)
                feat[f"score_dense_{mk}"] = info.get(f"dn_{mk}_s", 0.0)
            # RRF
            feat["rrf_sparse"] = rrf_sp.get(rid, 0.0)
            feat["rrf_dense"]  = rrf_dn.get(rid, 0.0)
            feat["rrf_total"]  = all_rrf.get(rid, 0.0)
            feat["rrf_rank"]   = rrf_rank_map.get(rid, len(pool))
            # Old CE
            feat["old_ce_score"] = ce_q.get(rid, -999.0)
            feat["old_ce_rank"]  = ce_rank_map.get(rid, len(pool))
            # BGE (NaN for non-BGE subsets — LGB handles natively)
            if use_bge:
                feat["bge_score"]  = bge_q.get(rid, -999.0)
                feat["bge_rank"]   = bge_rank_map.get(rid, len(pool))
                feat["bge_margin"] = bge_q.get(rid, -999.0) - bge_mean
                feat["bge_rank_delta_vs_ce"] = float(ce_rank_map.get(rid, len(pool)) - bge_rank_map.get(rid, len(pool)))
                feat["agreement_bge_rrf"] = int(bge_rank_map.get(rid, len(pool)) < 10
                                                 and rrf_rank_map.get(rid, len(pool)) < 10)
                feat["agreement_bge_ce"]  = int(bge_rank_map.get(rid, len(pool)) < 10
                                                 and ce_rank_map.get(rid, len(pool)) < 10)
            # else: NaN for bge features (pd.DataFrame will use NaN for missing keys)
            # Structural
            feat["subset_id"]    = SUBSETS.index(s) if s in SUBSETS else -1
            feat["same_subset"]  = int(cand_df.loc[rid, "subset"] == s)
            ans = str(cand_df.loc[rid, "output"])
            ans_len = len(ans.split())
            feat["query_len"]        = q_len
            feat["answer_len"]       = ans_len
            feat["answer_len_ratio"] = ans_len / (med_ans_len + 1)
            feat["n_sparse_found"]   = sum(1 for nm in rnames if f"sp_{nm}_r" in info)
            feat["n_dense_found"]    = sum(1 for mk,_ in dense_model_keys if f"dn_{mk}_r" in info)
            feat["n_total_found"]    = feat["n_sparse_found"] + feat["n_dense_found"]
            ans_w = ans.lower().split()[:100]
            a_bigrams = set(zip(ans_w, ans_w[1:]))
            feat["bigram_overlap"]   = len(q_bigrams & a_bigrams) / (len(q_bigrams) + 1)
            sp_ranks = [info.get(f"sp_{nm}_r", top_ks) for nm in rnames]
            feat["min_sparse_rank"]  = min(sp_ranks)
            feat["rank_stability"]   = float(np.std(sp_ranks))
            rows.append(feat)
    return pd.DataFrame(rows)

log.info("Building Stage9 features for Val...")
t0 = time.time()
feat9 = build_features_v9(val_sparse_ret, val_dense_ret,
                            val["input"].tolist(), val_subs,
                            val_ce, val_bge, train)
log.info(f"  {len(feat9)} rows, {len(feat9.columns)} cols in {time.time()-t0:.1f}s")

# ROUGE targets
log.info("Computing ROUGE targets...")
t0 = time.time()
doc_ids_v = feat9["doc_id"].astype(int).values
q_idxs_v  = feat9["query_idx"].astype(int).values
train_outs = train["output"].values
r1s_t = [rouge1_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
rls_t = [rougel_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
log.info(f"  done in {time.time()-t0:.1f}s")

feat_cols_all = [c for c in feat9.columns if c not in ("query_idx", "doc_id")]
log.info(f"Feature columns: {len(feat_cols_all)} ({feat_cols_all})")

# ── LGB training ────────────────────────────────────────────────────────────────
params_lgb = dict(objective="regression", metric="rmse", num_leaves=63,
                  learning_rate=0.05, verbosity=-1, random_state=42,
                  num_threads=4, min_data_in_leaf=5)

# 80/20 split on Val queries
n_q   = len(val)
split = int(n_q * 0.8)
tr_m  = q_idxs_v < split
vl_m  = ~tr_m

TARGET_VARIANTS = [
    ("rl_heavy",      lambda r1, rl: 0.35*r1 + 0.65*rl),
    ("balanced",      lambda r1, rl: 0.50*r1 + 0.50*rl),
    ("rougel_heavy",  lambda r1, rl: 0.30*r1 + 0.70*rl),
]

experiments9   = {}
trained_models = {}

def fallback_pred(i, sp_ret):
    s = val_subs[i]; rn = {"Aka_Gha":"char","Amh_Eth":"byte"}.get(s,"char_wb")
    rows, _ = sp_ret[rn][i]
    return train.loc[rows[0], "output"] if rows else ""

# ── Global LGB (all subsets, BGE features as NaN for non-BGE subsets) ──────────
log.info("Training global LGB variants (all subsets)...")
X_all = feat9[feat_cols_all].values

for tgt_name, tgt_fn in TARGET_VARIANTS:
    try:
        y = np.array([tgt_fn(r1, rl) for r1, rl in zip(r1s_t, rls_t)])
        dtrain = lgb.Dataset(X_all[tr_m], label=y[tr_m])
        dvalid = lgb.Dataset(X_all[vl_m], label=y[vl_m])
        t0 = time.time()
        model = lgb.train(params_lgb, dtrain, num_boost_round=400, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        log.info(f"  9A_global_{tgt_name}: {time.time()-t0:.1f}s iter={model.best_iteration}")
        lgb_scores = model.predict(X_all)
        preds = []
        for i in range(n_q):
            mask = q_idxs_v == i
            if not mask.any(): preds.append(fallback_pred(i, val_sparse_ret)); continue
            best = int(doc_ids_v[mask][np.argmax(lgb_scores[mask])])
            preds.append(train.loc[best, "output"])
        r1, rl, wr_v, _, _ = score_rows(val_refs, preds)
        exp_name = f"9A_global_{tgt_name}"
        experiments9[exp_name] = {"preds": preds, "r1": r1, "rl": rl, "wr": wr_v}
        trained_models[exp_name] = (model, feat_cols_all)
        lug_wr = subset_wr(preds, val_refs, val_subs, "Lug_Uga")
        swa_wr = subset_wr(preds, val_refs, val_subs, "Swa_Ken")
        log.info(f"  {exp_name}: WR={wr_v:.5f} Lug={lug_wr:.4f} Swa={swa_wr:.4f} "
                 f"(Δ={wr_v-BASELINE_WR_S5:+.5f})")
    except Exception as e:
        log.warning(f"9A_global_{tgt_name}: {e}\n{traceback.format_exc()[:200]}")

# ── Lug/Swa-specific LGB ────────────────────────────────────────────────────────
log.info("Training Lug/Swa-specific LGB variants...")
val_subs_arr     = np.array(val_subs)
lugswa_mask_feat = np.isin(val_subs_arr, list(BGE_SUBSETS))
lugswa_mask_rows = np.isin(val_subs_arr[q_idxs_v], list(BGE_SUBSETS))  # rows in feat9

# BGE features present for LugSwa only — use feat9 rows where subset in BGE_SUBSETS
feat9_lugswa = feat9[lugswa_mask_rows].reset_index(drop=True)
doc_ids_ls   = feat9_lugswa["doc_id"].astype(int).values
q_idxs_ls    = feat9_lugswa["query_idx"].astype(int).values
r1s_ls = [rouge1_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_ls, doc_ids_ls)]
rls_ls = [rougel_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_ls, doc_ids_ls)]

lugswa_q_set = sorted(set(q_idxs_ls))
split_ls     = int(len(lugswa_q_set) * 0.8)
tr_qs_ls     = set(lugswa_q_set[:split_ls])
tr_m_ls      = np.array([q in tr_qs_ls for q in q_idxs_ls])
vl_m_ls      = ~tr_m_ls

# Feature columns for LugSwa model: only include BGE cols if they have non-NaN values
bge_cols = ["bge_score", "bge_rank", "bge_margin", "bge_rank_delta_vs_ce",
            "agreement_bge_rrf", "agreement_bge_ce"]
feat_cols_ls = [c for c in feat_cols_all
                if c not in ("query_idx", "doc_id")
                and not (c in bge_cols and feat9_lugswa[c].isna().all())]
X_ls = feat9_lugswa[feat_cols_ls].values

for tgt_name, tgt_fn in TARGET_VARIANTS:
    try:
        y_ls = np.array([tgt_fn(r1, rl) for r1, rl in zip(r1s_ls, rls_ls)])
        dtrain_ls = lgb.Dataset(X_ls[tr_m_ls], label=y_ls[tr_m_ls])
        dvalid_ls = lgb.Dataset(X_ls[vl_m_ls], label=y_ls[vl_m_ls])
        t0 = time.time()
        model_ls = lgb.train(params_lgb, dtrain_ls, num_boost_round=400,
                              valid_sets=[dvalid_ls],
                              callbacks=[lgb.early_stopping(30, verbose=False),
                                         lgb.log_evaluation(-1)])
        log.info(f"  9B_lugswa_{tgt_name}: {time.time()-t0:.1f}s iter={model_ls.best_iteration}")

        lgb_scores_ls = model_ls.predict(X_ls)
        # Build predictions: LugSwa from this model, others from Stage5
        preds_ls = list(s5_routed)
        for idx, qi in enumerate(q_idxs_ls):
            # All rows for this query
            mask = q_idxs_ls == qi
            if not mask.any(): continue
            if qi not in [j for j in range(n_q) if val_subs[j] in BGE_SUBSETS]: continue
            best_doc = int(doc_ids_ls[mask][np.argmax(lgb_scores_ls[mask])])
            preds_ls[qi] = str(train.loc[best_doc, "output"])
        # Handle queries in BGE_SUBSETS that might not appear in feat9_lugswa
        for i, s in enumerate(val_subs):
            if s in BGE_SUBSETS and i not in set(q_idxs_ls):
                preds_ls[i] = fallback_pred(i, val_sparse_ret)

        r1, rl, wr_v, _, _ = score_rows(val_refs, preds_ls)
        exp_name = f"9B_lugswa_{tgt_name}"
        experiments9[exp_name] = {"preds": preds_ls, "r1": r1, "rl": rl, "wr": wr_v}
        trained_models[exp_name] = (model_ls, feat_cols_ls)
        lug_wr = subset_wr(preds_ls, val_refs, val_subs, "Lug_Uga")
        swa_wr = subset_wr(preds_ls, val_refs, val_subs, "Swa_Ken")
        log.info(f"  {exp_name}: WR={wr_v:.5f} Lug={lug_wr:.4f} Swa={swa_wr:.4f} "
                 f"(Δ={wr_v-BASELINE_WR_S5:+.5f})")
    except Exception as e:
        log.warning(f"9B_lugswa_{tgt_name}: {e}\n{traceback.format_exc()[:300]}")

# Add direct results to experiments9
experiments9.update(direct_results)

# ── Stage9E — Routing ────────────────────────────────────────────────────────────
log.info("\nStage9E: Subset-level routing...")

# Per-subset Stage4/Stage5 WR
s4_wr_by_sub = {s: subset_wr(s4_preds, val_refs, val_subs, s) for s in SUBSETS}
s5_wr_by_sub = {s: subset_wr(s5_routed, val_refs, val_subs, s) for s in SUBSETS}

routing_table = {}
for s in SUBSETS:
    # Baseline for this subset: best of Stage4/Stage5
    if s in S4_SUBSETS:
        base_wr = max(s4_wr_by_sub[s], s5_wr_by_sub[s])
        best_src, best_wr_s = ("stage4" if s4_wr_by_sub[s] >= s5_wr_by_sub[s] else "stage5"), base_wr
    else:
        best_src, best_wr_s = "stage5_routed", s5_wr_by_sub[s]

    best_exp_s = best_src
    for exp, sc in experiments9.items():
        if "preds" not in sc: continue
        exp_wr = subset_wr(sc["preds"], val_refs, val_subs, s)
        if exp_wr > best_wr_s + MIN_GAIN:
            best_src = "stage9"; best_wr_s = exp_wr; best_exp_s = exp
    routing_table[s] = {"source": best_src, "exp": best_exp_s, "wr": best_wr_s,
                         "s5_wr": s5_wr_by_sub[s], "delta": best_wr_s - s5_wr_by_sub[s]}
    log.info(f"  {s}: best={best_src}/{best_exp_s} WR={best_wr_s:.4f} "
             f"(S5={s5_wr_by_sub[s]:.4f} Δ={best_wr_s-s5_wr_by_sub[s]:+.4f})")

# Build routed predictions
def build_routed_preds(routing_table, experiments9, n_q, val_subs, s5_routed, s4_preds):
    preds = []
    for i, s in enumerate(val_subs):
        rt = routing_table[s]
        if rt["source"] in ("stage5_routed", "stage5") and s5_routed:
            preds.append(s5_routed[i])
        elif rt["source"] == "stage4" and s4_preds:
            preds.append(s4_preds[i])
        elif rt["source"] == "stage9" and rt["exp"] in experiments9:
            preds.append(experiments9[rt["exp"]]["preds"][i])
        elif s5_routed:
            preds.append(s5_routed[i])
        else:
            preds.append("")
    return preds

routed_preds = build_routed_preds(routing_table, experiments9, n_q, val_subs, s5_routed, s4_preds)
r1, rl, wr_rt, _, _ = score_rows(val_refs, routed_preds)
experiments9["9_routed"] = {"preds": routed_preds, "r1": r1, "rl": rl, "wr": wr_rt}
lug_wr_rt = subset_wr(routed_preds, val_refs, val_subs, "Lug_Uga")
swa_wr_rt = subset_wr(routed_preds, val_refs, val_subs, "Swa_Ken")
log.info(f"\n  9_routed: WR={wr_rt:.5f} Lug={lug_wr_rt:.4f} Swa={swa_wr_rt:.4f} "
         f"(Δ={wr_rt-BASELINE_WR_S5:+.5f})")

# ── Stage9F — Decision threshold ─────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("STAGE 9 — FINAL COMPARISON")
log.info(f"  Baseline (Stage5 routed): WR={BASELINE_WR_S5:.5f}")
log.info(f"  Submission threshold:     WR>={THRESHOLD_UPDATE:.5f} (+0.007)")

best9_exp = None; best9_wr = BASELINE_WR_S5
for exp in sorted(experiments9, key=lambda e: experiments9[e].get("wr", 0), reverse=True):
    sc = experiments9[exp]
    flag = "★" if sc.get("wr",0) >= THRESHOLD_UPDATE else ("△" if sc.get("wr",0) > BASELINE_WR_S5 else " ")
    log.info(f"  {flag} {exp:40s}: WR={sc.get('wr',0):.5f} (Δ={sc.get('wr',0)-BASELINE_WR_S5:+.5f})")
    if sc.get("wr", 0) > best9_wr:
        best9_wr = sc["wr"]; best9_exp = exp

log.info(f"\nBest Stage9: {best9_exp} WR={best9_wr:.5f}")

# Check public submission criteria
lug_s5 = s4_wr_by_sub.get("Lug_Uga", 0)
swa_s5 = s4_wr_by_sub.get("Swa_Ken", 0)
lug9_wr = subset_wr(experiments9.get(best9_exp or "", {}).get("preds", s5_routed),
                    val_refs, val_subs, "Lug_Uga")
swa9_wr = subset_wr(experiments9.get(best9_exp or "", {}).get("preds", s5_routed),
                    val_refs, val_subs, "Swa_Ken")
lug_delta = lug9_wr - lug_s5
swa_delta = swa9_wr - swa_s5
global_delta = best9_wr - BASELINE_WR_S5

crit1 = best9_wr >= CRIT_GLOBAL_WR
crit2 = lug_delta >= CRIT_LUG_DELTA and global_delta >= CRIT_LUG_GLOBAL
crit3 = (lug9_wr > lug_s5 + MIN_GAIN or swa9_wr > swa_s5 + MIN_GAIN) and global_delta >= CRIT_LUGSWA_BOTH

meets_threshold = crit1 or crit2 or crit3
log.info(f"  Crit1 (global WR≥0.38263): {crit1} (WR={best9_wr:.5f})")
log.info(f"  Crit2 (Lug Δ≥+0.06 AND global Δ≥+0.006): {crit2} (lug_Δ={lug_delta:+.4f})")
log.info(f"  Crit3 (Lug/Swa improve AND global Δ≥+0.007): {crit3}")
log.info(f"  Meets any threshold: {meets_threshold}")

# ── Output files ────────────────────────────────────────────────────────────────
# Validation scores global
rows_g = [{"experiment": "Stage5_routed_baseline", "rouge1": 0.53457, "rougel": 0.48063,
            "weighted_rouge": BASELINE_WR_S5}]
for exp, sc in experiments9.items():
    rows_g.append({"experiment": exp, "rouge1": sc.get("r1",0), "rougel": sc.get("rl",0),
                    "weighted_rouge": sc.get("wr",0)})
pd.DataFrame(rows_g).to_csv(OUT/"stage9_bge_validation_scores.csv", index=False)

# Validation scores by subset
sub_rows = []
for s in SUBSETS:
    mask = [i for i, ss in enumerate(val_subs) if ss == s]
    if not mask: continue
    srefs = [val_refs[i] for i in mask]
    for exp, sc in experiments9.items():
        if "preds" not in sc: continue
        sp = [sc["preds"][i] for i in mask]
        r1_, rl_, wr_, _, _ = score_rows(srefs, sp)
        sub_rows.append({"experiment": exp, "subset": s, "n": len(mask),
                          "rouge1": r1_, "rougel": rl_, "weighted_rouge": wr_})
pd.DataFrame(sub_rows).to_csv(OUT/"stage9_bge_validation_scores_by_subset.csv", index=False)

# Lug/Swa debug
def _wr(r1, rl): return 0.37*r1 + 0.37*rl

debug_rows = []
for i, (ref, s) in enumerate(zip(val_refs, val_subs)):
    if s not in BGE_SUBSETS: continue
    best_preds = {exp: sc["preds"][i] for exp, sc in experiments9.items() if "preds" in sc}
    best_preds["stage5_routed"] = s5_routed[i]
    row = {"idx": i, "subset": s, "ref": ref[:100]}
    for exp, pred in best_preds.items():
        row[f"wr_{exp}"] = round(_wr(rouge1_f1(ref, pred), rougel_f1(ref, pred)), 4)
    debug_rows.append(row)
pd.DataFrame(debug_rows).to_csv(OUT/"stage9_bge_lug_swa_debug.csv", index=False)

# Changed rows vs Stage5
if best9_exp and best9_exp in experiments9:
    bp = experiments9[best9_exp]["preds"]
    changed = [{"val_idx": i, "subset": val_subs[i]}
               for i in range(n_q) if s5_routed[i] != bp[i]]
    pd.DataFrame(changed).to_csv(OUT/"stage9_changed_rows_vs_stage5.csv", index=False)
    log.info(f"Changed rows vs Stage5: {len(changed)}/{n_q}")

# Routing table
pd.DataFrame([{"subset": s, "source": rt["source"], "experiment": rt["exp"],
                "wr_best": rt["wr"], "wr_stage5": rt["s5_wr"], "delta": rt["delta"]}
              for s, rt in routing_table.items()]).to_csv(OUT/"stage9_routing_by_subset.csv", index=False)

# Val debug (best Stage9)
if best9_exp and best9_exp in experiments9:
    bp = experiments9[best9_exp]["preds"]
    vd = val.copy(); vd["prediction"] = bp; vd["experiment"] = best9_exp
    vd["rouge1"] = [rouge1_f1(r, p) for r, p in zip(val_refs, bp)]
    vd["rougel"] = [rougel_f1(r, p) for r, p in zip(val_refs, bp)]
    vd.to_csv(OUT/"stage9_bge_val_predictions_debug.csv", index=False)

# ── Test processing (only if threshold met) ────────────────────────────────────
candidate_public = False
if meets_threshold:
    log.info(f"\nThreshold met! Building test submission for {best9_exp}...")

    # Refit sparse on Train+Val
    base_df = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
    for df in [base_df, test]:
        df["input_norm"] = df["input"].apply(norm)
        if "output" in df.columns: df["output_norm"] = df["output"].apply(norm)
    SUBSETS_BASE = sorted(base_df["subset"].unique())

    rvecs = {}
    for name, cfg in CFGS.items():
        v2 = TfidfVectorizer(**cfg); v2.fit(base_df["input_norm"]); rvecs[name] = v2
    rsub_i, rsub_r = {}, {}
    for name, v2 in rvecs.items():
        rsub_i[name] = {s: v2.transform(base_df[base_df["subset"]==s]["input_norm"])
                        for s in SUBSETS_BASE if len(base_df[base_df["subset"]==s]) > 0}
        rsub_r[name] = {s: base_df[base_df["subset"]==s].index.tolist() for s in SUBSETS_BASE}

    log.info("  Sparse retrieval for Test...")
    t0 = time.time()
    test_sparse_ret = {name: [] for name in rvecs}
    for q, s in zip(test["input_norm"], test_subs):
        for name in rvecs:
            v2 = rvecs[name]; X_q = v2.transform([q])
            if s in rsub_i[name]: X_s, rids = rsub_i[name][s], rsub_r[name][s]
            else: X_s = v2.transform(base_df["input_norm"]); rids = base_df.index.tolist()
            sim = (X_q @ X_s.T).toarray()[0]; tk = min(TOP_K_SPARSE, len(sim))
            top = np.argpartition(sim, -tk)[-tk:] if len(sim) > tk else np.argsort(sim)[::-1]
            top = top[np.argsort(sim[top])[::-1]]
            test_sparse_ret[name].append(([rids[j] for j in top], sim[top].tolist()))
    log.info(f"    done in {time.time()-t0:.1f}s")

    # Dense for test from cache
    test_dense_ret = {}
    rdense_sub_idx = {}; rdense_sub_rids = {}
    for model_key, model_name2, _, _ in S5_DENSE_CONFIGS:
        if model_key not in [k for k, _ in dense_model_keys]: continue
        try:
            base_embs_arr = np.load(str(CACHE_DIR / f"base_embs_{model_key}.npy")).astype(np.float32)
            te_embs_arr   = np.load(str(CACHE_DIR / f"test_embs_{model_key}.npy")).astype(np.float32)
            log.info(f"  Loaded base/test embs for {model_key}")
            dim3 = base_embs_arr.shape[1]
            rdense_sub_idx[model_key] = {}; rdense_sub_rids[model_key] = {}
            for s in SUBSETS_BASE:
                bmask = (base_df["subset"] == s).values
                if not bmask.any(): continue
                bidxs = base_df[base_df["subset"]==s].index.tolist()
                fi = faiss.IndexFlatIP(dim3); fi.add(base_embs_arr[bmask])
                rdense_sub_idx[model_key][s] = fi; rdense_sub_rids[model_key][s] = bidxs
            gfi = faiss.IndexFlatIP(dim3); gfi.add(base_embs_arr); grids = base_df.index.tolist()
            test_dense_ret[model_key] = []
            for i2, (q_emb, s) in enumerate(zip(te_embs_arr, test_subs)):
                qv = q_emb.reshape(1, -1)
                if s in rdense_sub_idx[model_key]: fi4 = rdense_sub_idx[model_key][s]; rids4 = rdense_sub_rids[model_key][s]
                else: fi4 = gfi; rids4 = grids
                tk = min(TOP_K_DENSE, fi4.ntotal); D, I = fi4.search(qv, tk)
                test_dense_ret[model_key].append(
                    ([rids4[j] for j in I[0] if j>=0],
                     [float(D[0][k]) for k in range(len(I[0])) if I[0][k]>=0]))
        except Exception as e: log.warning(f"Dense test {model_key}: {e}")

    # Old CE for test
    test_ce = {}
    test_ce_path = CACHE_DIR / "test_ce_scores_stage6.pkl"
    if test_ce_path.exists():
        with open(test_ce_path, "rb") as f: test_ce = pickle.load(f)
        log.info(f"  Reused Stage6 test CE scores for {len(test_ce)} queries")

    # BGE scoring for test Lug/Swa
    test_bge_cache = S9_CACHE / "test_bge_scores_stage9.pkl"
    test_bge = {}
    if test_bge_cache.exists():
        with open(test_bge_cache, "rb") as f: test_bge = pickle.load(f)
        log.info(f"  Loaded cached BGE test scores for {len(test_bge)} queries")
    elif bge_model is not None:
        log.info(f"  Collecting BGE test pairs for {BGE_SUBSETS}...")
        t0 = time.time()
        test_bge_meta = []; test_bge_pairs = []
        for i2, (q, s) in enumerate(zip(test["input"].tolist(), test_subs)):
            if s not in BGE_SUBSETS: continue
            pool = get_pool(i2, test_sparse_ret, test_dense_ret)
            for rid in pool:
                test_bge_meta.append((i2, rid))
                test_bge_pairs.append([q, str(base_df.loc[rid, "output"])])
        log.info(f"  {len(test_bge_pairs)} test pairs, scoring...")
        test_scores = bge_score_pairs(test_bge_pairs, batch_size=64)
        for (i2, rid), sc in zip(test_bge_meta, test_scores):
            if i2 not in test_bge: test_bge[i2] = {}
            test_bge[i2][rid] = sc
        with open(test_bge_cache, "wb") as f: pickle.dump(test_bge, f, protocol=4)
        elapsed = time.time()-t0
        log.info(f"  BGE test done: {len(test_bge_pairs)} pairs in {elapsed:.0f}s ({len(test_bge_pairs)/elapsed:.0f} pairs/s)")

    # Build test features
    log.info("  Building test features...")
    t0 = time.time()
    feat_test = build_features_v9(test_sparse_ret, test_dense_ret,
                                   test["input"].tolist(), test_subs,
                                   test_ce, test_bge, base_df)
    log.info(f"    {len(feat_test)} rows in {time.time()-t0:.1f}s")

    tq_arr = feat_test["query_idx"].astype(int).values
    td_arr = feat_test["doc_id"].astype(int).values

    # Load Stage5/Stage4 test submissions for routing
    sub5t = pd.read_csv(OUT/"submission_stage5_best.csv")
    sub4t = pd.read_csv(OUT/"submission_stage4_crossencoder.csv")
    test_ids = test["ID"].tolist()

    def get_stage5_pred(tid):
        row = sub5t[sub5t["ID"]==tid]
        return str(row["TargetRLF1"].values[0]) if len(row)>0 else ""
    def get_stage4_pred(tid):
        row = sub4t[sub4t["ID"]==tid]
        return str(row["TargetRLF1"].values[0]) if len(row)>0 else ""

    test_preds = []
    for i2 in range(len(test)):
        s = test_subs[i2]; rt = routing_table.get(s, {})
        src = rt.get("source", "stage5_routed")
        exp_k = rt.get("exp", "stage5_routed")

        if src in ("stage5_routed", "stage5"):
            test_preds.append(get_stage5_pred(test_ids[i2]))
        elif src == "stage4":
            test_preds.append(get_stage4_pred(test_ids[i2]))
        elif src == "stage9" and exp_k in trained_models:
            mdl, fc = trained_models[exp_k]
            for c in fc:
                if c not in feat_test.columns: feat_test[c] = 0.0
            mask = tq_arr == i2
            if mask.any():
                # Direct BGE: special case
                if "direct_BGE" in exp_k:
                    k_d = int(exp_k.split("top")[-1]) if "top" in exp_k else 20
                    pool = get_pool(i2, test_sparse_ret, test_dense_ret)
                    rrf_s = {r: sum(rrf_w_sparse[n]/(60+pool[r].get(f"sp_{n}_r",TOP_K_SPARSE))
                                    for n in rnames if f"sp_{n}_r" in pool[r]) +
                              sum(0.5/(60+pool[r].get(f"dn_{mk}_r",TOP_K_DENSE))
                                  for mk,_ in dense_model_keys if f"dn_{mk}_r" in pool[r])
                             for r in pool}
                    top_rids = sorted(pool.keys(), key=lambda r: rrf_s.get(r,0), reverse=True)[:k_d]
                    bge_q = test_bge.get(i2, {})
                    best_rid = max(top_rids, key=lambda r: bge_q.get(r,-999)) if top_rids else None
                    if best_rid is not None: test_preds.append(str(base_df.loc[best_rid, "output"])); continue
                sc_arr = mdl.predict(feat_test[fc].values[mask])
                best_doc = int(td_arr[mask][np.argmax(sc_arr)])
                test_preds.append(str(base_df.loc[best_doc, "output"]))
            else:
                test_preds.append(get_stage5_pred(test_ids[i2]))
        else:
            test_preds.append(get_stage5_pred(test_ids[i2]))

    sub9 = sample[["ID"]].copy()
    sub9["TargetRLF1"] = sub9["ID"].map(dict(zip(test_ids, test_preds)))
    sub9["TargetR1F1"] = sub9["TargetRLF1"]; sub9["TargetLLM"] = sub9["TargetRLF1"]

    assert list(sub9.columns) == ["ID","TargetRLF1","TargetR1F1","TargetLLM"]
    assert sub9.shape == sample.shape
    assert (sub9["ID"].values == sample["ID"].values).all()
    assert sub9[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
    assert (sub9["TargetRLF1"] == sub9["TargetR1F1"]).all()
    assert (sub9["TargetRLF1"] == sub9["TargetLLM"]).all()
    log.info("Submission checks PASSED")

    sub9.to_csv(OUT/"submission_stage9_bge_best.csv", index=False)
    shutil.copy(str(OUT/"submission_stage9_bge_best.csv"), str(OUT/"submission.csv"))
    log.info(f"submission.csv UPDATED: {BASELINE_WR_S5:.5f} → {best9_wr:.5f}")
    candidate_public = True
else:
    log.info(f"\nThreshold NOT met. Keeping Stage5 submission.")
    shutil.copy(str(backup_path), str(sub_path))
    candidate_public = False

# Final submission checks
final = pd.read_csv(OUT/"submission.csv")
assert list(final.columns) == ["ID","TargetRLF1","TargetR1F1","TargetLLM"]
assert final.shape == sample.shape
assert (final["ID"].values == sample["ID"].values).all()
assert final[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
assert (final["TargetRLF1"] == final["TargetR1F1"]).all()
assert (final["TargetRLF1"] == final["TargetLLM"]).all()
log.info("Final submission.csv checks PASSED")

# Report
routing_str = "; ".join(f"{s}→{rt['source']}" for s, rt in routing_table.items())
lug_s9 = subset_wr(experiments9.get(best9_exp or "", {}).get("preds", s5_routed), val_refs, val_subs, "Lug_Uga")
swa_s9 = subset_wr(experiments9.get(best9_exp or "", {}).get("preds", s5_routed), val_refs, val_subs, "Swa_Ken")

md = f"""# Stage 9 Report — BGE Reranker for Lug_Uga and Swa_Ken

## Summary
- Baseline: Stage5 routed | Val WR = {BASELINE_WR_S5:.5f} | Public = 0.732376
- BGE model: `{BGE_MODEL_USED}`
- Pool: Stage5 (sparse×4 + minilm + e5s, NO e5b)
- BGE scored only for: {sorted(BGE_SUBSETS)}
- Threshold to update: WR ≥ {THRESHOLD_UPDATE:.5f}

## Stage8 Oracle (context)
- Lug_Uga oracle@5 = 0.508 (+0.142 vs selected)
- Swa_Ken oracle@5 = 0.569 (+0.100 vs selected)

## All experiments

| Experiment | WR | Δ vs Stage5 | Lug_Uga | Swa_Ken |
|---|---|---|---|---|
| Stage5_routed | {BASELINE_WR_S5:.5f} | — | {s4_wr_by_sub.get("Lug_Uga",0):.4f} | {s4_wr_by_sub.get("Swa_Ken",0):.4f} |
"""
for exp in sorted(experiments9, key=lambda e: experiments9[e].get("wr",0), reverse=True):
    sc = experiments9[exp]
    lug = subset_wr(sc.get("preds", s5_routed), val_refs, val_subs, "Lug_Uga")
    swa = subset_wr(sc.get("preds", s5_routed), val_refs, val_subs, "Swa_Ken")
    ok = "★ YES" if sc.get("wr",0) >= THRESHOLD_UPDATE else "NO"
    md += f"| {exp} | {sc.get('wr',0):.5f} | {sc.get('wr',0)-BASELINE_WR_S5:+.5f} | {lug:.4f} | {swa:.4f} |\n"

md += f"""
## Routing
{routing_str}

## Decision
- Best: {best9_exp} WR={best9_wr:.5f}
- Meets threshold: {meets_threshold}
- Candidate for public: {'YES' if candidate_public else 'NO'}
"""
with open(OUT/"stage9_report.md", "w") as f: f.write(md)

# run_manifest.json update
try:
    rm = {}
    rm_path = OUT/"run_manifest.json"
    if rm_path.exists():
        with open(rm_path) as f: rm = json.load(f)
    rm["stage9"] = {"bge_model": BGE_MODEL_USED, "best_exp": best9_exp,
                    "best_wr": best9_wr, "threshold": THRESHOLD_UPDATE,
                    "meets_threshold": meets_threshold,
                    "candidate_public": candidate_public}
    with open(rm_path, "w") as f: json.dump(rm, f, indent=2)
except Exception as e: log.warning(f"run_manifest: {e}")

# Final message
elapsed = time.time() - START
log.info("\n" + "="*60)
log.info("DONE_STAGE9_BGE_RERANKER")
log.info(f"Best public baseline:")
log.info(f"0.732376")
log.info(f"Stage5 Val:")
log.info(f"{BASELINE_WR_S5:.5f}")
log.info(f"Best Stage9 Val:")
log.info(f"{best9_wr:.5f}")
log.info(f"Delta vs Stage5:")
log.info(f"{best9_wr-BASELINE_WR_S5:+.5f}")
log.info(f"Lug_Uga Stage5:")
log.info(f"{s4_wr_by_sub.get('Lug_Uga',0):.5f}")
log.info(f"Lug_Uga Stage9:")
log.info(f"{lug_s9:.5f}")
log.info(f"Swa_Ken Stage5:")
log.info(f"{s4_wr_by_sub.get('Swa_Ken',0):.5f}")
log.info(f"Swa_Ken Stage9:")
log.info(f"{swa_s9:.5f}")
log.info(f"Best strategy:")
log.info(f"{best9_exp}")
changed_count = len(pd.read_csv(OUT/"stage9_changed_rows_vs_stage5.csv")) if (OUT/"stage9_changed_rows_vs_stage5.csv").exists() else 0
log.info(f"Rows changed vs Stage5:")
log.info(f"{changed_count}/{len(val)}")
log.info(f"Final submission:")
log.info(f"outputs/latest/submission.csv")
log.info(f"Candidate for public submission:")
log.info(f"{'YES' if candidate_public else 'NO'}")
log.info(f"BGE model used: {BGE_MODEL_USED}")
log.info(f"Total elapsed: {elapsed:.0f}s")
log.info("="*60)
