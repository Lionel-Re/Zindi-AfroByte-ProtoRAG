"""
Stage 6A — multilingual-e5-base dense retrieval + enriched LGB + subset routing
Baseline: Stage5 routed (5A_LGB_rl_heavy + S4 for Swa/Lug), Val WR = 0.37563
"""
import os, sys, json, re, time, pickle, logging, warnings, traceback, unicodedata
from pathlib import Path
from datetime import datetime

os.environ["HF_HOME"]                = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"]     = "/tmp/hf_cache/transformers"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

WORK = Path("/home/onyxia/work")
OUT  = WORK / "outputs" / "latest"
CACHE_DIR = Path("/tmp/stage6_cache")
CACHE_DIR.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("STAGE 6A — e5-base dense + enriched LGB + subset routing")
log.info("=" * 60)
START = time.time()

BASELINE_WR_S5   = 0.37563   # Stage5 routed
BASELINE_WR_S4   = 0.34666   # Stage4 CE_LGB_all (used for routing fallback)
S4_SUBSETS       = {"Swa_Ken", "Lug_Uga"}   # Stage5 kept S4 for these
AKA_AMH          = {"Aka_Gha", "Amh_Eth"}

# ── ROUGE ─────────────────────────────────────────────────────────────────────
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

# ── Data ──────────────────────────────────────────────────────────────────────
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

# ── Load Stage5 reference predictions for routing ─────────────────────────────
log.info("Loading Stage5 reference predictions...")
try:
    s5_debug = pd.read_csv(OUT/"stage5_val_predictions_debug.csv")
    s5_preds_val = s5_debug["prediction"].tolist()
    log.info(f"  Stage5 val preds loaded: {len(s5_preds_val)}")
except Exception as e:
    log.warning(f"  Stage5 val preds not found: {e}"); s5_preds_val = None

try:
    s4_debug = pd.read_csv(OUT/"crossencoder_val_predictions_debug.csv")
    s4_preds_val = s4_debug["prediction"].tolist()
    log.info(f"  Stage4 val preds loaded: {len(s4_preds_val)}")
except Exception as e:
    log.warning(f"  Stage4 val preds not found: {e}"); s4_preds_val = None

# Stage5 routed baseline predictions (for comparison)
if s5_preds_val and s4_preds_val:
    s5_routed_val = [s4_preds_val[i] if val_subs[i] in S4_SUBSETS else s5_preds_val[i]
                     for i in range(len(val_subs))]
    r1b, rlb, wrb, _, _ = score_rows(val_refs, s5_routed_val)
    log.info(f"  Stage5 routed Val WR verify: {wrb:.5f} (expect {BASELINE_WR_S5:.5f})")
else:
    s5_routed_val = None

# ── Sparse retrievers ─────────────────────────────────────────────────────────
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
_wmap = {"char_wb": 0.40, "char": 0.30, "word": 0.20, "byte": 0.10}
tw = sum(_wmap[n] for n in rnames)
rrf_w_sparse = {n: _wmap[n]/tw for n in rnames}

def retrieve_sparse(q, s, name, topk=TOP_K_SPARSE, cv=None, csi=None, csr=None, cdf=None):
    cv_ = cv or vecs; csi_ = csi or sub_i; csr_ = csr or sub_r
    v = cv_[name]; X_q = v.transform([q])
    if s in csi_[name]: X_s, rids = csi_[name][s], csr_[name][s]
    else: X_s = v.transform((cdf or train)["input_norm"]); rids = (cdf or train).index.tolist()
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

# ── Dense retrieval ───────────────────────────────────────────────────────────
import torch
import faiss
from sentence_transformers import SentenceTransformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dense model configs: (key, model_name, use_e5_prefix, rrf_weight)
DENSE_CONFIGS = [
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False, 1/3),
    ("e5s",    "intfloat/multilingual-e5-small",                               True,  1/3),
    ("e5b",    "intfloat/multilingual-e5-base",                                True,  1/3),
]
TOP_K_DENSE = 20

dense_encoders  = {}   # key -> (model, use_prefix)
dense_sub_idx   = {}   # key -> {subset -> faiss.Index}
dense_sub_rids  = {}   # key -> {subset -> [rids]}
val_dense_ret   = {}   # key -> [(doc_ids, scores), ...]
dense_model_keys = []  # [(key, rrf_w)]
TRAIN_EMBS = {}        # key -> np.ndarray

for model_key, model_name, use_prefix, rrf_w_d in DENSE_CONFIGS:
    try:
        # --- Load or compute train embeddings ---
        emb_cache = CACHE_DIR / f"train_embs_{model_key}.npy"
        if emb_cache.exists():
            log.info(f"[6A] Loading cached train embeddings for {model_key}...")
            tr_embs = np.load(str(emb_cache))
            dm = SentenceTransformer(model_name, cache_folder="/tmp/hf_cache", device=DEVICE)
            log.info(f"  Model loaded, embs shape={tr_embs.shape}")
        else:
            log.info(f"[6A] Loading model {model_name}...")
            t0 = time.time()
            dm = SentenceTransformer(model_name, cache_folder="/tmp/hf_cache", device=DEVICE)
            log.info(f"  Loaded in {time.time()-t0:.1f}s")
            log.info(f"  Encoding Train ({len(train)})...")
            t0 = time.time()
            pfx = "passage: " if use_prefix else ""
            tr_texts = [pfx + t for t in train["input_norm"].tolist()]
            tr_embs = dm.encode(tr_texts, batch_size=256, show_progress_bar=False,
                                normalize_embeddings=True, device=DEVICE).astype(np.float32)
            np.save(str(emb_cache), tr_embs)
            log.info(f"  Train encoded in {time.time()-t0:.1f}s dim={tr_embs.shape[1]}, cached")

        TRAIN_EMBS[model_key] = tr_embs
        dim = tr_embs.shape[1]

        # Per-subset FAISS
        dense_sub_idx[model_key]  = {}
        dense_sub_rids[model_key] = {}
        for s in SUBSETS:
            mask = (train["subset"] == s).values
            if not mask.any(): continue
            idxs = train[train["subset"]==s].index.tolist()
            fi = faiss.IndexFlatIP(dim); fi.add(tr_embs[mask])
            dense_sub_idx[model_key][s]  = fi
            dense_sub_rids[model_key][s] = idxs

        dense_encoders[model_key] = (dm, use_prefix)
        dense_model_keys.append((model_key, rrf_w_d))

        # Encode Val and retrieve
        val_emb_cache = CACHE_DIR / f"val_embs_{model_key}.npy"
        if val_emb_cache.exists():
            val_embs = np.load(str(val_emb_cache))
            log.info(f"  Val embeddings loaded from cache")
        else:
            vpfx = "query: " if use_prefix else ""
            val_texts = [vpfx + t for t in val["input_norm"].tolist()]
            t0 = time.time()
            val_embs = dm.encode(val_texts, batch_size=256, show_progress_bar=False,
                                 normalize_embeddings=True, device=DEVICE).astype(np.float32)
            np.save(str(val_emb_cache), val_embs)
            log.info(f"  Val encoded in {time.time()-t0:.1f}s, cached")

        global_fi = faiss.IndexFlatIP(dim); global_fi.add(tr_embs)
        global_rids = train.index.tolist()

        val_dense_ret[model_key] = []
        for i, (q_emb, s) in enumerate(zip(val_embs, val_subs)):
            qv = q_emb.reshape(1, -1)
            if s in dense_sub_idx[model_key]:
                fi2  = dense_sub_idx[model_key][s]
                rids2 = dense_sub_rids[model_key][s]
            else:
                fi2 = global_fi; rids2 = global_rids
            tk = min(TOP_K_DENSE, fi2.ntotal)
            D, I = fi2.search(qv, tk)
            doc_ids = [rids2[j] for j in I[0] if j >= 0]
            scores  = [float(D[0][k]) for k in range(len(I[0])) if I[0][k] >= 0]
            val_dense_ret[model_key].append((doc_ids, scores))

        log.info(f"  Val dense retrieval done for {model_key}")

    except Exception as e:
        log.warning(f"Dense model {model_key} failed: {e}\n{traceback.format_exc()[:400]}")

log.info(f"Dense models active: {[k for k,_ in dense_model_keys]}")

# ── Candidate pool builder ─────────────────────────────────────────────────────
def get_pool(i, sparse_ret, dense_ret=None, top_ks=20, top_kd=20):
    seen = {}
    for name in rnames:
        rows, scs = sparse_ret[name][i]
        for rank, (rid, sc) in enumerate(zip(rows[:top_ks], scs[:top_ks])):
            if rid not in seen: seen[rid] = {}
            seen[rid][f"sp_{name}_r"] = rank
            seen[rid][f"sp_{name}_s"] = sc
    if dense_ret:
        for mk, _ in dense_model_keys:
            if mk not in dense_ret: continue
            rows, scs = dense_ret[mk][i]
            for rank, (rid, sc) in enumerate(zip(rows[:top_kd], scs[:top_kd])):
                if rid not in seen: seen[rid] = {}
                seen[rid][f"dn_{mk}_r"] = rank
                seen[rid][f"dn_{mk}_s"] = sc
    return seen

# ── CE model ──────────────────────────────────────────────────────────────────
log.info("Loading cross-encoder...")
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
try:
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(CE_MODEL, device=DEVICE, cache_folder="/tmp/hf_cache")
    CE_OK = True
    log.info(f"  CE loaded on {next(ce_model.model.parameters()).device}")
except Exception as e:
    log.warning(f"CE load failed: {e}"); CE_OK = False

# ── CE scoring (with disk cache) ──────────────────────────────────────────────
val_ce = {}
ce_cache_path = CACHE_DIR / "val_ce_scores_stage6.pkl"

if ce_cache_path.exists():
    log.info("Loading CE scores from cache...")
    with open(ce_cache_path, "rb") as f: val_ce = pickle.load(f)
    log.info(f"  Loaded CE scores for {len(val_ce)} queries")
    # Verify cache covers all current pool candidates
    pool_sample = get_pool(0, val_sparse_ret, val_dense_ret)
    cached_for_q0 = set(val_ce.get(0, {}).keys())
    pool_for_q0   = set(pool_sample.keys())
    new_uncached  = pool_for_q0 - cached_for_q0
    if len(new_uncached) > len(pool_for_q0) * 0.3:
        log.info(f"  Cache stale ({len(new_uncached)} uncached for q0). Recomputing...")
        val_ce = {}; ce_cache_path.unlink()
    else:
        log.info(f"  Cache valid for q0: {len(cached_for_q0)} cached, {len(new_uncached)} new")

if not val_ce and CE_OK:
    log.info("CE scoring val pool (all sparse+dense candidates)...")
    all_pairs = []
    for i, (q, s) in enumerate(zip(val["input"], val_subs)):
        pool = get_pool(i, val_sparse_ret, val_dense_ret)
        for rid in pool:
            all_pairs.append((i, rid, str(q), str(train.loc[rid, "output"])))
    log.info(f"  Total val pairs: {len(all_pairs)}")
    t0 = time.time(); BATCH = 8
    pair_inputs = [(q[:512], d[:512]) for _, _, q, d in all_pairs]
    all_ce = []
    for start in range(0, len(pair_inputs), BATCH):
        batch = pair_inputs[start:start+BATCH]
        try:   sc = ce_model.predict(batch, batch_size=BATCH, show_progress_bar=False)
        except RuntimeError: BATCH = 4; sc = ce_model.predict(batch, batch_size=BATCH, show_progress_bar=False)
        all_ce.extend(sc.tolist() if hasattr(sc, "tolist") else list(sc))
        if (start // BATCH) % 1000 == 0:
            elapsed = time.time()-t0
            eta = elapsed/(start+BATCH) * (len(pair_inputs)-start-BATCH) if start>0 else 0
            log.info(f"  CE: {start}/{len(pair_inputs)} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    elapsed = time.time()-t0
    log.info(f"  CE done: {len(all_ce)} pairs in {elapsed:.1f}s ({len(all_ce)/elapsed:.0f} p/s)")
    for (vi, rid, _, _), sc in zip(all_pairs, all_ce):
        if vi not in val_ce: val_ce[vi] = {}
        val_ce[vi][rid] = float(sc)
    # Save to cache
    with open(ce_cache_path, "wb") as f: pickle.dump(val_ce, f, protocol=4)
    log.info(f"  CE scores cached to {ce_cache_path}")

# ── Feature builder ────────────────────────────────────────────────────────────
def build_features_v6(sparse_ret, dense_ret, query_inputs, query_norms, query_subs,
                      ce_scores, cand_df, top_ks=10, top_kd=20):
    rows = []
    for i, (q_input, q_norm, s) in enumerate(zip(query_inputs, query_norms, query_subs)):
        pool = get_pool(i, sparse_ret, dense_ret, top_ks, top_kd)
        if not pool: continue
        q_words = q_norm.split(); q_len = len(q_words)

        # RRF sparse
        rrf_sp = {}
        for name in rnames:
            w = rrf_w_sparse[name]; r_list, _ = sparse_ret[name][i]
            for rank, rid in enumerate(r_list[:top_ks]):
                rrf_sp[rid] = rrf_sp.get(rid, 0.0) + w/(60+rank)

        # RRF dense
        rrf_dn = {}
        if dense_ret:
            for mk, rw in dense_model_keys:
                if mk not in dense_ret: continue
                r_list, _ = dense_ret[mk][i]
                for rank, rid in enumerate(r_list[:top_kd]):
                    rrf_dn[rid] = rrf_dn.get(rid, 0.0) + rw/(60+rank)

        # Median answer length in pool
        ans_lens = [len(str(cand_df.loc[rid, "output"]).split()) for rid in pool]
        med_ans_len = float(np.median(ans_lens)) if ans_lens else 1.0

        # Average dense score for centrality
        avg_e5b = float(np.mean([pool[rid].get("dn_e5b_s", 0.0) for rid in pool]))

        q_bigrams = set(zip(q_words, q_words[1:]))

        for rid, info in pool.items():
            feat = {"query_idx": i, "doc_id": rid}
            # Sparse
            for name in rnames:
                feat[f"rank_{name}"]  = info.get(f"sp_{name}_r", top_ks)
                feat[f"score_{name}"] = info.get(f"sp_{name}_s", 0.0)
            # Dense
            for mk, _ in dense_model_keys:
                feat[f"rank_dense_{mk}"]  = info.get(f"dn_{mk}_r", top_kd)
                feat[f"score_dense_{mk}"] = info.get(f"dn_{mk}_s", 0.0)
            # RRF
            feat["rrf_sparse"] = rrf_sp.get(rid, 0.0)
            feat["rrf_dense"]  = rrf_dn.get(rid, 0.0)
            feat["rrf_total"]  = feat["rrf_sparse"] + feat["rrf_dense"]
            # CE
            feat["ce_score"] = ce_scores.get(i, {}).get(rid, -999.0)
            # Subset
            feat["same_subset"] = int(cand_df.loc[rid, "subset"] == s)
            feat["subset_id"]   = SUBSETS.index(s) if s in SUBSETS else -1
            # Length features
            ans = str(cand_df.loc[rid, "output"])
            ans_len = len(ans.split())
            feat["query_len"]       = q_len
            feat["answer_len"]      = ans_len
            feat["answer_len_ratio"] = ans_len / (med_ans_len + 1)
            # Retriever agreement
            feat["n_sparse_found"] = sum(1 for nm in rnames   if f"sp_{nm}_r" in info)
            feat["n_dense_found"]  = sum(1 for mk, _ in dense_model_keys if f"dn_{mk}_r" in info)
            feat["n_total_found"]  = feat["n_sparse_found"] + feat["n_dense_found"]
            # Bigram overlap
            ans_words = ans.lower().split()[:100]
            a_bigrams = set(zip(ans_words, ans_words[1:]))
            feat["bigram_overlap"] = len(q_bigrams & a_bigrams) / (len(q_bigrams) + 1)
            # Rank stability (std of ranks across sparse)
            sp_ranks = [info.get(f"sp_{nm}_r", top_ks) for nm in rnames]
            feat["min_sparse_rank"]  = min(sp_ranks)
            feat["rank_stability"]   = float(np.std(sp_ranks))
            # Centrality: e5b score relative to pool mean
            feat["e5b_centrality"] = info.get("dn_e5b_s", 0.0) - avg_e5b
            rows.append(feat)
    return pd.DataFrame(rows).fillna(0)

# ── Build Val features ────────────────────────────────────────────────────────
log.info("Building Stage6 features for Val...")
t0 = time.time()
feat6 = build_features_v6(
    val_sparse_ret, val_dense_ret,
    val["input"].tolist(), val["input_norm"].tolist(), val_subs,
    val_ce, train, top_ks=10, top_kd=20
)
log.info(f"  {len(feat6)} rows, {len(feat6.columns)} cols in {time.time()-t0:.1f}s")

# ROUGE targets
log.info("Computing ROUGE targets...")
t0 = time.time()
doc_ids_v  = feat6["doc_id"].astype(int).values
q_idxs_v   = feat6["query_idx"].astype(int).values
train_outs  = train["output"].values
r1s_t = [rouge1_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
rls_t = [rougel_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
log.info(f"  done in {time.time()-t0:.1f}s")

feat_cols = [c for c in feat6.columns if c not in ("query_idx", "doc_id")]
X_all     = feat6[feat_cols].values
n_q       = len(val)
split     = int(n_q * 0.8)
tr_m      = q_idxs_v < split
vl_m      = ~tr_m

# ── LGB variants ──────────────────────────────────────────────────────────────
import lightgbm as lgb

params_lgb = dict(objective="regression", metric="rmse", num_leaves=63,
                  learning_rate=0.05, verbosity=-1, random_state=42,
                  num_threads=4, min_data_in_leaf=5)

TARGET_VARIANTS = [
    ("rl_heavy", lambda r1, rl: 0.35*r1 + 0.65*rl),
    ("balanced", lambda r1, rl: 0.50*r1 + 0.50*rl),
    ("comp",     lambda r1, rl: 0.37*r1 + 0.37*rl),
]

experiments6   = {}
trained_models = {}

def fallback_v3(i):
    s = val_subs[i]
    rn = {"Aka_Gha": "char", "Amh_Eth": "byte"}.get(s, "char_wb")
    rows, _ = val_sparse_ret[rn][i]
    return train.loc[rows[0], "output"] if rows else ""

for tgt_name, tgt_fn in TARGET_VARIANTS:
    try:
        y = np.array([tgt_fn(r1, rl) for r1, rl in zip(r1s_t, rls_t)])
        dtrain = lgb.Dataset(X_all[tr_m], label=y[tr_m])
        dvalid = lgb.Dataset(X_all[vl_m], label=y[vl_m])
        t0 = time.time()
        model = lgb.train(params_lgb, dtrain, num_boost_round=400, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(-1)])
        log.info(f"  LGB_{tgt_name}: {time.time()-t0:.1f}s iter={model.best_iteration}")

        lgb_scores = model.predict(X_all)
        preds = []
        for i in range(n_q):
            mask = q_idxs_v == i
            if not mask.any(): preds.append(fallback_v3(i)); continue
            best = int(doc_ids_v[mask][np.argmax(lgb_scores[mask])])
            preds.append(train.loc[best, "output"])

        r1, rl, wr_v, r1s, rls = score_rows(val_refs, preds)
        exp_name = f"6A_LGB_{tgt_name}"
        experiments6[exp_name] = {"preds": preds, "r1": r1, "rl": rl, "wr": wr_v,
                                   "r1s": r1s, "rls": rls}
        trained_models[exp_name] = (model, feat_cols)
        log.info(f"  {exp_name}: R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f} "
                 f"(Δ vs S5={wr_v-BASELINE_WR_S5:+.5f})")
    except Exception as e:
        log.warning(f"LGB_{tgt_name} failed: {e}\n{traceback.format_exc()[:300]}")

# ── Subset-level routing ──────────────────────────────────────────────────────
log.info("\nSubset-level routing analysis...")

# Per-subset WR for each experiment
def subset_wr_map(exp_name):
    sc = experiments6.get(exp_name, {})
    if "r1s" not in sc: return {}
    result = {}
    for s in SUBSETS:
        mask = [i for i, ss in enumerate(val_subs) if ss == s]
        if not mask: continue
        r1s_ = [sc["r1s"][i] for i in mask]
        rls_ = [sc["rls"][i] for i in mask]
        result[s] = 0.37*np.mean(r1s_) + 0.37*np.mean(rls_)
    return result

# Stage5 routed per-subset WR
def s5_subset_wr(s):
    if s5_routed_val is None: return 0.0
    mask = [i for i, ss in enumerate(val_subs) if ss == s]
    if not mask: return 0.0
    r1s_ = [rouge1_f1(val_refs[i], s5_routed_val[i]) for i in mask]
    rls_ = [rougel_f1(val_refs[i], s5_routed_val[i]) for i in mask]
    return 0.37*np.mean(r1s_) + 0.37*np.mean(rls_)

# Stage4 per-subset WR
def s4_subset_wr(s):
    if s4_preds_val is None: return 0.0
    mask = [i for i, ss in enumerate(val_subs) if ss == s]
    if not mask: return 0.0
    r1s_ = [rouge1_f1(val_refs[i], s4_preds_val[i]) for i in mask]
    rls_ = [rougel_f1(val_refs[i], s4_preds_val[i]) for i in mask]
    return 0.37*np.mean(r1s_) + 0.37*np.mean(rls_)

# Build routing table: for each subset, pick best source
routing_table = {}   # subset -> (source, exp_name, wr)
for s in SUBSETS:
    s5_wr = s5_subset_wr(s)
    s4_wr = s4_subset_wr(s)
    best_src = "stage5_routed"; best_wr_s = s5_wr; best_exp_s = "stage5_routed"
    if s4_wr > s5_wr: best_src = "stage4"; best_wr_s = s4_wr; best_exp_s = "stage4_CE_LGB"
    for exp in experiments6:
        wm = subset_wr_map(exp)
        if wm.get(s, 0) > best_wr_s + 0.003:  # min gain threshold
            best_src = "stage6"; best_wr_s = wm[s]; best_exp_s = exp
    routing_table[s] = {"source": best_src, "exp": best_exp_s, "wr": best_wr_s,
                         "s5_wr": s5_wr, "delta": best_wr_s - s5_wr}
    log.info(f"  {s}: best={best_src}/{best_exp_s} WR={best_wr_s:.4f} "
             f"(S5={s5_wr:.4f} Δ={best_wr_s-s5_wr:+.4f})")

# Build routed predictions
routed_preds = []
for i, s in enumerate(val_subs):
    rt = routing_table[s]
    if rt["source"] == "stage4" and s4_preds_val:
        routed_preds.append(s4_preds_val[i])
    elif rt["source"] == "stage5_routed" and s5_routed_val:
        routed_preds.append(s5_routed_val[i])
    elif rt["source"] == "stage6" and rt["exp"] in experiments6:
        routed_preds.append(experiments6[rt["exp"]]["preds"][i])
    elif s5_routed_val:
        routed_preds.append(s5_routed_val[i])
    else:
        routed_preds.append(fallback_v3(i))

r1, rl, wr_routed, r1s_rt, rls_rt = score_rows(val_refs, routed_preds)
experiments6["6A_routed"] = {"preds": routed_preds, "r1": r1, "rl": rl, "wr": wr_routed,
                               "r1s": r1s_rt, "rls": rls_rt}
log.info(f"\n  6A_routed: R1={r1:.4f} RL={rl:.4f} WR={wr_routed:.5f} "
         f"(Δ vs S5={wr_routed-BASELINE_WR_S5:+.5f})")

# ── Final comparison ───────────────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("STAGE 6 — FINAL COMPARISON")
log.info(f"  Stage5 routed baseline: WR={BASELINE_WR_S5:.5f}")
best6_exp = None; best6_wr = BASELINE_WR_S5
for exp in sorted(experiments6, key=lambda e: experiments6[e]["wr"], reverse=True):
    sc = experiments6[exp]
    flag = "★" if sc["wr"] > BASELINE_WR_S5 else " "
    log.info(f"  {flag} {exp:35s}: R1={sc['r1']:.4f} RL={sc['rl']:.4f} "
             f"WR={sc['wr']:.5f} (Δ={sc['wr']-BASELINE_WR_S5:+.5f})")
    if sc["wr"] > best6_wr: best6_wr = sc["wr"]; best6_exp = exp

log.info(f"\nBest Stage6: {best6_exp} WR={best6_wr:.5f}")

# ── Save routing CSV ───────────────────────────────────────────────────────────
routing_rows = []
for s, rt in routing_table.items():
    routing_rows.append({"subset": s, "source": rt["source"], "experiment": rt["exp"],
                          "wr_best": rt["wr"], "wr_stage5": rt["s5_wr"], "delta": rt["delta"]})
pd.DataFrame(routing_rows).to_csv(OUT/"stage6_routing_by_subset.csv", index=False)

# ── Per-subset scores ──────────────────────────────────────────────────────────
sub_rows = []
for s in SUBSETS:
    mask = [i for i, ss in enumerate(val_subs) if ss == s]
    if not mask: continue
    srefs = [val_refs[i] for i in mask]
    for exp, sc in experiments6.items():
        if "r1s" not in sc: continue
        sp = [sc["preds"][i] for i in mask]
        r1, rl, wr_v, _, _ = score_rows(srefs, sp)
        sub_rows.append({"experiment": exp, "subset": s, "n": len(mask),
                          "rouge1": r1, "rougel": rl, "weighted_rouge": wr_v})
df_sub = pd.DataFrame(sub_rows)
df_sub.to_csv(OUT/"stage6_validation_scores_by_subset.csv", index=False)
df_sub.to_csv(OUT/"stage6_e5base_validation_scores_by_subset.csv", index=False)
log.info("Saved per-subset scores")

# ── Build Test submission if improved ──────────────────────────────────────────
if best6_exp is None:
    log.info("Stage6 does NOT improve Stage5. Keeping Stage5 submission.")
else:
    log.info(f"\nBuilding Test predictions for {best6_exp}...")

    # Refit sparse on Train+Val
    base_df = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
    for df in [base_df, test]: df["input_norm"] = df["input"].apply(norm)
    base_df["output_norm"] = base_df["output"].apply(norm)
    SUBSETS_BASE = sorted(base_df["subset"].unique())

    log.info("  Refitting sparse on Train+Val...")
    rvecs = {}
    for name, cfg in CFGS.items():
        t0 = time.time(); v2 = TfidfVectorizer(**cfg); v2.fit(base_df["input_norm"])
        rvecs[name] = v2; log.info(f"    {name}: {time.time()-t0:.1f}s")

    rsub_i, rsub_r = {}, {}
    for name, v2 in rvecs.items():
        rsub_i[name] = {s: v2.transform(base_df[base_df["subset"]==s]["input_norm"])
                        for s in SUBSETS_BASE if len(base_df[base_df["subset"]==s]) > 0}
        rsub_r[name] = {s: base_df[base_df["subset"]==s].index.tolist()
                        for s in SUBSETS_BASE}

    def retrieve_test_sp(q, s, name, topk=TOP_K_SPARSE):
        v2 = rvecs[name]; X_q = v2.transform([q])
        if s in rsub_i[name]: X_s, rids = rsub_i[name][s], rsub_r[name][s]
        else: X_s = v2.transform(base_df["input_norm"]); rids = base_df.index.tolist()
        sim = (X_q @ X_s.T).toarray()[0]; tk = min(topk, len(sim))
        top = np.argpartition(sim, -tk)[-tk:] if len(sim) > tk else np.argsort(sim)[::-1]
        top = top[np.argsort(sim[top])[::-1]]
        return [rids[j] for j in top], sim[top].tolist()

    log.info("  Sparse retrieval for Test...")
    t0 = time.time()
    test_sparse_ret = {name: [] for name in rvecs}
    for q, s in zip(test["input_norm"], test_subs):
        for name in rvecs: test_sparse_ret[name].append(retrieve_test_sp(q, s, name))
    log.info(f"    done in {time.time()-t0:.1f}s")

    # Dense retrieval for test (refit on base_df)
    test_dense_ret = {}
    rdense_sub_idx = {}; rdense_sub_rids = {}; base_embs_dict = {}

    for model_key, (dm, use_prefix) in dense_encoders.items():
        try:
            base_emb_cache = CACHE_DIR / f"base_embs_{model_key}.npy"
            if base_emb_cache.exists():
                base_embs = np.load(str(base_emb_cache))
                log.info(f"  base_df embs loaded from cache for {model_key}")
            else:
                pfx = "passage: " if use_prefix else ""
                base_texts = [pfx + t for t in base_df["input_norm"].tolist()]
                t0 = time.time()
                base_embs = dm.encode(base_texts, batch_size=256, show_progress_bar=False,
                                      normalize_embeddings=True, device=DEVICE).astype(np.float32)
                np.save(str(base_emb_cache), base_embs)
                log.info(f"  base_df encoded {model_key} in {time.time()-t0:.1f}s, cached")

            base_embs_dict[model_key] = base_embs
            dim2 = base_embs.shape[1]
            rdense_sub_idx[model_key] = {}; rdense_sub_rids[model_key] = {}
            for s in SUBSETS_BASE:
                bmask = (base_df["subset"] == s).values
                if not bmask.any(): continue
                bidxs = base_df[base_df["subset"]==s].index.tolist()
                fi = faiss.IndexFlatIP(dim2); fi.add(base_embs[bmask])
                rdense_sub_idx[model_key][s]  = fi
                rdense_sub_rids[model_key][s] = bidxs

            # Test embeddings
            test_emb_cache = CACHE_DIR / f"test_embs_{model_key}.npy"
            if test_emb_cache.exists():
                te_embs = np.load(str(test_emb_cache))
                log.info(f"  test embs loaded from cache for {model_key}")
            else:
                vpfx = "query: " if use_prefix else ""
                test_texts = [vpfx + t for t in test["input_norm"].tolist()]
                t0 = time.time()
                te_embs = dm.encode(test_texts, batch_size=256, show_progress_bar=False,
                                    normalize_embeddings=True, device=DEVICE).astype(np.float32)
                np.save(str(test_emb_cache), te_embs)
                log.info(f"  test encoded {model_key} in {time.time()-t0:.1f}s, cached")

            gfi = faiss.IndexFlatIP(dim2); gfi.add(base_embs); grids = base_df.index.tolist()
            test_dense_ret[model_key] = []
            for i, (q_emb, s) in enumerate(zip(te_embs, test_subs)):
                qv = q_emb.reshape(1, -1)
                if s in rdense_sub_idx[model_key]:
                    fi4 = rdense_sub_idx[model_key][s]; rids4 = rdense_sub_rids[model_key][s]
                else: fi4 = gfi; rids4 = grids
                tk = min(TOP_K_DENSE, fi4.ntotal); D, I = fi4.search(qv, tk)
                doc_ids4 = [rids4[j] for j in I[0] if j >= 0]
                scores4  = [float(D[0][k]) for k in range(len(I[0])) if I[0][k] >= 0]
                test_dense_ret[model_key].append((doc_ids4, scores4))
            log.info(f"  Test dense done for {model_key}")
        except Exception as e:
            log.warning(f"Dense test {model_key} failed: {e}")

    # CE scoring for test (with cache)
    test_ce = {}
    test_ce_cache = CACHE_DIR / "test_ce_scores_stage6.pkl"
    if test_ce_cache.exists():
        log.info("  Loading test CE scores from cache...")
        with open(test_ce_cache, "rb") as f: test_ce = pickle.load(f)
        log.info(f"  Loaded for {len(test_ce)} test queries")
        # Quick validity check
        pool0 = get_pool(0, test_sparse_ret, test_dense_ret)
        cached0 = set(test_ce.get(0, {}).keys())
        if len(set(pool0.keys()) - cached0) > len(pool0)*0.3:
            log.info("  Test CE cache stale, recomputing...")
            test_ce = {}; test_ce_cache.unlink()

    if not test_ce and CE_OK:
        log.info("  CE scoring test pool...")
        test_pairs = []
        for i, (q, s) in enumerate(zip(test["input"], test_subs)):
            pool = get_pool(i, test_sparse_ret, test_dense_ret)
            for rid in pool:
                test_pairs.append((i, rid, str(q), str(base_df.loc[rid, "output"])))
        log.info(f"    Test pairs: {len(test_pairs)}")
        t0 = time.time(); BATCHT = 8
        tinputs = [(q[:512], d[:512]) for _, _, q, d in test_pairs]
        tscores = []
        for start in range(0, len(tinputs), BATCHT):
            b = tinputs[start:start+BATCHT]
            try:   sc = ce_model.predict(b, batch_size=BATCHT, show_progress_bar=False)
            except RuntimeError: BATCHT = 4; sc = ce_model.predict(b, batch_size=BATCHT, show_progress_bar=False)
            tscores.extend(sc.tolist() if hasattr(sc, "tolist") else list(sc))
            if (start//BATCHT) % 500 == 0:
                elapsed = time.time()-t0
                eta = elapsed/(start+BATCHT) * (len(tinputs)-start-BATCHT) if start > 0 else 0
                log.info(f"    CE test: {start}/{len(tinputs)} ({elapsed:.0f}s ETA {eta:.0f}s)")
        for (ti, rid, _, _), sc in zip(test_pairs, tscores):
            if ti not in test_ce: test_ce[ti] = {}
            test_ce[ti][rid] = float(sc)
        with open(test_ce_cache, "wb") as f: pickle.dump(test_ce, f, protocol=4)
        log.info(f"    Test CE cached. Done in {time.time()-t0:.1f}s")

    # Build test features
    log.info("  Building test features...")
    t0 = time.time()
    feat_test = build_features_v6(
        test_sparse_ret, test_dense_ret,
        test["input"].tolist(), test["input_norm"].tolist(), test_subs,
        test_ce, base_df, top_ks=10, top_kd=20
    )
    log.info(f"    {len(feat_test)} rows in {time.time()-t0:.1f}s")

    for c in feat_cols:
        if c not in feat_test.columns: feat_test[c] = 0
    tq_arr = feat_test["query_idx"].values.copy()
    td_arr = feat_test["doc_id"].astype(int).values.copy()
    X_test = feat_test[feat_cols].values

    # Determine test source per test query based on routing_table
    def get_test_s5_pred(i):
        """Stage5 test prediction for test row i"""
        # Load from Stage5 test submission (already built)
        return None  # will be filled from sub5 CSV

    sub5_test = pd.read_csv(OUT/"submission_stage5_best.csv")
    sub4_test  = pd.read_csv(OUT/"submission_stage4_crossencoder.csv")
    id2sub_test = dict(zip(test["ID"], test_subs))

    def fallback_test(i):
        s = test_subs[i]
        rn = {"Aka_Gha": "char", "Amh_Eth": "byte"}.get(s, "char_wb")
        rows, _ = test_sparse_ret[rn][i]
        return base_df.loc[rows[0], "output"] if rows else ""

    # Build test preds per the best model
    if "routed" in best6_exp:
        # Use routing table for test
        test_preds = []
        test_ids = test["ID"].tolist()
        for i in range(len(test)):
            s = test_subs[i]
            rt = routing_table.get(s, {})
            src = rt.get("source", "stage5_routed")
            if src == "stage4":
                row = sub4_test[sub4_test["ID"] == test_ids[i]]
                test_preds.append(str(row["TargetRLF1"].values[0]) if len(row) > 0 else fallback_test(i))
            elif src == "stage5_routed":
                row = sub5_test[sub5_test["ID"] == test_ids[i]]
                test_preds.append(str(row["TargetRLF1"].values[0]) if len(row) > 0 else fallback_test(i))
            else:  # stage6
                exp_key = rt.get("exp", best6_exp)
                if exp_key in trained_models:
                    mdl, fc = trained_models[exp_key]
                    mask = tq_arr == i
                    if mask.any():
                        sc6 = mdl.predict(feat_test[fc].values[mask])
                        best_doc = int(td_arr[mask][np.argmax(sc6)])
                        test_preds.append(base_df.loc[best_doc, "output"])
                    else: test_preds.append(fallback_test(i))
                else:
                    row = sub5_test[sub5_test["ID"] == test_ids[i]]
                    test_preds.append(str(row["TargetRLF1"].values[0]) if len(row) > 0 else fallback_test(i))
    else:
        # Use single best model
        best_model6, _ = trained_models[best6_exp]
        lgb_tscores = best_model6.predict(X_test)
        test_preds = []
        for i in range(len(test)):
            mask = tq_arr == i
            if not mask.any(): test_preds.append(fallback_test(i)); continue
            best_doc = int(td_arr[mask][np.argmax(lgb_tscores[mask])])
            test_preds.append(base_df.loc[best_doc, "output"])

    # Build submission
    sub6 = sample[["ID"]].copy()
    sub6["TargetRLF1"] = sub6["ID"].map(dict(zip(test["ID"], test_preds)))
    sub6["TargetR1F1"] = sub6["TargetRLF1"]
    sub6["TargetLLM"]  = sub6["TargetRLF1"]

    assert list(sub6.columns) == ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]
    assert sub6.shape == sample.shape
    assert (sub6["ID"].values == sample["ID"].values).all()
    assert sub6[["TargetRLF1", "TargetR1F1", "TargetLLM"]].notna().all().all()
    assert (sub6["TargetRLF1"] == sub6["TargetR1F1"]).all()
    assert (sub6["TargetRLF1"] == sub6["TargetLLM"]).all()
    log.info("Submission checks PASSED")

    sub6.to_csv(OUT/"submission_stage6_best.csv", index=False)
    sub6.to_csv(OUT/"submission.csv", index=False)
    log.info(f"submission.csv updated: {BASELINE_WR_S5:.5f} → {best6_wr:.5f}")

# ── Save Val debug + changed rows ─────────────────────────────────────────────
if best6_exp and best6_exp in experiments6:
    best_preds6 = experiments6[best6_exp]["preds"]
    vd = val.copy()
    vd["prediction"] = best_preds6; vd["experiment"] = best6_exp
    vd["rouge1"] = [rouge1_f1(r,p) for r,p in zip(val_refs, best_preds6)]
    vd["rougel"] = [rougel_f1(r,p) for r,p in zip(val_refs, best_preds6)]
    vd.to_csv(OUT/"stage6_val_predictions_debug.csv", index=False)
    vd.to_csv(OUT/"stage6_e5base_val_predictions_debug.csv", index=False)

    # Changed rows vs Stage5
    try:
        changed = []
        for i in range(n_q):
            p5 = str(s5_routed_val[i]) if s5_routed_val else ""
            p6 = str(best_preds6[i])
            if p5 != p6:
                changed.append({"val_idx": i, "subset": val_subs[i],
                                 "pred_stage5": p5[:100], "pred_stage6": p6[:100]})
        pd.DataFrame(changed).to_csv(OUT/"stage6_changed_rows_vs_stage5.csv", index=False)
        log.info(f"Changed rows vs Stage5: {len(changed)}/{n_q}")
    except Exception as e:
        log.warning(f"Changed rows: {e}")

# ── Global scores CSV ─────────────────────────────────────────────────────────
rows_g = [{"experiment": "Stage5_routed_baseline", "rouge1": 0.53457, "rougel": 0.48063,
            "weighted_rouge": BASELINE_WR_S5}]
for exp, sc in experiments6.items():
    rows_g.append({"experiment": exp, "rouge1": sc["r1"], "rougel": sc["rl"],
                    "weighted_rouge": sc["wr"]})
df_g = pd.DataFrame(rows_g)
df_g.to_csv(OUT/"stage6_validation_scores.csv", index=False)
df_g.to_csv(OUT/"stage6_e5base_validation_scores.csv", index=False)

# ── Update best_stage.json ────────────────────────────────────────────────────
try:
    with open(OUT/"best_stage.json") as f: bs = json.load(f)
    if best6_wr > bs.get("weighted_rouge_val", 0):
        bs["best_experiment"]    = best6_exp
        bs["weighted_rouge_val"] = best6_wr
        if best6_exp in experiments6:
            bs["rouge1_val"] = experiments6[best6_exp]["r1"]
            bs["rougel_val"] = experiments6[best6_exp]["rl"]
        bs.setdefault("stage_results", {})["stage6"] = {
            "experiment": best6_exp, "wr": best6_wr,
            "dense_models": [k for k, _ in dense_model_keys],
            "routing": {s: rt["source"] for s, rt in routing_table.items()}
        }
        with open(OUT/"best_stage.json", "w") as f: json.dump(bs, f, indent=2)
except Exception as e:
    log.warning(f"best_stage.json: {e}")

try:
    sc_df = pd.read_csv(OUT/"validation_scores_all_stages.csv")
    new = []
    for exp, sc in experiments6.items():
        if not (sc_df["experiment"] == exp).any():
            new.append({"experiment": exp, "rouge1": sc["r1"], "rougel": sc["rl"],
                         "weighted_rouge": sc["wr"]})
    if new: sc_df = pd.concat([sc_df, pd.DataFrame(new)], ignore_index=True)
    sc_df.to_csv(OUT/"validation_scores_all_stages.csv", index=False)
except Exception as e:
    log.warning(f"validation_scores: {e}")

# ── Routing report ────────────────────────────────────────────────────────────
routing_by_subset_str = "; ".join(f"{s}→{rt['source']}" for s, rt in routing_table.items())

md = f"""# Stage 6A Report — e5-base Dense Retrieval

## Summary
- Baseline (Stage5 routed): WR = {BASELINE_WR_S5:.5f}
- Dense models: {[k for k,_ in dense_model_keys]}
- Best Stage6 experiment: {best6_exp}
- Best Stage6 WR: {best6_wr:.5f} (Δ = {best6_wr - BASELINE_WR_S5:+.5f})

## All experiments

| Experiment | R1 | RL | WR | Δ vs Stage5 |
|---|---|---|---|---|
| Stage5_routed_baseline | 0.5346 | 0.4806 | {BASELINE_WR_S5:.5f} | — |
"""
for exp in sorted(experiments6, key=lambda e: experiments6[e]["wr"], reverse=True):
    sc = experiments6[exp]
    md += f"| {exp} | {sc['r1']:.4f} | {sc['rl']:.4f} | {sc['wr']:.5f} | {sc['wr']-BASELINE_WR_S5:+.5f} |\n"

md += "\n## Routing by subset\n\n| Subset | Source | WR | Delta vs S5 |\n|---|---|---|---|\n"
for s, rt in routing_table.items():
    md += f"| {s} | {rt['source']}/{rt['exp']} | {rt['wr']:.4f} | {rt['delta']:+.4f} |\n"

md += f"""
## Decision
- Stage6 improved: {'YES' if best6_exp else 'NO'}
- submission.csv: {'updated to ' + best6_exp if best6_exp else 'kept Stage5'}
- routing: {routing_by_subset_str}
"""
with open(OUT/"stage6_e5base_report.md", "w") as f: f.write(md)
with open(OUT/"stage6_report.md", "w") as f: f.write(md)
with open(OUT/"stage6_routing_report.md", "w") as f: f.write(md)

try:
    with open(OUT/"run_manifest.json") as f: mf = json.load(f)
    mf["stage6"] = {"best_exp": best6_exp, "best_wr": best6_wr,
                    "improved": best6_exp is not None,
                    "dense_models": [k for k, _ in dense_model_keys],
                    "routing": {s: rt["source"] for s, rt in routing_table.items()},
                    "timestamp": datetime.now().isoformat()}
    with open(OUT/"run_manifest.json", "w") as f: json.dump(mf, f, indent=2)
except Exception as e:
    log.warning(f"run_manifest: {e}")

# ── Final submission check ────────────────────────────────────────────────────
final = pd.read_csv(OUT/"submission.csv")
assert list(final.columns) == ["ID","TargetRLF1","TargetR1F1","TargetLLM"]
assert final.shape == sample.shape
assert (final["ID"].values == sample["ID"].values).all()
assert final[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
assert (final["TargetRLF1"] == final["TargetR1F1"]).all()
assert (final["TargetRLF1"] == final["TargetLLM"]).all()
log.info("Final submission.csv checks PASSED")

elapsed_total = time.time() - START
log.info("\n" + "="*60)
log.info("DONE_STAGE6")
log.info(f"Best previous public : 0.732376")
log.info(f"Best previous Val    : {BASELINE_WR_S5:.5f}")
log.info(f"Best Stage6 Val      : {best6_wr:.5f}")
log.info(f"Delta vs Stage5      : {best6_wr-BASELINE_WR_S5:+.5f}")
log.info(f"Best strategy        : {best6_exp or 'None (Stage5 kept)'}")
log.info(f"Routing by subset    : {routing_by_subset_str}")
log.info(f"Final submission     : {OUT}/submission.csv")
log.info(f"Candidate for public : {'YES' if best6_exp else 'NO'}")
log.info(f"Total elapsed        : {elapsed_total:.0f}s")
log.info("="*60)
