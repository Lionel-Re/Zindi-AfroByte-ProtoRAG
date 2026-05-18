"""
Stage 7 — AfroLM semantic reranking on Stage5 pool
Baseline: Stage5 routed, Val WR = 0.37563, Public = 0.732376
Stage6 REJECTED (public -0.0025 vs Stage5).
Uses Stage5 pool (sparse + minilm + e5s only, NO e5b).
Reuses Stage6 CE scores from cache.
AfroLM BERTScore proxy via cosine sim for LGB targets + features.
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

WORK      = Path("/home/onyxia/work")
OUT       = WORK / "outputs" / "latest"
CACHE_DIR = Path("/tmp/stage6_cache")   # reuse Stage6 cache
S7_CACHE  = Path("/tmp/stage7_cache")
S7_CACHE.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("STAGE 7 — AfroLM semantic reranking (Stage5 pool)")
log.info("=" * 60)
START = time.time()

BASELINE_WR_S5 = 0.37563    # Stage5 routed public=0.732376
THRESHOLD_UPDATE = 0.37563 + 0.007   # = 0.38263 min to replace submission
S4_SUBSETS = {"Swa_Ken", "Lug_Uga"}  # Stage5 routing: these used S4
AKA_AMH    = {"Aka_Gha", "Amh_Eth"}

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

# Load Stage5 reference predictions for baseline comparison
try:
    s5d  = pd.read_csv(OUT/"stage5_val_predictions_debug.csv")
    s4d  = pd.read_csv(OUT/"crossencoder_val_predictions_debug.csv")
    s5_preds = s5d["prediction"].tolist()
    s4_preds = s4d["prediction"].tolist()
    s5_routed = [s4_preds[i] if val_subs[i] in S4_SUBSETS else s5_preds[i]
                 for i in range(len(val_subs))]
    r1b, rlb, wrb, _, _ = score_rows(val_refs, s5_routed)
    log.info(f"Stage5 routed WR verify: {wrb:.5f}")
except Exception as e:
    log.warning(f"Stage5 ref preds not found: {e}"); s5_routed = None; s4_preds = None

# ── Sparse retrievers (Stage5 pool: sparse + minilm + e5s ONLY) ───────────────
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

def retrieve_sparse(q, s, name, topk=TOP_K_SPARSE, cv=None, si=None, sr=None, cdf=None):
    cv_ = cv or vecs; si_ = si or sub_i; sr_ = sr or sub_r
    v = cv_[name]; X_q = v.transform([q])
    if s in si_[name]: X_s, rids = si_[name][s], sr_[name][s]
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

# ── Dense retrieval (Stage5 pool: minilm + e5s ONLY) ─────────────────────────
import torch, faiss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stage5 dense models (no e5b)
S5_DENSE_CONFIGS = [
    ("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False, 0.5),
    ("e5s",    "intfloat/multilingual-e5-small",                               True,  0.5),
]
TOP_K_DENSE = 20

dense_model_keys = []   # [(key, rrf_w)]
dense_sub_idx    = {}   # key -> {subset -> faiss.Index}
dense_sub_rids   = {}   # key -> {subset -> [rids]}
val_dense_ret    = {}   # key -> [(doc_ids, scores), ...]

for model_key, model_name, use_prefix, rrf_w_d in S5_DENSE_CONFIGS:
    try:
        emb_cache = CACHE_DIR / f"train_embs_{model_key}.npy"
        val_emb_c = CACHE_DIR / f"val_embs_{model_key}.npy"
        if emb_cache.exists() and val_emb_c.exists():
            tr_embs  = np.load(str(emb_cache)).astype(np.float32)
            val_embs = np.load(str(val_emb_c)).astype(np.float32)
            log.info(f"  Loaded cached embeddings {model_key} dim={tr_embs.shape[1]}")
        else:
            from sentence_transformers import SentenceTransformer
            log.info(f"  Encoding {model_key} from scratch...")
            dm = SentenceTransformer(model_name, cache_folder="/tmp/hf_cache", device=DEVICE)
            pfx = "passage: " if use_prefix else ""
            tr_embs = dm.encode([pfx+t for t in train["input_norm"].tolist()],
                                 batch_size=256, show_progress_bar=False,
                                 normalize_embeddings=True, device=DEVICE).astype(np.float32)
            np.save(str(emb_cache), tr_embs)
            vpfx = "query: " if use_prefix else ""
            val_embs = dm.encode([vpfx+t for t in val["input_norm"].tolist()],
                                  batch_size=256, show_progress_bar=False,
                                  normalize_embeddings=True, device=DEVICE).astype(np.float32)
            np.save(str(val_emb_c), val_embs)

        dim = tr_embs.shape[1]
        dense_sub_idx[model_key] = {}; dense_sub_rids[model_key] = {}
        for s in SUBSETS:
            mask = (train["subset"] == s).values
            if not mask.any(): continue
            idxs = train[train["subset"]==s].index.tolist()
            fi = faiss.IndexFlatIP(dim); fi.add(tr_embs[mask])
            dense_sub_idx[model_key][s]  = fi
            dense_sub_rids[model_key][s] = idxs
        global_fi = faiss.IndexFlatIP(dim); global_fi.add(tr_embs)
        global_rids = train.index.tolist()

        val_dense_ret[model_key] = []
        for i, (q_emb, s) in enumerate(zip(val_embs, val_subs)):
            qv = q_emb.reshape(1, -1)
            if s in dense_sub_idx[model_key]:
                fi2 = dense_sub_idx[model_key][s]; rids2 = dense_sub_rids[model_key][s]
            else: fi2 = global_fi; rids2 = global_rids
            tk = min(TOP_K_DENSE, fi2.ntotal); D, I = fi2.search(qv, tk)
            val_dense_ret[model_key].append(
                ([rids2[j] for j in I[0] if j>=0],
                 [float(D[0][k]) for k in range(len(I[0])) if I[0][k]>=0]))
        log.info(f"  Val dense retrieval done for {model_key}")
        dense_model_keys.append((model_key, rrf_w_d))
    except Exception as e:
        log.warning(f"Dense {model_key} failed: {e}")

log.info(f"Dense models (Stage5 pool): {[k for k,_ in dense_model_keys]}")

# ── Pool builder ──────────────────────────────────────────────────────────────
def get_pool(i, sp_ret, dn_ret=None, top_ks=20, top_kd=20):
    seen = {}
    for name in rnames:
        rows, scs = sp_ret[name][i]
        for rank, (rid, sc) in enumerate(zip(rows[:top_ks], scs[:top_ks])):
            if rid not in seen: seen[rid] = {}
            seen[rid][f"sp_{name}_r"] = rank; seen[rid][f"sp_{name}_s"] = sc
    if dn_ret:
        for mk, _ in dense_model_keys:
            if mk not in dn_ret: continue
            rows, scs = dn_ret[mk][i]
            for rank, (rid, sc) in enumerate(zip(rows[:top_kd], scs[:top_kd])):
                if rid not in seen: seen[rid] = {}
                seen[rid][f"dn_{mk}_r"] = rank; seen[rid][f"dn_{mk}_s"] = sc
    return seen

# ── Load Stage6 CE scores (reuse — Stage5 pool ⊆ Stage6 pool) ─────────────────
val_ce = {}
ce_cache_path = CACHE_DIR / "val_ce_scores_stage6.pkl"
if ce_cache_path.exists():
    with open(ce_cache_path, "rb") as f: val_ce = pickle.load(f)
    log.info(f"Loaded Stage6 CE scores for {len(val_ce)} val queries (reusing)")
    # Verify coverage on Stage5 pool
    pool0 = get_pool(0, val_sparse_ret, val_dense_ret)
    missing = [rid for rid in pool0 if rid not in val_ce.get(0, {})]
    log.info(f"  Pool0: {len(pool0)} cands, {len(missing)} missing CE → {len(pool0)-len(missing)} covered")
else:
    log.warning("Stage6 CE cache not found — CE scores will be 0")

# ── AfroLM model ──────────────────────────────────────────────────────────────
AFROLM_MODEL_USED = None
afrolm_encode_fn  = None   # callable: list[str] -> np.ndarray [n, d] normalized

def make_encoder(model_name, use_prefix=False):
    """Return an encode function using mean-pooling of last hidden state."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/hf_cache")
    mdl = AutoModel.from_pretrained(model_name, cache_dir="/tmp/hf_cache").to(DEVICE)
    mdl.eval()
    def encode(texts, batch_size=128):
        if use_prefix:
            texts = ["query: " + t for t in texts]
        all_embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl(**enc)
            # Mean pool over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs = torch.nn.functional.normalize(embs, dim=-1)
            all_embs.append(embs.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)
    return encode

for model_name, use_prefix in [
    ("bonadossou/afrolm_active_learning", False),
    ("Davlan/afro-xlmr-large",            False),
    ("xlm-roberta-base",                  False),
]:
    try:
        log.info(f"Loading AfroLM model: {model_name}...")
        t0 = time.time()
        afrolm_encode_fn = make_encoder(model_name, use_prefix)
        # Quick test
        test_emb = afrolm_encode_fn(["health test sentence"], batch_size=1)
        assert test_emb.shape[1] > 100
        AFROLM_MODEL_USED = model_name
        log.info(f"  AfroLM loaded: {model_name}, dim={test_emb.shape[1]}, {time.time()-t0:.1f}s")
        break
    except Exception as e:
        log.warning(f"  {model_name} failed: {e}")

if afrolm_encode_fn is None:
    log.error("No AfroLM model could be loaded. AfroLM features will be 0.")

# ── Compute AfroLM embeddings ─────────────────────────────────────────────────
afrolm_q_embs    = None   # [n_val, d] — val query embeddings
afrolm_ref_embs  = None   # [n_val, d] — val reference embeddings
afrolm_cand_embs = {}     # {rid -> emb[d]} — candidate answer embeddings

if afrolm_encode_fn is not None:
    # Collect unique candidate rids in Stage5 pool
    all_rids = set()
    for i in range(len(val)):
        pool = get_pool(i, val_sparse_ret, val_dense_ret)
        all_rids.update(pool.keys())
    all_rids_list = sorted(all_rids)
    log.info(f"Unique candidate rids in Stage5 Val pool: {len(all_rids_list)}")

    # Cache check
    afrolm_cache = S7_CACHE / f"afrolm_embs_{AFROLM_MODEL_USED.replace('/','_')}.pkl"
    if afrolm_cache.exists():
        log.info("Loading AfroLM embeddings from cache...")
        with open(afrolm_cache, "rb") as f:
            cached = pickle.load(f)
        afrolm_q_embs   = cached["q"]
        afrolm_ref_embs = cached["ref"]
        afrolm_cand_embs = cached["cand"]
        log.info(f"  Loaded: q={afrolm_q_embs.shape}, ref={afrolm_ref_embs.shape}, cand={len(afrolm_cand_embs)}")
    else:
        log.info("Encoding Val questions with AfroLM...")
        t0 = time.time()
        afrolm_q_embs = afrolm_encode_fn(val["input"].tolist(), batch_size=128)
        log.info(f"  Val questions: {afrolm_q_embs.shape} in {time.time()-t0:.1f}s")

        log.info("Encoding Val references with AfroLM...")
        t0 = time.time()
        afrolm_ref_embs = afrolm_encode_fn(val_refs, batch_size=128)
        log.info(f"  Val references: {afrolm_ref_embs.shape} in {time.time()-t0:.1f}s")

        log.info(f"Encoding {len(all_rids_list)} unique candidate answers with AfroLM...")
        t0 = time.time()
        cand_texts = [str(train.loc[rid, "output"]) for rid in all_rids_list]
        cand_embs_arr = afrolm_encode_fn(cand_texts, batch_size=128)
        afrolm_cand_embs = {rid: cand_embs_arr[k] for k, rid in enumerate(all_rids_list)}
        log.info(f"  Candidates: {cand_embs_arr.shape} in {time.time()-t0:.1f}s")

        # Save cache
        with open(afrolm_cache, "wb") as f:
            pickle.dump({"q": afrolm_q_embs, "ref": afrolm_ref_embs,
                          "cand": afrolm_cand_embs}, f, protocol=4)
        log.info(f"  AfroLM embeddings cached to {afrolm_cache}")

# ── Feature builder ────────────────────────────────────────────────────────────
def build_features_v7(sp_ret, dn_ret, q_inputs, q_norms, q_subs,
                      ce_scores, cand_df,
                      a_q_embs, a_cand_embs, a_ref_embs=None,
                      top_ks=10, top_kd=20):
    rows = []
    for i, (q_in, q_norm, s) in enumerate(zip(q_inputs, q_norms, q_subs)):
        pool = get_pool(i, sp_ret, dn_ret, top_ks, top_kd)
        if not pool: continue
        q_words = q_norm.split(); q_len = len(q_words)

        # RRF sparse + dense
        rrf_sp = {}
        for name in rnames:
            w = rrf_w_sparse[name]; r_list, _ = sp_ret[name][i]
            for rank, rid in enumerate(r_list[:top_ks]):
                rrf_sp[rid] = rrf_sp.get(rid, 0.0) + w/(60+rank)
        rrf_dn = {}
        for mk, rw in dense_model_keys:
            if mk not in dn_ret: continue
            r_list, _ = dn_ret[mk][i]
            for rank, rid in enumerate(r_list[:top_kd]):
                rrf_dn[rid] = rrf_dn.get(rid, 0.0) + rw/(60+rank)

        # Answer length stats
        ans_lens = [len(str(cand_df.loc[rid, "output"]).split()) for rid in pool]
        med_ans_len = float(np.median(ans_lens)) if ans_lens else 1.0

        # AfroLM query embedding (for sim features)
        q_emb = a_q_embs[i] if a_q_embs is not None else None
        ref_emb = a_ref_embs[i] if a_ref_embs is not None else None

        # Compute afrolm_sim_q_cand for all candidates (for rank feature)
        if q_emb is not None:
            pool_sims = {}
            for rid in pool:
                c_emb = a_cand_embs.get(rid)
                if c_emb is not None:
                    pool_sims[rid] = float(np.dot(q_emb, c_emb))
                else:
                    pool_sims[rid] = 0.0
            sorted_rids_by_afrolm = sorted(pool_sims, key=lambda r: pool_sims[r], reverse=True)
            afrolm_rank_map = {rid: rank for rank, rid in enumerate(sorted_rids_by_afrolm)}
            mean_pool_sim = float(np.mean(list(pool_sims.values())))
        else:
            pool_sims = {rid: 0.0 for rid in pool}
            afrolm_rank_map = {rid: 0 for rid in pool}
            mean_pool_sim = 0.0

        q_bigrams = set(zip(q_words, q_words[1:]))

        for rid, info in pool.items():
            feat = {"query_idx": i, "doc_id": rid}
            # Sparse
            for name in rnames:
                feat[f"rank_{name}"]  = info.get(f"sp_{name}_r", top_ks)
                feat[f"score_{name}"] = info.get(f"sp_{name}_s", 0.0)
            # Dense (minilm + e5s only)
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
            # Length
            ans = str(cand_df.loc[rid, "output"])
            ans_len = len(ans.split())
            feat["query_len"]        = q_len
            feat["answer_len"]       = ans_len
            feat["answer_len_ratio"] = ans_len / (med_ans_len + 1)
            # Retriever agreement
            feat["n_sparse_found"] = sum(1 for nm in rnames if f"sp_{nm}_r" in info)
            feat["n_dense_found"]  = sum(1 for mk, _ in dense_model_keys if f"dn_{mk}_r" in info)
            feat["n_total_found"]  = feat["n_sparse_found"] + feat["n_dense_found"]
            # Bigram overlap
            ans_w = ans.lower().split()[:100]
            a_bigrams = set(zip(ans_w, ans_w[1:]))
            feat["bigram_overlap"] = len(q_bigrams & a_bigrams) / (len(q_bigrams) + 1)
            # Rank stability
            sp_ranks = [info.get(f"sp_{nm}_r", top_ks) for nm in rnames]
            feat["min_sparse_rank"] = min(sp_ranks)
            feat["rank_stability"]  = float(np.std(sp_ranks))
            # AfroLM features (reference-free → usable at test time)
            feat["afrolm_sim_q_cand"] = pool_sims.get(rid, 0.0)
            feat["afrolm_rank"]       = afrolm_rank_map.get(rid, len(pool))
            feat["afrolm_margin"]     = pool_sims.get(rid, 0.0) - mean_pool_sim
            # AfroLM candidate-reference similarity (Val only, used in target)
            if ref_emb is not None:
                c_emb = a_cand_embs.get(rid)
                feat["afrolm_sim_cand_ref"] = float(np.dot(ref_emb, c_emb)) if c_emb is not None else 0.0
            rows.append(feat)
    return pd.DataFrame(rows).fillna(0)

# ── Build Val features ────────────────────────────────────────────────────────
log.info("Building Stage7 features for Val...")
t0 = time.time()
feat7 = build_features_v7(
    val_sparse_ret, val_dense_ret,
    val["input"].tolist(), val["input_norm"].tolist(), val_subs,
    val_ce, train,
    afrolm_q_embs, afrolm_cand_embs, afrolm_ref_embs,
    top_ks=10, top_kd=20
)
log.info(f"  {len(feat7)} rows, {len(feat7.columns)} cols in {time.time()-t0:.1f}s")

# ROUGE targets
log.info("Computing ROUGE targets...")
t0 = time.time()
doc_ids_v = feat7["doc_id"].astype(int).values
q_idxs_v  = feat7["query_idx"].astype(int).values
train_outs = train["output"].values
r1s_t = [rouge1_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
rls_t = [rougel_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
log.info(f"  done in {time.time()-t0:.1f}s")

# AfroLM cand-ref similarity (already in feat7 if available)
has_afrolm_ref = "afrolm_sim_cand_ref" in feat7.columns and afrolm_encode_fn is not None
afrolm_sim_arr = feat7["afrolm_sim_cand_ref"].values if has_afrolm_ref else np.zeros(len(feat7))

# Feature columns for LGB (exclude training-only columns)
feat_cols_all = [c for c in feat7.columns if c not in ("query_idx", "doc_id", "afrolm_sim_cand_ref")]
X_all = feat7[feat_cols_all].values
n_q   = len(val)
split = int(n_q * 0.8)
tr_m  = q_idxs_v < split
vl_m  = ~tr_m

# ── LGB variants ──────────────────────────────────────────────────────────────
import lightgbm as lgb

params_lgb = dict(objective="regression", metric="rmse", num_leaves=63,
                  learning_rate=0.05, verbosity=-1, random_state=42,
                  num_threads=4, min_data_in_leaf=5)

TARGET_VARIANTS = [
    ("rl_heavy",        lambda r1, rl, al: 0.35*r1  + 0.65*rl),
    ("safe",            lambda r1, rl, al: 0.35*r1  + 0.35*rl + 0.30*al),
    ("semantic",        lambda r1, rl, al: 0.30*r1  + 0.30*rl + 0.40*al),
    ("rougel_semantic", lambda r1, rl, al: 0.25*r1  + 0.45*rl + 0.30*al),
]

experiments7   = {}
trained_models = {}

def fallback_v3(i):
    s = val_subs[i]; rn = {"Aka_Gha": "char", "Amh_Eth": "byte"}.get(s, "char_wb")
    rows, _ = val_sparse_ret[rn][i]
    return train.loc[rows[0], "output"] if rows else ""

for tgt_name, tgt_fn in TARGET_VARIANTS:
    try:
        y = np.array([tgt_fn(r1, rl, al) for r1, rl, al in zip(r1s_t, rls_t, afrolm_sim_arr)])
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
        exp_name = f"7_{tgt_name}"
        experiments7[exp_name] = {"preds": preds, "r1": r1, "rl": rl, "wr": wr_v,
                                   "r1s": r1s, "rls": rls}
        trained_models[exp_name] = (model, feat_cols_all)
        log.info(f"  {exp_name}: R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f} "
                 f"(Δ vs S5={wr_v-BASELINE_WR_S5:+.5f})")
    except Exception as e:
        log.warning(f"LGB_{tgt_name} failed: {e}\n{traceback.format_exc()[:300]}")

# ── Subset-level routing ──────────────────────────────────────────────────────
log.info("\nSubset-level routing analysis...")

def subset_wr_for(preds_list, ref_list, subs_list, s):
    mask = [i for i, ss in enumerate(subs_list) if ss == s]
    if not mask: return 0.0
    r1s_ = [rouge1_f1(ref_list[i], preds_list[i]) for i in mask]
    rls_ = [rougel_f1(ref_list[i], preds_list[i]) for i in mask]
    return 0.37*np.mean(r1s_) + 0.37*np.mean(rls_)

routing_table = {}
MIN_GAIN = 0.005  # conservative threshold per subset

for s in SUBSETS:
    # Stage5 routed baseline for this subset
    s5_wr = subset_wr_for(s5_routed, val_refs, val_subs, s) if s5_routed else 0.0
    # Stage4 for this subset
    s4_wr = subset_wr_for(s4_preds, val_refs, val_subs, s) if s4_preds else 0.0

    best_src = "stage5_routed"; best_wr_s = s5_wr; best_exp_s = "stage5_routed"
    if s4_wr > s5_wr + MIN_GAIN:
        best_src = "stage4"; best_wr_s = s4_wr; best_exp_s = "stage4_CE_LGB"
    for exp, sc in experiments7.items():
        if "r1s" not in sc: continue
        exp_wr = subset_wr_for(sc["preds"], val_refs, val_subs, s)
        if exp_wr > best_wr_s + MIN_GAIN:
            best_src = "stage7"; best_wr_s = exp_wr; best_exp_s = exp
    routing_table[s] = {"source": best_src, "exp": best_exp_s, "wr": best_wr_s,
                         "s5_wr": s5_wr, "delta": best_wr_s - s5_wr}
    log.info(f"  {s}: best={best_src}/{best_exp_s} WR={best_wr_s:.4f} "
             f"(S5={s5_wr:.4f} Δ={best_wr_s-s5_wr:+.4f})")

# Build routed predictions
routed_preds = []
for i, s in enumerate(val_subs):
    rt = routing_table[s]
    if rt["source"] == "stage4" and s4_preds:
        routed_preds.append(s4_preds[i])
    elif rt["source"] == "stage5_routed" and s5_routed:
        routed_preds.append(s5_routed[i])
    elif rt["source"] == "stage7" and rt["exp"] in experiments7:
        routed_preds.append(experiments7[rt["exp"]]["preds"][i])
    elif s5_routed: routed_preds.append(s5_routed[i])
    else: routed_preds.append(fallback_v3(i))

r1, rl, wr_rt, r1s_rt, rls_rt = score_rows(val_refs, routed_preds)
experiments7["7_routed"] = {"preds": routed_preds, "r1": r1, "rl": rl, "wr": wr_rt,
                              "r1s": r1s_rt, "rls": rls_rt}
log.info(f"\n  7_routed: R1={r1:.4f} RL={rl:.4f} WR={wr_rt:.5f} "
         f"(Δ vs S5={wr_rt-BASELINE_WR_S5:+.5f})")

# ── Final comparison ───────────────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("STAGE 7 — FINAL COMPARISON")
log.info(f"  Baseline (Stage5 routed): WR={BASELINE_WR_S5:.5f}")
log.info(f"  Submission threshold:     WR>={THRESHOLD_UPDATE:.5f} (+0.007)")
best7_exp = None; best7_wr = BASELINE_WR_S5
for exp in sorted(experiments7, key=lambda e: experiments7[e]["wr"], reverse=True):
    sc = experiments7[exp]
    flag = "★" if sc["wr"] >= THRESHOLD_UPDATE else ("△" if sc["wr"] > BASELINE_WR_S5 else " ")
    log.info(f"  {flag} {exp:35s}: R1={sc['r1']:.4f} RL={sc['rl']:.4f} "
             f"WR={sc['wr']:.5f} (Δ={sc['wr']-BASELINE_WR_S5:+.5f})")
    if sc["wr"] > best7_wr: best7_wr = sc["wr"]; best7_exp = exp

log.info(f"\nBest Stage7: {best7_exp} WR={best7_wr:.5f}")
meets_threshold = best7_wr >= THRESHOLD_UPDATE
log.info(f"Meets threshold (≥{THRESHOLD_UPDATE:.5f}): {meets_threshold}")

# ── AfroLM BERTScore stats on Val ─────────────────────────────────────────────
if has_afrolm_ref:
    # For the top-1 predictions, compute mean afrolm_sim_cand_ref
    afrolm_preds_sims = []
    for i in range(n_q):
        mask = q_idxs_v == i
        if not mask.any(): afrolm_preds_sims.append(0.0); continue
        if best7_exp and best7_exp != "7_routed" and best7_exp in trained_models:
            mdl, fc = trained_models[best7_exp]
            sc_arr = mdl.predict(feat7[fc].values[mask])
        else:
            sc_arr = np.zeros(mask.sum())  # dummy
        best_idx = np.argmax(sc_arr)
        best_rid = int(doc_ids_v[mask][best_idx])
        cand_emb = afrolm_cand_embs.get(best_rid)
        if cand_emb is not None and afrolm_ref_embs is not None:
            afrolm_preds_sims.append(float(np.dot(afrolm_ref_embs[i], cand_emb)))
        else:
            afrolm_preds_sims.append(0.0)
    log.info(f"\nAfroLM BERTScore proxy (mean sim cand-ref):")
    log.info(f"  Stage7 best preds: {np.mean(afrolm_preds_sims):.4f}")

# ── Save outputs ───────────────────────────────────────────────────────────────
# Per-subset scores
sub_rows = []
for s in SUBSETS:
    mask = [i for i, ss in enumerate(val_subs) if ss == s]
    if not mask: continue
    srefs = [val_refs[i] for i in mask]
    for exp, sc in experiments7.items():
        if "r1s" not in sc: continue
        sp = [sc["preds"][i] for i in mask]
        r1_, rl_, wr_, _, _ = score_rows(srefs, sp)
        sub_rows.append({"experiment": exp, "subset": s, "n": len(mask),
                          "rouge1": r1_, "rougel": rl_, "weighted_rouge": wr_})
df_sub = pd.DataFrame(sub_rows)
df_sub.to_csv(OUT/"stage7_afrolm_validation_scores_by_subset.csv", index=False)

# Global scores
rows_g = [{"experiment": "Stage5_routed_baseline", "rouge1": 0.53457, "rougel": 0.48063,
            "weighted_rouge": BASELINE_WR_S5}]
for exp, sc in experiments7.items():
    rows_g.append({"experiment": exp, "rouge1": sc["r1"], "rougel": sc["rl"],
                    "weighted_rouge": sc["wr"]})
df_g = pd.DataFrame(rows_g)
df_g.to_csv(OUT/"stage7_afrolm_validation_scores.csv", index=False)

# Routing table
pd.DataFrame([{"subset": s, "source": rt["source"], "experiment": rt["exp"],
                "wr_best": rt["wr"], "wr_stage5": rt["s5_wr"], "delta": rt["delta"]}
              for s, rt in routing_table.items()]).to_csv(
    OUT/"stage7_routing_by_subset.csv", index=False)

# Val predictions debug (best Stage7)
if best7_exp and best7_exp in experiments7:
    bp = experiments7[best7_exp]["preds"]
    vd = val.copy(); vd["prediction"] = bp; vd["experiment"] = best7_exp
    vd["rouge1"] = [rouge1_f1(r, p) for r, p in zip(val_refs, bp)]
    vd["rougel"] = [rougel_f1(r, p) for r, p in zip(val_refs, bp)]
    vd.to_csv(OUT/"stage7_afrolm_val_predictions_debug.csv", index=False)

# BERTScore proxy file
afrolm_rows = []
for i, (ref, s) in enumerate(zip(val_refs, val_subs)):
    pool = get_pool(i, val_sparse_ret, val_dense_ret)
    ref_emb = afrolm_ref_embs[i] if afrolm_ref_embs is not None else None
    q_emb   = afrolm_q_embs[i]   if afrolm_q_embs   is not None else None
    for rid, info in list(pool.items())[:5]:  # top-5 only for file size
        cand = str(train.loc[rid, "output"])
        c_emb = afrolm_cand_embs.get(rid)
        al_ref = float(np.dot(ref_emb, c_emb)) if (ref_emb is not None and c_emb is not None) else 0.0
        al_q   = float(np.dot(q_emb,   c_emb)) if (q_emb   is not None and c_emb is not None) else 0.0
        afrolm_rows.append({"ID": val.iloc[i]["ID"], "subset": s, "candidate_id": rid,
                              "afrolm_bertscore_f1": al_ref, "afrolm_sim_q_cand": al_q,
                              "rouge1": rouge1_f1(ref, cand), "rougel": rougel_f1(ref, cand)})
pd.DataFrame(afrolm_rows).to_csv(OUT/"stage7_afrolm_bertscore_scores.csv", index=False)

# Changed rows vs Stage5
if best7_exp and s5_routed:
    changed = [{"val_idx": i, "subset": val_subs[i],
                 "pred_s5": str(s5_routed[i])[:100],
                 "pred_s7": str(experiments7[best7_exp]["preds"][i])[:100]}
               for i in range(n_q) if s5_routed[i] != experiments7[best7_exp]["preds"][i]]
    pd.DataFrame(changed).to_csv(OUT/"stage7_changed_rows_vs_stage5.csv", index=False)
    log.info(f"Changed rows vs Stage5: {len(changed)}/{n_q}")

# ── Build test submission ONLY if threshold met ────────────────────────────────
if meets_threshold:
    log.info(f"\nThreshold met! Building Test submission for {best7_exp}...")

    # Refit sparse on Train+Val
    base_df = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
    for df in [base_df, test]: df["input_norm"] = df["input"].apply(norm)
    base_df["output_norm"] = base_df["output"].apply(norm)
    SUBSETS_BASE = sorted(base_df["subset"].unique())

    rvecs = {}
    for name, cfg in CFGS.items():
        v2 = TfidfVectorizer(**cfg); v2.fit(base_df["input_norm"]); rvecs[name] = v2
    rsub_i, rsub_r = {}, {}
    for name, v2 in rvecs.items():
        rsub_i[name] = {s: v2.transform(base_df[base_df["subset"]==s]["input_norm"])
                        for s in SUBSETS_BASE if len(base_df[base_df["subset"]==s]) > 0}
        rsub_r[name] = {s: base_df[base_df["subset"]==s].index.tolist() for s in SUBSETS_BASE}

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

    # Dense for test (refit base_df, use cached base embs if available)
    test_dense_ret = {}
    rdense_sub_idx = {}; rdense_sub_rids = {}
    for model_key, (_, use_prefix) in [(k, (None, up)) for k, _, up, _ in S5_DENSE_CONFIGS]:
        model_key2, _, use_prefix2, _ = next((x for x in S5_DENSE_CONFIGS if x[0]==model_key), (None,)*4)
        if model_key2 is None: continue
    for model_key, model_name2, use_prefix2, _ in S5_DENSE_CONFIGS:
        if model_key not in [k for k, _ in dense_model_keys]: continue
        try:
            base_ec = CACHE_DIR / f"base_embs_{model_key}.npy"
            test_ec = CACHE_DIR / f"test_embs_{model_key}.npy"
            if base_ec.exists() and test_ec.exists():
                base_embs = np.load(str(base_ec)).astype(np.float32)
                te_embs   = np.load(str(test_ec)).astype(np.float32)
                log.info(f"  Loaded cached base/test embs for {model_key}")
            else:
                from sentence_transformers import SentenceTransformer
                dm2 = SentenceTransformer(model_name2, cache_folder="/tmp/hf_cache", device=DEVICE)
                pfx2 = "passage: " if use_prefix2 else ""
                base_embs = dm2.encode([pfx2+t for t in base_df["input_norm"].tolist()],
                                        batch_size=256, show_progress_bar=False,
                                        normalize_embeddings=True, device=DEVICE).astype(np.float32)
                np.save(str(base_ec), base_embs)
                vpfx2 = "query: " if use_prefix2 else ""
                te_embs = dm2.encode([vpfx2+t for t in test["input_norm"].tolist()],
                                      batch_size=256, show_progress_bar=False,
                                      normalize_embeddings=True, device=DEVICE).astype(np.float32)
                np.save(str(test_ec), te_embs)

            dim3 = base_embs.shape[1]
            rdense_sub_idx[model_key] = {}; rdense_sub_rids[model_key] = {}
            for s in SUBSETS_BASE:
                bmask = (base_df["subset"] == s).values
                if not bmask.any(): continue
                bidxs = base_df[base_df["subset"]==s].index.tolist()
                fi = faiss.IndexFlatIP(dim3); fi.add(base_embs[bmask])
                rdense_sub_idx[model_key][s]  = fi
                rdense_sub_rids[model_key][s] = bidxs
            gfi = faiss.IndexFlatIP(dim3); gfi.add(base_embs); grids = base_df.index.tolist()
            test_dense_ret[model_key] = []
            for i, (q_emb, s) in enumerate(zip(te_embs, test_subs)):
                qv = q_emb.reshape(1, -1)
                if s in rdense_sub_idx[model_key]:
                    fi4 = rdense_sub_idx[model_key][s]; rids4 = rdense_sub_rids[model_key][s]
                else: fi4 = gfi; rids4 = grids
                tk = min(TOP_K_DENSE, fi4.ntotal); D, I = fi4.search(qv, tk)
                test_dense_ret[model_key].append(
                    ([rids4[j] for j in I[0] if j>=0],
                     [float(D[0][k]) for k in range(len(I[0])) if I[0][k]>=0]))
        except Exception as e:
            log.warning(f"Dense test {model_key}: {e}")

    # CE for test (reuse Stage6 cache)
    test_ce = {}
    test_ce_cache = CACHE_DIR / "test_ce_scores_stage6.pkl"
    if test_ce_cache.exists():
        with open(test_ce_cache, "rb") as f: test_ce = pickle.load(f)
        log.info(f"  Reused Stage6 test CE scores for {len(test_ce)} queries")

    # AfroLM embeddings for test queries
    test_afrolm_q_embs  = None
    test_afrolm_cand_embs = {}
    if afrolm_encode_fn is not None:
        log.info("  Encoding test queries with AfroLM...")
        t0 = time.time()
        test_afrolm_q_embs = afrolm_encode_fn(test["input"].tolist(), batch_size=128)
        log.info(f"    {test_afrolm_q_embs.shape} in {time.time()-t0:.1f}s")

        all_test_rids = set()
        for i in range(len(test)):
            pool = get_pool(i, test_sparse_ret, test_dense_ret)
            all_test_rids.update(pool.keys())
        new_rids = all_test_rids - set(afrolm_cand_embs.keys())
        if new_rids:
            log.info(f"  Encoding {len(new_rids)} new candidate answers for test...")
            t0 = time.time()
            new_rids_list = sorted(new_rids)
            new_texts = [str(base_df.loc[rid, "output"]) for rid in new_rids_list]
            new_embs  = afrolm_encode_fn(new_texts, batch_size=128)
            test_afrolm_cand_embs = {rid: new_embs[k] for k, rid in enumerate(new_rids_list)}
            log.info(f"    done in {time.time()-t0:.1f}s")
        # Merge cand embs (train+val pool)
        combined_cand_embs = {**afrolm_cand_embs, **test_afrolm_cand_embs}
    else:
        combined_cand_embs = {}; test_afrolm_q_embs = None

    log.info("  Building test features...")
    t0 = time.time()
    feat_test = build_features_v7(
        test_sparse_ret, test_dense_ret,
        test["input"].tolist(), test["input_norm"].tolist(), test_subs,
        test_ce, base_df,
        test_afrolm_q_embs, combined_cand_embs, a_ref_embs=None,  # no ref at test time
        top_ks=10, top_kd=20
    )
    log.info(f"    {len(feat_test)} rows in {time.time()-t0:.1f}s")

    for c in feat_cols_all:
        if c not in feat_test.columns: feat_test[c] = 0
    tq_arr = feat_test["query_idx"].values.copy()
    td_arr = feat_test["doc_id"].astype(int).values.copy()
    X_test = feat_test[feat_cols_all].values

    # Determine test strategy
    if "routed" in best7_exp:
        # Load Stage5 and Stage4 test submissions for routing
        sub5t = pd.read_csv(OUT/"submission_stage5_best.csv")
        sub4t = pd.read_csv(OUT/"submission_stage4_crossencoder.csv")
        test_ids = test["ID"].tolist()
        test_preds = []
        for i in range(len(test)):
            s = test_subs[i]; rt = routing_table.get(s, {})
            src = rt.get("source", "stage5_routed")
            exp_k = rt.get("exp", "stage5_routed")
            if src == "stage4":
                row = sub4t[sub4t["ID"] == test_ids[i]]
                test_preds.append(str(row["TargetRLF1"].values[0]) if len(row) > 0 else "")
            elif src == "stage5_routed":
                row = sub5t[sub5t["ID"] == test_ids[i]]
                test_preds.append(str(row["TargetRLF1"].values[0]) if len(row) > 0 else "")
            else:
                best_mdl, best_fc = trained_models.get(exp_k, (None, None))
                if best_mdl is not None:
                    mask = tq_arr == i
                    if mask.any():
                        sc_arr = best_mdl.predict(feat_test[best_fc].values[mask])
                        best_doc = int(td_arr[mask][np.argmax(sc_arr)])
                        test_preds.append(base_df.loc[best_doc, "output"])
                    else:
                        row = sub5t[sub5t["ID"] == test_ids[i]]
                        test_preds.append(str(row["TargetRLF1"].values[0]) if len(row) > 0 else "")
                else:
                    row = sub5t[sub5t["ID"] == test_ids[i]]
                    test_preds.append(str(row["TargetRLF1"].values[0]) if len(row) > 0 else "")
    else:
        best_mdl7, best_fc7 = trained_models[best7_exp]
        lgb_ts = best_mdl7.predict(X_test)
        test_preds = []
        for i in range(len(test)):
            mask = tq_arr == i
            if not mask.any():
                s = test_subs[i]; rn = {"Aka_Gha": "char", "Amh_Eth": "byte"}.get(s, "char_wb")
                rows_s, _ = test_sparse_ret[rn][i]
                test_preds.append(base_df.loc[rows_s[0], "output"] if rows_s else ""); continue
            best_doc = int(td_arr[mask][np.argmax(lgb_ts[mask])])
            test_preds.append(base_df.loc[best_doc, "output"])

    # Build submission
    sub7 = sample[["ID"]].copy()
    sub7["TargetRLF1"] = sub7["ID"].map(dict(zip(test["ID"], test_preds)))
    sub7["TargetR1F1"] = sub7["TargetRLF1"]; sub7["TargetLLM"] = sub7["TargetRLF1"]

    assert list(sub7.columns) == ["ID","TargetRLF1","TargetR1F1","TargetLLM"]
    assert sub7.shape == sample.shape
    assert (sub7["ID"].values == sample["ID"].values).all()
    assert sub7[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
    assert (sub7["TargetRLF1"] == sub7["TargetR1F1"]).all()
    assert (sub7["TargetRLF1"] == sub7["TargetLLM"]).all()
    log.info("Submission checks PASSED")

    sub7.to_csv(OUT/"submission_stage7_afrolm_best.csv", index=False)
    sub7.to_csv(OUT/"submission.csv", index=False)
    log.info(f"submission.csv UPDATED: {BASELINE_WR_S5:.5f} → {best7_wr:.5f}")
    candidate_public = True
else:
    log.info(f"\nThreshold NOT met (best={best7_wr:.5f} < {THRESHOLD_UPDATE:.5f}). "
             f"Keeping Stage5 submission.")
    import shutil
    shutil.copy(OUT/"submission_backup_before_stage7_afrolm.csv", OUT/"submission.csv")
    candidate_public = False

# ── Update metadata ────────────────────────────────────────────────────────────
try:
    with open(OUT/"best_stage.json") as f: bs = json.load(f)
    bs.setdefault("stage_results", {})["stage7"] = {
        "best_exp": best7_exp, "best_wr": best7_wr,
        "threshold": THRESHOLD_UPDATE, "met_threshold": meets_threshold,
        "afrolm_model": AFROLM_MODEL_USED,
        "routing": {s: rt["source"] for s, rt in routing_table.items()}
    }
    if meets_threshold and best7_wr > bs.get("weighted_rouge_val", 0):
        bs["best_experiment"] = best7_exp; bs["weighted_rouge_val"] = best7_wr
        if best7_exp in experiments7:
            bs["rouge1_val"] = experiments7[best7_exp]["r1"]
            bs["rougel_val"] = experiments7[best7_exp]["rl"]
    with open(OUT/"best_stage.json", "w") as f: json.dump(bs, f, indent=2)
except Exception as e: log.warning(f"best_stage.json: {e}")

try:
    sc_df = pd.read_csv(OUT/"validation_scores_all_stages.csv")
    new = [{"experiment": exp, "rouge1": sc["r1"], "rougel": sc["rl"], "weighted_rouge": sc["wr"]}
           for exp, sc in experiments7.items() if not (sc_df["experiment"] == exp).any()]
    if new: sc_df = pd.concat([sc_df, pd.DataFrame(new)], ignore_index=True)
    sc_df.to_csv(OUT/"validation_scores_all_stages.csv", index=False)
except Exception as e: log.warning(f"validation_scores: {e}")

# Stage 7 report
routing_str = "; ".join(f"{s}→{rt['source']}" for s, rt in routing_table.items())
md = f"""# Stage 7 Report — AfroLM Semantic Reranking

## Summary
- Baseline (Stage5 routed): WR = {BASELINE_WR_S5:.5f} | Public = 0.732376
- Stage6 e5-base: REJECTED (public = 0.729871)
- AfroLM model used: `{AFROLM_MODEL_USED}`
- Pool: Stage5 (sparse × 4 + minilm + e5s, NO e5b)
- CE scores: reused from Stage6 cache
- Threshold to update submission: WR ≥ {THRESHOLD_UPDATE:.5f} (+0.007)

## All experiments

| Experiment | R1 | RL | WR | Δ vs Stage5 | Meets threshold |
|---|---|---|---|---|---|
| Stage5_routed | 0.5346 | 0.4806 | {BASELINE_WR_S5:.5f} | — | baseline |
"""
for exp in sorted(experiments7, key=lambda e: experiments7[e]["wr"], reverse=True):
    sc = experiments7[exp]
    ok = "YES ★" if sc["wr"] >= THRESHOLD_UPDATE else "NO"
    md += f"| {exp} | {sc['r1']:.4f} | {sc['rl']:.4f} | {sc['wr']:.5f} | {sc['wr']-BASELINE_WR_S5:+.5f} | {ok} |\n"

md += f"""
## Routing by subset

| Subset | Source | WR | Delta |
|---|---|---|---|
"""
for s, rt in routing_table.items():
    md += f"| {s} | {rt['source']}/{rt['exp']} | {rt['wr']:.4f} | {rt['delta']:+.4f} |\n"

md += f"""
## Decision
- Threshold met: {meets_threshold}
- submission.csv: {'updated to ' + best7_exp if meets_threshold else 'kept Stage5 (threshold not met)'}
- Candidate for public submission: {'YES' if candidate_public else 'NO'}
"""
with open(OUT/"stage7_report.md", "w") as f: f.write(md)

# Final check
final = pd.read_csv(OUT/"submission.csv")
assert list(final.columns) == ["ID","TargetRLF1","TargetR1F1","TargetLLM"]
assert final.shape == sample.shape
assert (final["ID"].values == sample["ID"].values).all()
assert final[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
assert (final["TargetRLF1"] == final["TargetR1F1"]).all()
assert (final["TargetRLF1"] == final["TargetLLM"]).all()
log.info("Final submission.csv checks PASSED")

elapsed = time.time() - START
log.info("\n" + "="*60)
log.info("DONE_STAGE7_AFROLM")
log.info(f"Best public baseline : 0.732376")
log.info(f"Stage6 public        : 0.729871 REJECTED")
log.info(f"Best Stage5 Val      : {BASELINE_WR_S5:.5f}")
log.info(f"Best Stage7 Val      : {best7_wr:.5f}")
log.info(f"Delta vs Stage5      : {best7_wr-BASELINE_WR_S5:+.5f}")
log.info(f"AfroLM model used    : {AFROLM_MODEL_USED}")
log.info(f"Threshold (>=+0.007) : {THRESHOLD_UPDATE:.5f}")
log.info(f"Threshold met        : {meets_threshold}")
log.info(f"Best strategy        : {best7_exp or 'None'}")
log.info(f"Routing by subset    : {routing_str}")
log.info(f"Final submission     : {OUT}/submission.csv")
log.info(f"Candidate for public : {'YES' if candidate_public else 'NO'}")
log.info(f"Total elapsed        : {elapsed:.0f}s")
log.info("="*60)
