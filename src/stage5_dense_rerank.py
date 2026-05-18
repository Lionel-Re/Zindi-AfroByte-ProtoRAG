"""
Stage 5 — Dense retrieval (5A) + RougeL-oriented targets (5B) + Phrase composer (5C)
Baseline: Stage 4 CE_LGB_all, WR=0.34666 on Val
"""
import os, sys, json, re, time, logging, warnings, traceback, unicodedata, math
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
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "execution.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("STAGE 5 — Dense + RougeL reranking + Phrase composer")
log.info("=" * 60)
START = time.time()

BASELINE_WR = 0.34666      # Stage 4 CE_LGB_all
AKA_AMH     = {"Aka_Gha", "Amh_Eth"}

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
    p, rv = hits / len(h), hits / len(r)
    return 2 * p * rv / (p + rv) if p + rv else 0.0

def _lcs(a, b):
    if not a or not b: return 0
    prev = [0] * (len(b) + 1)
    for ai in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b):
            cur[j+1] = prev[j]+1 if ai == bj else max(cur[j], prev[j+1])
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
val_refs = val["output"].tolist()
val_subs = val["subset"].tolist()
test_subs = test["subset"].tolist()
SUBSETS = sorted(train["subset"].unique())
log.info(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

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

def retrieve_sparse(q, s, name, topk=TOP_K_SPARSE, cand_df=None, cand_sub_i=None, cand_sub_r=None):
    si = cand_sub_i or sub_i; sr = cand_sub_r or sub_r
    v = vecs[name]; X_q = v.transform([q])
    if s in si[name]: X_s, rids = si[name][s], sr[name][s]
    else: X_s = v.transform((cand_df or train)["input_norm"]); rids = (cand_df or train).index.tolist()
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

# ── CE model ──────────────────────────────────────────────────────────────────
import torch
log.info("Loading cross-encoder...")
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
try:
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(CE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
                            cache_folder="/tmp/hf_cache")
    CE_OK = True
    log.info(f"  CE loaded on {next(ce_model.model.parameters()).device}")
except Exception as e:
    log.warning(f"CE load failed: {e}"); CE_OK = False

# ── Stage 5A: Dense retrieval ─────────────────────────────────────────────────
DENSE_OK = False
val_dense_ret = {}          # model_key -> list of (doc_ids, scores)
dense_encoders = {}         # model_key -> (SentenceTransformer, use_prefix)
dense_sub_idx  = {}         # model_key -> {subset -> faiss.IndexFlatIP}
dense_sub_rids = {}         # model_key -> {subset -> [train_rids]}
dense_model_keys = []       # [(model_key, rrf_weight)]
TRAIN_EMBS = {}             # model_key -> np.ndarray [n_train, dim]

DENSE_CONFIGS = [
    ("minilm",   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False, 0.5),
    ("e5s",      "intfloat/multilingual-e5-small",                               True,  0.5),
]
TOP_K_DENSE = 20

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    for model_key, model_name, use_prefix, rrf_w_d in DENSE_CONFIGS:
        try:
            log.info(f"[5A] Loading {model_name}...")
            t0 = time.time()
            dm = SentenceTransformer(model_name, cache_folder="/tmp/hf_cache", device=DEVICE)
            log.info(f"  loaded in {time.time()-t0:.1f}s")

            log.info(f"  Encoding Train ({len(train)})...")
            t0 = time.time()
            pfx = "passage: " if use_prefix else ""
            train_texts = [pfx + t for t in train["input_norm"].tolist()]
            tr_embs = dm.encode(train_texts, batch_size=512, show_progress_bar=False,
                                normalize_embeddings=True, device=DEVICE).astype(np.float32)
            TRAIN_EMBS[model_key] = tr_embs
            dim = tr_embs.shape[1]
            log.info(f"  Train encoded in {time.time()-t0:.1f}s dim={dim}")

            # Per-subset FAISS indices
            dense_sub_idx[model_key] = {}; dense_sub_rids[model_key] = {}
            for s in SUBSETS:
                mask = (train["subset"] == s).values
                if not mask.any(): continue
                idxs = train[train["subset"]==s].index.tolist()
                fi = faiss.IndexFlatIP(dim); fi.add(tr_embs[mask])
                dense_sub_idx[model_key][s]  = fi
                dense_sub_rids[model_key][s] = idxs

            dense_encoders[model_key] = (dm, use_prefix)
            dense_model_keys.append((model_key, rrf_w_d))

            # Encode val
            log.info(f"  Encoding Val for {model_key}...")
            t0 = time.time()
            vpfx = "query: " if use_prefix else ""
            val_texts = [vpfx + t for t in val["input_norm"].tolist()]
            val_embs = dm.encode(val_texts, batch_size=512, show_progress_bar=False,
                                 normalize_embeddings=True, device=DEVICE).astype(np.float32)

            val_dense_ret[model_key] = []
            global_fi = faiss.IndexFlatIP(dim); global_fi.add(tr_embs)
            global_rids = train.index.tolist()

            for i, (q_emb, s) in enumerate(zip(val_embs, val_subs)):
                qv = q_emb.reshape(1, -1)
                if s in dense_sub_idx[model_key]:
                    fi2 = dense_sub_idx[model_key][s]; rids2 = dense_sub_rids[model_key][s]
                else:
                    fi2 = global_fi; rids2 = global_rids
                tk = min(TOP_K_DENSE, fi2.ntotal)
                D, I = fi2.search(qv, tk)
                doc_ids = [rids2[j] for j in I[0] if j >= 0]
                scores  = [float(D[0][k]) for k in range(len(I[0])) if I[0][k] >= 0]
                val_dense_ret[model_key].append((doc_ids, scores))

            log.info(f"  Val dense retrieval done in {time.time()-t0:.1f}s")
            DENSE_OK = True

        except Exception as e:
            log.warning(f"Dense model {model_key} failed: {e}\n{traceback.format_exc()[:300]}")

except ImportError as e:
    log.warning(f"faiss/SentenceTransformer import failed: {e}")

log.info(f"Dense models active: {[k for k,_ in dense_model_keys]}")

# ── Build candidate pool ───────────────────────────────────────────────────────
def get_pool(i, sparse_ret, dense_ret=None, top_ks=20, top_kd=20):
    """Return {rid: info_dict} for query i"""
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

# ── CE scoring of full pool ────────────────────────────────────────────────────
val_ce = {}   # val_idx -> {rid -> score}
if CE_OK:
    log.info("CE scoring val pool (sparse+dense)...")
    all_pairs = []
    for i, (q, s) in enumerate(zip(val["input"], val_subs)):
        pool = get_pool(i, val_sparse_ret, val_dense_ret if DENSE_OK else None)
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
            log.info(f"  CE progress: {start}/{len(pair_inputs)} ({time.time()-t0:.0f}s)")
    elapsed = time.time() - t0
    log.info(f"  CE done: {len(all_ce)} pairs in {elapsed:.1f}s ({len(all_ce)/elapsed:.0f} p/s)")
    for (vi, rid, _, _), sc in zip(all_pairs, all_ce):
        if vi not in val_ce: val_ce[vi] = {}
        val_ce[vi][rid] = float(sc)

# ── Feature builder V5 ────────────────────────────────────────────────────────
def build_features_v5(sparse_ret, dense_ret, query_inputs, query_norms, query_subs,
                      ce_scores, cand_df, top_ks=10, top_kd=20):
    rows = []
    for i, (q_input, q_norm, s) in enumerate(zip(query_inputs, query_norms, query_subs)):
        pool = get_pool(i, sparse_ret, dense_ret, top_ks, top_kd)
        if not pool: continue
        q_words = q_norm.split()
        q_len   = len(q_words)

        # RRF sparse
        rrf_sp = {}
        for name in rnames:
            w = rrf_w_sparse[name]; r_list, _ = sparse_ret[name][i]
            for rank, rid in enumerate(r_list[:top_ks]):
                rrf_sp[rid] = rrf_sp.get(rid, 0.0) + w / (60 + rank)

        # RRF dense
        rrf_dn = {}
        if dense_ret:
            for mk, rrf_w_d in dense_model_keys:
                if mk not in dense_ret: continue
                r_list, _ = dense_ret[mk][i]
                for rank, rid in enumerate(r_list[:top_kd]):
                    rrf_dn[rid] = rrf_dn.get(rid, 0.0) + rrf_w_d / (60 + rank)

        # Median answer length in pool (for ratio feature)
        ans_lens = [len(str(cand_df.loc[rid, "output"]).split()) for rid in pool]
        med_ans_len = float(np.median(ans_lens)) if ans_lens else 1.0

        q_bigrams = set(zip(q_words, q_words[1:]))

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
            # RRF combined
            feat["rrf_sparse"] = rrf_sp.get(rid, 0.0)
            feat["rrf_dense"]  = rrf_dn.get(rid, 0.0)
            feat["rrf_total"]  = feat["rrf_sparse"] + feat["rrf_dense"]
            # CE
            feat["ce_score"] = ce_scores.get(i, {}).get(rid, -999.0)
            # Subset
            feat["same_subset"] = int(cand_df.loc[rid, "subset"] == s)
            feat["subset_id"]   = SUBSETS.index(s) if s in SUBSETS else -1
            # Lengths
            ans = str(cand_df.loc[rid, "output"])
            ans_len = len(ans.split())
            feat["query_len"]       = q_len
            feat["answer_len"]      = ans_len
            feat["answer_len_ratio"] = ans_len / (med_ans_len + 1)
            # Multi-retriever agreement
            feat["n_sparse_found"] = sum(1 for nm in rnames if f"sp_{nm}_r" in info)
            feat["n_dense_found"]  = sum(1 for mk, _ in dense_model_keys if f"dn_{mk}_r" in info)
            feat["n_total_found"]  = feat["n_sparse_found"] + feat["n_dense_found"]
            # Bigram overlap (query vs answer)
            ans_words = ans.lower().split()[:100]
            a_bigrams = set(zip(ans_words, ans_words[1:]))
            feat["bigram_overlap"] = len(q_bigrams & a_bigrams) / (len(q_bigrams) + 1)
            # Min rank across all sparse retrievers (rank stability)
            sp_ranks = [info.get(f"sp_{nm}_r", top_ks) for nm in rnames]
            feat["min_sparse_rank"] = min(sp_ranks)
            feat["rank_stability"]  = float(np.std(sp_ranks))
            rows.append(feat)
    return pd.DataFrame(rows).fillna(0)

# ── Build features for Val ────────────────────────────────────────────────────
log.info("Building Stage 5 features for Val...")
t0 = time.time()
feat5 = build_features_v5(
    val_sparse_ret, val_dense_ret if DENSE_OK else {},
    val["input"].tolist(), val["input_norm"].tolist(), val_subs,
    val_ce, train, top_ks=10, top_kd=20
)
log.info(f"  {len(feat5)} rows, {len(feat5.columns)} cols in {time.time()-t0:.1f}s")

# ROUGE targets
log.info("Computing ROUGE targets...")
t0 = time.time()
doc_ids_v = feat5["doc_id"].astype(int).values
q_idxs_v  = feat5["query_idx"].astype(int).values
train_outs = train["output"].values
r1s_t = [rouge1_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
rls_t = [rougel_f1(val_refs[qi], train_outs[di]) for qi, di in zip(q_idxs_v, doc_ids_v)]
log.info(f"  done in {time.time()-t0:.1f}s")

feat_cols = [c for c in feat5.columns if c not in ("query_idx", "doc_id")]
X_all = feat5[feat_cols].values
n_q   = len(val)

# Train/val split for LGB (80/20 by query index)
split = int(n_q * 0.8)
tr_m  = q_idxs_v < split
vl_m  = ~tr_m

# ── LGB variants ─────────────────────────────────────────────────────────────
import lightgbm as lgb

params_lgb = dict(objective="regression", metric="rmse", num_leaves=63,
                  learning_rate=0.05, verbosity=-1, random_state=42,
                  num_threads=4, min_data_in_leaf=5)

TARGET_VARIANTS = [
    ("avg",      lambda r1, rl: 0.5*r1   + 0.5*rl),
    ("rl_heavy", lambda r1, rl: 0.35*r1  + 0.65*rl),
    ("comp",     lambda r1, rl: 0.37*r1  + 0.37*rl),
]

experiments5 = {}
trained_models = {}   # exp_name -> lgb model

def predict_from_model(model, feat_df, feat_c, q_idx, doc_id, n_queries,
                       fallback_fn=None):
    scores = model.predict(feat_df[feat_c].values)
    preds = []
    for i in range(n_queries):
        mask = q_idx == i
        if not mask.any():
            preds.append(fallback_fn(i) if fallback_fn else "")
        else:
            best = int(doc_id[mask][np.argmax(scores[mask])])
            preds.append(train.loc[best, "output"])
    return preds, scores

def fallback_v3(i, subset_list=None, sparse_ret=None):
    s = (subset_list or val_subs)[i]
    rn = {"Aka_Gha": "char", "Amh_Eth": "byte"}.get(s, "char_wb")
    rows, _ = (sparse_ret or val_sparse_ret)[rn][i]
    return train.loc[rows[0], "output"] if rows else ""

for tgt_name, tgt_fn in TARGET_VARIANTS:
    try:
        y = np.array([tgt_fn(r1, rl) for r1, rl in zip(r1s_t, rls_t)])
        dtrain = lgb.Dataset(X_all[tr_m], label=y[tr_m])
        dvalid = lgb.Dataset(X_all[vl_m], label=y[vl_m])
        t0 = time.time()
        model = lgb.train(params_lgb, dtrain, num_boost_round=300, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(25, verbose=False),
                                     lgb.log_evaluation(-1)])
        log.info(f"  LGB_{tgt_name}: {time.time()-t0:.1f}s iter={model.best_iteration}")

        preds, _ = predict_from_model(model, feat5, feat_cols, q_idxs_v,
                                      doc_ids_v, n_q,
                                      fallback_fn=lambda i: fallback_v3(i))
        r1, rl, wr_v, r1s, rls = score_rows(val_refs, preds)
        suffix = "5A" if DENSE_OK else "5B"
        exp_name = f"{suffix}_LGB_{tgt_name}"
        experiments5[exp_name] = {"preds": preds, "r1": r1, "rl": rl, "wr": wr_v,
                                   "r1s": r1s, "rls": rls}
        trained_models[exp_name] = (model, feat_cols)
        log.info(f"  {exp_name}: R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f} "
                 f"(Δ={wr_v-BASELINE_WR:+.5f})")
    except Exception as e:
        log.warning(f"LGB_{tgt_name} failed: {e}\n{traceback.format_exc()[:300]}")

# ── Stage 5C: Phrase composer ─────────────────────────────────────────────────
def compose_answer(q_norm, cand_answers, cand_scores, target_len=None):
    """Extract and assemble fragments from top candidates."""
    if not cand_answers: return ""
    # Target length: median of top-5 answers
    top5_lens = [len(str(a).split()) for a in cand_answers[:5]]
    tgt = target_len or (int(np.median(top5_lens)) if top5_lens else 50)

    fragments = []   # (score, text, word_list)
    q_words   = set(q_norm.split())

    for rank, (ans, sc) in enumerate(zip(cand_answers, cand_scores)):
        src_sc = float(sc) * (1.0 - 0.08 * rank)
        text   = str(ans).strip()
        # Split on sentence delimiters; fallback to word windows
        sents  = re.split(r'(?<=[.!?;])\s+', text)
        if len(sents) < 2:
            words = text.split()
            sents = [" ".join(words[j:j+15]) for j in range(0, len(words), 12)]
        for sent in sents:
            sent  = sent.strip()
            words = sent.split()
            if len(words) < 4 or len(words) > 70: continue
            wset  = set(words)
            q_ov  = len(q_words & wset) / (len(q_words) + 1)
            score = src_sc + 0.25 * q_ov
            fragments.append((score, sent, words))

    if not fragments: return str(cand_answers[0])
    fragments.sort(reverse=True)

    used_words = set(); chosen = []; total_w = 0
    for sc2, text, words in fragments:
        if total_w >= tgt * 1.3: break
        ov = len(set(words) & used_words) / (len(words) + 1)
        if ov > 0.55: continue
        chosen.append(text); used_words.update(words); total_w += len(words)

    return " ".join(chosen) if chosen else str(cand_answers[0])

try:
    # Use best LGB model to rank candidates before composing
    best5_so_far = max(experiments5, key=lambda e: experiments5[e]["wr"]) if experiments5 else None
    if best5_so_far:
        log.info(f"\nStage 5C: Phrase composer (base: {best5_so_far})...")
        mdl5c, fc5c = trained_models[best5_so_far]
        lgb_scores_all = mdl5c.predict(feat5[fc5c].values)

        composer_preds = []
        for i in range(n_q):
            mask = q_idxs_v == i
            if not mask.any():
                composer_preds.append(fallback_v3(i)); continue
            top_idx    = np.argsort(lgb_scores_all[mask])[::-1][:10]
            top_ids    = doc_ids_v[mask][top_idx]
            top_scs    = lgb_scores_all[mask][top_idx]
            top_answers= [str(train.loc[rid, "output"]) for rid in top_ids]
            composed   = compose_answer(val["input_norm"].iloc[i], top_answers, list(top_scs))
            composer_preds.append(composed)

        r1, rl, wr_v, r1s, rls = score_rows(val_refs, composer_preds)
        experiments5["5C_composer"] = {"preds": composer_preds, "r1": r1, "rl": rl,
                                        "wr": wr_v, "r1s": r1s, "rls": rls}
        log.info(f"  5C_composer: R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f} "
                 f"(Δ={wr_v-BASELINE_WR:+.5f})")

        # Hybrid: composer for AKA/AMH, best LGB for others
        hyb1 = []
        base_preds = experiments5[best5_so_far]["preds"]
        for i, s in enumerate(val_subs):
            hyb1.append(composer_preds[i] if s in AKA_AMH else base_preds[i])
        r1, rl, wr_v, r1s, rls = score_rows(val_refs, hyb1)
        experiments5["5C_hyb_aka"] = {"preds": hyb1, "r1": r1, "rl": rl, "wr": wr_v,
                                       "r1s": r1s, "rls": rls}
        log.info(f"  5C_hyb_aka : R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f} "
                 f"(Δ={wr_v-BASELINE_WR:+.5f})")
except Exception as e:
    log.warning(f"Stage 5C failed: {e}\n{traceback.format_exc()[:300]}")

# ── Final comparison ──────────────────────────────────────────────────────────
log.info("\n" + "="*60)
log.info("STAGE 5 — FINAL COMPARISON")
log.info(f"  Baseline (Stage4 CE_LGB_all): WR={BASELINE_WR:.5f}")
best5_exp = None; best5_wr = BASELINE_WR
for exp in sorted(experiments5, key=lambda e: experiments5[e]["wr"], reverse=True):
    sc = experiments5[exp]
    flag = "★" if sc["wr"] > BASELINE_WR else " "
    log.info(f"  {flag} {exp:35s}: R1={sc['r1']:.4f} RL={sc['rl']:.4f} "
             f"WR={sc['wr']:.5f} (Δ={sc['wr']-BASELINE_WR:+.5f})")
    if sc["wr"] > best5_wr:
        best5_wr = sc["wr"]; best5_exp = exp

if best5_exp is None:
    log.info("No Stage 5 variant improves Stage 4 baseline. Keeping Stage 4 submission.")
else:
    log.info(f"\nBest Stage 5 experiment: {best5_exp} WR={best5_wr:.5f} "
             f"(+{best5_wr-BASELINE_WR:.5f} vs Stage 4)")

# ── Per-subset scores ─────────────────────────────────────────────────────────
sub_rows = []
for s in SUBSETS:
    mask = [i for i, ss in enumerate(val_subs) if ss == s]
    if not mask: continue
    srefs = [val_refs[i] for i in mask]
    for exp, sc in experiments5.items():
        if "r1s" not in sc: continue
        sp = [sc["preds"][i] for i in mask]
        r1, rl, wr_v, _, _ = score_rows(srefs, sp)
        sub_rows.append({"experiment": exp, "subset": s, "n": len(mask),
                          "rouge1": r1, "rougel": rl, "weighted_rouge": wr_v})
df_sub = pd.DataFrame(sub_rows)
df_sub.to_csv(OUT/"stage5_validation_scores_by_subset.csv", index=False)
log.info("Saved stage5_validation_scores_by_subset.csv")

# ── Build test predictions if improved ───────────────────────────────────────
if best5_exp is not None:
    log.info(f"\nBuilding Test predictions for {best5_exp}...")

    # Refit vectorizers on Train+Val
    base_df = pd.concat([train, val], ignore_index=True).reset_index(drop=True)
    for df in [base_df, test]: df["input_norm"] = df["input"].apply(norm)
    base_df["output_norm"] = base_df["output"].apply(norm)
    SUBSETS_BASE = sorted(base_df["subset"].unique())

    log.info("  Refitting sparse vectorizers on Train+Val...")
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

    # Dense retrieval for test (if dense was used)
    test_dense_ret = {}
    if DENSE_OK and "5A" in best5_exp:
        log.info("  Encoding base_df for dense (Train+Val)...")
        rdense_sub_idx = {}; rdense_sub_rids = {}
        base_embs_dict = {}
        for model_key, (dm, use_prefix) in dense_encoders.items():
            t0 = time.time()
            pfx = "passage: " if use_prefix else ""
            base_texts = [pfx + t for t in base_df["input_norm"].tolist()]
            base_embs = dm.encode(base_texts, batch_size=512, show_progress_bar=False,
                                  normalize_embeddings=True, device=DEVICE).astype(np.float32)
            base_embs_dict[model_key] = base_embs
            dim2 = base_embs.shape[1]
            rdense_sub_idx[model_key] = {}; rdense_sub_rids[model_key] = {}
            for s in SUBSETS_BASE:
                bmask = (base_df["subset"] == s).values
                if not bmask.any(): continue
                bidxs = base_df[base_df["subset"]==s].index.tolist()
                fi3 = faiss.IndexFlatIP(dim2); fi3.add(base_embs[bmask])
                rdense_sub_idx[model_key][s]  = fi3
                rdense_sub_rids[model_key][s] = bidxs
            log.info(f"    base_df encoded {model_key} in {time.time()-t0:.1f}s")

        for model_key, (dm, use_prefix) in dense_encoders.items():
            log.info(f"  Dense retrieval test for {model_key}...")
            t0 = time.time()
            vpfx = "query: " if use_prefix else ""
            test_texts = [vpfx + t for t in test["input_norm"].tolist()]
            te_embs = dm.encode(test_texts, batch_size=512, show_progress_bar=False,
                                normalize_embeddings=True, device=DEVICE).astype(np.float32)
            dim2 = te_embs.shape[1]
            gfi = faiss.IndexFlatIP(dim2); gfi.add(base_embs_dict[model_key])
            grids = base_df.index.tolist()
            test_dense_ret[model_key] = []
            for i, (q_emb, s) in enumerate(zip(te_embs, test_subs)):
                qv = q_emb.reshape(1, -1)
                if s in rdense_sub_idx[model_key]:
                    fi4 = rdense_sub_idx[model_key][s]; rids4 = rdense_sub_rids[model_key][s]
                else:
                    fi4 = gfi; rids4 = grids
                tk = min(TOP_K_DENSE, fi4.ntotal)
                D, I = fi4.search(qv, tk)
                doc_ids4 = [rids4[j] for j in I[0] if j >= 0]
                scores4  = [float(D[0][k]) for k in range(len(I[0])) if I[0][k] >= 0]
                test_dense_ret[model_key].append((doc_ids4, scores4))
            log.info(f"    done in {time.time()-t0:.1f}s")

    # CE scoring for test
    test_ce = {}
    if CE_OK:
        log.info("  CE scoring test pool...")
        test_pairs = []
        for i, (q, s) in enumerate(zip(test["input"], test_subs)):
            pool = get_pool(i, test_sparse_ret, test_dense_ret if test_dense_ret else None)
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
                log.info(f"    CE test: {start}/{len(tinputs)} ({time.time()-t0:.0f}s)")
        for (ti, rid, _, _), sc in zip(test_pairs, tscores):
            if ti not in test_ce: test_ce[ti] = {}
            test_ce[ti][rid] = float(sc)
        log.info(f"    Test CE done in {time.time()-t0:.1f}s")

    # Build test features
    log.info("  Building test features...")
    t0 = time.time()
    feat_test = build_features_v5(
        test_sparse_ret, test_dense_ret if test_dense_ret else {},
        test["input"].tolist(), test["input_norm"].tolist(), test_subs,
        test_ce, base_df, top_ks=10, top_kd=20
    )
    log.info(f"    {len(feat_test)} rows in {time.time()-t0:.1f}s")

    # Align columns
    for c in feat_cols:
        if c not in feat_test.columns: feat_test[c] = 0
    tq_arr = feat_test["query_idx"].values.copy()
    td_arr = feat_test["doc_id"].astype(int).values.copy()
    X_test = feat_test[feat_cols].values

    # Pick model
    if best5_exp in trained_models:
        best_model5, _ = trained_models[best5_exp]
    else:
        # composer: use best LGB model
        best_model5, _ = trained_models.get(
            max((k for k in trained_models), key=lambda k: experiments5.get(k, {}).get("wr", 0)),
            (None, None)
        )

    def fallback_test(i):
        s = test_subs[i]
        rn = {"Aka_Gha": "char", "Amh_Eth": "byte"}.get(s, "char_wb")
        rows, _ = test_sparse_ret[rn][i]
        return base_df.loc[rows[0], "output"] if rows else ""

    if best5_exp == "5C_composer":
        # Use best LGB to rank, then compose
        lgb_tscores = best_model5.predict(X_test)
        test_preds = []
        for i in range(len(test)):
            mask = tq_arr == i
            if not mask.any(): test_preds.append(fallback_test(i)); continue
            top_idx = np.argsort(lgb_tscores[mask])[::-1][:10]
            top_ids = td_arr[mask][top_idx]
            top_scs = lgb_tscores[mask][top_idx]
            top_ans = [str(base_df.loc[rid, "output"]) for rid in top_ids]
            test_preds.append(compose_answer(test["input_norm"].iloc[i], top_ans, list(top_scs)))
    elif best5_exp == "5C_hyb_aka":
        lgb_tscores = best_model5.predict(X_test)
        test_preds = []
        for i in range(len(test)):
            s = test_subs[i]; mask = tq_arr == i
            if not mask.any(): test_preds.append(fallback_test(i)); continue
            if s in AKA_AMH:
                top_idx = np.argsort(lgb_tscores[mask])[::-1][:10]
                top_ids = td_arr[mask][top_idx]; top_scs = lgb_tscores[mask][top_idx]
                top_ans = [str(base_df.loc[rid, "output"]) for rid in top_ids]
                test_preds.append(compose_answer(test["input_norm"].iloc[i], top_ans, list(top_scs)))
            else:
                best_doc = int(td_arr[mask][np.argmax(lgb_tscores[mask])])
                test_preds.append(base_df.loc[best_doc, "output"])
    else:
        # Standard LGB
        lgb_tscores = best_model5.predict(X_test)
        test_preds = []
        for i in range(len(test)):
            mask = tq_arr == i
            if not mask.any(): test_preds.append(fallback_test(i)); continue
            best_doc = int(td_arr[mask][np.argmax(lgb_tscores[mask])])
            test_preds.append(base_df.loc[best_doc, "output"])

    # Build submission
    sub5 = sample[["ID"]].copy()
    sub5["TargetRLF1"] = sub5["ID"].map(dict(zip(test["ID"], test_preds)))
    sub5["TargetR1F1"] = sub5["TargetRLF1"]
    sub5["TargetLLM"]  = sub5["TargetRLF1"]

    assert list(sub5.columns) == ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]
    assert sub5.shape == sample.shape
    assert (sub5["ID"].values == sample["ID"].values).all()
    assert sub5[["TargetRLF1", "TargetR1F1", "TargetLLM"]].notna().all().all()
    assert (sub5["TargetRLF1"] == sub5["TargetR1F1"]).all()
    assert (sub5["TargetRLF1"] == sub5["TargetLLM"]).all()
    log.info("Submission checks PASSED")

    sub5.to_csv(OUT/"submission_stage5_best.csv", index=False)
    sub5.to_csv(OUT/"submission.csv", index=False)
    log.info(f"submission.csv updated: {BASELINE_WR:.5f} → {best5_wr:.5f}")

# ── Save Val predictions debug ────────────────────────────────────────────────
if best5_exp and best5_exp in experiments5:
    best_preds = experiments5[best5_exp]["preds"]
    vd = val.copy()
    vd["prediction"] = best_preds
    vd["experiment"] = best5_exp
    vd["rouge1"] = [rouge1_f1(r, p) for r, p in zip(val_refs, best_preds)]
    vd["rougel"] = [rougel_f1(r, p) for r, p in zip(val_refs, best_preds)]
    vd.to_csv(OUT/"stage5_val_predictions_debug.csv", index=False)

# ── Changed rows vs Stage 4 ───────────────────────────────────────────────────
try:
    s4 = pd.read_csv(OUT/"submission_stage4_crossencoder.csv")
    if best5_exp and best5_exp in experiments5:
        stage4_val = pd.read_csv(OUT/"crossencoder_val_predictions_debug.csv")
        changed = []
        for i in range(n_q):
            p4 = str(stage4_val["prediction"].iloc[i]) if i < len(stage4_val) else ""
            p5 = experiments5[best5_exp]["preds"][i]
            if p4 != p5:
                changed.append({"val_idx": i, "subset": val_subs[i],
                                 "pred_stage4": p4[:100], "pred_stage5": str(p5)[:100]})
        pd.DataFrame(changed).to_csv(OUT/"stage5_changed_rows_vs_stage4.csv", index=False)
        log.info(f"Changed rows vs Stage 4: {len(changed)}/{n_q}")
except Exception as e:
    log.warning(f"Changed rows diff failed: {e}")

# ── Global scores CSV ─────────────────────────────────────────────────────────
rows_g = [{"experiment": "Stage4_CE_LGB_all_baseline", "rouge1": 0.4974,
            "rougel": 0.4396, "weighted_rouge": BASELINE_WR}]
for exp, sc in experiments5.items():
    rows_g.append({"experiment": exp, "rouge1": sc["r1"], "rougel": sc["rl"],
                    "weighted_rouge": sc["wr"]})
df_g = pd.DataFrame(rows_g)
df_g.to_csv(OUT/"stage5_validation_scores.csv", index=False)

# ── Update best_stage.json ────────────────────────────────────────────────────
try:
    with open(OUT/"best_stage.json") as f: bs = json.load(f)
    if best5_wr > bs.get("weighted_rouge_val", 0):
        bs["best_experiment"]  = best5_exp
        bs["weighted_rouge_val"] = best5_wr
        if best5_exp and best5_exp in experiments5:
            bs["rouge1_val"] = experiments5[best5_exp]["r1"]
            bs["rougel_val"] = experiments5[best5_exp]["rl"]
        bs.setdefault("stage_results", {})["stage5"] = {
            "experiment": best5_exp, "wr": best5_wr,
            "dense_ok": DENSE_OK, "dense_models": [k for k, _ in dense_model_keys]
        }
        with open(OUT/"best_stage.json", "w") as f: json.dump(bs, f, indent=2)
except Exception as e:
    log.warning(f"best_stage.json update: {e}")

# ── Update validation_scores_all_stages.csv ───────────────────────────────────
try:
    sc_df = pd.read_csv(OUT/"validation_scores_all_stages.csv")
    new = []
    for exp, sc in experiments5.items():
        if not (sc_df["experiment"] == exp).any():
            new.append({"experiment": exp, "rouge1": sc["r1"], "rougel": sc["rl"],
                         "weighted_rouge": sc["wr"]})
    if new: sc_df = pd.concat([sc_df, pd.DataFrame(new)], ignore_index=True)
    sc_df.to_csv(OUT/"validation_scores_all_stages.csv", index=False)
except Exception as e:
    log.warning(f"validation_scores update: {e}")

# ── Stage 5 report ────────────────────────────────────────────────────────────
def subset_wr(exp_name, subset):
    sc = experiments5.get(exp_name, {})
    if "r1s" not in sc: return None
    mask = [i for i, ss in enumerate(val_subs) if ss == subset]
    if not mask: return None
    r1s = [sc["r1s"][i] for i in mask]; rls = [sc["rls"][i] for i in mask]
    return 0.37*np.mean(r1s) + 0.37*np.mean(rls)

md = f"""# Stage 5 Report

## Summary
- Baseline (Stage 4 CE_LGB_all): WR = {BASELINE_WR:.5f}
- Dense models active: {[k for k, _ in dense_model_keys]}
- Best Stage 5 experiment: {best5_exp}
- Best Stage 5 WR: {best5_wr:.5f} (Δ = {best5_wr - BASELINE_WR:+.5f})

## All experiments

| Experiment | R1 | RL | WR | Δ vs Stage4 |
|---|---|---|---|---|
| Stage4_CE_LGB_all | 0.4974 | 0.4396 | {BASELINE_WR:.5f} | — |
"""
for exp in sorted(experiments5, key=lambda e: experiments5[e]["wr"], reverse=True):
    sc = experiments5[exp]
    md += f"| {exp} | {sc['r1']:.4f} | {sc['rl']:.4f} | {sc['wr']:.5f} | {sc['wr']-BASELINE_WR:+.5f} |\n"

md += "\n## Per-subset (best Stage 5 experiment)\n\n| Subset | WR |\n|---|---|\n"
for s in SUBSETS:
    v = subset_wr(best5_exp, s) if best5_exp else None
    md += f"| {s} | {v:.4f} |\n" if v is not None else f"| {s} | N/A |\n"

md += f"""
## Decision
- Stage 5 improved: {'YES' if best5_exp else 'NO'}
- submission.csv: {'updated to ' + best5_exp if best5_exp else 'kept Stage 4'}
"""

with open(OUT/"stage5_report.md", "w") as f: f.write(md)

# ── run_manifest ──────────────────────────────────────────────────────────────
try:
    with open(OUT/"run_manifest.json") as f: mf = json.load(f)
    mf["stage5"] = {
        "best_exp": best5_exp, "best_wr": best5_wr,
        "improved": best5_exp is not None,
        "dense_ok": DENSE_OK,
        "dense_models": [k for k, _ in dense_model_keys],
        "experiments": list(experiments5.keys()),
        "timestamp": datetime.now().isoformat()
    }
    with open(OUT/"run_manifest.json", "w") as f: json.dump(mf, f, indent=2)
except Exception as e:
    log.warning(f"run_manifest: {e}")

# ── Final submission check ────────────────────────────────────────────────────
final = pd.read_csv(OUT/"submission.csv")
assert list(final.columns) == ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]
assert final.shape == sample.shape
assert (final["ID"].values == sample["ID"].values).all()
assert final[["TargetRLF1", "TargetR1F1", "TargetLLM"]].notna().all().all()
assert (final["TargetRLF1"] == final["TargetR1F1"]).all()
assert (final["TargetRLF1"] == final["TargetLLM"]).all()
log.info("Final submission.csv checks PASSED")

elapsed_total = time.time() - START
log.info("\n" + "="*60)
log.info("DONE_STAGE5")
log.info(f"Best previous public : 0.703354")
log.info(f"Best previous Val    : {BASELINE_WR:.5f}")
log.info(f"Best Stage5 Val      : {best5_wr:.5f}")
log.info(f"Delta vs Stage4      : {best5_wr-BASELINE_WR:+.5f}")
log.info(f"Best strategy        : {best5_exp or 'None (Stage 4 kept)'}")
log.info(f"Dense active         : {DENSE_OK} {[k for k,_ in dense_model_keys]}")
log.info(f"Final submission     : {OUT}/submission.csv")
log.info(f"Candidate for public : {'YES' if best5_exp else 'NO'}")
log.info(f"Total elapsed        : {elapsed_total:.0f}s")
log.info("="*60)
