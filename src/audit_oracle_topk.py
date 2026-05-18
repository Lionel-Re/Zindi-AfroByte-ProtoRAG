"""
Audit oracle top-k : évalue le plafond WR si le reranker était parfait.
Pour chaque query Val, prend l'union top-k des retrievers actifs,
sélectionne le candidat avec le meilleur 0.5*rouge1+0.5*rougel.
"""
import unicodedata, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

WORK = Path("/home/onyxia/work")
OUT  = WORK / "outputs" / "latest"

def norm(t): return unicodedata.normalize("NFC", str(t)).lower()

def _tok(t, ml=None):
    t2 = str(t).lower().split()
    return t2[:ml] if ml else t2

def rouge1(ref, hyp, ml=200):
    r,h = _tok(ref,ml), _tok(hyp,ml)
    if not r or not h: return 0.0
    cnt = {}
    for t in r: cnt[t] = cnt.get(t,0)+1
    hits = 0
    for t in h:
        if cnt.get(t,0)>0: hits+=1; cnt[t]-=1
    p,rv = hits/len(h), hits/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def _lcs(a,b):
    if not a or not b: return 0
    prev=[0]*(len(b)+1)
    for ai in a:
        cur=[0]*(len(b)+1)
        for j,bj in enumerate(b):
            cur[j+1]=prev[j]+1 if ai==bj else max(cur[j],prev[j+1])
        prev=cur
    return prev[len(b)]

def rougel(ref, hyp, ml=200):
    r,h = _tok(ref,ml), _tok(hyp,ml)
    if not r or not h: return 0.0
    lcs=_lcs(r,h); p,rv=lcs/len(h),lcs/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def wr(r1s, rls): return 0.37*np.mean(r1s)+0.37*np.mean(rls)

print("Loading data...")
train = pd.read_csv(WORK/"Train.csv")
val   = pd.read_csv(WORK/"Val.csv")
for df in [train, val]: df["input_norm"] = df["input"].apply(norm)

val_refs  = val["output"].tolist()
val_subs  = val["subset"].tolist()
SUBSETS   = sorted(train["subset"].unique())

def make_byte_analyzer(n_min=3, n_max=5):
    def analyzer(s):
        b = s.encode("utf-8"); out=[]
        for n in range(n_min,n_max+1):
            for i in range(len(b)-n+1): out.append(b[i:i+n].hex())
        return out
    return analyzer

CFGS = {
    "char_wb": dict(analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True, dtype=np.float32),
    "char":    dict(analyzer="char",    ngram_range=(3,4), min_df=2, sublinear_tf=True, dtype=np.float32, max_features=150000),
    "word":    dict(analyzer="word",    ngram_range=(1,2), min_df=1, sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w+\b"),
    "byte":    dict(analyzer=make_byte_analyzer(3,5), min_df=2, sublinear_tf=True, dtype=np.float32, max_features=200000),
}

TOP_K_MAX = 50

print("Fitting vectorizers...")
vecs = {}
for name, cfg in CFGS.items():
    t0=time.time(); v=TfidfVectorizer(**cfg); v.fit(train["input_norm"])
    print(f"  {name}: {time.time()-t0:.1f}s vocab={len(v.vocabulary_)}")
    vecs[name] = v

print("Building per-subset indices...")
sub_i, sub_r = {}, {}
for name, v in vecs.items():
    sub_i[name]={s: v.transform(train[train["subset"]==s]["input_norm"])
                 for s in SUBSETS if len(train[train["subset"]==s])>0}
    sub_r[name]={s: train[train["subset"]==s].index.tolist() for s in SUBSETS}

def retrieve(q, s, name, topk):
    v = vecs[name]
    X_q = v.transform([q])
    if s in sub_i[name]:
        X_s, rids = sub_i[name][s], sub_r[name][s]
    else:
        X_s = v.transform(train["input_norm"]); rids = train.index.tolist()
    sim = (X_q @ X_s.T).toarray()[0]
    tk = min(topk, len(sim))
    top = np.argpartition(sim,-tk)[-tk:] if len(sim)>tk else np.argsort(sim)[::-1]
    top = top[np.argsort(sim[top])[::-1]]
    return [rids[j] for j in top], sim[top].tolist()

print("Retrieving top-50 for Val...")
t0=time.time()
retrievals = {name:[] for name in vecs}
for q, s in zip(val["input_norm"], val_subs):
    for name in vecs:
        retrievals[name].append(retrieve(q, s, name, TOP_K_MAX))
print(f"  done in {time.time()-t0:.1f}s")

# Current WR (LGB best = MoE fallback: pick best retriever per subset from V3 results)
# Use V3 subset_best mapping from report
v3_subset_best = {'Aka_Gha':'E_char_top1','Amh_Eth':'E_byte_top1','Eng_Eth':'E_byte_top1',
                  'Eng_Gha':'E_rrf','Eng_Ken':'E_char_top1','Eng_Uga':'E_char_wb_top1',
                  'Lug_Uga':'E_char_top1','Swa_Ken':'E_char_wb_top1'}
retriever_map = {'E_char_top1':'char','E_byte_top1':'byte','E_char_wb_top1':'char_wb','E_rrf':'char_wb'}

current_preds=[]
for i, s in enumerate(val_subs):
    exp = v3_subset_best.get(s, 'E_char_wb_top1')
    rn  = retriever_map.get(exp, 'char_wb')
    rows, _ = retrievals[rn][i]
    current_preds.append(train.loc[rows[0], "output"] if rows else "")
cur_r1s=[rouge1(ref,pred) for ref,pred in zip(val_refs,current_preds)]
cur_rls=[rougel(ref,pred) for ref,pred in zip(val_refs,current_preds)]
current_wr = wr(cur_r1s, cur_rls)
print(f"Current (MoE) WR = {current_wr:.5f}")

# Oracle at each k
Ks = [1, 5, 10, 20, 50]
oracle_results = {}

print("Computing oracle scores...")
t0=time.time()
oracle_by_query = {k: {"r1":[], "rl":[]} for k in Ks}

for i, (ref, s) in enumerate(zip(val_refs, val_subs)):
    # union of all candidates up to top-50
    all_cands = {}
    for name in vecs:
        rows, scs = retrievals[name][i]
        for rank, (rid, sc) in enumerate(zip(rows, scs)):
            if rid not in all_cands:
                all_cands[rid] = {"min_rank": rank, "max_score": sc}
            else:
                all_cands[rid]["min_rank"] = min(all_cands[rid]["min_rank"], rank)
                all_cands[rid]["max_score"] = max(all_cands[rid]["max_score"], sc)

    # Precompute ROUGE for all unique candidates
    rouge_cache = {}
    for rid in all_cands:
        pred = train.loc[rid, "output"]
        r1 = rouge1(ref, pred); rl = rougel(ref, pred)
        rouge_cache[rid] = (r1, rl, 0.5*r1+0.5*rl)

    # Oracle at each k: union of top-k per retriever
    for k in Ks:
        union_k = set()
        for name in vecs:
            rows, _ = retrievals[name][i]
            union_k.update(rows[:k])
        # Pick best in union
        best_r1, best_rl = 0.0, 0.0
        for rid in union_k:
            if rid in rouge_cache:
                r1, rl, _ = rouge_cache[rid]
                if 0.5*r1+0.5*rl > 0.5*best_r1+0.5*best_rl:
                    best_r1, best_rl = r1, rl
        oracle_by_query[k]["r1"].append(best_r1)
        oracle_by_query[k]["rl"].append(best_rl)

print(f"  oracle computed in {time.time()-t0:.1f}s")

# Global oracle WR per k
oracle_wrs = {}
for k in Ks:
    r1s = oracle_by_query[k]["r1"]
    rls = oracle_by_query[k]["rl"]
    oracle_wrs[k] = wr(r1s, rls)
    print(f"  Oracle top-{k:2d}: WR={oracle_wrs[k]:.5f}  gap vs current={oracle_wrs[k]-current_wr:+.5f}")

# Per-subset oracle (top-20)
print("\nOracle top-20 by subset:")
subset_rows = []
for s in SUBSETS:
    mask = [i for i,ss in enumerate(val_subs) if ss==s]
    if not mask: continue
    sub_r1s_cur = [cur_r1s[i] for i in mask]
    sub_rls_cur = [cur_rls[i] for i in mask]
    cur_wr_s = wr(sub_r1s_cur, sub_rls_cur)
    for k in [5, 10, 20]:
        sub_r1s_or = [oracle_by_query[k]["r1"][i] for i in mask]
        sub_rls_or = [oracle_by_query[k]["rl"][i] for i in mask]
        or_wr_s = wr(sub_r1s_or, sub_rls_or)
        subset_rows.append({"subset":s,"k":k,"n":len(mask),
                            "current_wr":cur_wr_s,"oracle_wr":or_wr_s,
                            "gap":or_wr_s-cur_wr_s})
    print(f"  {s}: current={cur_wr_s:.4f}  oracle@5={wr([oracle_by_query[5]['r1'][i] for i in mask],[oracle_by_query[5]['rl'][i] for i in mask]):.4f}  oracle@20={wr([oracle_by_query[20]['r1'][i] for i in mask],[oracle_by_query[20]['rl'][i] for i in mask]):.4f}")

# Save CSV
rows = []
for k in Ks:
    r1s = oracle_by_query[k]["r1"]
    rls = oracle_by_query[k]["rl"]
    rows.append({"k":k,"oracle_wr":oracle_wrs[k],"oracle_r1":float(np.mean(r1s)),
                 "oracle_rl":float(np.mean(rls)),"gap_vs_current":oracle_wrs[k]-current_wr,
                 "relative_gain_pct":(oracle_wrs[k]-current_wr)/current_wr*100})
rows.insert(0,{"k":0,"oracle_wr":current_wr,"oracle_r1":float(np.mean(cur_r1s)),
               "oracle_rl":float(np.mean(cur_rls)),"gap_vs_current":0.0,"relative_gain_pct":0.0})
df_oracle = pd.DataFrame(rows)
df_oracle.to_csv(OUT/"audit_oracle_topk.csv", index=False)

pd.DataFrame(subset_rows).to_csv(OUT/"audit_oracle_topk_by_subset.csv", index=False)

# Interpretation
gap_5  = oracle_wrs[5]  - current_wr
gap_20 = oracle_wrs[20] - current_wr
gap_50 = oracle_wrs[50] - current_wr

if gap_5 / max(gap_20, 1e-9) > 0.7:
    reco = "cross-encoder (gain concentré dans top-5, reranking vaut plus)"
elif gap_50 / max(gap_5, 1e-9) > 2.5:
    reco = "dense retrieval (beaucoup de gain au-delà du top-5, recall limité)"
else:
    reco = "cross-encoder + dense retrieval (gain distribué, les deux aident)"

# Per-subset dense vs cross-encoder signal
dense_subsets = [r["subset"] for r in subset_rows
                 if r["k"]==5 and r["gap"]<(oracle_wrs[20]-oracle_wrs[5])/4]

md = f"""# Audit Oracle Top-k

## Scores globaux (Val, n={len(val)})

| k | oracle_WR | gap vs current |
|---|-----------|----------------|
| current | {current_wr:.5f} | — |
"""
for k in Ks:
    md += f"| top-{k} | {oracle_wrs[k]:.5f} | +{oracle_wrs[k]-current_wr:.5f} |\n"

md += f"""
## Interprétation

- Gain oracle@5 vs current : **+{gap_5:.5f}** ({gap_5/current_wr*100:.1f}%)
- Gain oracle@20 vs current : **+{gap_20:.5f}** ({gap_20/current_wr*100:.1f}%)
- Gain oracle@50 vs current : **+{gap_50:.5f}** ({gap_50/current_wr*100:.1f}%)

**Signal :**
- Si gain concentré en top-5 → reranker (cross-encoder) limiterait le plafond
- Si gain croît fortement au-delà de top-5 → recall insuffisant (dense retrieval)

**Recommandation : {reco}**

## Par subset (oracle@20 vs current)
"""
for s in SUBSETS:
    sr = [r for r in subset_rows if r["subset"]==s and r["k"]==20]
    if sr: md += f"- {s}: current={sr[0]['current_wr']:.4f} → oracle@20={sr[0]['oracle_wr']:.4f} (+{sr[0]['gap']:.4f})\n"

with open(OUT/"audit_oracle_interpretation.md","w") as f: f.write(md)

print(f"\n{'='*50}")
print(f"current WR   = {current_wr:.5f}")
print(f"oracle top5  = {oracle_wrs[5]:.5f}  (+{oracle_wrs[5]-current_wr:.5f})")
print(f"oracle top10 = {oracle_wrs[10]:.5f}  (+{oracle_wrs[10]-current_wr:.5f})")
print(f"oracle top20 = {oracle_wrs[20]:.5f}  (+{oracle_wrs[20]-current_wr:.5f})")
print(f"oracle top50 = {oracle_wrs[50]:.5f}  (+{oracle_wrs[50]-current_wr:.5f})")
print(f"Recommandation: {reco}")
print(f"Files saved: audit_oracle_topk.csv, audit_oracle_topk_by_subset.csv, audit_oracle_interpretation.md")
