"""
Stage 4 — Cross-encoder reranking
Modèle: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingue)
Expériences: CE_top5/10/20 × all / no_Aka_Amh
Stage 4B: CE score + LightGBM
"""
import os, sys, json, time, logging, warnings, traceback, unicodedata
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
log.info("STAGE 4 — Cross-encoder reranking")
log.info("=" * 60)
START = time.time()

BASELINE_LGB_WR = 0.31242476415340303
AKA_AMH = {"Aka_Gha", "Amh_Eth"}   # subsets with recall-limited scores

# ── ROUGE ──────────────────────────────────────────────────────────────────────
def _tok(t, ml=None):
    t2 = str(t).lower().split(); return t2[:ml] if ml else t2

def rouge1_f1(ref, hyp, ml=200):
    r,h=_tok(ref,ml),_tok(hyp,ml)
    if not r or not h: return 0.0
    cnt={}
    for t in r: cnt[t]=cnt.get(t,0)+1
    hits=0
    for t in h:
        if cnt.get(t,0)>0: hits+=1; cnt[t]-=1
    p,rv=hits/len(h),hits/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def _lcs(a,b):
    if not a or not b: return 0
    prev=[0]*(len(b)+1)
    for ai in a:
        cur=[0]*(len(b)+1)
        for j,bj in enumerate(b): cur[j+1]=prev[j]+1 if ai==bj else max(cur[j],prev[j+1])
        prev=cur
    return prev[len(b)]

def rougel_f1(ref, hyp, ml=200):
    r,h=_tok(ref,ml),_tok(hyp,ml)
    if not r or not h: return 0.0
    lcs=_lcs(r,h); p,rv=lcs/len(h),lcs/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def score_rows(refs, preds):
    r1s=[rouge1_f1(str(a),str(b)) for a,b in zip(refs,preds)]
    rls=[rougel_f1(str(a),str(b)) for a,b in zip(refs,preds)]
    r1,rl=float(np.mean(r1s)),float(np.mean(rls))
    return r1,rl,0.37*r1+0.37*rl,r1s,rls

def norm(t): return unicodedata.normalize("NFC",str(t)).lower()

# ── Data ──────────────────────────────────────────────────────────────────────
log.info("Loading data...")
train  = pd.read_csv(WORK/"Train.csv")
val    = pd.read_csv(WORK/"Val.csv")
test   = pd.read_csv(WORK/"Test.csv")
sample = pd.read_csv(WORK/"SampleSubmission.csv")
for df in [train,val,test]: df["input_norm"]=df["input"].apply(norm)
for df in [train,val]:      df["output_norm"]=df["output"].apply(norm)
val_refs=val["output"].tolist(); val_ids=val["ID"].tolist(); val_subs=val["subset"].tolist()
SUBSETS=sorted(train["subset"].unique())
log.info(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

# ── Retrievers ────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer

def make_byte_analyzer(n_min=3,n_max=5):
    def analyzer(s):
        b=s.encode("utf-8"); out=[]
        for n in range(n_min,n_max+1):
            for i in range(len(b)-n+1): out.append(b[i:i+n].hex())
        return out
    return analyzer

CFGS={
    "char_wb": dict(analyzer="char_wb",ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32),
    "char":    dict(analyzer="char",   ngram_range=(3,4),min_df=2,sublinear_tf=True,dtype=np.float32,max_features=150000),
    "word":    dict(analyzer="word",   ngram_range=(1,2),min_df=1,sublinear_tf=True,dtype=np.float32,token_pattern=r"(?u)\b\w+\b"),
    "byte":    dict(analyzer=make_byte_analyzer(3,5),min_df=2,sublinear_tf=True,dtype=np.float32,max_features=200000),
}

log.info("Fitting vectorizers on Train...")
vecs={}
for name,cfg in CFGS.items():
    t0=time.time(); v=TfidfVectorizer(**cfg); v.fit(train["input_norm"])
    log.info(f"  {name}: {time.time()-t0:.1f}s"); vecs[name]=v

log.info("Building per-subset indices...")
sub_i,sub_r={},{}
for name,v in vecs.items():
    sub_i[name]={s:v.transform(train[train["subset"]==s]["input_norm"]) for s in SUBSETS if len(train[train["subset"]==s])>0}
    sub_r[name]={s:train[train["subset"]==s].index.tolist() for s in SUBSETS}

TOP_K_MAX=20  # retrieve top-20 per retriever

def retrieve(q,s,name,topk=TOP_K_MAX):
    v=vecs[name]; X_q=v.transform([q])
    if s in sub_i[name]: X_s,rids=sub_i[name][s],sub_r[name][s]
    else: X_s=v.transform(train["input_norm"]); rids=train.index.tolist()
    sim=(X_q@X_s.T).toarray()[0]; tk=min(topk,len(sim))
    top=np.argpartition(sim,-tk)[-tk:] if len(sim)>tk else np.argsort(sim)[::-1]
    top=top[np.argsort(sim[top])[::-1]]
    return [rids[j] for j in top],sim[top].tolist()

log.info("Retrieving top-20 for Val...")
t0=time.time()
val_ret={name:[] for name in vecs}
for q,s in zip(val["input_norm"],val_subs):
    for name in vecs: val_ret[name].append(retrieve(q,s,name))
log.info(f"Val retrieval done in {time.time()-t0:.1f}s")

# RRF weights (same as V3)
rnames=list(vecs.keys())
_wmap={"char_wb":0.40,"char":0.30,"word":0.20,"byte":0.10}
tw=sum(_wmap[n] for n in rnames)
rrf_w={n:_wmap[n]/tw for n in rnames}

# V3 MoE predictions (baseline for no_Aka_Amh fallback)
v3_subset_best={'Aka_Gha':'char','Amh_Eth':'byte','Eng_Eth':'byte','Eng_Gha':'char_wb',
                'Eng_Ken':'char','Eng_Uga':'char_wb','Lug_Uga':'char','Swa_Ken':'char_wb'}

def v3_moe_pred(i):
    s=val_subs[i]; rn=v3_subset_best.get(s,'char_wb')
    rows,_=val_ret[rn][i]
    return train.loc[rows[0],"output"] if rows else ""

# ── Load Cross-encoder ─────────────────────────────────────────────────────────
import torch
log.info("Loading cross-encoder...")
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
try:
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(CE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
                            cache_folder="/tmp/hf_cache")
    log.info(f"  Loaded {CE_MODEL} on {next(ce_model.model.parameters()).device}")
    CE_OK = True
except Exception as e:
    log.warning(f"  Primary CE failed: {e}. Trying fallback...")
    try:
        CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ce_model = CrossEncoder(CE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu",
                                cache_folder="/tmp/hf_cache")
        log.info(f"  Loaded fallback {CE_MODEL}")
        CE_OK = True
    except Exception as e2:
        log.error(f"  Fallback CE failed: {e2}"); CE_OK = False

# ── Score all top-20 union candidates with CE ─────────────────────────────────
CE_SCORES = {}  # val_idx -> {doc_id -> ce_score}

if CE_OK:
    log.info("Scoring all (query, candidate) pairs with CE...")
    # Collect all unique pairs
    all_pairs = []   # (val_idx, doc_id, query_text, doc_text)
    for i,(q,s) in enumerate(zip(val["input"],val_subs)):
        seen=set()
        for name in rnames:
            rows,_=val_ret[name][i]
            for rid in rows[:TOP_K_MAX]:
                if rid not in seen:
                    seen.add(rid)
                    all_pairs.append((i, rid, str(q), str(train.loc[rid,"output"])))

    log.info(f"  Total pairs: {len(all_pairs)}")
    BATCH=8
    t0=time.time()
    pair_inputs=[(q[:512],d[:512]) for _,_,q,d in all_pairs]
    all_scores=[]
    for start in range(0,len(pair_inputs),BATCH):
        batch=pair_inputs[start:start+BATCH]
        try:
            scores=ce_model.predict(batch,batch_size=BATCH,show_progress_bar=False)
        except RuntimeError:  # OOM
            BATCH=4
            scores=ce_model.predict(batch,batch_size=BATCH,show_progress_bar=False)
        all_scores.extend(scores.tolist() if hasattr(scores,'tolist') else list(scores))
        if (start//BATCH) % 1000 == 0:
            log.info(f"  CE progress: {start}/{len(pair_inputs)} ({time.time()-t0:.0f}s)")

    elapsed=time.time()-t0
    log.info(f"  CE scoring done: {len(all_scores)} scores in {elapsed:.1f}s ({len(all_scores)/elapsed:.0f} pairs/s)")

    for (vi,rid,_,_),sc in zip(all_pairs,all_scores):
        if vi not in CE_SCORES: CE_SCORES[vi]={}
        CE_SCORES[vi][rid]=float(sc)

# ── Build union sets per k ────────────────────────────────────────────────────
def get_union_topk(i, k):
    seen={}
    for name in rnames:
        rows,scs=val_ret[name][i]
        for rank,(rid,sc) in enumerate(zip(rows[:k],scs[:k])):
            if rid not in seen: seen[rid]={"min_rank":rank,"max_sc":sc}
            else:
                seen[rid]["min_rank"]=min(seen[rid]["min_rank"],rank)
                seen[rid]["max_sc"]=max(seen[rid]["max_sc"],sc)
    return seen  # dict rid->info

def ce_best(i, k):
    union=get_union_topk(i,k)
    scores=CE_SCORES.get(i,{})
    best_rid,best_sc=None,-1e9
    for rid in union:
        sc=scores.get(rid,-1e9)
        if sc>best_sc: best_sc=sc; best_rid=rid
    if best_rid is None:
        rows,_=val_ret[rnames[0]][i]; best_rid=rows[0] if rows else 0
    return train.loc[best_rid,"output"],best_rid,best_sc

# ── Evaluate experiments on Val ───────────────────────────────────────────────
log.info("Evaluating experiments on Val...")
experiments={}

if CE_OK:
    for k in [5,10,20]:
        # CE all subsets
        preds=[ce_best(i,k)[0] for i in range(len(val))]
        r1,rl,wr_v,r1s,rls=score_rows(val_refs,preds)
        experiments[f"CE_top{k}_all"]={"preds":preds,"r1":r1,"rl":rl,"wr":wr_v,"r1s":r1s,"rls":rls}
        log.info(f"  CE_top{k}_all: R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f} (baseline={BASELINE_LGB_WR:.5f})")

        # CE no_Aka_Amh: keep V3 MoE for Aka_Gha and Amh_Eth
        preds_mixed=[]
        for i,s in enumerate(val_subs):
            if s in AKA_AMH: preds_mixed.append(v3_moe_pred(i))
            else: preds_mixed.append(ce_best(i,k)[0])
        r1,rl,wr_v,r1s,rls=score_rows(val_refs,preds_mixed)
        experiments[f"CE_top{k}_no_Aka_Amh"]={"preds":preds_mixed,"r1":r1,"rl":rl,"wr":wr_v,"r1s":r1s,"rls":rls}
        log.info(f"  CE_top{k}_no_Aka_Amh: R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f}")

# ── Stage 4B: CE + LightGBM ───────────────────────────────────────────────────
lgb_ce_wr=None
lgb_ce_preds=None

try:
    import lightgbm as lgb
    log.info("Stage 4B: CE + LightGBM...")
    TOP_K_LGB=5

    def build_features_ce(retrievals, query_norms, query_subs, ce_scores_dict, cand_df):
        rows=[]
        for i,(q_norm,s) in enumerate(zip(query_norms,query_subs)):
            q_len=len(q_norm.split())
            cands={}
            for name in rnames:
                r_list,s_list=retrievals[name][i]
                for rank,(rid,sc) in enumerate(zip(r_list[:TOP_K_LGB],s_list[:TOP_K_LGB])):
                    if rid not in cands: cands[rid]={}
                    cands[rid][f"rank_{name}"]=rank
                    cands[rid][f"score_{name}"]=sc
            rrf_sc={}
            for name in rnames:
                w=rrf_w[name]; r_list,_=retrievals[name][i]
                for rank,rid in enumerate(r_list[:TOP_K_LGB]):
                    rrf_sc[rid]=rrf_sc.get(rid,0.0)+w/(60+rank)
            q_ce=ce_scores_dict.get(i,{})
            for rid,feat in cands.items():
                feat["query_idx"]=i; feat["doc_id"]=rid
                feat["rrf_score"]=rrf_sc.get(rid,0.0)
                feat["ce_score"]=q_ce.get(rid,-999.0)
                feat["same_subset"]=int(cand_df.loc[rid,"subset"]==s)
                feat["subset_id"]=SUBSETS.index(s) if s in SUBSETS else -1
                feat["query_len"]=q_len
                feat["answer_len"]=len(str(cand_df.loc[rid,"output"]).split())
                rows.append(feat)
        return pd.DataFrame(rows).fillna(0)

    log.info("  Building CE+LGB features for Val...")
    t0=time.time()
    feat_df=build_features_ce(val_ret, val["input_norm"].tolist(), val_subs, CE_SCORES, train)
    log.info(f"  Features: {len(feat_df)} rows in {time.time()-t0:.1f}s")

    # ROUGE targets (capped at 200 words)
    log.info("  Computing ROUGE targets...")
    t0=time.time()
    doc_ids=feat_df["doc_id"].astype(int).values
    q_idxs=feat_df["query_idx"].astype(int).values
    train_outs=train["output"].values
    r1s_t=[rouge1_f1(val_refs[qi],train_outs[di]) for qi,di in zip(q_idxs,doc_ids)]
    rls_t=[rougel_f1(val_refs[qi],train_outs[di]) for qi,di in zip(q_idxs,doc_ids)]
    feat_df["target"]=[0.5*r1+0.5*rl for r1,rl in zip(r1s_t,rls_t)]
    log.info(f"  Targets computed in {time.time()-t0:.1f}s")

    feat_cols=[c for c in feat_df.columns if c not in ("query_idx","doc_id","target")]
    X=feat_df[feat_cols].values; y=feat_df["target"].values
    n_q=len(val); split=int(n_q*0.8)
    tr_m=feat_df["query_idx"].values<split; vl_m=~tr_m

    dtrain_lgb=lgb.Dataset(X[tr_m],label=y[tr_m])
    dvalid_lgb=lgb.Dataset(X[vl_m],label=y[vl_m])
    params=dict(objective="regression",metric="rmse",num_leaves=31,learning_rate=0.1,
                verbosity=-1,random_state=42,num_threads=4)

    log.info("  Training CE+LGB model...")
    t0=time.time()
    model_lgb=lgb.train(params,dtrain_lgb,num_boost_round=200,valid_sets=[dvalid_lgb],
                        callbacks=[lgb.early_stopping(15,verbose=False),lgb.log_evaluation(-1)])
    log.info(f"  LGB trained in {time.time()-t0:.1f}s, iter={model_lgb.best_iteration}")

    # Predict CE+LGB scores
    feat_df["lgb_score"]=model_lgb.predict(feat_df[feat_cols].values)
    q_idx_arr=feat_df["query_idx"].values; doc_id_arr=feat_df["doc_id"].astype(int).values
    lgb_score_arr=feat_df["lgb_score"].values

    lgb_ce_preds=[]
    for i in range(n_q):
        mask=q_idx_arr==i
        if not mask.any(): lgb_ce_preds.append(v3_moe_pred(i))
        else:
            best_doc=int(doc_id_arr[mask][np.argmax(lgb_score_arr[mask])])
            lgb_ce_preds.append(train.loc[best_doc,"output"])

    r1,rl,lgb_ce_wr,r1s_ce,rls_ce=score_rows(val_refs,lgb_ce_preds)
    experiments["CE_LGB_all"]={"preds":lgb_ce_preds,"r1":r1,"rl":rl,"wr":lgb_ce_wr,"r1s":r1s_ce,"rls":rls_ce}
    log.info(f"  CE_LGB_all: R1={r1:.4f} RL={rl:.4f} WR={lgb_ce_wr:.5f}")

    # CE+LGB no_Aka_Amh
    preds_ce_lgb_mixed=[]
    for i,s in enumerate(val_subs):
        if s in AKA_AMH: preds_ce_lgb_mixed.append(v3_moe_pred(i))
        else: preds_ce_lgb_mixed.append(lgb_ce_preds[i])
    r1,rl,wr_v,_,_=score_rows(val_refs,preds_ce_lgb_mixed)
    experiments["CE_LGB_no_Aka_Amh"]={"preds":preds_ce_lgb_mixed,"r1":r1,"rl":rl,"wr":wr_v}
    log.info(f"  CE_LGB_no_Aka_Amh: R1={r1:.4f} RL={rl:.4f} WR={wr_v:.5f}")

except Exception as e:
    log.warning(f"Stage 4B CE+LGB failed: {e}\n{traceback.format_exc()}")

# ── Select best experiment ────────────────────────────────────────────────────
best_exp=None; best_wr=BASELINE_LGB_WR
for exp,sc in experiments.items():
    if sc["wr"]>best_wr:
        best_wr=sc["wr"]; best_exp=exp

log.info(f"\nBest experiment: {best_exp} WR={best_wr:.5f} (V3 LGB={BASELINE_LGB_WR:.5f})")

if best_exp is None:
    log.info("CE does NOT improve V3 LGB. Restoring backup.")
    import shutil; shutil.copy(OUT/"submission_backup_before_crossencoder.csv", OUT/"submission.csv")
    final_preds_val=None
else:
    final_preds_val=experiments[best_exp]["preds"]
    # ── Test retrieval ────────────────────────────────────────────────────────
    log.info("Refitting on Train+Val for Test...")
    base_df=pd.concat([train,val],ignore_index=True).reset_index(drop=True)
    for df in [base_df,test]: df["input_norm"]=df["input"].apply(norm)
    base_df["output_norm"]=base_df["output"].apply(norm)

    refit={};  rsub_i={}; rsub_r={}
    for name,cfg in CFGS.items():
        t0=time.time(); v2=TfidfVectorizer(**cfg); v2.fit(base_df["input_norm"])
        refit[name]=v2
        rsub_i[name]={s:v2.transform(base_df[base_df["subset"]==s]["input_norm"])
                      for s in SUBSETS if len(base_df[base_df["subset"]==s])>0}
        rsub_r[name]={s:base_df[base_df["subset"]==s].index.tolist() for s in SUBSETS}
        log.info(f"  refit {name} {time.time()-t0:.1f}s")

    def retrieve_test(q,s,name,topk=TOP_K_MAX):
        v=refit[name]; X_q=v.transform([q])
        if s in rsub_i[name]: X_s,rids=rsub_i[name][s],rsub_r[name][s]
        else: X_s=v.transform(base_df["input_norm"]); rids=base_df.index.tolist()
        sim=(X_q@X_s.T).toarray()[0]; tk=min(topk,len(sim))
        top=np.argpartition(sim,-tk)[-tk:] if len(sim)>tk else np.argsort(sim)[::-1]
        top=top[np.argsort(sim[top])[::-1]]
        return [rids[j] for j in top],sim[top].tolist()

    log.info("Test retrieval...")
    test_ret={name:[] for name in refit}
    for q,s in zip(test["input_norm"],test["subset"]):
        for name in refit: test_ret[name].append(retrieve_test(q,s,name))
    log.info("  Test retrieval done")

    # CE scoring for test
    log.info("Scoring test pairs with CE...")
    test_pairs=[]; test_ce={}
    for i,(q,s) in enumerate(zip(test["input"],test["subset"])):
        seen=set()
        for name in refit:
            rows,_=test_ret[name][i]
            for rid in rows[:TOP_K_MAX]:
                if rid not in seen:
                    seen.add(rid)
                    test_pairs.append((i,rid,str(q),str(base_df.loc[rid,"output"])))

    log.info(f"  Test pairs: {len(test_pairs)}")
    t0=time.time()
    test_inputs=[(q[:512],d[:512]) for _,_,q,d in test_pairs]
    test_scores_list=[]
    BATCH_CE=8
    for start in range(0,len(test_inputs),BATCH_CE):
        batch=test_inputs[start:start+BATCH_CE]
        try: sc=ce_model.predict(batch,batch_size=BATCH_CE,show_progress_bar=False)
        except RuntimeError: BATCH_CE=4; sc=ce_model.predict(batch,batch_size=BATCH_CE,show_progress_bar=False)
        test_scores_list.extend(sc.tolist() if hasattr(sc,'tolist') else list(sc))
        if (start//BATCH_CE)%500==0: log.info(f"  Test CE: {start}/{len(test_inputs)} ({time.time()-t0:.0f}s)")
    for (ti,rid,_,_),sc in zip(test_pairs,test_scores_list):
        if ti not in test_ce: test_ce[ti]={}
        test_ce[ti][rid]=float(sc)
    log.info(f"  Test CE done in {time.time()-t0:.1f}s")

    def get_test_union(i,k):
        seen={}
        for name in refit:
            rows,scs=test_ret[name][i]
            for rank,(rid,sc) in enumerate(zip(rows[:k],scs[:k])):
                if rid not in seen: seen[rid]={"min_rank":rank,"max_sc":sc}
        return seen

    def ce_best_test(i,k):
        union=get_test_union(i,k); scores=test_ce.get(i,{})
        best_rid,best_sc=None,-1e9
        for rid in union:
            sc=scores.get(rid,-1e9)
            if sc>best_sc: best_sc=sc; best_rid=rid
        if best_rid is None:
            rows,_=test_ret[list(refit.keys())[0]][i]; best_rid=rows[0] if rows else 0
        return base_df.loc[best_rid,"output"],best_rid

    # Determine test prediction strategy from best_exp
    k_str=[c for c in ["top5","top10","top20"] if c in best_exp]
    best_k=int(k_str[0].replace("top","")) if k_str else 10

    # V3 MoE fallback for test (Aka_Amh subsets if no_Aka_Amh variant)
    test_v3_subset_best={'Aka_Gha':'char','Amh_Eth':'byte','Eng_Eth':'byte','Eng_Gha':'char_wb',
                         'Eng_Ken':'char','Eng_Uga':'char_wb','Lug_Uga':'char','Swa_Ken':'char_wb'}
    def test_v3_pred(i):
        s=test["subset"].iloc[i]; rn=test_v3_subset_best.get(s,'char_wb')
        rows,_=test_ret[rn][i]; return base_df.loc[rows[0],"output"] if rows else ""

    if "LGB" in best_exp:
        # CE + LGB for test
        log.info(f"  Using CE+LGB (k={best_k}) for Test...")
        test_feat=build_features_ce(test_ret, test["input_norm"].tolist(), test["subset"].tolist(),
                                    test_ce, base_df)
        # Align columns
        tq=test_feat["query_idx"].values.copy(); td=test_feat["doc_id"].astype(int).values.copy()
        for c in feat_cols:
            if c not in test_feat.columns: test_feat[c]=0
        X_test=test_feat[feat_cols].values
        ts=model_lgb.predict(X_test)
        test_preds=[]
        for i in range(len(test)):
            mask=tq==i
            if not mask.any(): test_preds.append(test_v3_pred(i))
            else:
                best_doc=int(td[mask][np.argmax(ts[mask])])
                test_preds.append(base_df.loc[best_doc,"output"])
        if "no_Aka_Amh" in best_exp:
            for i,s in enumerate(test["subset"].tolist()):
                if s in AKA_AMH: test_preds[i]=test_v3_pred(i)
    else:
        log.info(f"  Using CE_top{best_k} for Test...")
        test_preds=[]
        for i,s in enumerate(test["subset"].tolist()):
            if "no_Aka_Amh" in best_exp and s in AKA_AMH:
                test_preds.append(test_v3_pred(i))
            else:
                pred,_=ce_best_test(i,best_k)
                test_preds.append(pred)

    # Build submission
    sub=sample[["ID"]].copy()
    sub["TargetRLF1"]=sub["ID"].map(dict(zip(test["ID"],test_preds)))
    sub["TargetR1F1"]=sub["TargetRLF1"]; sub["TargetLLM"]=sub["TargetRLF1"]

    assert list(sub.columns)==["ID","TargetRLF1","TargetR1F1","TargetLLM"]
    assert sub.shape==sample.shape
    assert (sub["ID"].values==sample["ID"].values).all()
    assert sub[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
    assert (sub["TargetRLF1"]==sub["TargetR1F1"]).all()
    assert (sub["TargetRLF1"]==sub["TargetLLM"]).all()
    log.info("Submission checks PASSED")

    sub.to_csv(OUT/"submission_stage4_crossencoder.csv", index=False)
    sub.to_csv(OUT/"submission.csv", index=False)
    log.info(f"submission.csv updated: WR {BASELINE_LGB_WR:.5f} → {best_wr:.5f}")

# ── Per-subset scores ──────────────────────────────────────────────────────────
sub_rows=[]
for s in SUBSETS:
    mask=[i for i,ss in enumerate(val_subs) if ss==s]
    if not mask: continue
    srefs=[val_refs[i] for i in mask]
    for exp,sc in experiments.items():
        sp=[sc["preds"][i] for i in mask]
        r1,rl,wr_v,_,_=score_rows(srefs,sp)
        sub_rows.append({"experiment":exp,"subset":s,"n":len(mask),"rouge1":r1,"rougel":rl,"weighted_rouge":wr_v})

pd.DataFrame(sub_rows).to_csv(OUT/"crossencoder_validation_scores_by_subset.csv",index=False)

# Global scores CSV
rows_g=[{"experiment":"V3_LGB_baseline","rouge1":0.4524,"rougel":0.3920,"weighted_rouge":BASELINE_LGB_WR}]
for exp,sc in experiments.items():
    rows_g.append({"experiment":exp,"rouge1":sc["r1"],"rougel":sc["rl"],"weighted_rouge":sc["wr"]})
pd.DataFrame(rows_g).to_csv(OUT/"crossencoder_validation_scores.csv",index=False)

# Val predictions debug (best exp or V3)
if best_exp and final_preds_val:
    val_debug=val.copy()
    val_debug["prediction"]=final_preds_val
    val_debug["experiment"]=best_exp
    val_debug["rouge1"]=[rouge1_f1(r,p) for r,p in zip(val_refs,final_preds_val)]
    val_debug["rougel"]=[rougel_f1(r,p) for r,p in zip(val_refs,final_preds_val)]
    val_debug.to_csv(OUT/"crossencoder_val_predictions_debug.csv",index=False)

# ── Update best_stage.json, run_manifest, validation_scores ───────────────────
try:
    with open(OUT/"best_stage.json") as f: bs=json.load(f)
    if best_wr>bs.get("weighted_rouge_val",0):
        bs["best_experiment"]=best_exp or "V3_LGB"
        bs["weighted_rouge_val"]=best_wr
        bs["rouge1_val"]=experiments[best_exp]["r1"] if best_exp else bs.get("rouge1_val")
        bs["rougel_val"]=experiments[best_exp]["rl"] if best_exp else bs.get("rougel_val")
        bs.setdefault("stage_results",{})["stage4_ce"]={"experiment":best_exp,"wr":best_wr}
        with open(OUT/"best_stage.json","w") as f: json.dump(bs,f,indent=2)
except Exception as e: log.warning(f"best_stage.json: {e}")

try:
    sc_df=pd.read_csv(OUT/"validation_scores_all_stages.csv")
    new=[]
    for exp,sc in experiments.items():
        if not any(sc_df["experiment"]==exp):
            new.append({"experiment":exp,"rouge1":sc["r1"],"rougel":sc["rl"],"weighted_rouge":sc["wr"]})
    if new: sc_df=pd.concat([sc_df,pd.DataFrame(new)],ignore_index=True)
    sc_df.to_csv(OUT/"validation_scores_all_stages.csv",index=False)
except Exception as e: log.warning(f"validation_scores: {e}")

try:
    with open(OUT/"run_manifest.json") as f: mf=json.load(f)
    mf["stage4_ce"]={"best_exp":best_exp,"best_wr":best_wr,"improved":best_exp is not None,
                     "experiments_tried":list(experiments.keys()),"timestamp":datetime.now().isoformat()}
    with open(OUT/"run_manifest.json","w") as f: json.dump(mf,f,indent=2)
except Exception as e: log.warning(f"run_manifest: {e}")

# ── Final check ───────────────────────────────────────────────────────────────
final=pd.read_csv(OUT/"submission.csv")
assert list(final.columns)==["ID","TargetRLF1","TargetR1F1","TargetLLM"]
assert final.shape==sample.shape
assert (final["ID"].values==sample["ID"].values).all()
assert final[["TargetRLF1","TargetR1F1","TargetLLM"]].notna().all().all()
assert (final["TargetRLF1"]==final["TargetR1F1"]).all()
assert (final["TargetRLF1"]==final["TargetLLM"]).all()
log.info("Final submission.csv checks PASSED")

log.info("\n" + "=" * 60)
log.info("DONE_STAGE4_CE")
log.info(f"V3 LGB baseline WR: {BASELINE_LGB_WR:.5f}")
log.info(f"Best CE experiment: {best_exp}")
log.info(f"Best val WR:        {best_wr:.5f}")
log.info(f"Delta vs V3 LGB:    {best_wr-BASELINE_LGB_WR:+.5f}")
log.info(f"CE improved:        {best_exp is not None}")
log.info(f"Final submission:   {OUT}/submission.csv")
log.info("=" * 60)

# Print experiments summary
log.info("\nAll experiments:")
for exp in sorted(experiments,key=lambda e:experiments[e]["wr"],reverse=True):
    flag="★" if exp==best_exp else " "
    log.info(f"  {flag} {exp:35s}: WR={experiments[exp]['wr']:.5f}")
