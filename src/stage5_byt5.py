"""
Stage 5 — ByT5 contrôlé
Fine-tune google/byt5-small sur Train.csv, évaluer sur Val.csv,
comparer avec baseline retrieval/RRF/MoE.
Activer uniquement si amélioration sur Val.csv.
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

# ── Cache HuggingFace ────────────────────────────────────────────────────────
os.environ["HF_HOME"]                = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"]     = "/tmp/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"]      = "/tmp/hf_cache/datasets"
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
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
log.info("=" * 60)
log.info("STAGE 5 — ByT5 contrôlé")
log.info("=" * 60)

START = time.time()

# ── ROUGE helpers ─────────────────────────────────────────────────────────────
def _tok(text):
    return str(text).lower().split()

def rouge1_f1(ref, hyp):
    r, h = _tok(ref), _tok(hyp)
    if not r or not h: return 0.0
    cnt = {}
    for t in r: cnt[t] = cnt.get(t, 0) + 1
    hits = sum(1 for t in h if cnt.get(t, 0) > 0 and not cnt.__setitem__(t, cnt[t]-1))
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

def rougel_f1(ref, hyp):
    r, h = _tok(ref), _tok(hyp)
    if not r or not h: return 0.0
    lcs = _lcs(r, h)
    p, rv = lcs/len(h), lcs/len(r)
    return 2*p*rv/(p+rv) if p+rv else 0.0

def score_df(refs, preds):
    r1s = [rouge1_f1(str(a), str(b)) for a, b in zip(refs, preds)]
    rls = [rougel_f1(str(a), str(b)) for a, b in zip(refs, preds)]
    r1, rl = float(np.mean(r1s)), float(np.mean(rls))
    return r1, rl, 0.37*r1+0.37*rl, r1s, rls

# ── Load data ─────────────────────────────────────────────────────────────────
log.info("Loading data...")
train  = pd.read_csv(WORK / "Train.csv")
val    = pd.read_csv(WORK / "Val.csv")
test   = pd.read_csv(WORK / "Test.csv")
sample = pd.read_csv(WORK / "SampleSubmission.csv")
val_debug = pd.read_csv(OUT / "val_predictions_debug.csv")

log.info(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

# ── Baseline scores ───────────────────────────────────────────────────────────
baseline_r1 = val_debug["rouge1"].mean()
baseline_rl = val_debug["rougel"].mean()
baseline_wr = 0.37*baseline_r1 + 0.37*baseline_rl
log.info(f"Baseline: WR={baseline_wr:.5f} R1={baseline_r1:.5f} RL={baseline_rl:.5f}")

# ── Report init ───────────────────────────────────────────────────────────────
report = {
    "stage": "stage5_byt5",
    "timestamp": datetime.now().isoformat(),
    "byt5_status": "disabled",
    "baseline_weighted_rouge": baseline_wr,
    "baseline_rouge1": baseline_r1,
    "baseline_rougel": baseline_rl,
    "byt5_alone_wr": None,
    "hybrid_wr": None,
    "best_mode": "baseline",
    "best_wr": baseline_wr,
    "decision": "keep_baseline",
    "error": None,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ByT5 STAGE
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
        DataCollatorForSeq2Seq,
    )
    from torch.utils.data import Dataset as TorchDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Download / load ByT5-small ────────────────────────────────────────────
    MODEL_NAME = "google/byt5-small"
    log.info(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/tmp/hf_cache")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir="/tmp/hf_cache")
    model = model.to(device)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── Prompt builders ───────────────────────────────────────────────────────
    def build_prompt_modeA(subset, question):
        return (
            f"task: answer_health_same_language\n"
            f"subset: {subset}\n"
            f"question: {question}\n"
            f"answer:"
        )

    def build_prompt_modeB(subset, question, retrieved_answer, neighbor_answers=""):
        nb = neighbor_answers[:200] if neighbor_answers else ""
        return (
            f"task: answer_health_same_language\n"
            f"subset: {subset}\n"
            f"question: {question}\n"
            f"retrieved_answer: {retrieved_answer[:200]}\n"
            f"neighbor_answers: {nb}\n"
            f"answer:"
        )

    # ── Training config ───────────────────────────────────────────────────────
    MAX_IN   = 512
    MAX_TGT  = 192
    BSZ      = 2
    GRAD_ACC = 8
    LR       = 5e-5
    EPOCHS   = 1
    MAX_STEPS = 2000   # cap to avoid timeout

    # ── Dataset ───────────────────────────────────────────────────────────────
    class QADataset(TorchDataset):
        def __init__(self, df, tokenizer, max_in, max_tgt):
            self.data = df.reset_index(drop=True)
            self.tok  = tokenizer
            self.max_in  = max_in
            self.max_tgt = max_tgt

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            inp = build_prompt_modeA(row["subset"], str(row["input"]))
            tgt = str(row["output"])
            enc = self.tok(inp,  max_length=self.max_in,  truncation=True, padding=False)
            lab = self.tok(tgt, max_length=self.max_tgt, truncation=True, padding=False)
            enc["labels"] = lab["input_ids"]
            return {k: v for k, v in enc.items()}

    log.info(f"Building train dataset ({len(train)} rows, max_steps={MAX_STEPS})...")
    train_ds = QADataset(train, tokenizer, MAX_IN, MAX_TGT)

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/byt5_ckpt",
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BSZ,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        fp16=(device == "cuda"),
        gradient_checkpointing=(device == "cuda"),
        predict_with_generate=False,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    log.info("Starting ByT5 fine-tuning...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info(f"Training done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    report["training_time_s"] = elapsed
    report["byt5_status"] = "trained"

    # ── Inference helper ──────────────────────────────────────────────────────
    model.eval()

    def generate_answers(prompts, batch_size=8):
        answers = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            enc = tokenizer(
                batch,
                max_length=MAX_IN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_TGT,
                    num_beams=2,
                    do_sample=False,
                )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            answers.extend(decoded)
        return answers

    # ═══ STEP 1: Evaluate on 200-sample Val (stratified) ════════════════════
    log.info("Step 1: ByT5 eval on 200-sample Val (stratified by subset)...")
    SUBSETS = sorted(val["subset"].unique())
    n_per_sub = max(1, 200 // len(SUBSETS))
    sample_val = val.groupby("subset", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n_per_sub), random_state=42)
    ).reset_index(drop=True)
    log.info(f"Sample size: {len(sample_val)}")

    prompts_sample = [
        build_prompt_modeA(row["subset"], str(row["input"]))
        for _, row in sample_val.iterrows()
    ]
    preds_sample = generate_answers(prompts_sample, batch_size=8)

    r1_s, rl_s, wr_s, _, _ = score_df(sample_val["output"].tolist(), preds_sample)
    log.info(f"Sample ByT5-alone: WR={wr_s:.5f} R1={r1_s:.5f} RL={rl_s:.5f}")

    # Quick baseline on same sample
    sample_val_debug = val_debug[val_debug["ID"].isin(sample_val["ID"])].set_index("ID")
    sample_preds_base = [
        sample_val_debug.loc[row["ID"], "prediction"] if row["ID"] in sample_val_debug.index else ""
        for _, row in sample_val.iterrows()
    ]
    r1_base_s, rl_base_s, wr_base_s, _, _ = score_df(
        sample_val["output"].tolist(), sample_preds_base
    )
    log.info(f"Sample Baseline:   WR={wr_base_s:.5f} R1={r1_base_s:.5f} RL={rl_base_s:.5f}")

    SAMPLE_PROMISING = wr_s > wr_base_s * 0.90  # within 10% → proceed to full eval

    if not SAMPLE_PROMISING:
        log.info(f"ByT5 sample score too low ({wr_s:.5f} vs {wr_base_s:.5f}). Skipping full eval.")
        report["byt5_alone_sample_wr"] = wr_s
        report["baseline_sample_wr"]   = wr_base_s
        report["decision"] = "keep_baseline"
    else:
        # ═══ STEP 2: Full Val eval ═══════════════════════════════════════════
        log.info("Step 2: ByT5 full Val eval...")
        val_debug_idx = val_debug.set_index("ID")

        prompts_full = [
            build_prompt_modeA(row["subset"], str(row["input"]))
            for _, row in val.iterrows()
        ]
        preds_full = generate_answers(prompts_full, batch_size=8)

        r1_byt5, rl_byt5, wr_byt5, r1s_byt5, rls_byt5 = score_df(
            val["output"].tolist(), preds_full
        )
        log.info(f"Full ByT5-alone: WR={wr_byt5:.5f} R1={r1_byt5:.5f} RL={rl_byt5:.5f}")
        report["byt5_alone_wr"] = wr_byt5

        # ── Hybrid: use ByT5 only where retrieval confidence is low ───────────
        # confidence threshold: top_neighbor_score < 0.5
        CONF_THRESHOLD = 0.5
        preds_hybrid = []
        val_ids = val["ID"].tolist()
        for i, (vid, byt5_pred) in enumerate(zip(val_ids, preds_full)):
            if vid in val_debug_idx.index:
                score = val_debug_idx.loc[vid, "top_neighbor_score"]
                base_pred = val_debug_idx.loc[vid, "prediction"]
                if float(score) < CONF_THRESHOLD:
                    preds_hybrid.append(byt5_pred)
                else:
                    preds_hybrid.append(base_pred)
            else:
                preds_hybrid.append(byt5_pred)

        r1_hyb, rl_hyb, wr_hyb, r1s_hyb, rls_hyb = score_df(
            val["output"].tolist(), preds_hybrid
        )
        log.info(f"Full Hybrid:     WR={wr_hyb:.5f} R1={r1_hyb:.5f} RL={rl_hyb:.5f}")
        report["hybrid_wr"]   = wr_hyb
        report["hybrid_r1"]   = r1_hyb
        report["hybrid_rl"]   = rl_hyb

        # ── Choose best ───────────────────────────────────────────────────────
        best_wr   = baseline_wr
        best_mode = "baseline"
        best_preds_val = val_debug_idx.reindex(val_ids)["prediction"].tolist()

        if wr_byt5 > best_wr:
            best_wr   = wr_byt5
            best_mode = "byt5_alone"
            best_preds_val = preds_full
        if wr_hyb > best_wr:
            best_wr   = wr_hyb
            best_mode = "hybrid"
            best_preds_val = preds_hybrid

        log.info(f"Best mode: {best_mode} WR={best_wr:.5f}")
        report["best_mode"]   = best_mode
        report["best_wr"]     = best_wr
        report["best_r1"]     = r1_byt5 if best_mode == "byt5_alone" else (r1_hyb if best_mode == "hybrid" else baseline_r1)
        report["best_rl"]     = rl_byt5 if best_mode == "byt5_alone" else (rl_hyb if best_mode == "hybrid" else baseline_rl)

        # Save val predictions debug
        val_byt5_debug = val.copy()
        val_byt5_debug["byt5_prediction"]     = preds_full
        val_byt5_debug["hybrid_prediction"]   = preds_hybrid
        val_byt5_debug["baseline_prediction"] = [
            val_debug_idx.loc[vid, "prediction"] if vid in val_debug_idx.index else ""
            for vid in val_ids
        ]
        val_byt5_debug["byt5_rouge1"]  = r1s_byt5
        val_byt5_debug["byt5_rougel"]  = rls_byt5
        val_byt5_debug["hybrid_rouge1"]= r1s_hyb
        val_byt5_debug["hybrid_rougel"]= rls_hyb
        val_byt5_debug.to_csv(OUT / "byt5_val_predictions_debug.csv", index=False)
        log.info("Saved byt5_val_predictions_debug.csv")

        if best_mode != "baseline":
            report["decision"] = f"use_{best_mode}"

            # ── Generate Test predictions ─────────────────────────────────────
            log.info(f"Generating Test predictions with mode={best_mode}...")

            # Load baseline test predictions
            test_debug = pd.read_csv(OUT / "test_predictions_debug.csv")
            test_debug_idx = test_debug.set_index("ID")

            test_prompts = [
                build_prompt_modeA(row["subset"], str(row["input"]))
                for _, row in test.iterrows()
            ]
            test_preds_byt5 = generate_answers(test_prompts, batch_size=8)

            if best_mode == "byt5_alone":
                final_test_preds = test_preds_byt5
            else:  # hybrid
                final_test_preds = []
                for i, (row, byt5_pred) in enumerate(zip(test.itertuples(), test_preds_byt5)):
                    tid = row.ID
                    if tid in test_debug_idx.index:
                        score = test_debug_idx.loc[tid, "top_neighbor_score"]
                        base_pred = test_debug_idx.loc[tid, "prediction"]
                        if float(score) < CONF_THRESHOLD:
                            final_test_preds.append(byt5_pred)
                        else:
                            final_test_preds.append(base_pred)
                    else:
                        final_test_preds.append(byt5_pred)

            # Save test debug
            test_byt5_debug = test.copy()
            test_byt5_debug["byt5_prediction"]  = test_preds_byt5
            test_byt5_debug["final_prediction"] = final_test_preds
            test_byt5_debug.to_csv(OUT / "byt5_test_predictions_debug.csv", index=False)
            log.info("Saved byt5_test_predictions_debug.csv")

            # ── Build submission ──────────────────────────────────────────────
            sub = sample[["ID"]].copy()
            pred_map = dict(zip(test["ID"], final_test_preds))
            sub["TargetRLF1"] = sub["ID"].map(pred_map)
            sub["TargetR1F1"] = sub["TargetRLF1"]
            sub["TargetLLM"]  = sub["TargetRLF1"]

            sub.to_csv(OUT / "submission_stage5_byt5.csv", index=False)
            log.info("Saved submission_stage5_byt5.csv")

            # Copy as best submission
            sub.to_csv(OUT / "submission.csv", index=False)
            log.info("Updated submission.csv with ByT5 results")

        else:
            report["decision"] = "keep_baseline"
            log.info("ByT5 does not improve Val. Keeping baseline submission.")
            # Restore backup
            import shutil
            shutil.copy(OUT / "submission_backup_before_byt5.csv", OUT / "submission.csv")
            log.info("Restored submission.csv from backup")

except Exception as exc:
    log.error(f"Stage 5 ByT5 FAILED: {exc}")
    log.error(traceback.format_exc())
    report["byt5_status"] = "failed"
    report["error"] = str(exc)
    report["decision"] = "keep_baseline"
    # Restore backup
    import shutil
    shutil.copy(OUT / "submission_backup_before_byt5.csv", OUT / "submission.csv")
    log.info("Restored submission.csv from backup (after error)")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
log.info("Running final submission checks...")
try:
    submission = pd.read_csv(OUT / "submission.csv")
    assert list(submission.columns) == ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"], \
        f"Wrong columns: {list(submission.columns)}"
    assert submission.shape == sample.shape, \
        f"Shape mismatch: {submission.shape} vs {sample.shape}"
    assert (submission["ID"].values == sample["ID"].values).all(), "ID mismatch"
    assert submission[["TargetRLF1", "TargetR1F1", "TargetLLM"]].notna().all().all(), "NaN values"
    assert (submission["TargetRLF1"] == submission["TargetR1F1"]).all(), "TargetRLF1 != TargetR1F1"
    assert (submission["TargetRLF1"] == submission["TargetLLM"]).all(),  "TargetRLF1 != TargetLLM"
    log.info(f"submission.csv checks PASSED. Shape: {submission.shape}")
    report["submission_checks"] = "passed"
except Exception as e:
    log.error(f"Submission check FAILED: {e}")
    report["submission_checks"] = f"FAILED: {e}"

# ── Update validation_scores_all_stages.csv ───────────────────────────────────
try:
    scores_df = pd.read_csv(OUT / "validation_scores_all_stages.csv")
    new_rows = []
    if report.get("byt5_alone_wr"):
        new_rows.append({
            "experiment": "stage5_byt5_alone",
            "rouge1": report.get("best_r1", baseline_r1),
            "rougel": report.get("best_rl", baseline_rl),
            "weighted_rouge": report["byt5_alone_wr"],
        })
    if report.get("hybrid_wr"):
        new_rows.append({
            "experiment": "stage5_byt5_hybrid",
            "rouge1": report.get("hybrid_r1", baseline_r1),
            "rougel": report.get("hybrid_rl", baseline_rl),
            "weighted_rouge": report["hybrid_wr"],
        })
    if new_rows:
        scores_df = pd.concat([scores_df, pd.DataFrame(new_rows)], ignore_index=True)
        scores_df.to_csv(OUT / "validation_scores_all_stages.csv", index=False)
        log.info("Updated validation_scores_all_stages.csv")
except Exception as e:
    log.warning(f"Could not update validation_scores_all_stages.csv: {e}")

# ── Update best_stage.json ────────────────────────────────────────────────────
try:
    with open(OUT / "best_stage.json") as f:
        best = json.load(f)

    if report["best_wr"] > best.get("weighted_rouge_val", 0):
        best["best_experiment"] = f"stage5_{report['best_mode']}"
        best["weighted_rouge_val"] = report["best_wr"]
        if "byt5" in report.get("best_mode", ""):
            best["stage_results"]["stage5_byt5"] = {
                "experiment": f"stage5_{report['best_mode']}",
                "wr": report["best_wr"],
            }
        with open(OUT / "best_stage.json", "w") as f:
            json.dump(best, f, indent=2)
        log.info(f"Updated best_stage.json: {best['best_experiment']} WR={best['weighted_rouge_val']:.5f}")
except Exception as e:
    log.warning(f"Could not update best_stage.json: {e}")

# ── Save ByT5 report ──────────────────────────────────────────────────────────
report["elapsed_s"] = time.time() - START
with open(OUT / "byt5_stage_report.json", "w") as f:
    json.dump(report, f, indent=2)
log.info("Saved byt5_stage_report.json")

# ── Update run_manifest ───────────────────────────────────────────────────────
try:
    with open(OUT / "run_manifest.json") as f:
        manifest = json.load(f)
    manifest["stage5_byt5"] = {
        "status": report["decision"],
        "byt5_status": report["byt5_status"],
        "best_mode": report["best_mode"],
        "best_wr": report["best_wr"],
        "timestamp": report["timestamp"],
    }
    with open(OUT / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
except Exception as e:
    log.warning(f"Could not update run_manifest.json: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("")
log.info("=" * 60)
log.info("DONE_STAGE5_BYT5")
log.info("")
log.info(f"ByT5 status:  {report['byt5_status']}")
log.info(f"Best stage:   {report['best_mode']}")
log.info(f"Best val WR:  {report['best_wr']:.5f}")
log.info(f"Baseline WR:  {report['baseline_weighted_rouge']:.5f}")
log.info(f"Decision:     {report['decision']}")
log.info(f"Final submission: {OUT}/submission.csv")
log.info(f"ByT5 report:      {OUT}/byt5_stage_report.json")
log.info("=" * 60)
