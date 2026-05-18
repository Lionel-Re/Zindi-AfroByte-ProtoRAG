# Stage 5 Report

## Summary
- Baseline (Stage 4 CE_LGB_all): WR = 0.34666
- Dense models active: ['minilm', 'e5s']
- Best Stage 5 experiment: 5A_LGB_rl_heavy
- Best Stage 5 WR: 0.37148 (Δ = +0.02482)

## All experiments

| Experiment | R1 | RL | WR | Δ vs Stage4 |
|---|---|---|---|---|
| Stage4_CE_LGB_all | 0.4974 | 0.4396 | 0.34666 | — |
| 5A_LGB_rl_heavy | 0.5292 | 0.4748 | 0.37148 | +0.02482 |
| 5C_hyb_aka | 0.5297 | 0.4724 | 0.37077 | +0.02411 |
| 5A_LGB_avg | 0.5273 | 0.4725 | 0.36992 | +0.02326 |
| 5A_LGB_comp | 0.5273 | 0.4725 | 0.36992 | +0.02326 |
| 5C_composer | 0.4728 | 0.3234 | 0.29458 | -0.05208 |

## Per-subset (best Stage 5 experiment)

| Subset | WR |
|---|---|
| Aka_Gha | 0.1852 |
| Amh_Eth | 0.1281 |
| Eng_Eth | 0.4687 |
| Eng_Gha | 0.1894 |
| Eng_Ken | 0.5744 |
| Eng_Uga | 0.5901 |
| Lug_Uga | 0.3434 |
| Swa_Ken | 0.4520 |

## Decision
- Stage 5 improved: YES
- submission.csv: updated to 5A_LGB_rl_heavy
