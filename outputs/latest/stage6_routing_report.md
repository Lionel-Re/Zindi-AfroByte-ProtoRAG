# Stage 6A Report — e5-base Dense Retrieval

## Summary
- Baseline (Stage5 routed): WR = 0.37563
- Dense models: ['minilm', 'e5s', 'e5b']
- Best Stage6 experiment: 6A_routed
- Best Stage6 WR: 0.37891 (Δ = +0.00328)

## All experiments

| Experiment | R1 | RL | WR | Δ vs Stage5 |
|---|---|---|---|---|
| Stage5_routed_baseline | 0.5346 | 0.4806 | 0.37563 | — |
| 6A_routed | 0.5391 | 0.4850 | 0.37891 | +0.00328 |
| 6A_LGB_balanced | 0.5364 | 0.4812 | 0.37650 | +0.00087 |
| 6A_LGB_comp | 0.5364 | 0.4812 | 0.37650 | +0.00087 |
| 6A_LGB_rl_heavy | 0.5348 | 0.4800 | 0.37548 | -0.00015 |

## Routing by subset

| Subset | Source | WR | Delta vs S5 |
|---|---|---|---|
| Aka_Gha | stage6/6A_LGB_rl_heavy | 0.1883 | +0.0032 |
| Amh_Eth | stage5_routed/stage5_routed | 0.1281 | +0.0000 |
| Eng_Eth | stage5_routed/stage5_routed | 0.4687 | +0.0000 |
| Eng_Gha | stage5_routed/stage5_routed | 0.1894 | +0.0000 |
| Eng_Ken | stage6/6A_LGB_rl_heavy | 0.5841 | +0.0097 |
| Eng_Uga | stage5_routed/stage5_routed | 0.5901 | +0.0000 |
| Lug_Uga | stage5_routed/stage5_routed | 0.3657 | +0.0000 |
| Swa_Ken | stage6/6A_LGB_rl_heavy | 0.4974 | +0.0283 |

## Decision
- Stage6 improved: YES
- submission.csv: updated to 6A_routed
- routing: Aka_Gha→stage6; Amh_Eth→stage5_routed; Eng_Eth→stage5_routed; Eng_Gha→stage5_routed; Eng_Ken→stage6; Eng_Uga→stage5_routed; Lug_Uga→stage5_routed; Swa_Ken→stage6
