# Stage 7 Report — AfroLM Semantic Reranking

## Summary
- Baseline (Stage5 routed): WR = 0.37563 | Public = 0.732376
- Stage6 e5-base: REJECTED (public = 0.729871)
- AfroLM model used: `bonadossou/afrolm_active_learning`
- Pool: Stage5 (sparse × 4 + minilm + e5s, NO e5b)
- CE scores: reused from Stage6 cache
- Threshold to update submission: WR ≥ 0.38263 (+0.007)

## All experiments

| Experiment | R1 | RL | WR | Δ vs Stage5 | Meets threshold |
|---|---|---|---|---|---|
| Stage5_routed | 0.5346 | 0.4806 | 0.37563 | — | baseline |
| 7_routed | 0.5353 | 0.4815 | 0.37620 | +0.00057 | NO |
| 7_safe | 0.5348 | 0.4794 | 0.37526 | -0.00037 | NO |
| 7_semantic | 0.5305 | 0.4748 | 0.37195 | -0.00368 | NO |
| 7_rougel_semantic | 0.5288 | 0.4735 | 0.37086 | -0.00477 | NO |
| 7_rl_heavy | 0.5207 | 0.4653 | 0.36484 | -0.01079 | NO |

## Routing by subset

| Subset | Source | WR | Delta |
|---|---|---|---|
| Aka_Gha | stage5_routed/stage5_routed | 0.1852 | +0.0000 |
| Amh_Eth | stage5_routed/stage5_routed | 0.1281 | +0.0000 |
| Eng_Eth | stage7/7_safe | 0.4755 | +0.0069 |
| Eng_Gha | stage5_routed/stage5_routed | 0.1894 | +0.0000 |
| Eng_Ken | stage5_routed/stage5_routed | 0.5744 | +0.0000 |
| Eng_Uga | stage5_routed/stage5_routed | 0.5901 | +0.0000 |
| Lug_Uga | stage5_routed/stage5_routed | 0.3657 | +0.0000 |
| Swa_Ken | stage5_routed/stage5_routed | 0.4691 | +0.0000 |

## Decision
- Threshold met: False
- submission.csv: kept Stage5 (threshold not met)
- Candidate for public submission: NO
