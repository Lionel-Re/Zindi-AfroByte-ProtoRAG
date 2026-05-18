# Stage 9 Report — BGE Reranker for Lug_Uga and Swa_Ken

## Summary
- Baseline: Stage5 routed | Val WR = 0.37563 | Public = 0.732376
- BGE model: `BAAI/bge-reranker-v2-m3`
- Pool: Stage5 (sparse×4 + minilm + e5s, NO e5b)
- BGE scored only for: ['Lug_Uga', 'Swa_Ken']
- Threshold to update: WR ≥ 0.38263

## Stage8 Oracle (context)
- Lug_Uga oracle@5 = 0.508 (+0.142 vs selected)
- Swa_Ken oracle@5 = 0.569 (+0.100 vs selected)

## All experiments

| Experiment | WR | Δ vs Stage5 | Lug_Uga | Swa_Ken |
|---|---|---|---|---|
| Stage5_routed | 0.37563 | — | 0.3657 | 0.4691 |
| 9B_lugswa_rl_heavy | 0.40779 | +0.03216 | 0.5633 | 0.5614 |
| 9_routed | 0.40779 | +0.03216 | 0.5633 | 0.5614 |
| 9B_lugswa_rougel_heavy | 0.40684 | +0.03121 | 0.5575 | 0.5587 |
| 9B_lugswa_balanced | 0.40674 | +0.03111 | 0.5555 | 0.5606 |
| 9C_direct_BGE_top5 | 0.37303 | -0.00260 | 0.3541 | 0.4544 |
| 9C_direct_BGE_top10 | 0.37009 | -0.00554 | 0.3353 | 0.4473 |
| 9A_global_rougel_heavy | 0.36803 | -0.00760 | 0.3460 | 0.4624 |
| 9C_direct_BGE_top20 | 0.36715 | -0.00848 | 0.3165 | 0.4399 |
| 9A_global_balanced | 0.36675 | -0.00888 | 0.3491 | 0.4701 |
| 9A_global_rl_heavy | 0.36565 | -0.00998 | 0.3476 | 0.4689 |

## Routing
Aka_Gha→stage5_routed; Amh_Eth→stage5_routed; Eng_Eth→stage5_routed; Eng_Gha→stage5_routed; Eng_Ken→stage5_routed; Eng_Uga→stage5_routed; Lug_Uga→stage9; Swa_Ken→stage9

## Decision
- Best: 9B_lugswa_rl_heavy WR=0.40779
- Meets threshold: True
- Candidate for public: YES
