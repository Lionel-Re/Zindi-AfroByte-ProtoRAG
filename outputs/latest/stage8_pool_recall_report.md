# Stage 8 — Pool Recall Diagnosis Report

Generated: 2026-05-18 02:15 UTC
Baseline: Stage5 routed | Val WR = 0.37563 | Public = 0.732376

## 1. Global Oracle Table

| K | oracle_WR | oracle_WR(CE) | gap_vs_selected | hit@thr=0.30 | hit@thr=0.40 | hit@thr=0.50 |
|---|-----------|--------------|-----------------|--------------|--------------|--------------|
| 1 | 0.3218 | 0.2675 | +-0.0538 | 0.340 | 0.304 | 0.288 |
| 5 | 0.4326 | 0.3488 | +0.0570 | 0.490 | 0.453 | 0.441 |
| 10 | 0.4574 | 0.3966 | +0.0818 | 0.527 | 0.487 | 0.475 |
| 20 | 0.4728 | 0.4427 | +0.0972 | 0.553 | 0.506 | 0.494 |
| 50 | 0.4844 | 0.4818 | +0.1088 | 0.573 | 0.521 | 0.509 |

Selected (Stage5 routed): **0.37563**

## 2. Per-Subset Oracle Table

| Subset | N | Selected | oracle@5 | oracle@20 | oracle@50 | gap@20 | bottleneck |
|--------|---|----------|----------|-----------|-----------|--------|------------|
| Aka_Gha | 1114 | 0.1852 | 0.2160 | 0.2341 | 0.2406 | +0.0489 | recall |
| Amh_Eth | 462 | 0.1281 | 0.1730 | 0.1995 | 0.2114 | +0.0714 | recall |
| Eng_Eth | 564 | 0.4687 | 0.5219 | 0.5434 | 0.5535 | +0.0747 | mixed |
| Eng_Gha | 1104 | 0.1894 | 0.2180 | 0.2368 | 0.2440 | +0.0474 | recall |
| Eng_Ken | 390 | 0.5744 | 0.6195 | 0.6708 | 0.6749 | +0.0963 | mixed |
| Eng_Uga | 1688 | 0.5901 | 0.6344 | 0.6758 | 0.6874 | +0.0857 | mixed |
| Lug_Uga | 846 | 0.3657 | 0.5079 | 0.6019 | 0.6182 | +0.2363 | mixed |
| Swa_Ken | 518 | 0.4691 | 0.5686 | 0.6350 | 0.6659 | +0.1659 | mixed |

## 3. Aka_Gha Diagnosis

- Selected WR: 0.1852
- oracle@5: 0.2160 (+0.0308)
- oracle@20: 0.2341 (+0.0489)
- oracle@50 (full pool): 0.2406
- hit@20 (WR≥0.30): 0.105
- hit@20 (WR≥0.40): 0.013
- avg pool size: 66.0
- avg best rank (RRF): 23.3
- % oracle from sparse only: 47.5%
- % oracle from dense only:  22.1%
- % oracle same subset:      100.0%
- **Bottleneck: recall**

Error type breakdown:
  - A_solved: 2 (0.2%)
  - A_near_solved: 3 (0.3%)
  - B_reranking_failure: 8 (0.7%)
  - C_recall_failure: 985 (88.4%)
  - D_weak_pool: 112 (10.1%)
  - E_ambiguous_medium: 4 (0.4%)

## 4. Amh_Eth Diagnosis

- Selected WR: 0.1281
- oracle@5: 0.1730 (+0.0448)
- oracle@20: 0.1995 (+0.0714)
- oracle@50 (full pool): 0.2114
- hit@20 (WR≥0.30): 0.193
- hit@20 (WR≥0.40): 0.110
- avg pool size: 63.9
- avg best rank (RRF): 22.4
- % oracle from sparse only: 35.1%
- % oracle from dense only:  22.9%
- % oracle same subset:      100.0%
- **Bottleneck: recall**

Error type breakdown:
  - A_solved: 21 (4.5%)
  - A_near_solved: 19 (4.1%)
  - B_reranking_failure: 9 (1.9%)
  - C_recall_failure: 366 (79.2%)
  - D_weak_pool: 43 (9.3%)
  - E_ambiguous_medium: 4 (0.9%)

## 5. Ranking vs Recall Conclusion

- Global selected WR:   0.3756
- Global oracle@5:      0.4326  (headroom +0.0570)
- Global oracle@20:     0.4728 (headroom +0.0972)
- Global oracle@50:     0.4844 (headroom +0.1088)

**Interpretation:**
- Max theoretical gain with perfect reranker (same pool): +0.1088
- Gain achievable with oracle@5 (top-5 candidates): +0.0570

If oracle@5 >> selected: **reranking failure** (good candidates exist but not selected).
If oracle@50 is low: **recall failure** (good candidates not in pool at all).

## 6. Recommendations

- **Aka_Gha** (sel=0.185, oracle@50=0.241): RECALL bottleneck — better retrieval needed (BM25+, language-specific models)
- **Amh_Eth** (sel=0.128, oracle@50=0.211): RECALL bottleneck — better retrieval needed (BM25+, language-specific models)
- **Eng_Eth** (sel=0.469, oracle@20=0.543): MIXED — consider both recall improvement and reranking (+0.075 headroom)
- **Eng_Gha** (sel=0.189, oracle@50=0.244): RECALL bottleneck — better retrieval needed (BM25+, language-specific models)
- **Eng_Ken** (sel=0.574, oracle@20=0.671): MIXED — consider both recall improvement and reranking (+0.096 headroom)
- **Eng_Uga** (sel=0.590, oracle@20=0.676): MIXED — consider both recall improvement and reranking (+0.086 headroom)
- **Lug_Uga** (sel=0.366, oracle@20=0.602): MIXED — consider both recall improvement and reranking (+0.236 headroom)
- **Swa_Ken** (sel=0.469, oracle@20=0.635): MIXED — consider both recall improvement and reranking (+0.166 headroom)

### Global error type breakdown:
- A_solved: 2521 / 6686 (37.7%)
- A_near_solved: 73 / 6686 (1.1%)
- B_reranking_failure: 762 / 6686 (11.4%)
- C_recall_failure: 2858 / 6686 (42.7%)
- D_weak_pool: 364 / 6686 (5.4%)
- E_ambiguous_medium: 108 / 6686 (1.6%)
