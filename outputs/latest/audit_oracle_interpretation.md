# Audit Oracle Top-k

## Scores globaux (Val, n=6686)

| k | oracle_WR | gap vs current |
|---|-----------|----------------|
| current | 0.30839 | — |
| top-1 | 0.34988 | +0.04149 |
| top-5 | 0.43518 | +0.12678 |
| top-10 | 0.45599 | +0.14759 |
| top-20 | 0.47057 | +0.16217 |
| top-50 | 0.48518 | +0.17679 |

## Interprétation

- Gain oracle@5 vs current : **+0.12678** (41.1%)
- Gain oracle@20 vs current : **+0.16217** (52.6%)
- Gain oracle@50 vs current : **+0.17679** (57.3%)

**Signal :**
- Si gain concentré en top-5 → reranker (cross-encoder) limiterait le plafond
- Si gain croît fortement au-delà de top-5 → recall insuffisant (dense retrieval)

**Recommandation : cross-encoder (gain concentré dans top-5, reranking vaut plus)**

## Par subset (oracle@20 vs current)
- Aka_Gha: current=0.1785 → oracle@20=0.2398 (+0.0613)
- Amh_Eth: current=0.1195 → oracle@20=0.2093 (+0.0898)
- Eng_Eth: current=0.3935 → oracle@20=0.5464 (+0.1529)
- Eng_Gha: current=0.1678 → oracle@20=0.2364 (+0.0686)
- Eng_Ken: current=0.4347 → oracle@20=0.6429 (+0.2082)
- Eng_Uga: current=0.4061 → oracle@20=0.6613 (+0.2552)
- Lug_Uga: current=0.3779 → oracle@20=0.6129 (+0.2350)
- Swa_Ken: current=0.4363 → oracle@20=0.6328 (+0.1965)
