# Zindi / ITU Multilingual Health QA — AfroByte-ProtoRAG

Private competition repository for the Zindi/ITU Multilingual Health Question Answering challenge.

## Best public submission so far

Submission ID: TzmGb4De  
Public Score: 0.732376  
ROUGE-L F1: 0.6334  
ROUGE-1 F1: 0.8118  
LLM Judge: 0.7602  

## Best validated local stage

Stage5 routed  
Validation WR: 0.37563  

## Current best architecture

- Sparse retrieval:
  - word TF-IDF
  - char n-grams
  - char_wb
  - byte n-grams
- Dense retrieval:
  - paraphrase-multilingual-MiniLM-L12-v2
  - multilingual-e5-small
- RRF candidate fusion
- Cross-encoder score used as LightGBM feature
- LightGBM reranker with ROUGE-L-heavy target
- Routing by subset:
  - Stage5 for Eng_*/Aka/Amh
  - Stage4 for Swa_Ken/Lug_Uga

## Important notes

ByT5 was tested and disabled.  
Phrase composer was tested and disabled.  
The current best solution is retrieval + dense retrieval + reranking + routing, not free generation.

