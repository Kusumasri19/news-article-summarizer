# News Article Summarization: TextRank vs. BART vs. Fine-tuned T5

This project compares three approaches for summarizing news articles and evaluates them with ROUGE.

## Approaches

| Approach | Type | Description |
|---|---|---|
| TextRank | Extractive | Selects the 3 most central sentences using a TF-IDF similarity graph and PageRank |
| BART (`facebook/bart-large-cnn`) | Abstractive | Pretrained summarization model, used without fine-tuning |
| T5-small (fine-tuned) | Abstractive | `t5-small` fine-tuned on the news dataset |

## Dataset

[News Summary (Kaggle)](https://www.kaggle.com/datasets/sunnysai12345/news-summary): news articles with short human-written summaries. A subset was used: 2,000 articles for training and 100 for testing (fixed random seed).

## Results

ROUGE scores on the 100 held-out test articles:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| TextRank (extractive) | 38.32 | 19.21 | 26.78 |
| BART (pretrained) | 42.62 | 21.76 | 31.53 |
| **T5-small (fine-tuned)** | **50.90** | **29.78** | **39.31** |

The fine-tuned T5-small achieved the highest scores on all three metrics on this dataset.

## Tech stack

Python, PyTorch, Hugging Face Transformers, scikit-learn, NetworkX, Gradio

## How to run

1. Open `News_Article_Summarization.ipynb` in Google Colab.
2. Select a GPU runtime (Runtime > Change runtime type > T4 GPU).
3. Download `news_summary.csv` from the Kaggle dataset linked above.
4. Run all cells and upload `news_summary.csv` when prompted.

## Limitations

- The test set has only 100 articles and results come from a single run, so small differences should not be over-interpreted.
- The pretrained BART model was trained on CNN/DailyMail, which has longer summaries than this dataset. Part of its gap to the fine-tuned T5 is likely due to this domain and length mismatch.
- ROUGE measures word overlap, not meaning or factual consistency. Hallucination was not evaluated.

## Future work

Train on the full dataset, try larger models (`t5-base`, `bart-base`), and add BERTScore and a factuality check.
