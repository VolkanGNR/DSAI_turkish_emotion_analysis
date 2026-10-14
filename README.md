# Emotion Analysis of Turkish User Comments

**Natural Language Processing Capstone Project · Data Science and Artificial Intelligence Minor · MEF University**  
**Author:** Volkan Güner · **Project date:** February 2026

## Project overview

This project develops a **Python-based NLP and machine-learning pipeline** for Turkish user comments. It moves beyond the source dataset's positive/negative sentiment labels by constructing **six lexicon-based emotion categories** and comparing classical supervised classifiers using TF-IDF text features.

**Interpretation note:** The emotion labels were generated automatically from a manually constructed keyword lexicon, **not independently annotated by human reviewers**. The model scores reported here therefore measure agreement with these **pseudo-labels**, rather than verified accuracy at detecting human emotion.

## Dataset and files

The source file, **`social_media_comments.csv`**, contains **11,119 Turkish comments**:

- `Paylaşım` — comment text.
- `Tip` — existing positive/negative sentiment label (`Pozitif` / `Negatif`).

The repository may also include these two processed datasets:

| File | Contents | Role |
|---|---|---|
| `social_media_comments.csv` | 11,119 source comments and sentiment labels | **Required input** for the original notebook |
| `emotion_analysis_with_labels.csv` | 11,119 comments with normalized text, generated emotions, and text-length features | Processed export matching the notebook's saved emotion counts |
| `emotion_analysis_output.csv` | 11,119 comments with normalized text, emotion labels, and an `emotion_scores` field | **Separate processed export** with a different label distribution |

**The two processed exports are not identical:** `emotion_analysis_with_labels.csv` contains **9,515 `neutral`** labels, while `emotion_analysis_output.csv` contains **9,613 `neutral`** labels. They should not be presented as interchangeable results of the same labeling run. The README's statistics and model results below follow the **unchanged notebook** and the distribution in `emotion_analysis_with_labels.csv`; the original notebook only reads `social_media_comments.csv` and does not require either processed export to run.

### Label distribution used in the notebook

| Emotion | Number of comments |
|---|---:|
| Neutral | 9,515 |
| Anger | 938 |
| Joy | 543 |
| Disgust | 69 |
| Sadness | 46 |
| Fear | 8 |
| **Total** | **11,119** |

The data are strongly imbalanced: most comments receive the `neutral` pseudo-label.

**Data-use note:** The source and processed CSV files contain user-generated comments. Before publishing them in a public GitHub repository, verify the original dataset's license, reuse permissions, and applicable privacy obligations. Including a CSV in a repository does not itself establish that it can be redistributed publicly.

## Methodology

1. Read the Turkish CSV, accounting for possible encoding and delimiter differences.
2. Identify the comment and existing sentiment columns.
3. Normalize comments by removing URLs, user mentions, hashtags, and punctuation while preserving Turkish letters.
4. Construct a Turkish keyword lexicon for **anger, joy, sadness, fear, and disgust**, using `neutral` when no keyword is matched.
5. Explore comment-length distributions and class imbalance.
6. Extract **TF-IDF unigram and bigram** features.
7. Split the examples into **stratified 80% training / 20% test** groups.
8. Compare a majority-class **Dummy** classifier with **Logistic Regression** and **Linear SVM**, using class weights for the two trained linear models.
9. Evaluate accuracy, macro F1, weighted F1, classification reports, and a confusion matrix.

## Saved experiment results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Dummy (most frequent) | 85.57% | 0.1537 | 0.7891 |
| Logistic Regression | 98.79% | **0.7403** | 0.9872 |
| Linear SVM | **98.92%** | 0.7393 | 0.9884 |

The **Linear SVM** has the highest recorded accuracy, while **Logistic Regression** has the highest recorded macro F1. The notebook sorts its model comparison by macro F1 and uses Logistic Regression for the final confusion-matrix plot.

### How to interpret the scores

The high overall accuracies are influenced by the dominance of `neutral` comments and by the fact that the emotion targets were created using keyword rules applied to the same text later used for prediction. These numbers **do not establish 98–99% real-world emotion-recognition accuracy**. The extremely small `fear` class (eight comments in total) is another important limitation.

## Repository files

- [`capstone02.ipynb`](capstone02%281%29.ipynb) — original, **unchanged** Jupyter Notebook supplied for this repository.
- `social_media_comments.csv` — raw input needed by the notebook.
- `emotion_analysis_with_labels.csv` — additional processed export consistent with the notebook's recorded label counts.
- `emotion_analysis_output.csv` — separate processed export with different label counts; supplied for reference, not required by the notebook.

## Running the original notebook

Open `capstone02.ipynb` in Jupyter Notebook or JupyterLab. Install the Python packages it imports, including pandas, NumPy, Matplotlib, and scikit-learn.

Place `social_media_comments.csv` in the notebook's **working directory**. The original code expects this exact filename:

```python
CSV_PATH = "social_media_comments.csv"
```

The notebook attempts multiple Turkish-compatible file encodings and separators; the supplied source file was readable with `cp1254` encoding. The code and its saved outputs have **not** been changed for this README, and no new model training was performed. The processed CSV exports are optional for rerunning the notebook.

## Limitations

- The target labels are generated using a keyword lexicon, not independently human-verified annotations.
- Sarcasm, negation, context, and implicit emotions can be missed by simple keyword matching.
- Class imbalance makes overall accuracy less informative than macro F1 and per-class evaluation.
- The `fear` category has only eight samples and one test example in the notebook's recorded split.
- A separately labeled evaluation dataset would be needed to assess real emotion-detection performance.
- Public redistribution of user-generated comments requires a separate license/privacy check.

## Project status

Completed undergraduate capstone project demonstrating a Python-based exploratory analysis, Turkish text preprocessing, feature engineering, pseudo-label generation, and supervised-model comparison workflow.
