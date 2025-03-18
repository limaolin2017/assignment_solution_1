# ADP assignment

## Experiment
I tested **Logistic Regression** on dataset, comparing three parameters:
- **C** (0.5, 1, 2)  
- **class_weight** (`None` or `"balanced"`)  
- **max_iter** (100 or 300)

## Method

1. **Data**  
   - I read two CSV files (`available_conversations.csv` and `available_topics.csv`) and replace `topic_id` with `topic_name`.

2. **Split & Vectorize**  
   - I split messages into 80% training and 20% testing.  
   - I apply `TfidfVectorizer(max_features=5000)` to convert text into numeric vectors.

3. **Logistic Regression**  
   - I vary `C` (inverse regularization strength), `class_weight`, and `max_iter`.  
   - I train on the TF-IDF features and predict on the test set.

4. **Evaluation**  
   - I compute accuracy, precision, recall, and F1-score (weighted average).  
   - I log metrics to **WandB** to compare runs.

## Results

| C   | class_weight | max_iter | Accuracy | Precision | Recall  | F1-score |
|-----|--------------|----------|----------|----------|---------|---------|
| 2   | None/balanced | 100/300 | 0.954   | ~0.953   | ~0.954  | 0.954   |
| 1   | None         | 100/300 | 0.946   | 0.944    | 0.947   | 0.946   |
| 1   | balanced     | 100/300 | 0.944   | 0.943    | 0.944   | 0.944   |
| 0.5 | None         | 100/300 | 0.940   | 0.938    | 0.942   | 0.940   |
| 0.5 | balanced     | 100/300 | 0.934   | 0.933    | 0.935   | 0.934   |

- **C=2** yields the highest accuracy (0.954).  
- **class_weight="balanced"** does not significantly help.  
- **max_iter=100 vs 300** shows no difference.

## Conclusion

Iaker regularization (C=2) provides the best accuracy for this dataset, while `class_weight="balanced"` offers no clear advantage. The model converges quickly, so increasing `max_iter` beyond 100 does not improve results.
