# Assignment

# 逻辑回归实验详情

## 实验设置
本实验测试了**逻辑回归**模型在特定数据集上的表现，主要对比了以下三个参数的不同组合：
- **C值** (正则化强度的倒数)：测试了 0.5、1、2 这三个值。
- **类别权重 (class_weight)**：测试了 `None` (无权重) 和 `"balanced"` (平衡权重) 两种选项。
- **最大迭代次数 (max_iter)** (求解器收敛的最大迭代次数)：测试了 100 和 300 两个值。

## 方法步骤

1.  **数据准备**:
    *   读取了两个 CSV 文件：`available_conversations.csv` 和 `available_topics.csv`。
    *   通过将 `available_conversations.csv` 中的 `topic_id` 替换为 `available_topics.csv` 中的 `topic_name`，对数据进行了合并或关联处理。

2.  **数据划分与向量化**:
    *   将消息数据（推测来自合并后的数据）划分为训练集 (80%) 和测试集 (20%)。
    *   应用 `TfidfVectorizer` (TF-IDF 词频-逆文档频率向量化方法) 将文本数据转换为数值向量，其中 `max_features` (最大特征数) 设置为 5000。

3.  **逻辑回归模型训练**:
    *   通过改变 `C`值、`class_weight` (类别权重) 和 `max_iter` (最大迭代次数) 等参数来训练逻辑回归模型。
    *   模型在经过 TF-IDF 特征转换后的数据上进行训练。
    *   在测试集上进行预测。

4.  **评估**:
    *   使用以下指标评估模型性能：
        *   准确率 (Accuracy)
        *   精确率 (Precision) (加权平均)
        *   召回率 (Recall) (加权平均)
        *   F1分数 (F1-score) (加权平均)
    *   所有评估指标均记录到 **WandB** (Weights & Biases) 平台，以便于比较不同实验运行的结果。

## 实验结果

实验结果汇总如下表：

| C值 | 类别权重 (class_weight) | 最大迭代次数 (max_iter) | 准确率 (Accuracy) | 精确率 (Precision) | 召回率 (Recall) | F1分数 (F1-score) |
|-----|----------------------|--------------------|-----------------|-------------------|----------------|-----------------|
| 2   | 无权重/平衡权重        | 100/300            | 0.954           | ~0.953            | ~0.954         | 0.954           |
| 1   | 无权重               | 100/300            | 0.946           | 0.944             | 0.947          | 0.946           |
| 1   | 平衡权重             | 100/300            | 0.944           | 0.943             | 0.944          | 0.944           |
| 0.5 | 无权重               | 100/300            | 0.940           | 0.938             | 0.942          | 0.940           |
| 0.5 | 平衡权重             | 100/300            | 0.934           | 0.933             | 0.935          | 0.934           |

从结果中可以观察到：
- **C=2** 时获得了最高的准确率 (0.954)。
- 使用 **`class_weight="balanced"`** (平衡类别权重) 并未显著提升模型性能。
- 对比 **`max_iter=100`** 和 **`max_iter=300`** (最大迭代次数)，两者结果没有差异。

## 结论
- 对于本实验所使用的数据集，较高的正则化强度 (特别是 **C=2**) 能够带来最佳的准确率。
- `class_weight="balanced"` (平衡类别权重) 参数在本实验中未显示出明显优势。
- 模型收敛速度较快，因此将 `max_iter` (最大迭代次数) 增加到100以上并不能改善结果。

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

Higher regularization (C=2) provides the best accuracy for this dataset, while `class_weight="balanced"` offers no clear advantage. The model converges quickly, so increasing `max_iter` beyond 100 does not improve results.
