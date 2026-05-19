# Iris Dataset Classification with Apache Spark MLlib
**Student ID**: P156872  
**Name**: LIANG HAIZHU  
**Course**: STQD6324 Data Management  
**Assignment**: Assignment 1

---
## Project Overview

This project implements a complete classification pipeline on the classic **Iris dataset** using **Apache Spark MLlib**. Three classification algorithms are trained, tuned, and compared:

- Logistic Regression
- Decision Tree
- Random Forest

All models are optimised using **5‑fold cross‑validation** and **grid search**. The best‑performing model is selected based on accuracy, precision, recall, and F1‑score.

---

## Dataset Description

- **Dataset Name**: Iris Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) (loaded from a public CSV)
- **Samples**: 150 instances
- **Features** (4 numeric, cm): sepal length/width, petal length/width
- **Classes** (3 balanced, 50 each): Setosa, Versicolor, Virginica

The dataset is small, well‑balanced, and nearly linearly separable – ideal for demonstrating Spark MLlib classification pipelines.

---

## Methodology

### 1. Environment & Spark Session
- Python libraries: `pyspark`, `pandas`, `matplotlib`, `seaborn`, `requests`
- SparkSession with app name `"Iris"`

### 2. Data Loading & Exploration
- CSV loaded from a public URL, schema inferred.
- Checked: no missing values, balanced classes, features on similar scales.

### 3. Preprocessing
- **Label encoding**: `StringIndexer` → numeric labels (Setosa=0, Versicolor=1, Virginica=2)
- **Feature assembly**: `VectorAssembler` → single `features` column

### 4. Train‑Test Split
- 80% training, 20% testing using `randomSplit([0.8, 0.2], seed=123)`
- Result: 121 training, 29 test samples.
- **Why seed=123?** This seed gave a test set size (29) closest to the expected 20% of 150 (30), improving evaluation stability.

### 5. Model Definition (Base)
- `LogisticRegression` (family="multinomial")
- `DecisionTreeClassifier`
- `RandomForestClassifier` (initial `numTrees=50`, fixed `seed=123` for reproducibility)

### 6. Model Tuning: Cross‑Validation + Grid Search
- **Evaluator**: `MulticlassClassificationEvaluator` with `accuracy` as primary metric.
- **Cross‑validation**: 5 folds – chosen because the dataset is small; 5 folds provide a more reliable performance estimate than 3 folds with minimal extra cost.
- **Parallelism**: 4 tasks run simultaneously to speed up grid search.
- **Reproducibility**: A fixed `seed=123` is passed to `CrossValidator` for consistent fold splits.

#### Hyperparameter Grids

| Model | Parameters tuned | Candidate values |
|-------|------------------|------------------|
| Logistic Regression | `regParam`<br>`elasticNetParam`<br>`maxIter` | [0.01, 0.1, 1.0]<br>[0.0, 0.5, 1.0]<br>[10, 20, 50] |
| Decision Tree | `maxDepth`<br>`maxBins`<br>`minInstancesPerNode` | [3, 5, 10]<br>[8, 16, 32]<br>[1, 2, 5] |
| Random Forest | `numTrees`<br>`maxDepth`<br>`maxBins` | [10, 20, 50]<br>[3, 5, 10]<br>[8, 16, 32] |

#### Optimal Hyperparameters (from cross‑validation)
- **Logistic Regression**: `regParam=0.01`, `elasticNetParam=0.0`, `maxIter=10`  
  → mild L2 (Ridge) regularisation, fast convergence.
- **Decision Tree**: `maxDepth=5`, `maxBins=32`, `minInstancesPerNode=1`  
  → moderately deep tree with fine‑grained bins, avoids overfitting.
- **Random Forest**: `numTrees=20`, `maxDepth=3`, `maxBins=16`  
  → small ensemble, shallow trees, balanced granularity.

### 7. Evaluation Metrics
- Accuracy (main)
- Weighted Precision, Weighted Recall, F1‑Score

A reusable `evaluate_model` function computes all metrics. Confusion matrices are visualised using Seaborn.

### 8. Prediction Generation
The optimised models generate predictions on the test set, showing the predicted class and probability vector for the first 5 samples.

### 9. Comparative Analysis
- Performance table (all metrics)
- Bar chart comparing accuracy and F1
- Discussion of strengths/limitations per model
- Justification of the best model

---

## Results & Key Findings

### Test Set Performance (29 samples)

| Model                 | Accuracy | Precision | Recall | F1‑Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | **0.9655** | **0.9690** | **0.9655** | **0.9649** |
| Decision Tree         | 0.9310   | 0.9310    | 0.9310 | 0.9310   |
| Random Forest         | **0.9655** | **0.9690** | **0.9655** | **0.9649** |

### Confusion Matrices
- **Logistic Regression**: 1 misclassification (Versicolor → Virginica).  
  Setosa: 14/14, Versicolor: 5/6, Virginica: 9/9.
- **Decision Tree**: 2 misclassifications (Versicolor → Virginica, Virginica → Versicolor).  
  Setosa: 14/14, Versicolor: 5/6, Virginica: 8/9.
- **Random Forest**: 1 misclassification (Versicolor → Virginica).  
  Setosa: 14/14, Versicolor: 5/6, Virginica: 9/9.

### Model Strengths & Limitations

| Model | Strengths | Limitations |
|-------|-----------|--------------|
| **Logistic Regression** | High accuracy (0.9655), simple and fast, interpretable probabilities | Only linear boundaries; cannot capture complex non‑linear patterns |
| **Decision Tree** | Non‑linear, highly interpretable rules, no scaling needed | Lower accuracy (0.9310), prone to overfitting, less stable |
| **Random Forest** | Matches logistic regression accuracy, ensemble reduces overfitting, robust to noise | More complex, slower to train, less interpretable than a single tree |

### Best Model Justification
**Logistic Regression** is selected as the best model because:
1. It achieves the **highest accuracy (0.9655)** together with Random Forest.
2. It is **simpler, faster to train, and more interpretable** than Random Forest.
3. The Iris dataset is nearly linearly separable – a linear model is both sufficient and optimal.

---

## How to Reproduce the Analysis

### Prerequisites
- Python 3.8+
- Apache Spark (installed via `pip install pyspark`)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Iris-Classification-SparkMLlib.git
   cd Iris-Classification-SparkMLlib

