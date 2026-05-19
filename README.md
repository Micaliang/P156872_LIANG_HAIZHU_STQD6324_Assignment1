# Iris-Classification-SparkMLlib
STQD6324 Assignment 1 - Iris Classification with Spark MLlib
# Iris Dataset Classification with Apache Spark MLlib

## Project Overview
This project builds a complete end-to-end classification pipeline on the classic **Iris dataset** using **Apache Spark MLlib**.

Three classification models are implemented, tuned and compared:
- Multinomial Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

All models are optimised using **5-fold cross-validation** and **grid search**. Performance is evaluated using accuracy, precision, recall and F1-score. This assignment fully meets the requirements of **STQD6324 Data Management (2025/2026 Semester 2)**.

---

## Dataset Description
- **Dataset Name**: Fisher’s Iris Dataset
- **Source**: UCI Machine Learning Repository
- **Total Samples**: 150
- **Features**: 4 continuous features (sepal length, sepal width, petal length, petal width) in cm
- **Classes**: 3 species, 50 samples each
  - Iris Setosa
  - Iris Versicolor
  - Iris Virginica

- No missing values
- Class distribution is fully balanced
- Setosa is linearly separable; Versicolor and Virginica have partial feature overlap.

---

## Methodology
### 1. Environment
- PySpark
- Pandas
- Matplotlib & Seaborn for visualisation

### 2. Workflow
1. Load dataset from public source
2. Data exploration and preprocessing
3. Label encoding and feature assembling
4. Train-test split (80% : 20%)
5. Model training with grid search
6. 5-fold cross-validation for hyperparameter tuning
7. Model evaluation using multi-class metrics
8. Confusion matrix analysis
9. Model comparison and best model selection

### 3. Train-Test Split
- Training set: 121 samples
- Test set: 29 samples
- Fixed random seed for reproducibility

---

## Hyperparameter Tuning
### Parameter Search Grid
| Model | Tuned Parameters |
|-------|------------------|
| Logistic Regression | regParam, elasticNetParam, maxIter |
| Decision Tree | maxDepth, maxBins, minInstancesPerNode |
| Random Forest | numTrees, maxDepth, maxBins |

### Optimal Hyperparameters
- **Logistic Regression**: regParam=0.01, elasticNetParam=0.0, maxIter=10
- **Decision Tree**: maxDepth=5, maxBins=32, minInstancesPerNode=1
- **Random Forest**: numTrees=20, maxDepth=3, maxBins=16

---

## Model Performance Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.9655 | 0.9690 | 0.9655 | 0.9649 |
| Decision Tree | 0.9310 | 0.9310 | 0.9310 | 0.9310 |
| Random Forest | 0.9655 | 0.9690 | 0.9655 | 0.9649 |

---

## Confusion Matrix Summary
- **Logistic Regression**: Only 1 misclassification
- **Decision Tree**: 2 misclassifications between Versicolor and Virginica
- **Random Forest**: Same performance as Logistic Regression (1 misclassification)

---

## Model Strengths and Limitations
### Logistic Regression
- Strengths: High accuracy, simple structure, fast training, good interpretability
- Limitations: Only linear decision boundaries

### Decision Tree
- Strengths: Captures non-linear patterns, clear decision rules
- Limitations: Lower accuracy, easy overfitting, unstable

### Random Forest
- Strengths: High accuracy, strong generalisation, robust to noise
- Limitations: Higher complexity, less interpretable

---

## Best Model Justification
**Logistic Regression** is selected as the optimal model:
1. Achieves the highest accuracy of **0.9655**
2. Simpler, faster and more interpretable than Random Forest
3. Iris dataset is nearly linearly separable, making logistic regression the most suitable choice.

---

## Reproduction Instruction
1. Install required libraries:
```bash
pip install pyspark pandas matplotlib seaborn
