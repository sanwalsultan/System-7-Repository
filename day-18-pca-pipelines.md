## 1. Day 18 Overview

Day 18 continues advanced machine learning with **Principal Component Analysis (PCA)**, feature engineering and selection, class imbalance, data leakage, and scikit-learn pipelines. The Week 3 syllabus assigns **questions 11–13** and a practical **customer-segmentation or student-performance analysis** task.

### Learning objectives

By the end of this day, I should be able to:

1. Explain why reducing feature dimensions can simplify analysis—and what information can be lost.
2. Scale features correctly and use PCA to represent data with two principal components.
3. Engineer new features, select informative features, and separate those two operations.
4. Recognize why high classification accuracy is misleading when classes are imbalanced.
5. Use `class_weight="balanced"` and compare **minority-class recall** with the unweighted baseline.
6. Prevent data leakage by placing all learned preprocessing **inside a scikit-learn pipeline**.
7. Explain why fitting a scaler **before** the train/test split invalidates model evaluation.
8. Use a pipeline in a small, reproducible customer-segmentation exercise.

## 2. Core Concepts and Learning Notes

### 2.1 What is PCA?

**Principal Component Analysis (PCA)** is an **unsupervised dimensionality-reduction** technique. It finds new orthogonal directions (*principal components*) that capture the largest amount of variance in the input features.

- **PC1** captures the greatest available variance.
- **PC2** captures the greatest remaining variance in a direction orthogonal to PC1.
- Each principal component is a weighted combination of the original features.
- PCA does **not** use class labels and does **not** guarantee better classification performance.
- PCA may discard information helpful for predicting a target, even when it retains much of the total variance.

**Example:** A dataset with 13 original features can be projected onto two principal components for a 2D plot. The plot is easier to inspect but cannot preserve every detail of the original 13-dimensional representation.

**Important:** Fit a scaler on training data *only* when PCA is part of a predictive model. In a purely exploratory unsupervised visualization with no held-out predictive evaluation, fitting PCA to all observations is acceptable if described clearly.

### 2.2 Explained variance and the curse of dimensionality

The **explained variance ratio** tells us what share of the data's total variance is represented by each principal component:

\[
\text{Cumulative explained variance (first 2 PCs)} = r_1 + r_2
\]

With many features, observations can become sparse in feature space, distance comparisons can become less useful, and models may require more data to generalize. This collection of issues is often called the **curse of dimensionality**.

**How PCA helps:** It can reduce noise, computation, and dimensionality when much of the information lies in a few directions. **How feature selection helps:** It retains a subset of original features instead of creating new component directions. Neither method always improves performance; validate the choice with appropriate cross-validation.

### 2.3 Feature engineering versus feature selection

| Concept | Meaning | Example |
|---|---|---|
| Feature engineering | Creating or transforming input variables | `average_spend_per_visit = annual_spend / visits_per_year` |
| Feature selection | Keeping a subset of original variables | Selecting the six strongest features with `SelectKBest` |
| Feature extraction | Constructing a lower-dimensional representation | PCA converts 13 variables into PC1 and PC2 |

A useful engineered feature should have an interpretable relationship to the problem and be available at prediction time. Avoid creating any feature from the target label or future information unavailable when the prediction is made.

**Supervised feature selection** (such as `SelectKBest(f_classif)`) uses the class labels during fitting. Therefore, it must run **inside each training fold**, never on the full dataset before cross-validation.

### 2.4 Imbalanced datasets

A dataset is *imbalanced* when one class is much less common than another. If only 7% of examples are positive, predicting the majority class for every case may yield about **93% accuracy** while missing every positive example.

For the minority (positive) class:

\[
\text{Recall}=\frac{TP}{TP+FN},\qquad
\text{Precision}=\frac{TP}{TP+FP}
\]

- **Recall:** Of the actual positives, how many did the model find?
- **Precision:** Of the predicted positives, how many were correct?
- **Balanced accuracy:** Mean recall across classes; useful under class imbalance.
- **`class_weight="balanced"`:** In supported classifiers, makes errors on less frequent classes count more during training. For scikit-learn, each class weight is proportional to `n_samples / (n_classes * n_samples_in_class)`.

**Trade-off:** Class weighting often increases minority recall but may produce more false positives, lower minority precision, or lower overall accuracy. Check the confusion matrix and the application's real-world costs.

### 2.5 Data leakage

**Data leakage** occurs when information not legitimately available to the model during training affects model development or evaluation. This can make reported scores unreliable.

Examples include:

- Fitting `StandardScaler` to **all** samples before the train/test split. The scaler's mean and standard deviation already include held-out test observations.
- Fitting PCA or selecting features before cross-validation, then validating the transformed data.
- Using future information or the target itself to engineer a predictive feature.
- Repeatedly adjusting a model according to results on the final test set.

**Correct order:** `split raw data → fit preprocessing and model on training folds → evaluate once on held-out test data`.

> Leakage does not have to improve a score in every experiment. A leaky evaluation is **invalid even when it performs the same or worse** than the proper pipeline.

### 2.6 Scikit-learn pipelines

A `Pipeline` links sequential transformations to an estimator and ensures the fitted transformation parameters are learned **within the training portion of each cross-validation split**.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=5)),
    ("classifier", LogisticRegression(max_iter=2000)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

If features have different types, a `ColumnTransformer` can scale numeric columns and one-hot encode categorical columns in one workflow before PCA or another estimator.

**Rule:** `fit` a pipeline only on training data. Use its `predict` or `transform` methods on test data; do not refit it on the test set.

## 3. Required Code Practice — Questions 11, 12 and 13

The Week 3 syllabus assigns these exact three questions to Day 18. Each has its own Python script.

| Question | Level | Python file | Required task |
|---|---|---|---|
| **Q11** | Medium | [`q11_imbalanced_class_weight.py`](q11_imbalanced_class_weight.py) | Compare minority recall before and after using `class_weight` |
| **Q12** | Hard | [`q12_pca_2d_clusters.py`](q12_pca_2d_clusters.py) | Reduce to two principal components, plot clusters, report explained variance |
| **Q13** | Hard | [`q13_pipeline_data_leakage.py`](q13_pipeline_data_leakage.py) | Build a scaler + PCA + classifier pipeline; intentionally scale before the split and compare scores |

### Q11: Imbalanced classification and class weighting

**Data:** Reproducible **synthetic** binary-classification dataset with a large majority and small minority class (`make_classification`). No real-world measurements or labels are implied.

**Method:** Stratified 75/25 train/test split. Fit logistic regression with and without `class_weight="balanced"`, always inside a `StandardScaler` pipeline. Use 5-fold stratified cross-validation on training data; report minority recall, precision, F1, balanced accuracy, and test confusion matrices.

```bash
python q11_imbalanced_class_weight.py
```

**Outputs:** `results/q11_class_weight_comparison.csv` and `results/q11_confusion_matrices.png`.

### Q12: PCA to two components and cluster plot

**Data:** Built-in **Wine** dataset (178 observations, 13 numeric features).

**Method:** Standardize the features, project into 2D with `PCA(n_components=2)`, use K-Means with `k=3` on the PCA representation, and draw discovered groups. A second panel uses the known wine classes **only as a reference**. True labels are **not used** by the PCA or K-Means model.

```bash
python q12_pca_2d_clusters.py
```

**Outputs:** `results/q12_pca_clusters.png` and `results/q12_pca_2d_coordinates.csv`.

**Note:** This is **descriptive unsupervised analysis**, not a predictive train/test evaluation.

### Q13: Safe pipeline versus leakage

**Data:** Built-in **Wine** dataset with a 70/30 stratified raw-data split.

**Correct approach:** Fit `Pipeline(StandardScaler → PCA(5) → LogisticRegression)` on training data only. Use 5-fold cross-validation on the training set and evaluate on the held-out test set.

**Incorrect approach (deliberate demonstration):** Fit `StandardScaler` to **the full dataset before splitting**, then fit PCA and the classifier to the training portion. The fitted scaler includes statistics from held-out samples, so both the reported CV score and test score are leakage-contaminated.

```bash
python q13_pipeline_data_leakage.py
```

**Outputs:** `results/q13_pipeline_leakage_comparison.csv` and `results/q13_pipeline_leakage_comparison.png`.

**Conclusion:** Trust the correct pipeline's results, not the leaky results. There is no rule that leaking test-set statistics must increase accuracy in a specific run.

## 4. Practical Task — Customer Segmentation

**Required syllabus practical:** Customer segmentation **or** student-performance analysis. This folder implements the **customer segmentation** option.

**Data source:** 360 reproducible **synthetic** customer records with annual income, annual spending, visit frequency, digital engagement, and preferred channel. The generated data is strictly for teaching and is **not** a real customer dataset.

**Workflow:**

1. Generate synthetic behavior fields with reproducible random noise.
2. Engineer `average_spend_per_visit`.
3. Use `ColumnTransformer` to standardize numerical features and one-hot encode `channel`.
4. Apply PCA to extract two components.
5. Run K-Means with three clusters.
6. Calculate silhouette score, inspect cluster summary statistics, and visualize the result.

**Script:** [`customer_segmentation_pipeline.py`](customer_segmentation_pipeline.py)

```bash
python customer_segmentation_pipeline.py
```

**Outputs:** `results/customer_segments_demo.csv`, `results/customer_cluster_summary.csv`, and `results/customer_segmentation_pca.png`.

**Interpretation caution:** These clusters are unsupervised groupings of deliberately structured simulated data. Cluster IDs such as 0, 1, and 2 are arbitrary and are **not verified business personas**.

### Optional extra — Feature selection

The Module B syllabus also names feature selection. The additional [`feature_selection_demo.py`](feature_selection_demo.py) compares logistic regression on all 13 Wine features with a `SelectKBest(k=6)` pipeline. The selector is fit only within training folds; selected features are exported to CSV.

```bash
python feature_selection_demo.py
```

## 5. Setup and Run Instructions

**Requirements:** Python 3.10+ and the Python packages listed in `requirements.txt`. The code is CPU-friendly and uses built-in or generated data; **no dataset downloads or API keys are required**.

From the `day-18-pca-pipelines/` folder:

```bash
python -m pip install -r requirements.txt

python q11_imbalanced_class_weight.py
python q12_pca_2d_clusters.py
python q13_pipeline_data_leakage.py
python customer_segmentation_pipeline.py
python feature_selection_demo.py
```

All charts, CSV files, and text logs are stored in `results/`. The script output is reproducible with the same package versions and specified random seeds; different library versions can change some numeric results.

## 6. Executed Results and My Interpretation

The Python scripts were successfully executed with **scikit-learn 1.8.0**. These are example results from the included code, not general guarantees.


