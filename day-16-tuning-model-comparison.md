

## 1. Day 16 Overview

Day 16 focuses on recognizing underfitting and overfitting, preparing numerical and categorical features, tuning model hyperparameters, and selecting a model using fair evaluation.

### Learning objectives

By the end of this day, I should be able to:

1. Explain underfitting, overfitting, generalization, and the bias–variance trade-off.
2. Inspect how decision tree complexity affects training and test accuracy.
3. Use `ColumnTransformer` to scale numerical columns and one-hot encode categorical columns.
4. Tune an SVM using `GridSearchCV` and `RandomizedSearchCV`.
5. Compare several classifiers with cross-validation and justify the selected model.

## 2. Core Concepts and Learning Notes

### 2.1 Underfitting

Underfitting happens when a model is too simple to learn the important relationships in the training data. **Both training and validation scores are often low.** For example, a decision tree with maximum depth 1 may overlook useful feature combinations.

**Possible remedies:** try a more expressive model, add informative features, or reduce excessive regularization.

### 2.2 Overfitting

Overfitting happens when a model learns the training examples too closely, including noise. **Training performance may be very high while performance on unseen data is weaker.** A very deep decision tree can memorize individual observations.

**Possible remedies:** use cross-validation, limit tree depth, apply regularization, collect more representative data, or simplify features.

### 2.3 Generalization

Generalization is the ability to perform well on **new, unseen examples**, not just on the dataset used to fit the model. A model with the highest training accuracy is not necessarily the most useful model.

### 2.4 Bias–Variance Trade-Off (Required Day 16 Learning Topic)

- **High bias:** the model is too rigid and misses genuine patterns (often underfitting).
- **High variance:** the model changes too much with small changes in the training data (often overfitting).
- **Trade-off:** increasing complexity may reduce bias while increasing variance. The goal is reliable performance on new data, not zero training error.

**Example:** a shallow decision tree may underfit; a deep tree may fit training data perfectly but generalize poorly. An intermediate depth may give a better balance. This depends on the dataset and must be measured.

### 2.5 Feature Scaling

Scaling puts numerical features on comparable scales. For standardization:

\[
z = \frac{x - \mu}{\sigma}
\]

`StandardScaler` subtracts the training mean and divides by the training standard deviation. Scaling is especially useful for SVMs, logistic regression, and distance-based methods.

**Important:** fit the scaler on training data only. Putting preprocessing inside a `Pipeline` helps avoid data leakage during cross-validation.

### 2.6 Categorical Encoding

Machine-learning estimators usually require numeric input. **One-hot encoding** converts each category into indicator columns—for example, a `city` column containing Lahore or Karachi becomes separate indicators. `OneHotEncoder(handle_unknown="ignore")` can handle categories not seen during training.

`ColumnTransformer` applies the appropriate transformation to each group of columns in one preprocessing step:

- Numerical columns → `StandardScaler`
- Categorical columns → `OneHotEncoder`

### 2.7 Hyperparameters and Tuning

**Parameters** are learned from training data (e.g., coefficients). **Hyperparameters** are selected before fitting (e.g., SVM `C`, kernel, and `gamma`).

| Method | How it searches | Trade-off |
|---|---|---|
| `GridSearchCV` | Evaluates all specified combinations | Exhaustive over the given grid, but costs grow quickly |
| `RandomizedSearchCV` | Samples a fixed number of combinations | Can explore broader ranges with a search budget, but may miss strong settings |

Both should use the same metric and cross-validation strategy for a meaningful comparison. Their measured run times depend on hardware, data size, and search spaces.

### 2.8 Model Comparison Without Test-Set Leakage

1. Split data into training and final test sets.
2. Compare candidate models using cross-validation **only on training data**.
3. Select a model according to the mean validation score, its variability, and practical needs.
4. Fit the selected model on all training data.
5. Evaluate once on the previously untouched test set.

Use a suitable metric. The exercises use **balanced accuracy** for model selection because both classes should matter. Accuracy alone may hide poor minority-class performance.

> **Caution on Question 6:** The syllabus explicitly asks for a *training-versus-test* plot as a learning illustration. Do not use that repeatedly inspected test curve to choose a model in real projects. Use validation data or cross-validation to choose depth, then evaluate on a separate test set.

## 3. Required Code Practice

The Week 3 syllabus assigns **Questions 6–8** to Day 16. Each solution is in its own Python file.

| Question | File | What it does | Output |
|---|---|---|---|
| **Q6 — Medium** | [`q06_overfitting_tree.py`](q06_overfitting_tree.py) | Trains decision trees with `max_depth` from 1 to 15; plots training vs. test scores to illustrate overfitting | `results/q06_tree_train_vs_test.png`, console table |
| **Q7 — Medium** | [`q07_column_transformer.py`](q07_column_transformer.py) | Uses `ColumnTransformer` to scale numeric features and one-hot encode categorical features in one pipeline | Original and transformed shapes, feature names, test accuracy |
| **Q8 — Medium** | [`q08_svm_search.py`](q08_svm_search.py) | Tunes a scaled SVM with grid and randomized search; compares best CV score and runtime | Printed best CV scores, parameter sets, runtime and test scores |

### Q6: Decision Tree Depth and Overfitting

**Task:** Train a decision tree with `max_depth` from 1 to 15 and plot training versus test score.

**Method:** Use the built-in scikit-learn breast cancer dataset, an 80/20 stratified split, and a fixed seed (`random_state=42`). Interpret small-depth scores, score gaps, and the behavior of deeper trees. A larger training/test gap suggests overfitting, but a gap alone does not prove that depth is the only cause.

**Run:**

```bash
python q06_overfitting_tree.py
```

### Q7: ColumnTransformer for Mixed Features

**Task:** Scale numerical columns and one-hot encode categorical columns in one step.

**Method:** Create reproducible **synthetic demonstration data** with `age`, `income`, `city`, and `plan`. Apply `StandardScaler` and `OneHotEncoder` through a `ColumnTransformer` inside a `Pipeline`, and fit only on the training split. The example does **not** claim to represent real customers.

**Run:**

```bash
python q07_column_transformer.py
```

### Q8: GridSearchCV versus RandomizedSearchCV

**Task:** Tune an SVM using both methods and compare the best score and run time.

**Method:** Use an SVM with an RBF kernel. Keep `StandardScaler` inside the search pipeline. Use the same training split, five-fold stratified CV, and balanced-accuracy metric for both approaches. Record the number of configurations each explores and their actual elapsed times.

**Run:**

```bash
python q08_svm_search.py
```

**Interpretation:** The method with the better CV score in one run is not guaranteed to be better in general. The search spaces and budgets differ, so do not treat this as an equal-compute benchmark.

## 4. Practical Assignment: Compare Multiple Models

**Syllabus task:** Compare multiple models and justify the best-performing model.

**Models:** Logistic Regression, Random Forest, and SVM.  
**Dataset:** scikit-learn breast cancer classification dataset (included with scikit-learn; no download).  
**Evaluation:** 80/20 stratified split with five-fold cross-validation on the training subset, using balanced accuracy. Report the mean and standard deviation of the five folds. Select the strongest mean-CV candidate and evaluate it on the held-out test set.

```bash
python model_comparison.py
```

**Output:** `results/model_comparison_cv.csv` and console test metrics.

**How to justify selection:** Report the actual cross-validation score, variability across folds, independent test result, and relevant considerations such as prediction speed or interpretability. Do **not** invent numbers or claim a universal best model.

## 5. Setup and Run Instructions

Requires **Python 3.10+**. From this folder:

```bash
python -m pip install -r requirements.txt
python q06_overfitting_tree.py
python q07_column_transformer.py
python q08_svm_search.py
python model_comparison.py
```

All data is available locally or generated by the scripts. No API key, GPU, or internet dataset download is required (package installation may require internet). Generated outputs go into `results/`.

## 6. Executed Results and Reflection

The included scripts were executed successfully in a local Python environment with **scikit-learn 1.8.0**. These are **measured example results**, not performance guarantees; results and runtimes can differ by environment. Add your own observations before your GitHub submission.


---

**Source:** *System 7 — Week 3 Syllabus*, Module A, Daily Plan (Day 2 / Day 16), Learning Topics, Code Practice Questions 6–8, and GitHub Instructions.  
**Note:** Explanations, demonstrations, code, and reflection prompts expand the syllabus; the generated experimental results are meant to be calculated rather than assumed.
