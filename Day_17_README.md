# System 7 — Day 17: SVM, Naive Bayes & Clustering

**Week:** 3 — Model Evaluation, Advanced ML & Deep Learning  
**Module:** B — Advanced Machine Learning  
**Mentor:** Saad Hassan Saadi  
**GitHub folder:** `day-17-svm-clustering`  
**Suggested commit message:** `day-17-svm-clustering`

## 1. Day 17 Overview

Day 17 introduces two **supervised classification** methods (Support Vector Machines and Naive Bayes) and three **unsupervised clustering** approaches (K-Means, hierarchical clustering, and DBSCAN). We practice choosing the number of clusters and comparing classifiers using precision and recall.

### Learning objectives

By the end of this day, I should be able to:

1. Explain the SVM margin, support vectors, kernels, and main hyperparameters.
2. Describe how Naive Bayes predicts classes using probabilities.
3. Apply K-Means and decide on `k` using elbow and silhouette analysis.
4. Explain hierarchical clustering, dendrograms, DBSCAN, and noise points.
5. Compare SVM and Naive Bayes on the **same data split** using macro precision and recall.
6. State when DBSCAN is a more suitable option than K-Means—and when it is not.

## 2. Core Concepts and Learning Notes

### 2.1 Supervised versus unsupervised learning

| Type | Has target labels? | Algorithms in this lesson | Typical question |
|---|---|---|---|
| Supervised classification | Yes | SVM, Naive Bayes | "Which category does this example belong to?" |
| Unsupervised clustering | No | K-Means, hierarchical clustering, DBSCAN | "Which observations are similar to one another?" |

Clustering IDs are **arbitrary group identifiers**, not automatically known class labels. When exploring the Iris dataset, the true species are intentionally **not** supplied to K-Means.

### 2.2 Support Vector Machine (SVM)

An SVM classification model tries to find a boundary separating the classes while maximizing the **margin**—the distance to the closest training points. Those influential points are called **support vectors**.

- **Linear kernel:** for data with an approximately linear boundary.
- **RBF kernel:** permits nonlinear decision boundaries using similarity between observations.
- **`C`:** controls the trade-off between a wider margin and classification errors. A very high value may increase overfitting.
- **`gamma` (RBF):** controls how locally each point influences the decision boundary. Extremely large values can yield overly intricate boundaries.
- **Scaling:** important because SVM distance and similarity calculations depend on feature magnitudes.

**In code:** `make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))` keeps scaling inside each training or cross-validation fold to avoid data leakage.

### 2.3 Naive Bayes

Naive Bayes uses Bayes' theorem to estimate class probabilities:

\[
P(y \mid x) \propto P(y) P(x \mid y)
\]

Its **naive** assumption treats the features as *conditionally independent given the class*. Real features may be correlated, so the assumption is an approximation, not a guarantee.

- `GaussianNB`: often used with continuous numerical features; assumes a Gaussian distribution of each feature within a class.
- Naive Bayes is usually quick to fit and can work well with modest amounts of data.
- It may perform worse if the distributional assumptions are poorly matched to the features.

### 2.4 K-Means clustering

K-Means partitions data into `k` groups by iteratively assigning observations to the nearest centroid and updating the centroids. It aims to minimize **within-cluster squared distances** (inertia):

\[
\text{Inertia}=\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|^2
\]

**Strengths:** simple, fast on many datasets, interpretable cluster centers.  
**Limitations:** you must choose `k` in advance, results depend on scaling and initialization, and it favors roughly compact/convex groups. Outliers can affect the centroids.

### 2.5 Choosing `k` — Required Day 17 learning topic

**Elbow method:** Fit K-Means with several `k` values and plot inertia. Inertia decreases (or stays the same) as `k` increases; look for a bend where additional groups yield much smaller improvements. The elbow is **not always clear**.

**Silhouette score:** Measures how close each observation is to its own cluster compared with neighboring clusters. Its mean is typically between **−1 and +1**; higher scores usually mean better separation under this distance-based criterion. It may favor fewer or more compact clusters than the real-world concept of "correct" groups.

**Our practical decision:** Examine both plots for `k = 2 ... 10`. State which `k` achieves the highest silhouette score, and mention whether the elbow suggests another plausible choice. Neither method proves the existence of a true number of clusters.

### 2.6 Hierarchical clustering

Hierarchical clustering builds a hierarchy of groups. **Agglomerative** clustering starts with individual observations and repeatedly merges groups until a stopping rule is met.

- A **dendrogram** shows the merging structure.
- A horizontal cut through the dendrogram helps visualize a possible number of clusters.
- **Ward linkage** attempts to minimize the increase in within-cluster variance when groups merge; it commonly prefers compact groups.
- Hierarchical methods can be costly on large datasets.

### 2.7 DBSCAN

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** groups observations in dense regions and can mark sparse observations as **noise** (`-1`).

- **`eps`:** neighborhood distance threshold.
- **`min_samples`:** minimum number of observations for a dense core neighborhood (including the point itself).
- **No need to provide `k`:** the number of clusters emerges from the data and selected parameters.
- **Advantage:** can recover irregularly shaped clusters and identify noise/outliers.
- **Limitation:** sensitive to `eps`, `min_samples`, scaling, and varying cluster densities; performance can worsen in high dimensions.

**When DBSCAN may beat K-Means:** for non-convex shapes (such as two interlocking moons), clusters of similar density, and noisy data. **When not:** when density varies greatly or cluster centers and a fixed `k` are the practical objective. There is **no universal best clustering algorithm**.

### 2.8 Choosing precision and recall

For a given class:

\[
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN}
\]

- **Precision** answers: Of the observations predicted as this class, what fraction were correct?
- **Recall** answers: Of the observations truly in this class, what fraction did we find?
- **Macro average** calculates a separate metric for each class, then averages them equally. This is useful for comparison across multiple classes.

Use the **same dataset and train/test split** to compare classifiers. Conduct cross-validation on the training split; do not select a model by repeatedly using the held-out test set.

## 3. Required Code Practice — Questions 9 and 10

The Week 3 syllabus assigns **Q9 and Q10** to Day 17. Each question has its own Python solution.

| Question | File | Task | Saved output |
|---|---|---|---|
| **Q9 — Medium** | [`q09_kmeans_elbow_silhouette.py`](q09_kmeans_elbow_silhouette.py) | Fit K-Means for `k=2` to `10`, plot inertia and silhouette, pick `k` | `results/q09_elbow_silhouette.png`, `results/q09_kmeans_metrics.csv` |
| **Q10 — Medium** | [`q10_naive_bayes_vs_svm.py`](q10_naive_bayes_vs_svm.py) | Train Naive Bayes and SVM on the same data, compare precision and recall | `results/q10_confusion_matrices.png`, `results/q10_naive_bayes_vs_svm_metrics.csv` |

### Q9: Elbow curve and silhouette analysis

**Dataset:** built-in *Iris* (150 observations, four numeric features). Features are standardized before clustering; known species labels are **not** used to train the clustering algorithm.

**Method:** Try `k=2...10`, fit `KMeans(n_init=10, random_state=42)` at each value, calculate inertia and mean silhouette, and save the plots and CSV table.

```bash
python q09_kmeans_elbow_silhouette.py
```

**Interpretation:** Use the maximum silhouette score as the explicit selection rule for this exercise, then discuss whether the elbow curve points to the same or a different plausible `k`.

### Q10: Naive Bayes versus SVM

**Dataset:** built-in *Wine* classification (178 observations, 13 numeric features, 3 target classes).

**Method:** Use an 80/20 **stratified split** and five-fold stratified cross-validation on the training subset. Compare `GaussianNB` with a scaled RBF SVM. Calculate class-specific metrics, **macro precision**, **macro recall**, F1, and confusion matrices on the held-out test set.

```bash
python q10_naive_bayes_vs_svm.py
```

**Interpretation:** State which model has the stronger **training-cross-validation macro recall**, then report each model's held-out test precision and recall. Explain any mismatch. Do not claim either algorithm universally wins.

## 4. Additional Practical — Hierarchical Clustering and DBSCAN

The syllabus includes hierarchical clustering and DBSCAN as Day 17 topics, even though it does **not** assign a separate numbered code question for them. This extra exercise covers both.

**Demo data:** reproducible **synthetic two-moons** data (`make_moons`, 450 observations) so curved groups are visible. It is a conceptual illustration, **not** a real-world dataset.

**Algorithms:** K-Means with `k=2`, agglomerative clustering with Ward linkage and two groups, and DBSCAN with `eps=0.18`, `min_samples=6`. Show a small-sample dendrogram and a side-by-side cluster comparison.

```bash
python hierarchical_dbscan_demo.py
```

**Generated charts:** `results/clustering_methods_comparison.png` and `results/hierarchical_dendrogram.png`.

Compare the *shape* of the discovered groups, not just their number. DBSCAN can label some observations `-1` rather than forcing each point into a group. The algorithms have not undergone equal-budget hyperparameter tuning, so the plot is **not a fair leaderboard**.

## 5. Setup and Run Instructions

**Requires:** Python 3.10+, CPU only; no API keys or external dataset downloads. Packages may require internet for first-time installation.

Open a terminal in `day-17-svm-clustering/` and run:

```bash
python -m pip install -r requirements.txt
python q09_kmeans_elbow_silhouette.py
python q10_naive_bayes_vs_svm.py
python hierarchical_dbscan_demo.py
```

All outputs are written to the `results/` folder. If using a notebook or Google Colab, upload the folder or adapt the file paths before running.

## 6. Executed Results and Interpretation

The three scripts were executed with **scikit-learn 1.8.0**. These are measured example results, not guarantees. Run the scripts yourself before committing and add your observations.

| Experiment | Measured result | What it means |
|---|---|---|
| Q9: K-Means, `k=2` | Inertia **222.3617**, silhouette **0.5818** | Highest silhouette score among `k=2...10` |
| Q9: K-Means, `k=3` | Inertia **139.8205**, silhouette **0.4599** | Lower inertia but also lower silhouette; smaller inertia alone is not enough to pick `k` |
| Q10: Naive Bayes, 5-fold training CV | Macro recall **0.9762** | Strong performance on this dataset and split |
| Q10: SVM, 5-fold training CV | Macro recall **0.9878** | SVM is selected by the predefined CV macro-recall rule |
| Q10: Naive Bayes, held-out test | Accuracy **0.9722**, macro precision **0.9744**, macro recall **0.9762** | Better test macro recall of the two in this run |
| Q10: SVM, held-out test | Accuracy **0.9722**, macro precision **0.9778**, macro recall **0.9667** | Better test macro precision, but lower test macro recall |
| Clustering: K-Means / hierarchical | **2 clusters** each | Both were configured to return two groups |
| Clustering: DBSCAN | **2 clusters**, **8 noise points** | Found two density-connected groups and marked some observations as noise |

**My interpretation of Q9:** `k=2` wins on the mean silhouette metric for the standardized Iris features, even though the dataset contains three known Iris species. This does **not** mean that Iris has only two real species. The elbow curve can be interpreted differently, illustrating why cluster selection depends on the objective.

**My interpretation of Q10:** The SVM had the higher cross-validation macro recall and would be selected by that rule. Both models had the same test accuracy, but Naive Bayes had higher **test recall** and SVM higher **test precision**. A model chosen on one split is not guaranteed to win every metric on new data.

**My interpretation of clustering:** K-Means and Ward hierarchical clustering create compact partitioning structures. On the curved two-moons illustration, DBSCAN can better follow density-connected shapes and separately label noise. A different `eps` or dataset may reverse that outcome.

### Reflection questions

1. How does SVM's `C` differ from RBF `gamma`?
2. What does the conditional-independence assumption mean in Naive Bayes?
3. Why does inertia always decrease or stay constant as `k` increases?
4. Why can the silhouette-optimal `k` differ from the number of known classes?
5. When is a dendrogram useful, and how does DBSCAN identify noise?
6. Why might an SVM win on cross-validation recall but lose on test recall?

## 7. Folder Structure

```text
day-17-svm-clustering/
├── README.md
├── requirements.txt
├── q09_kmeans_elbow_silhouette.py
├── q10_naive_bayes_vs_svm.py
├── hierarchical_dbscan_demo.py
└── results/
    ├── q09_kmeans_metrics.csv
    ├── q09_elbow_silhouette.png
    ├── q09_kmeans_elbow_silhouette_run.txt
    ├── q10_naive_bayes_vs_svm_metrics.csv
    ├── q10_confusion_matrices.png
    ├── q10_naive_bayes_vs_svm_run.txt
    ├── clustering_methods_comparison.png
    ├── hierarchical_dendrogram.png
    └── hierarchical_dbscan_demo_run.txt
```

## 8. Daily GitHub Commit

Place `day-17-svm-clustering/` inside the **existing System 7 repository** (do not start a new repository). After checking the scripts and adding your reflections:

```bash
git add day-17-svm-clustering/
git commit -m "day-17-svm-clustering"
git push
```

Make **one day-wise commit** for Day 17; do not batch multiple days into one commit.

---

**Source:** *System 7 — Week 3 Syllabus*, Module B, Daily Plan (Day 3 / Day 17), Learning Topic "Choosing k in K-Means", Code Practice Questions 9–10, and GitHub Instructions.  
**Scope note:** The extra demonstrations, examples, calculations, and reflection questions expand the syllabus for practice. The practical datasets and results are explicitly identified above.
