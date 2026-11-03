# System 7 — Day 19: Deep Learning & Neural Networks

**Week:** 3 — Model Evaluation, Advanced ML & Deep Learning  
**Module:** C — Deep Learning & Neural Networks  
**Mentor:** Saad Hassan Saadi  
**GitHub folder:** `day-19-neural-networks`  
**Suggested commit message:** `day-19-neural-networks`

## 1. Day 19 Overview

The Week 3 syllabus assigns **perceptrons, neural-network fundamentals, forward propagation, backpropagation, gradient descent, SGD and Adam, and the TensorFlow/Keras workflow** to Day 19. The required code questions are **Q14 (NumPy AND-gate perceptron)** and **Q15 (Keras Sequential model on tabular data; plot loss; compare SGD and Adam)**. The practical task is to **predict student results with a neural network**.

### Learning objectives

By the end of this day, I should be able to:

1. Explain what an artificial neuron, weight, bias, activation, layer, and perceptron are.
2. Calculate a neuron's weighted sum and activation output.
3. Distinguish forward propagation from backpropagation.
4. Explain loss functions and how gradient descent updates parameters.
5. Recognize when to use **ReLU**, **sigmoid**, **softmax**, and **linear** outputs.
6. Describe how **learning rate**, **batch size**, and **epochs** affect training.
7. Implement a single perceptron in NumPy and verify its AND-gate truth table.
8. Build, compile, fit, and evaluate a `keras.Sequential` model on tabular data.
9. Compare SGD and Adam using validation and held-out test data without leakage.
10. Predict scores in a **synthetic student-results demonstration**, while acknowledging that the results do not establish real-world predictive validity.

## 2. Core Concepts and Learning Notes

### 2.1 Artificial neural networks and the perceptron

An **artificial neural network (ANN)** contains connected computation units called **neurons**. Every neuron receives input values, multiplies them by trainable **weights**, adds a trainable **bias**, and applies an **activation function**.

For one neuron:

\[
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
\]

\[
a = f(z)
\]

| Term | Plain-English meaning |
|---|---|
| Input `x` | A number describing an example, such as study hours |
| Weight `w` | How strongly an input contributes to the neuron's calculation |
| Bias `b` | A trainable offset, allowing the decision boundary to shift |
| Activation `f` | A function applied after the weighted sum |
| Layer | A collection of neurons whose outputs feed another layer |
| Epoch | One complete pass through the training dataset |

A **single-layer perceptron** produces a binary decision using a threshold (step) activation. It can learn linearly separable logic functions such as **AND**. It cannot solve **XOR** by itself because XOR is not linearly separable.

**AND truth table:**

| `x1` | `x2` | AND output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

The perceptron update rule used in Q14 is:

\[
w \leftarrow w + \eta (y-\hat{y})x,\qquad
b \leftarrow b + \eta(y-\hat{y})
\]

where `η` is the learning rate, `y` the correct answer, and `ŷ` the predicted answer. This is the **perceptron learning rule**, not gradient-based backpropagation through a step activation.

### 2.2 Activation functions: ReLU, sigmoid, softmax, and linear

| Activation | Definition / behavior | Typical use |
|---|---|---|
| **ReLU** | `max(0, z)` | Hidden layers; introduces nonlinearity |
| **Sigmoid** | `1 / (1 + exp(-z))`; output is between 0 and 1 | Binary-classification output |
| **Softmax** | Converts a vector of logits to nonnegative values that sum to 1 | Mutually exclusive multiclass outputs |
| **Linear** | Outputs `z` without a nonlinear transformation | Regression, such as predicting a numeric exam score |
| **Step** | Outputs 0 or 1 based on a threshold | Classic perceptron; not used for ordinary backpropagation |

**Rule:** Match the output activation and loss to the target. For this day's **continuous-score regression**, the Keras model uses a **linear output** and **mean squared error** loss. For binary classification, sigmoid with binary cross-entropy is common. For mutually exclusive multiclass classification, softmax with categorical or sparse categorical cross-entropy is common.

### 2.3 Forward propagation

**Forward propagation** is the process of calculating predictions from input to output:

`Input features → weighted sums → activation functions → output prediction`

For two layers:

\[
h = f(XW_1+b_1),\qquad \hat{y}=g(hW_2+b_2)
\]

The script [`nn_forward_backward_numpy.py`](nn_forward_backward_numpy.py) demonstrates this using a small network with a hidden layer for XOR. This is an **extra learning demonstration**, not one of the two mandatory syllabus questions.

### 2.4 Loss functions and backpropagation

A **loss function** measures prediction error. Regression commonly uses **mean squared error (MSE)**:

\[
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Classification often uses **cross-entropy**. With binary targets:

\[
\mathrm{BCE}=-\frac{1}{n}\sum_{i=1}^{n}\big[y_i\ln(\hat{y}_i)+(1-y_i)\ln(1-\hat{y}_i)\big]
\]

**Backpropagation** applies the chain rule to compute how changing each network parameter changes the loss. The optimizer then updates those parameters:

\[
\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\eta\nabla_{\theta}L
\]

The NumPy demonstration includes a **finite-difference gradient check**: it compares derivatives calculated by backpropagation with derivatives approximated by small parameter changes. This is a useful way to test a handwritten derivative implementation.

### 2.5 Optimizers: SGD versus Adam

- **Stochastic Gradient Descent (SGD):** Updates parameters using gradients computed on a batch (or individual example). Simple and widely used; learning-rate selection strongly affects convergence.
- **Adam:** Uses moving averages of the first and second moments of gradients to adapt update magnitudes. It often trains quickly with reasonable defaults, but is not always the best optimizer.

**Important comparison limitation:** The exercise compares **SGD at learning rate 0.025** with **Adam at learning rate 0.005** on the same architecture, initialization, dataset split, epoch count, and batch size. Because the learning rates differ, the result compares **these training configurations**, not the optimizer algorithms in isolation. A more rigorous conclusion would require multiple seeds and learning-rate tuning restricted to training/validation data.

### 2.6 Learning rate, batch size, and epochs

| Hyperparameter | What it controls | Possible problem |
|---|---|---|
| **Learning rate** | Size of parameter-update steps | Too high: unstable/diverging loss; too low: slow progress |
| **Batch size** | Number of training samples used for one gradient estimate | Changes gradient noise, memory usage, and update frequency |
| **Epochs** | Number of full passes through the training set | Too many may overfit; too few may underfit |

Track **training and validation loss**. Lower training loss with rising validation loss can indicate overfitting. Do not select the model using the final test set.

### 2.7 Keras model-building workflow

The usual sequence is **build → compile → fit → evaluate/predict**:

```python
import keras

model = keras.Sequential([
    keras.layers.Input(shape=(6,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(1, activation="linear"),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss="mse",
    metrics=["mae"],
)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=65, batch_size=32)
predictions = model.predict(X_test)
```

**Keras backend note:** Keras 3 supports several backends. The code automatically uses **TensorFlow when installed**, otherwise **PyTorch**. The experiments included in this ZIP were executed on **Keras 3.13.2 with PyTorch backend** because TensorFlow was not installed in the execution environment. Therefore, the actual observed outcomes are **not** claimed as a TensorFlow execution or benchmark. The Keras API shown above is the common model-building workflow.

## 3. Required Code Practice — Questions 14 and 15

| Question | Level | File | Syllabus requirement |
|---|---|---|---|
| **Q14** | Easy | [`q14_perceptron_and_gate.py`](q14_perceptron_and_gate.py) | NumPy perceptron learns the AND gate |
| **Q15** | Medium | [`q15_keras_sgd_vs_adam.py`](q15_keras_sgd_vs_adam.py) | Keras Sequential model on tabular data; plot loss; compare SGD and Adam |

### Q14: Perceptron learns the AND gate

**What the program does:** Initializes two weights and a bias at zero; computes predictions using a step activation; updates the weights on mistakes; stops when it can correctly classify all four AND-gate inputs. The script checks the truth table using `assert`.

```bash
python q14_perceptron_and_gate.py
```

**Recorded result:** Converged after **6 epochs**, with weights `[0.4, 0.2]`, bias `-0.4`, and **4/4 truth-table entries correct**.

**Generated files:**

- `results/q14_training_history.csv`
- `results/q14_truth_table.csv`

### Q15: Train a Keras network and compare SGD with Adam

**Data:** 900 **synthetic** student records generated by [`student_data.py`](student_data.py). The input variables are weekly study hours, attendance percentage, average assignment score, average quiz score, previous-term average, and weekly practice sessions. The target is a fabricated numeric final score from a transparent formula plus random noise. These are **not actual student records**.

**Method:**

1. Split records into **630 training**, **135 validation**, and **135 test** rows, all with a fixed random seed.
2. Fit `StandardScaler` on training inputs only, then transform validation and test inputs without refitting.
3. Scale the numeric score target to the range 0–1 for optimization; convert model predictions back to scores out of 100 for the final metrics.
4. Use the same network architecture and initial model weights for SGD and Adam.
5. Train each for **65 epochs** with **batch size 32** and plot training/validation MSE against epoch.
6. Select the optimizer configuration based on **validation MAE**, not test accuracy.
7. Evaluate both configurations on the untouched test set and compare them with a **Ridge regression** reference model on the same data.

```bash
python q15_keras_sgd_vs_adam.py
```

**Observed run — Keras 3.13.2, PyTorch backend, seed 19:**

| Training configuration | Validation MAE (points) | Test MAE (points) | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| SGD, learning rate 0.025 | 5.6776 | 5.7317 | 7.5572 | 0.7909 |
| Adam, learning rate 0.005 | **4.5809** | **4.7112** | **5.9476** | **0.8705** |
| Ridge baseline | — | 5.0692 | 6.5307 | 0.8439 |

**Selection:** Adam had the lowest *validation* MAE in this run, so it was chosen **before** looking at held-out test metrics. A real deployment would require real student records, proper permissions, subgroup/temporal validation, and checks for spurious correlations and bias. These synthetic results do not establish real student prediction quality.

**Generated files:**

- `results/q15_sgd_vs_adam_loss.png` — training and validation loss curves
- `results/student_actual_vs_predicted.png` — test predictions versus synthetic targets
- `results/q15_validation_comparison.csv` — optimizer-selection data
- `results/q15_test_metrics.csv` — held-out evaluation and reference baseline
- `results/student_prediction_examples.csv` — 30 synthetic test predictions
- `results/synthetic_student_scores.csv` — complete generated synthetic dataset
- `results/q15_summary.json` — settings and numerical results

## 4. Practical Task — Predict Student Results Using a Neural Network

**Required syllabus task:** Predict student results using a neural network. The **Q15 script also implements this practical**, so there is no redundant second training script.

**Input:** Six features available before the simulated final assessment.  
**Target:** Simulated numeric final score on a 0–100 scale.  
**Model:** `Dense(24, ReLU) → Dense(12, ReLU) → Dense(1, Linear)`.  
**Loss:** MSE; **evaluation:** MAE, RMSE, R².  
**Baseline:** Ridge regression, trained on the identical training split.

**Interpretation:** A test MAE of 4.7112 points means the model's predictions differed from the generated test scores by **4.7112 points on average** in this specific run. Do **not** interpret this as evidence that the model can estimate real students' performance to that precision.

## 5. Additional Learning Exercise — NumPy Forward/Backpropagation

The syllabus lists forward propagation and backpropagation as learning topics. To demonstrate both without hiding the math inside a library, the optional script [`nn_forward_backward_numpy.py`](nn_forward_backward_numpy.py) constructs a tiny network and trains it to solve **XOR**, which a single perceptron cannot solve.

```bash
python nn_forward_backward_numpy.py
```

**Observed result:** The maximum absolute discrepancy between analytic and finite-difference gradients was about `7.7e-12`. After training, the network classified all **four XOR inputs correctly**; binary cross-entropy decreased from approximately `0.69799` to `0.00087`.

**Generated files:** `results/numpy_backprop_loss.csv` and `results/numpy_backprop_predictions.csv`.

This is supplemental code; it does not replace Q14 or Q15.

## 6. Repository Structure

```text
day-19-neural-networks/
├── README.md
├── requirements.txt
├── requirements-torch.txt
├── .gitignore
├── q14_perceptron_and_gate.py
├── q15_keras_sgd_vs_adam.py
├── nn_forward_backward_numpy.py
├── student_data.py
├── tests/
│   └── test_day19.py
└── results/
    ├── execution_log.txt
    ├── q14_training_history.csv
    ├── q14_truth_table.csv
    ├── numpy_backprop_loss.csv
    ├── numpy_backprop_predictions.csv
    ├── synthetic_student_scores.csv
    ├── q15_validation_comparison.csv
    ├── q15_test_metrics.csv
    ├── q15_summary.json
    ├── q15_sgd_vs_adam_loss.png
    ├── student_actual_vs_predicted.png
    └── student_prediction_examples.csv
```

## 7. How to Run

### Option A — TensorFlow backend (recommended for the syllabus)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python q14_perceptron_and_gate.py
python nn_forward_backward_numpy.py
python q15_keras_sgd_vs_adam.py
python -m unittest discover -s tests -v
```

The Q15 code detects TensorFlow automatically if it is installed and uses Keras with the TensorFlow backend. **TensorFlow was not installed in the tested environment; do not treat its execution as verified here.**

### Option B — PyTorch backend (tested in the current environment)

```bash
python -m pip install -r requirements-torch.txt
# The script will choose PyTorch automatically if TensorFlow is absent.
python q14_perceptron_and_gate.py
python nn_forward_backward_numpy.py
python q15_keras_sgd_vs_adam.py
python -m unittest discover -s tests -v
```

If you have **both backends installed** and want to explicitly choose PyTorch:

```powershell
# Windows PowerShell
$env:KERAS_BACKEND="torch"
python q15_keras_sgd_vs_adam.py
```

```bash
# macOS / Linux
KERAS_BACKEND=torch python q15_keras_sgd_vs_adam.py
```

**Dependency note:** Both TensorFlow and PyTorch can be large installations. Use **one backend** unless you specifically need both. Execution time and scores may differ across software versions, hardware, and backends.

## 8. Questions for Self-Assessment

1. Why does a neuron need weights and a bias?
2. What changes during training: the inputs or the model parameters?
3. Why can one perceptron learn AND but not XOR?
4. What is forward propagation? What does backpropagation calculate?
5. Why are activation functions necessary in a multilayer network?
6. When should I use ReLU, sigmoid, softmax, or a linear output?
7. How do learning rate, batch size, and epochs affect training?
8. What is the difference between training loss and validation loss?
9. Why must the scaler be fitted on training samples only?
10. Why must I compare with a simple baseline and avoid selecting by test score?

## 9. Verification and Limitations

**Executed in the prepared environment:**

- Q14 passed the AND truth-table assertion.
- The NumPy backpropagation exercise passed finite-difference gradient checks and the XOR truth-table assertion.
- Q15 completed 65-epoch runs for both SGD and Adam using **Keras 3.13.2 with PyTorch backend**, exported summary metrics, and saved both figures.
- All three standard-library unit tests passed.

**Limitations:** Data is synthetic; one random split and seed are not robust external validation; optimizer learning rates differ; TensorFlow backend is included as a supported installation path but was **not executed** in the available environment. Do not treat the generated numbers as evidence about real schools or students.

## 10. GitHub Daily Commit

The syllabus requires **one day-wise commit** in the existing System 7 repository; keep daily code, learning notes, data, and charts in the corresponding folder.

```bash
git add day-19-neural-networks/
git commit -m "day-19-neural-networks"
git push
```

**Day 19 complete:** Q14, Q15, neural-network theory notes, student-result practical, generated charts, recorded results, and repeatable test commands.
