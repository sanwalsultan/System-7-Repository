
## Day 13: Classification Models


## 1. Objective

The objective of this assignment is to understand four classification algorithms: Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forests.

## 2. What Is Classification?

Classification is a type of supervised Machine Learning used to predict categories or classes.

**Examples:**
- Email: Spam or Not Spam
- Student: Pass or Fail
- Review: Positive or Negative
- Transaction: Fraudulent or Legitimate

Unlike regression, which predicts numerical values, classification predicts categories.

## 3. Logistic Regression

Logistic Regression is a classification algorithm commonly used to predict the probability that an example belongs to a class.

**Example:** Predicting whether a student will pass or fail based on study hours and attendance.

The model estimates a probability and uses a decision threshold to assign a class.

**Key point:** Despite its name, Logistic Regression is commonly used for classification.

## 4. K-Nearest Neighbors (KNN)

KNN predicts a class by examining the nearest examples in the training data.

**How it works:**
1. Choose the number of neighbors, called `K`.
2. Calculate the distances between the new example and training examples.
3. Find the K nearest neighbors.
4. Assign the class based on the neighbors' classes, commonly using a majority vote.

**Example:** A new fruit may be classified by comparing its features with nearby examples of known fruits.

## 5. Decision Trees

A Decision Tree makes predictions by following a series of questions or conditions.

**Example:**
- Is attendance above 75%?
- Are marks above the required threshold?
- Predict Pass or Fail based on the learned decision rules.

Decision Trees are often easy to understand because their decisions can be represented as branches.

## 6. Random Forests

A Random Forest combines multiple decision trees to make a prediction.

For classification, the trees commonly vote on the predicted class, and the majority class is selected.

**Advantages:**
- Can learn complex patterns.
- Combines predictions from multiple trees.
- Often performs better than a single decision tree on suitable datasets.

## 7. Comparison of Classification Algorithms

| Algorithm | Basic idea | Example |
|---|---|---|
| Logistic Regression | Estimates class probability | Spam detection |
| KNN | Uses nearby examples | Fruit classification |
| Decision Tree | Follows learned decision rules | Pass/Fail prediction |
| Random Forest | Combines multiple trees | Customer churn prediction |

## 8. What Is Scikit-learn?

Scikit-learn is a Python library that provides tools for training classification models, making predictions, and evaluating their performance.

A typical workflow is:

1. Import the required libraries.
2. Load and explore a dataset.
3. Separate features and target labels.
4. Split the data into training and testing sets.
5. Select a classification algorithm.
6. Train the model using the training data.
7. Predict classes for test examples.
8. Evaluate the predictions.

## 9. Practical Activity

Build a simple classification model using Scikit-learn.

Complete these tasks:

1. Choose a beginner-friendly classification dataset.
2. Identify the input features and target labels.
3. Split the dataset into training and testing sets.
4. Train a Logistic Regression model.
5. Train a KNN or Decision Tree model.
6. Generate predictions using both models.
7. Compare their accuracy on the test set.
8. Write a short explanation of which model performed better on your test data.

**Note:** The best-performing algorithm depends on the dataset. One model is not always better than the others.

## 10. Practice Questions and Answers

**Q1. What is classification?**  
Classification predicts categories or classes from data.

**Q2. What is Logistic Regression commonly used for?**  
Predicting class probabilities and assigning examples to classes.

**Q3. What does KNN stand for?**  
K-Nearest Neighbors.

**Q4. How does KNN make predictions?**  
It examines nearby training examples and commonly uses majority voting.

**Q5. What is a Decision Tree?**  
A model that makes predictions using learned decision rules.

**Q6. What is a Random Forest?**  
A model that combines multiple decision trees.

**Q7. What is a target label?**  
The category that the model is trained to predict.

**Q8. What is model accuracy?**  
The proportion of predictions that are correct.

**Q9. Why do we use training and testing sets?**  
To train the model and evaluate it on data not used for training.

**Q10. Can one classification algorithm perform best on every dataset?**  
No. Performance depends on the data and the problem.

## 11. Conclusion

Classification models predict categories from data. Logistic Regression, KNN, Decision Trees, and Random Forests use different approaches to make predictions. Scikit-learn makes it easier to train and evaluate these models.


