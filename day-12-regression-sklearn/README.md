
## Day 12: Regression Models and Scikit-learn Workflow


## 1. Objective

The objective of this assignment is to understand linear regression, multiple linear regression, the difference between regression and classification, and the basic workflow of training a Machine Learning model using Scikit-learn.

## 2. What Is Regression?

Regression is a type of supervised Machine Learning used to predict numerical values.

**Examples:**
- Predicting house prices.
- Predicting a student's marks.
- Predicting monthly sales.
- Predicting temperature.

The model learns patterns from existing data and uses them to predict values for new data.

## 3. What Is Linear Regression?

Linear regression is a Machine Learning algorithm that predicts a numerical value by learning a linear relationship between an input feature and a target.

**Example:** Predicting a house price based on its size.

A simple linear regression equation is:

`y = mx + b`

Where:
- `y` = predicted value
- `x` = input feature
- `m` = slope or coefficient
- `b` = intercept

**Example:** If house prices generally increase with house size, a linear regression model can learn this relationship from previous house data.

## 4. What Is Multiple Linear Regression?

Multiple linear regression predicts a numerical value using two or more input features.

**Example:** Predicting a house price using:
- House size
- Number of bedrooms
- Number of bathrooms
- House location, if represented appropriately as data

The model considers multiple features to estimate the target value.

**Difference:**
- Linear regression in the simple case uses one input feature.
- Multiple linear regression uses multiple input features.

## 5. Regression vs. Classification

| Regression | Classification |
|---|---|
| Predicts numerical values | Predicts categories or classes |
| Example: House price | Example: Spam or not spam |
| Example: Monthly sales | Example: Pass or fail |
| Example: Temperature | Example: Disease category |

**Remember:** Regression predicts a number, while classification predicts a class.

## 6. What Is Scikit-learn?

Scikit-learn is a Python library used for Machine Learning. It provides tools for preparing data, training models, making predictions, and evaluating performance.

It supports algorithms such as:
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forests

## 7. Basic Machine Learning Workflow

A typical Machine Learning workflow includes the following steps:

1. **Collect data:** Obtain a suitable dataset.
2. **Explore data:** Understand its features, values, and missing data.
3. **Prepare data:** Clean the data and select the features and target.
4. **Split data:** Separate training data from test data.
5. **Choose a model:** Select an appropriate algorithm.
6. **Train the model:** Use the training data to learn patterns.
7. **Make predictions:** Use the trained model on new or test inputs.
8. **Evaluate the model:** Measure how well its predictions perform.

## 8. Practical Activity: House Price Prediction

Complete the following tasks:

1. Create or find a small dataset containing house sizes and prices.
2. Identify house size as the input feature and house price as the target.
3. Divide the dataset into training and testing sets.
4. Train a Linear Regression model using Scikit-learn.
5. Predict the price of a house using its size.
6. Compare the predicted price with the actual price, where available.
7. Write a short explanation of your results.

**Expected learning outcome:** Understand how a regression model learns from existing data and predicts numerical values.

## 9. Practice Questions and Answers

**Q1. What is regression?**  
Regression predicts numerical values from data.

**Q2. What is linear regression?**  
It is an algorithm that learns a linear relationship to predict a numerical target.

**Q3. What is multiple linear regression?**  
It predicts a numerical target using multiple input features.

**Q4. What does the equation `y = mx + b` represent?**  
A simple linear relationship used for prediction.

**Q5. What is a feature?**  
A feature is an input variable used by a model.

**Q6. What is a target?**  
The target is the value the model is trained to predict.

**Q7. What is the difference between regression and classification?**  
Regression predicts numerical values; classification predicts categories.

**Q8. What is Scikit-learn?**  
It is a Python library for building and evaluating Machine Learning models.

**Q9. Why do we split data into training and testing sets?**  
To train the model and evaluate how well it performs on data not used for training.

**Q10. What is model training?**  
Model training is the process of learning patterns from training data.

## 10. Conclusion

Regression models help predict numerical values from data. Linear regression uses a linear relationship, while multiple linear regression uses multiple features. Scikit-learn provides tools to train, predict, and evaluate Machine Learning models.


