
## Day 11: Probability, Correlation, and Data Splitting


## 1. Objective

The objective of this assignment is to understand basic probability, correlation between variables, and how datasets are divided for Machine Learning.

## 2. What Is Probability?

Probability measures how likely an event is to happen. Its value ranges from 0 to 1.

- **0:** The event is impossible.
- **0.5:** The event has a 50% chance of happening.
- **1:** The event is certain.

**Example:** When tossing a fair coin, the probability of getting heads is 1/2, or 0.5.

### Formula

Probability = Number of favorable outcomes ÷ Total possible outcomes

## 3. What Is Correlation?

Correlation measures the relationship between two variables.

There are three basic types:

- **Positive correlation:** Both variables tend to increase together.
- **Negative correlation:** One variable tends to increase when the other decreases.
- **Zero or no linear correlation:** There is no clear linear relationship between the variables.

**Example:** As the number of hours studied increases, marks may also increase. This can indicate a positive correlation, although it does not prove that studying alone caused the increase.

Correlation values range from -1 to +1.

- **+1:** Perfect positive linear correlation.
- **-1:** Perfect negative linear correlation.
- **0:** No linear correlation.

**Important:** Correlation does not necessarily mean causation.

## 4. What Is Dataset Splitting?

Dataset splitting means dividing available data into separate parts for different purposes.

The three common parts are:

### Training Set

The training set is used to teach the Machine Learning model patterns from data.

### Validation Set

The validation set is used to evaluate and tune the model during development.

### Test Set

The test set is used to evaluate the final model on data that was not used for training or model tuning.

A common example is:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

These percentages are examples; the best split depends on the dataset and problem.

## 5. Why Is Dataset Splitting Important?

Dataset splitting helps us:
- Check how well a model performs on unseen data.
- Reduce the risk of overfitting.
- Tune model settings using validation data.
- Estimate final performance using test data.

**Important:** Do not use the test set to repeatedly tune the model, because this can make the final evaluation unreliable.

## 6. Practical Activity

Complete the following tasks:

1. Calculate the probability of getting a 4 when rolling a fair six-sided die.
2. Give one example of positive correlation.
3. Give one example of negative correlation.
4. Explain why correlation does not always mean causation.
5. Divide a dataset of 1,000 records into training, validation, and test sets using a 70:15:15 split.
6. Explain the purpose of each dataset split.

## 7. Practice Questions and Answers

**Q1. What is probability?**  
Probability measures how likely an event is to happen.

**Q2. What is the probability of getting heads on a fair coin?**  
1/2, or 0.5.

**Q3. What is positive correlation?**  
When two variables tend to increase together.

**Q4. What is negative correlation?**  
When one variable tends to increase as the other decreases.

**Q5. What is the range of correlation values?**  
From -1 to +1.

**Q6. Does correlation prove causation?**  
No. Correlation alone does not prove that one variable causes another.

**Q7. What is a training set?**  
The data used to train a model.

**Q8. What is a validation set?**  
The data used to evaluate and tune a model during development.

**Q9. What is a test set?**  
The data used for final evaluation of the model.

**Q10. Why do we split a dataset?**  
To assess how well the model performs on data it has not learned from.

## 8. Conclusion

Probability helps us understand uncertainty, correlation describes relationships between variables, and dataset splitting helps us evaluate Machine Learning models more reliably.

