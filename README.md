Dataset Allotment

The datasets are assigned as follows:

Roll Number % 4 | Datasets
0 | Advertising, WineQuality A, Student Performance B
1 | California Housing, Student Performance A, WineQuality B
2 | Advertising, Student Performance A, WineQuality B
3 | California Housing, WineQuality A, Student Performance B

Table 1: The first dataset in each row is for Linear Regression, the second is for Logistic Regression,
and the third is for Decision Tree.

Example:
If Roll Number % 4 = 0, then choose:
- Advertising for Linear Regression
- WineQuality Dataset A for Logistic Regression
- Student Performance Dataset B for Decision Tree


--------------------------------------------------
1. Linear Regression
--------------------------------------------------

1.1 Understanding the Data and Simple Curve Fitting

(a) Download the Advertising dataset. This dataset contains data on product sales along with
advertising budgets spent on TV, radio, and newspapers. It is used to explore the relationship
between advertising efforts and sales.

Download the California Housing dataset. This dataset was collected by the U.S. Census in 1990
and includes attributes such as median house value, median income, housing age, total rooms,
total bedrooms, population, households, latitude, and longitude.

(b) Plot each feature vs. label graph for both the training data and the test data.

(c) Write code to fit a curve that minimizes the squared error cost function using gradient descent
(learning rate = 0.05) on the training set. The model is defined as:

y = W^T Φ_n(x), where W ∈ R^(n+1)

Here, n is the degree of the polynomial.

For the i-th training example, the polynomial feature mapping is:

Φ_n(x_i) = [1, x_i, x_i^2, x_i^3, ..., x_i^n]^T

Let m be the number of training examples.

The squared error cost function is:

J(W) = (1 / 2m) * Σ (W^T Φ_n(x_i) − y_i)^2   for i = 1 to m

Vary n from 1 to 9, i.e., fit 9 different polynomial curves of degree 1 to 9.
Estimate the parameters W for each case.

Use the learned W to predict labels on the test data and compute the squared error on the test set.
Call this value the test error.


--------------------------------------------------
1.2 Visualization of the Fitted Curves
--------------------------------------------------

(a) Draw separate plots of all 9 fitted curves obtained for the training dataset.

(b) Plot the squared error for both training and test data against n (from 1 to 9).
The x-axis represents n, and the y-axis represents error.

Explain which value of n is most suitable for the dataset and justify your answer.


--------------------------------------------------
1.3 Regularization
--------------------------------------------------

Perform regularization on the curves that resulted in the minimum and maximum training error.

(a) Lasso Regression

Vary λ = 0, 0.25, 0.5, 0.75, 1.

J(W) = (1 / 2m) * Σ (W^T Φ_n(x_i) − y_i)^2 + λ ||W||_1

(b) Ridge Regression

Vary λ = 0, 0.25, 0.5, 0.75, 1.

J(W) = (1 / 2m) * Σ (W^T Φ_n(x_i) − y_i)^2 + λ ||W||_2^2

Plot both training and test errors for Lasso and Ridge regression.
Discuss the differences between the two methods.
State which method you prefer for this problem and why.


--------------------------------------------------
2. Logistic Regression
--------------------------------------------------

2.1 Dataset Generation

Download the WineQuality dataset. It contains chemical properties of red wine samples along with
their quality ratings.

Download the Student Performance dataset. It consists of two files which must be combined to
form the final dataset. Attributes include grades, demographics, and social and school-related
features.

The target variables are:
- "quality" for WineQuality dataset
- "G3" for Student Performance dataset

Create two modified datasets (A and B) from each dataset.


WineQuality Dataset

Dataset A (for Logistic Regression):
- Convert quality ≤ 6 to 0 (bad)
- Convert quality > 6 to 1 (good)
- Normalize all other attributes using min-max scaling to [0, 1]

Dataset B (for Decision Tree):
- Convert quality < 5 to 0 (bad)
- Convert quality = 5 or 6 to 1 (good)
- Convert quality > 6 to 2 (great)
- Normalize attributes using Z-score normalization
- Discretize each attribute into 4 equal-width bins labeled 0 to 3

Example:
If normalized values range from -0.5 to 1.5, create bins:
- Bin 0: [-0.5, 0.0]
- Bin 1: [0.0, 0.5]
- Bin 2: [0.5, 1.0]
- Bin 3: [1.0, 1.5]

Replace each value with its corresponding bin index.


Student Performance Dataset

Dataset A (for Logistic Regression):
- Convert G3 ≤ 12 to 0 (bad)
- Convert G3 > 12 to 1 (good)
- Normalize other attributes using min-max scaling

Dataset B (for Decision Tree):
- Convert G3 < 10 to 0 (bad)
- Convert G3 between 10 and 15 (inclusive) to 1 (good)
- Convert G3 > 15 to 2 (great)
- Normalize using Z-score normalization
- Discretize attributes into 4 equal-width bins (0 to 3), same as above


--------------------------------------------------
2.2 Logistic Regression
--------------------------------------------------

Use Dataset A for this section.

1. Implement a standard Logistic Regression classifier from scratch.
   Do NOT use scikit-learn.

2. Implement Logistic Regression using scikit-learn with:
   - saga solver
   - no regularization penalty

3. Perform 3-fold cross-validation for both classifiers.
   Report the mean accuracy, precision, and recall for class 1 (good).
   You may use scikit-learn utilities for computing metrics and cross-validation.
