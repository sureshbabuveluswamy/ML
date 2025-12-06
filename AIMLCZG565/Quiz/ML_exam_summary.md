# Machine Learning Exam: Complete Summary

## Question 1: Sigmoid Function Properties
**Question:** Choose the appropriate option which indicate the correct properties of a sigmoid function?

**Correct Answer:** (b) It acts as a squashing function because it maps the whole real axis into a finite interval.

**Explanation:**
- Sigmoid is a non-linear function that maps all real inputs (-∞ to +∞) to (0, 1)
- It is NOT linear (rules out option a)
- It is used for binary classification, NOT multiclass (rules out option c)
- Its derivative is simple to compute: σ'(x) = σ(x)(1-σ(x)), NOT complex (rules out option d)
- The squashing property is the defining characteristic - it bounds unbounded inputs to finite interval

---

## Question 2: Regression Evaluation Metric with Outliers
**Question:** Which regression evaluation metric should you consider when you have many outliers and don't want to account for them?

**Correct Answer:** (a) Mean Absolute Error

**Explanation:**
- **MAE** treats all errors linearly without amplification - robust to outliers
- **MSE** squares errors, amplifying large errors dramatically - sensitive to outliers
- **RMSE** still involves squaring, making it also sensitive to outliers
- When outliers exist and shouldn't dominate evaluation, MAE is the correct choice
- Formula: MAE = (1/n)∑|y_i - ŷ_i|

---

## Question 3: Linear vs Quadratic Curve for Noisy Data
**Question:** True/False - For very noisy data, it is better to use a quadratic curve rather than a linear curve

**Correct Answer:** False

**Explanation:**
- **More complex models overfit noisy data** - quadratic would fit noise, not true pattern
- **Bias-variance tradeoff**: simpler linear model has lower variance, better generalization
- Simpler models perform better on noisy data by avoiding overfitting
- Increasing complexity only justified when underlying relationship is actually non-linear
- For noisy data, use regularization or simpler models

---

## Question 4: Linear Basis Functions
**Question:** Which of these is a linear basis function?

**Correct Answer:** (d) All of the above

**Explanation:**
- **Linear basis function** means linear in parameters, not input
- Model form: y(x,w) = w₀ + Σ w_j φ_j(x)
- **Polynomial**: x^j terms are linear in parameters w
- **Gaussian**: RBF functions are linear in parameters w
- **Sigmoidal**: σ(x) terms are linear in parameters w
- All three are used in linear basis function models while being non-linear in input x

---

## Question 5: Components of a Learning Problem
**Question:** What are the things we must identify to have a learning problem?

**Correct Answer:** (b) All of the above

**Explanation:**
Based on Tom Mitchell's definition of learning: "A computer program learns from experience E with respect to some class of tasks T and performance measure P"
- **(a) Class of tasks (T)**: What the system should learn to do
- **(c) Performance measure (P)**: How to quantify improvement
- **(d) Source of experience (E)**: Training data
- All three are mandatory for a well-defined learning problem

---

## Question 6: Logistic Regression and Output Categories
**Question:** Logistic regression can be applied only when output Y is Boolean?

**Correct Answer:** False

**Explanation:**
- **Binary logistic regression**: 2 categories using sigmoid
- **Multinomial logistic regression**: 3+ categories using softmax
- **One-vs-Rest strategy**: Decomposes multiclass into multiple binary problems
- Sigmoid for binary (Y ∈ {0,1}), Softmax for multiclass
- Logistic regression extends to any number of classes

---

## Question 7: Evaluation Metrics for Continuous Output
**Question:** If y is continuous, which evaluation metric is correct?

**Correct Answer:** (d) SSD (Sum of Squared Differences)

**Explanation:**
- **Classification metrics** (Recall, Accuracy, Precision) require discrete categories - not for continuous
- **SSD/SSE** measures squared differences for continuous values: Σ(y_i - ŷ_i)²
- Other continuous metrics: MAE, MSE, RMSE, R²
- SSD is fundamental regression metric for continuous outputs

---

## Question 8: Basis Functions in Linear Regression
**Question:** Which can be used as basis functions in linear regression models?

**Correct Answer:** (d) All of the above

**Explanation:**
- All three are valid basis functions used in linear regression with basis function expansion
- **Polynomial**: y = w₀ + w₁x + w₂x² + ... (standard polynomial regression)
- **Gaussian**: RBF networks, radial basis function regression
- **Sigmoidal**: Local basis functions for complex curve fitting
- All maintain linearity in parameters while enabling non-linear input-output relationships

---

## Question 9: Sigmoid Function Classification
**Question:** Which functions belong to the sigmoid class?

**Correct Answer:** (a) σ(x) = 1/(1 + e^(-x))

**Explanation:**
- This is the **logistic sigmoid** - the standard sigmoid function
- **Properties of sigmoid**: S-shaped curve, bounded (0,1), monotonically increasing, single inflection point
- **sin(x)**: Oscillating wave, not monotonic, multiple inflections - NOT sigmoid
- **cos(x)**: Oscillating wave, not monotonic, multiple inflections - NOT sigmoid
- **|x|**: V-shaped, not smooth (not differentiable at 0) - NOT sigmoid

---

## Question 10: Improper Learning Rate Consequences
**Question:** What happens with improper learning rate in gradient descent?

**Correct Answer:** (d) All the given options

**Explanation:**
- **Too high learning rate**:
  - Causes oscillations around minimum
  - May diverge entirely
- **Too low learning rate**:
  - Causes very slow convergence
  - May get stuck in local minima with inadequate exploration
- All three consequences are possible depending on how improper the rate is

---

## Question 11: Parameter Calculation with Large Features
**Question:** m=20,000 examples, n=100,000 features. Calculate θ using linear regression?

**Correct Answer:** (b) Use gradient descent; (X^T X)^(-1) will be very slow

**Explanation:**
- **Normal equation complexity**: O(n³) = O(10^15) operations for matrix inversion
- **Gradient descent complexity**: O(t × n × m) ≈ O(1.2 × 10^13) for ~1000 iterations
- Gradient descent is ~80-100× faster with 100,000 features
- Recommendation: Use GD when n ≥ 10,000 features
- With this many features, matrix inversion is computationally prohibitive

---

## Question 12: Cross-Entropy in Logistic Regression
**Question:** Choose the correct statement for logistic regression cost function?

**Correct Answer:** (a) If P(Y=1|X,θ) = 0 and actual label is 1, then cross entropy cost will be very high

**Explanation:**
- **Cross-entropy formula**: Cost = -log(p) when y=1
- When p=0 and y=1: Cost = -log(0) = ∞ (infinite penalty)
- This is the worst possible prediction (completely wrong with high confidence)
- **Option b**: p=0, y=0 → Cost = 0 (perfect prediction)
- **Option c**: P(Y=0)=0 means P(Y=1)=1, with y=1 → Cost = 0 (perfect)
- **Option d**: p=1, y=1 → Cost = 0 (perfect prediction)

---

## Question 13: False Statement About Regression
**Question:** Which statement is FALSE regarding regression?

**Correct Answer:** (a) It discovers causal relationships

**Explanation:**
- **Correlation ≠ Causation**: Regression identifies associations, not causality
- Causality requires theoretical justification BEFORE analysis, cannot be discovered by regression
- **Option b** TRUE: Regression relates inputs to outputs
- **Option c** TRUE: Regression used for interpretation of relationships
- **Option d** TRUE: Regression used for prediction
- Only option (a) is false - regression cannot discover causal relationships

---

## Question 14: Logistic Regression Model Type
**Question:** Logistic regression is a discriminative model?

**Correct Answer:** True

**Explanation:**
- **Discriminative model**: Models P(Y|X) directly
- Learns decision boundary between classes
- Cannot generate new data
- Logistic regression directly estimates: P(Y=1|X,θ) = 1/(1+e^(-θ^T X))
- Contrast: Naive Bayes is generative, models P(X|Y) and P(Y)

---

## Question 15: Gradient Descent with Non-Differentiable Functions
**Question:** Can gradient descent be used for non-differentiable functions?

**Correct Answer:** False

**Explanation:**
- **Gradient descent requires differentiable functions** - it's defined as first-order iterative algorithm
- **Why**: Algorithm computes gradient ∇f(θ) to determine descent direction
- At non-differentiable points, gradient doesn't exist (e.g., |x| at x=0)
- **Alternatives for non-differentiable functions**:
  - Subgradient methods
  - Smoothing techniques
  - Proximal methods
- These are specialized extensions, not standard gradient descent

---

## Question 16: Learning Rate Convergence Pattern
**Question:** J(θ) decreases quickly then levels off with α=0.3. Conclusion?

**Correct Answer:** (d) α=0.3 is an effective choice

**Explanation:**
- **Observed pattern**: Quick decrease → smooth convergence → levels off
- This is **ideal convergence behavior** indicating good learning rate selection
- **Too low α**: Would cause very slow initial decrease
- **Too high α**: Would cause oscillations, not smooth convergence
- **Perfect LR**: Rapid initial progress with smooth asymptotic approach to minimum
- No adjustment needed - continue with α=0.3

---

## Question 17: MLE Assumptions in Linear Regression
**Question:** Choose correct statement for maximum likelihood estimation?

**Correct Answer:** (a) MLE assumes noise in the target variable

**Explanation:**
- **Standard linear regression model**: y_i = X_i β + ε_i where ε ~ N(0, σ²)
- **Noise assumption**: Only in target variable (y), NOT in features (X)
- Features are treated as fixed/observed without measurement error
- **Option b** FALSE: X variables assumed noise-free
- **Option c** & **d** FALSE: MLE explicitly requires Gaussian noise in y
- Extension: Errors-in-variables models would assume noise in both X and y

---

## Question 18: Correct Data Preprocessing Order
**Question:** Which is the correct preprocessing order?

**Correct Answer:** (d) Normalize the data → PCA → training

**Explanation:**
- **MUST normalize BEFORE PCA** - critical requirement
- **Why**: PCA is sensitive to scale; features with larger scale dominate
- Example: Unscaled PCA accuracy 35%, Scaled PCA accuracy 96%
- **Option b/c**: Re-normalizing PCA output not standard practice
- **Option c**: Never apply PCA to unnormalized data (biased components)
- Standard sklearn pipeline: StandardScaler → PCA → Train

---

## Question 19: Cross-Entropy Loss Divergence
**Question:** What happens to cross-entropy loss as predicted probability diverges from actual label?

**Correct Answer:** (d) Increases

**Explanation:**
- **Fundamental property**: Cross-entropy loss increases with divergence
- **Formula**: Loss = -log(p) when y=1
- **Example**: 
  - p=0.99, y=1 → Loss ≈ 0.01 (low)
  - p=0.50, y=1 → Loss ≈ 0.69 (higher)
  - p=0.01, y=1 → Loss ≈ 4.61 (very high)
- **Design purpose**: Heavily penalizes confident wrong predictions
- **During training**: As predictions improve (converge to labels), loss decreases

---

## Summary of Key Concepts

### Activation Functions & Loss Functions
- Sigmoid: Non-linear, bounded, used for binary classification
- Cross-entropy: Increases with prediction divergence, optimal loss for classification

### Model Selection & Complexity
- Simpler models preferred for noisy data (avoid overfitting)
- Basis functions enable non-linear modeling while staying linear in parameters

### Preprocessing & Optimization
- Always normalize BEFORE PCA (critical for correct components)
- Learning rate tuning: Quick convergence + smooth leveling = good choice
- Gradient descent requires differentiable functions

### Regression vs Classification
- Regression metrics: MAE, MSE, RMSE for continuous outputs
- Classification metrics: Accuracy, Precision, Recall for discrete outputs
- Logistic regression is discriminative, models P(Y|X)

### Fundamental Principles
- Learning problems need: Task (T), Performance measure (P), Experience (E)
- Logistic regression works for binary and multiclass problems
- Regression identifies associations, not causation
- MLE in linear regression assumes noise only in target variable

