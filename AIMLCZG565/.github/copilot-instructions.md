# AI Coding Agent Instructions for ML Learning Workspace

This workspace contains machine learning educational content and a bike-sharing demand prediction assignment. AI agents should understand the following key patterns and structures:

## Project Structure

- **`Assignment/`**: Primary machine learning project for bike-sharing demand prediction using regression
  - `2025ab05129_assignment.ipynb` - Main working notebook with full ML pipeline
  - CSV files: `bike_train.csv`, `bike_test.csv`, `SampleSubmission.csv`
  - Multiple submission versions tracking project iterations
  
- **`Labware_Machine Learning/`**: Educational lab materials organized by ML concepts
  - **LabCapsule 1**: Pandas basics and ML fundamentals (DecisionTreeClassifier on Zoo dataset)
  - **LabCapsule 2**: Regression techniques (Simple Linear, Least Squares, Polynomial, Gradient Descent)
  - **LabCapsule 3**: Classification techniques (Logistic Regression - binary and multinomial)

## Bike-Sharing Assignment Architecture

The assignment follows a systematic ML workflow in Jupyter notebooks:

1. **Data Loading & Exploration**: Read CSV files, examine shape, dtypes, missing values
2. **Feature Engineering**: 
   - Parse datetime column (`dayfirst=True` for DD/MM/YY format)
   - Extract temporal features: hour, day, month, year, weekday, is_weekend
   - **Cyclical encoding** for periodic features using sin/cos transformation (e.g., `hour_sin = sin(2π*hour/24)`)
   - Drop data-leakage columns: `casual`, `registered` (not in test set)
3. **Preprocessing Pipeline**:
   - Numeric features: SimpleImputer(median) → StandardScaler
   - Categorical features: SimpleImputer(most_frequent) → OneHotEncoder(sparse_output=False)
   - Use sklearn's Pipeline + ColumnTransformer for modular transforms
4. **Train-Validation Split**: 80/20 split with `random_state=42` for reproducibility
5. **Model Training**: Multiple regression models compared
   - LinearRegression, RidgeCV, LassoCV, RandomForestRegressor, GradientBoostingRegressor
   - Models wrapped in Pipeline with preprocessor for clean train/predict workflow
6. **Evaluation Metric**: **RMSLE** (Root Mean Squared Logarithmic Error)
   - Formula: `sqrt(mean((log(pred+1) - log(actual+1))²))`
   - Handles skewed distributions and zero-clipping: `pred = max(0, pred)`

## Common Patterns

- **Random state**: Always use `random_state=42` for reproducible splits and model initialization
- **Datetime handling**: Input format is `DD/MM/YY HH:MM` - use `pd.to_datetime(..., dayfirst=True)`
- **Target variable**: `count` column in bike dataset; separate into y for modeling
- **Test predictions**: Generate predictions on test set without target for submission (see `submission.csv`)
- **Jupyter workflow**: Use `%matplotlib inline` for inline plots; organize cells by task (import → explore → preprocess → train → evaluate)

## Lab Material Conventions

- Each lab notebook follows: Task 1 (Import) → Task 2 (Prepare) → Task 3 (Build) → Task 4 (Visualize) → Task 5 (Evaluate)
- Imports are verbose with comments explaining each library's role
- Models accessed via sklearn: e.g., `from sklearn.linear_model import LinearRegression`
- Mathematical notation included in markdown cells to explain algorithms

## Tips for Agents

- When assisting with the assignment, focus on the main notebook (`2025ab05129_assignment.ipynb`) for model improvements
- Lab materials are reference implementations—use them to understand regression/classification patterns, not to modify
- For model debugging: Check preprocessing output shapes, handle missing values, verify feature scales before training
- The runtime warnings about "divide by zero" and "overflow" suggest numerical instability in predictions—investigate model coefficient magnitudes and feature scaling
- CSV files are included locally; no external data fetching needed
- Submission format: Predictions for test set aligned with `SampleSubmission.csv`
