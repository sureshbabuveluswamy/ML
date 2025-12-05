# %% [markdown]
# Linear Regression Assignment

# %%
## Import all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
%matplotlib inline

# %%
## import model related libraries

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from math import pi

# %%

# RMSLE function definition
#RMSLE = sqrt( (1/n) * Σ (log(pred+1) - log(actual+1))² ) 

def rmsle(y_true, y_pred):
    y_pred = np.where(y_pred < 0, 0, y_pred)  # avoid negative predictions
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# %%
# importing data to dataframe using pandas  
train = pd.read_csv("bike_train.csv")
test = pd.read_csv("bike_test.csv")

# %%
# Display the shape of the dataset
print(train.shape)

# %%
# Display the first 5 rows of the training dataset
train.head(5)

# %%
print (train.info()) #2.4 Summary of  Training data set

# %%
print (train.describe()) #2.5 Statistical Summary of all numerical attributes

# %% [markdown]
# Q1. Examine dataset size, missing values, and feature types. 
# answer : no missing value ,  categorical feature are season, holiday, workingday, weather & continuous variable are temp, atemp, humidity, windspeed
# Q2. Visualize relationships between key features and the target variable (count).
# 
# Q3: Suggest which variables are likely to be most informative.
# Holidays and working day plays important in categorical data
#  

# %%
#visualization of data is kept standlone for better readability
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('bike_train.csv')

# Convert datetime column to datetime type and extract hour and month for temporal analysis
train_df['datetime'] = pd.to_datetime(train_df['datetime'], format='%d/%m/%y %H:%M')
train_df['hour'] = train_df['datetime'].dt.hour
train_df['month'] = train_df['datetime'].dt.month

# Set plotting style
sns.set_style('whitegrid')

# 1. Histograms for continuous features and target
continuous_features = ['temp', 'atemp', 'humidity', 'windspeed', 'count']
train_df[continuous_features].hist(bins=30, figsize=(12, 8))
plt.suptitle('Distribution of Continuous Features and Target')
plt.show()

# 2. Boxplots of count by categorical features
categorical_features = ['season', 'holiday', 'workingday', 'weather']
plt.figure(figsize=(14, 10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=feature, y='count', data=train_df)
    plt.title(f'Count by {feature.capitalize()}')
plt.tight_layout()
plt.show()

# 3. Scatter plots for continuous features vs count
plt.figure(figsize=(15, 5))
for i, feature in enumerate(['temp', 'humidity', 'windspeed'], 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=feature, y='count', data=train_df, alpha=0.3)
    plt.title(f'Count vs {feature.capitalize()}')
plt.tight_layout()
plt.show()

# 4. Line plot for average count by hour of the day
plt.figure(figsize=(8, 5))
hourly_counts = train_df.groupby('hour')['count'].mean()
sns.lineplot(x=hourly_counts.index, y=hourly_counts.values)
plt.title('Average Bike Count by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Average Count')
plt.show()

# 5. Correlation heatmap for continuous variables
plt.figure(figsize=(8, 6))
corr = train_df[continuous_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Continuous Features')
plt.show()

# 6. Pairplot of key features and count (sample for speed)
sns.pairplot(train_df.sample(500), vars=['temp', 'humidity', 'windspeed', 'count'])
plt.suptitle('Pairplot of Selected Features and Count', y=1.02)
plt.show()


# %%
# Your dataset uses DD/MM/YY HH:MM format, so dayfirst=True is appropriate
train['datetime_parsed'] = pd.to_datetime(
    train['datetime'], 
    dayfirst=True, 
    errors='coerce'
)

test['datetime_parsed'] = pd.to_datetime(
    test['datetime'], 
    dayfirst=True, 
    errors='coerce'
)
#  Extract datetime features
def add_datetime_features(df):
    # Basic time units
    df['hour'] = df['datetime_parsed'].dt.hour
    df['day'] = df['datetime_parsed'].dt.day
    df['month'] = df['datetime_parsed'].dt.month
    df['year'] = df['datetime_parsed'].dt.year
    
    # Monday = 0 ... Sunday = 6
    df['weekday'] = df['datetime_parsed'].dt.weekday
    
    # Weekend flag
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    return df


# Apply for both train and test
train = add_datetime_features(train)
test = add_datetime_features(test)

# %%
def add_interaction_features(df):
    df["temp_feels_diff"] = df["temp"] - df["atemp"]
    df["bad_weather_peak"] = df["weather"] * df["hour"]
    df["humid_temp"] = df["humidity"] * df["temp"]
    df["work_hour"] = df["workingday"] * df["hour"]
    df["weekend_hour"] = df["is_weekend"] * df["hour"]
    df["wind_weather"] = df["windspeed"] * df["weather"]
    df["temp_hour"] = df["temp"] * df["hour"]
    df["peak_morning"] = df["hour"].isin([7,8,9]).astype(int)
    df["peak_evening"] = df["hour"].isin([17,18,19]).astype(int)
    return df

train = add_interaction_features(train)
test = add_interaction_features(test)

# %%
#  causal and registered columns are not present in test set, so drop them from train set , also target is to get predicated count , 
# hence dropping these columns will avoid data leakage  
for c in ['casual','registered']:
    if c in train.columns:
        train = train.drop(columns=[c])

# %%

#  Target & base features
y = train['count']
X = train.drop(columns=['count','datetime'])

# %%

# Cyclical encoding of time features
def cyclical_encode(series, period, prefix):
    radians = 2 * np.pi * series / period
    return pd.DataFrame({
        f'{prefix}_sin': np.sin(radians),
        f'{prefix}_cos': np.cos(radians)
    })

# Apply cyclical enc
X_cyc = pd.concat([
    cyclical_encode(X['hour'], 24, 'hour'),
    cyclical_encode(X['month'], 12, 'month'),
    cyclical_encode(X['weekday'], 7, 'weekday')], axis=1)

X_model_fe = X.drop(columns=['hour','day','month','weekday']).reset_index(drop=True)
X_model_fe = pd.concat([X_model_fe, X_cyc.reset_index(drop=True)], axis=1)

# same for test
test_model = test.drop(columns=['datetime']).copy()
test_cyc = pd.concat([
    cyclical_encode(test_model['hour'], 24, 'hour'),
    cyclical_encode(test_model['month'], 12, 'month'),
    cyclical_encode(test_model['weekday'], 7, 'weekday')], axis=1)
test_model = test_model.drop(columns=['hour','day','month','weekday']).reset_index(drop=True)
test_model = pd.concat([test_model, test_cyc.reset_index(drop=True)], axis=1)

# %%
train.head(5)

# %%
# Define columns (adjust if necessary)
numeric_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'year', 'day','temp_feels_diff','bad_weather_peak','humid_temp','work_hour','weekend_hour','wind_weather','temp_hour','peak_morning','peak_evening']
cat_cols = ['season', 'weather', 'holiday', 'workingday', 'is_weekend']

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
])




# %%
# Split the data into training and validation sets
#randoem_state=42 for reproducibility
#test_size=0.2 means 20% data will be used for validation
#Q5. Split data into training and validation sets and build a simple Linear Regression model.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
assert not np.any(np.isnan(X_train)), "NaNs found in X_train"
assert not np.any(np.isinf(X_train)), "Infinite values found in X_train"
assert not np.any(np.isnan(y_train)), "NaNs found in y_train"
assert not np.any(np.isinf(y_train)), "Infinite values found in y_train"

# Adjust feature columns to match X_train columns
numeric_cols = [col for col in numeric_cols if col in X_train.columns]
cat_cols = [col for col in cat_cols if col in X_train.columns]

# %%
print(X_train.columns)

# %%
# Define your models dictionary with all models (including polynomial with scaler after poly)
models = {
    'Linear': Pipeline([
        ('pre', preprocessor), 
        ('model', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('pre', preprocessor), 
        ('model', RidgeCV(alphas=[0.1,1.0,10.0]))
    ]),
    'Polynomial Regression (Degree 2)': Pipeline([
        ('pre', preprocessor),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('scaler', StandardScaler()),  # ITSR fine tuning
        ('model', RidgeCV(alphas=[0.1, 1.0, 10.0]))
    ]),
    'Lasso Polynomial Regression': Pipeline([
        ('pre', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),  # ITSR fine tuning
        ('model', Lasso(alpha=0.001, max_iter=10000))  # Increased max_iter, ITSR fine tuning
    ]),
    'RandomForest': Pipeline([
        ('pre', preprocessor), 
        ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ]),
    # GradientBoosting wrapped with log-transform for further tuning added below
}

# %%
#tuning: Wrap GradientBoosting with TransformedTargetRegressor for log-transform of target
from sklearn.compose import TransformedTargetRegressor

gb_pipe = Pipeline([
    ('pre', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

regressor_log = TransformedTargetRegressor(
    regressor=gb_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

# Add tuned GradientBoosting regressors to models dict for easy access if needed
models['GradientBoosting (Tuned Log-Transform)'] = regressor_log

# Hyperparameter tuning for the GradientBoosting regressor with log-transform target
param_grid = {
    'regressor__model__n_estimators': [100, 200, 300],
    'regressor__model__max_depth': [3, 4, 5],
    'regressor__model__learning_rate': [0.01, 0.1, 0.2],
    'regressor__model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    regressor_log,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_log_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)


#print("Best parameters for GradientBoosting:", grid_search.best_params_)

best_gb_model = grid_search.best_estimator_

# %%
results = {}
warnings.filterwarnings
for name, pipe in models.items():
    if name == 'GradientBoosting (Tuned Log-Transform)':
        # Use the best tuned Gradient Boosting model
        y_pred = best_gb_model.predict(X_val)
    
    else:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
    score = rmsle(y_val, y_pred)
    results[name] = score
    print(f"{name} RMSLE: {score}")

# Convert results dictionary to DataFrame and display
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'RMSLE']).sort_values(by='RMSLE')
print(results_df)

# %%
X_test=test
train.head(5)



# %%
test_preds = best_gb_model.predict(X_test)
test_preds = np.where(test_preds < 0, 0, test_preds)
test_preds_rounded = np.round(test_preds).astype(int)

submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': test_preds_rounded
})

submission.to_csv('submission.csv', index=False)
print("Saved submission to submission.csv")


