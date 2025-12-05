# %% [markdown]
# Linear Regression Assignment

# %%
## Import all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline

# %%
## import model related libraries

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV,  Lasso 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from math import pi

# %%

# RMSLE function definition
#RMSLE = sqrt( (1/n) * Σ (log(pred+1) - log(actual+1))² ) 

def rmsle(y_true, y_pred):
    y_pred = np.where(y_pred<0, 0, y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

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
# model feature columns
numeric_cols = ['temp','atemp','humidity','windspeed','year','day']
cyc_cols = ['hour_sin','hour_cos','month_sin','month_cos','weekday_sin','weekday_cos']
cat_cols = ['season','weather','holiday','workingday','is_weekend']

# %%

# Ensure expected columns exist
for col in numeric_cols + cyc_cols + cat_cols:
    if col not in X_model_fe.columns:
        if col in cyc_cols: X_model_fe[col] = 0.0
        elif col in numeric_cols: X_model_fe[col] = X_model_fe[col] if col in X_model_fe.columns else 0.0
        else: X_model_fe[col] = 0
    if col not in test_model.columns:
        test_model[col] = 0

# %%
# Preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols + cyc_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# %%
# Split the data into training and validation sets
#randoem_state=42 for reproducibility
#test_size=0.2 means 20% data will be used for validation
#Q5. Split data into training and validation sets and build a simple Linear Regression model.
X_train, X_val, y_train, y_val = train_test_split(X_model_fe, y, test_size=0.2, random_state=42)

# %%
#liner regression , ridge regression , random forest and gradient boosting models
#Q6. To improve model performance, you may try to different models and tune hyperparameters.:

models = {
    'Linear': Pipeline([('pre', preprocessor), ('model', LinearRegression())]),
    'Ridge': Pipeline([('pre', preprocessor), ('model', RidgeCV(alphas=[0.1,1.0,10.0]))]),
    'RandomForest': Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))]),
    'GradientBoosting': Pipeline([('pre', preprocessor), ('model', GradientBoostingRegressor(n_estimators=200, random_state=42))])
}


# %%
target = 'count'

# Feature list used by models (keep these in the same order for reproducibility)
features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'hour', 'weekday', 'month', 'year'
]
X = train[features]
y = train[target]
X_test_final = test[features]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# MODEL 1: Linear Regression
# ---------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_valid)
score_lr = rmsle(y_valid, pred_lr)


# ---------------------------------------------
# MODEL 2: Polynomial Regression (Degree 2)
# ---------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_valid_poly = poly.transform(X_valid)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
pred_poly = poly_model.predict(X_valid_poly)
score_poly = rmsle(y_valid, pred_poly)

# ---------------------------------------------
# MODEL 3: Lasso Regression with Polynomial Features
# ---------------------------------------------
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_poly, y_train)
pred_lasso = lasso.predict(X_valid_poly)
score_lasso = rmsle(y_valid, pred_lasso)


# ---------------------------------------------
# MODEL 4: Random Forest
# ---------------------------------------------
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_valid)
score_rf = rmsle(y_valid, pred_rf)


# %%
print("RMSLE Scores:")
print(f"Linear Regression:        {score_lr:.5f}")
print(f"Polynomial Regression:    {score_poly:.5f}")
print(f"Lasso Polynomial:         {score_lasso:.5f}")
print(f"Random Forest:            {score_rf:.5f}")


# %%
#Q7. Summarize all results (of different models tried out) in one table (RMSLE, key observations).
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    print(name, "RMSLE:", rmsle(y_val, preds))

# %%


# %%
#based on 
final_pipe = Pipeline([('pre', preprocessor), ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
final_pipe.fit(X_model_fe, y)
test_preds = final_pipe.predict(test_model[numeric_cols + cyc_cols + cat_cols])
test_preds = np.where(test_preds < 0, 0, test_preds)
test_preds_rounded = np.round(test_preds).astype(int)

# %%
# output submission file
submission_out = "submission.csv"  

# %%
submission = pd.DataFrame({'datetime': test['datetime'], 'count_predicted': test_preds_rounded})
submission.to_csv(submission_out, index=False)
print("Saved submission to:", submission_out)


