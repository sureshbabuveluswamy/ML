import json

notebook_path = '2025ab05129_assignment.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# The complete code block to restore
restored_source = [
    "# Split the data into training and validation sets\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "assert not np.any(np.isnan(X_train)), \"NaNs found in X_train\"\n",
    "assert not np.any(np.isinf(X_train)), \"Infinite values found in X_train\"\n",
    "assert not np.any(np.isnan(y_train)), \"NaNs found in y_train\"\n",
    "assert not np.any(np.isinf(y_train)), \"Infinite values found in y_train\"\n",
    "\n",
    "# Adjust feature columns to match X_train columns\n",
    "numeric_cols = [col for col in numeric_cols if col in X_train.columns]\n",
    "cat_cols = [col for col in cat_cols if col in X_train.columns]\n",
    "\n",
    "# Re-define preprocessor with filtered columns to avoid ValueError\n",
    "preprocessor = ColumnTransformer([\n",
    "    ('num', numeric_transformer, numeric_cols),\n",
    "    ('cat', categorical_transformer, cat_cols)\n",
    "])\n",
    "\n",
    "# Define models\n",
    "models = {\n",
    "    'Linear': Pipeline([\n",
    "        ('pre', preprocessor), \n",
    "        ('model', LinearRegression())\n",
    "    ]),\n",
    "    'Ridge': Pipeline([\n",
    "        ('pre', preprocessor), \n",
    "        ('model', RidgeCV(alphas=[0.1,1.0,10.0]))\n",
    "    ]),\n",
    "    'Polynomial Regression (Degree 2)': Pipeline([\n",
    "        ('pre', preprocessor),\n",
    "        ('poly', PolynomialFeatures(degree=3, include_bias=False)),\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('model', RidgeCV(alphas=[0.1, 1.0, 10.0]))\n",
    "    ]),\n",
    "    'Lasso Polynomial Regression': Pipeline([\n",
    "        ('pre', preprocessor),\n",
    "        ('poly', PolynomialFeatures(degree=2, include_bias=False)),\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('model', Lasso(alpha=0.001, max_iter=20000))\n",
    "    ]),\n",
    "    'RandomForest': Pipeline([\n",
    "        ('pre', preprocessor), \n",
    "        ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))\n",
    "    ])\n",
    "}\n",
    "\n",
    "# GradientBoosting setup\n",
    "from sklearn.compose import TransformedTargetRegressor\n",
    "gb_pipe = Pipeline([\n",
    "    ('pre', preprocessor),\n",
    "    ('model', GradientBoostingRegressor(random_state=42))\n",
    "])\n",
    "\n",
    "regressor_log = TransformedTargetRegressor(\n",
    "    regressor=gb_pipe,\n",
    "    func=np.log1p,\n",
    "    inverse_func=np.expm1\n",
    ")\n",
    "\n",
    "models['GradientBoosting (Tuned Log-Transform)'] = regressor_log\n",
    "\n",
    "# Hyperparameter tuning\n",
    "param_grid = {\n",
    "    'regressor__model__n_estimators': [100, 200, 300],\n",
    "    'regressor__model__max_depth': [3, 4, 5],\n",
    "    'regressor__model__learning_rate': [0.01, 0.1, 0.2],\n",
    "    'regressor__model__min_samples_split': [2, 5, 10]\n",
    "}\n",
    "\n",
    "grid_search = GridSearchCV(\n",
    "    regressor_log,\n",
    "    param_grid,\n",
    "    cv=5,\n",
    "    scoring='neg_mean_squared_log_error',\n",
    "    n_jobs=-1,\n",
    "    verbose=2\n",
    ")\n",
    "\n",
    "print(\"Starting Grid Search...\")\n",
    "grid_search.fit(X_train, y_train)\n",
    "best_gb_model = grid_search.best_estimator_\n",
    "\n",
    "# Evaluation Loop\n",
    "results = {}\n",
    "warnings.filterwarnings('ignore')\n",
    "for name, pipe in models.items():\n",
    "    if name == 'GradientBoosting (Tuned Log-Transform)':\n",
    "        y_pred = best_gb_model.predict(X_val)\n",
    "    else:\n",
    "        pipe.fit(X_train, y_train)\n",
    "        y_pred = pipe.predict(X_val)\n",
    "    score = rmsle(y_val, y_pred)\n",
    "    results[name] = score\n",
    "    print(f\"{name} RMSLE: {score}\")\n",
    "\n",
    "results_df = pd.DataFrame(list(results.items()), columns=['Model', 'RMSLE']).sort_values(by='RMSLE')\n",
    "print(results_df)\n"
]

fixed = False
for cell in cells:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Find the cell I previously broke (it has the loop but missing the rest)
        if any("if name == 'GradientBoosting (Tuned Log-Transform)':" in line for line in source):
            cell['source'] = restored_source
            cell['outputs'] = [] # Clear outputs to reduce file size and confusion
            fixed = True
            break

if fixed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook restored. Code block reconstructed.")
else:
    print("Could not find the target cell to restore.")
