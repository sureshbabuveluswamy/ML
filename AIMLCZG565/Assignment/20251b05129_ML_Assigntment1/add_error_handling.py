import json

file_path = '/Users/ps/Documents/ML/AIMLCZG565/Assignment/20251b05129_ML_Assigntment1/working.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

# Find the cell with the results loop
target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "results = {}" in source and "for name, pipe in models.items():" in source:
            target_cell = cell
            break

if target_cell:
    new_source = [
        "results = {}\n",
        "\n",
        "for name, pipe in models.items():\n",
        "    try:\n",
        "        if name == 'GradientBoosting (Tuned Log-Transform)':\n",
        "            # Use the best tuned Gradient Boosting model\n",
        "            y_pred = best_gb_model.predict(X_val)\n",
        "        else:\n",
        "            pipe.fit(X_train, y_train)\n",
        "            y_pred = pipe.predict(X_val)\n",
        "        score = rmsle(y_val, y_pred)\n",
        "        results[name] = score\n",
        "        print(f\"{name} RMSLE: {score}\")\n",
        "    except Exception as e:\n",
        "        print(f\"Error processing {name}: {e}\")\n",
        "        results[name] = np.nan\n",
        "\n",
        "# Convert results dictionary to DataFrame and display\n",
        "results_df = pd.DataFrame(list(results.items()), columns=['Model', 'RMSLE']).sort_values(by='RMSLE')\n",
        "print(results_df)"
    ]
    target_cell['source'] = new_source
    print("Modified Results Cell")
else:
    print("Results Cell not found")

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Done")
