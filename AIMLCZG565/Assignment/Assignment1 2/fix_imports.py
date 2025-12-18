import json

notebook_path = '2025ab05129_assignment.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the cell with imports
for cell in cells:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this is the import cell
        if any('import pandas' in line for line in source):
             # Add warnings import if not present
             if not any('import warnings' in line for line in source):
                 source.insert(0, "import warnings\n")
                 print("Added warnings import.")
        
        for i, line in enumerate(source):
            if 'from sklearn.preprocessing import' in line:
                if 'PolynomialFeatures' not in line:
                    source.insert(i+1, "from sklearn.preprocessing import PolynomialFeatures\n")
                    print("Added PolynomialFeatures import.")
            
            if 'from sklearn.model_selection import' in line:
                if 'GridSearchCV' not in line:
                     source.insert(i+1, "from sklearn.model_selection import GridSearchCV\n")
                     print("Added GridSearchCV import.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
