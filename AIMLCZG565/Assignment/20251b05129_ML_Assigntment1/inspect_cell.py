import json

file_path = '/Users/ps/Documents/ML/AIMLCZG565/Assignment/20251b05129_ML_Assigntment1/working.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

# Find the first code cell
target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        target_cell = cell
        break

if target_cell:
    print("First cell source:")
    print(target_cell['source'])
else:
    print("No code cell found")
