import json

notebook_path = '2025ab05129_assignment.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

target_id = "d8ff01ef"
found = False

for i, cell in enumerate(cells):
    if cell.get('id') == target_id:
        print(f"Found cell {i} with ID {target_id}")
        source = cell['source']
        print(f"Line count: {len(source)}")
        print("--- Content ---")
        for j, line in enumerate(source):
            print(f"{j+1}: {repr(line)}")
        found = True
        break

if not found:
    print(f"Cell with ID {target_id} not found.")
