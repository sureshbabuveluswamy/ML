import json

notebook_path = '2025ab05129_assignment.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

print(f"Total cells: {len(cells)}")

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = cell['source']
        num_lines = len(source)
        if num_lines >= 70:
            print(f"Cell {i} has {num_lines} lines.")
            # Check for indentation issues or 'else:' around line 79
            if num_lines >= 79:
                line_79 = source[78] # 0-indexed
                print(f"  Line 79: {repr(line_79)}")
                # Print context around line 79
                start = max(0, 75)
                end = min(num_lines, 85)
                print("  Context:")
                for j in range(start, end):
                    print(f"    {j+1}: {repr(source[j])}")
