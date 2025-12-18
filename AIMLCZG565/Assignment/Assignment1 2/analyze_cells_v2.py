import json

notebook_path = '2025ab05129_assignment.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

print(f"Total cells: {len(cells)}")

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = cell['source']
        exec_count = cell.get('execution_count')
        
        if isinstance(source, str):
            lines = source.splitlines()
        else:
            lines = source # list of strings
            
        num_lines = len(lines)
        print(f"Cell {i} (Exec {exec_count}): {num_lines} lines. Type of source: {type(source)}")
        
        if num_lines >= 70:
            print(f"  [MATCH] Cell {i} has {num_lines} lines.")
            if num_lines >= 79:
                line_79 = lines[78]
                print(f"  Line 79: {repr(line_79)}")
