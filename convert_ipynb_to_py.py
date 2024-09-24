import nbformat
from nbconvert import PythonExporter

# Path to your Jupyter notebook file
notebook_path = 'rag_from_scratch_1_to_4.ipynb'
python_script_path = 'rag_from_scratch_1_to_4.py'

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert the notebook to Python script
exporter = PythonExporter()
python_script, _ = exporter.from_notebook_node(notebook_content)

# Save the Python script
with open(python_script_path, 'w', encoding='utf-8') as f:
    f.write(python_script)

print(f"Notebook has been successfully saved as Python script at {python_script_path}")
