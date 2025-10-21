#!/usr/bin/env python3
"""
Script to convert kmeans_script.py to Jupyter Notebook format
Usage: python convert_to_notebook.py
"""

import json
import re

def python_to_notebook(python_file, notebook_file):
    """Convert Python script to Jupyter notebook"""
    
    with open(python_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by cell markers (## Cell X: comments)
    cells = []
    current_cell = {"type": "code", "content": []}
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for markdown cell marker (# # Title or similar)
        if line.strip().startswith('# #'):
            # Save previous cell if it has content
            if current_cell["content"]:
                cells.append(current_cell)
            
            # Start new markdown cell
            markdown_lines = []
            # Remove the comment markers
            markdown_lines.append(line.strip()[2:].strip())
            i += 1
            
            # Collect markdown lines
            while i < len(lines) and (lines[i].strip().startswith('#') or lines[i].strip() == ''):
                if lines[i].strip().startswith('# '):
                    markdown_lines.append(lines[i].strip()[2:])
                elif lines[i].strip() == '#':
                    markdown_lines.append('')
                i += 1
            
            cells.append({
                "type": "markdown",
                "content": markdown_lines
            })
            current_cell = {"type": "code", "content": []}
            continue
        
        # Check for ## Cell marker (code cell separator)
        if line.strip().startswith('## Cell'):
            # Save previous cell if it has content
            if current_cell["content"]:
                cells.append(current_cell)
            current_cell = {"type": "code", "content": []}
            i += 1
            continue
        
        # Regular code line
        if line.strip() or current_cell["content"]:  # Include empty lines if cell started
            current_cell["content"].append(line)
        
        i += 1
    
    # Save last cell
    if current_cell["content"]:
        cells.append(current_cell)
    
    # Create notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Convert cells to notebook format
    for cell in cells:
        if cell["type"] == "markdown":
            nb_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in cell["content"]]
            }
        else:
            # Clean up code cell
            source = []
            for line in cell["content"]:
                if not line.strip().startswith('## Cell'):
                    source.append(line + "\n")
            
            if source:  # Only add if there's actual content
                nb_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source
                }
        
        if "nb_cell" in locals():
            notebook["cells"].append(nb_cell)
            del nb_cell
    
    # Write notebook
    with open(notebook_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Successfully converted {python_file} to {notebook_file}")
    print(f"Created {len(notebook['cells'])} cells")

if __name__ == "__main__":
    python_to_notebook('kmeans_script.py', 'KMeansClustering.ipynb')
    print("\n📝 Next steps:")
    print("1. Open KMeansClustering.ipynb in Jupyter")
    print("2. Run all cells to see the results")
    print("3. Modify as needed for your specific use case")



