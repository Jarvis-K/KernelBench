#!/usr/bin/env python3
"""
Script to convert KernelBench level1 Python files into Hugging Face dataset format.
Each Python file becomes one data entry with its index and operator name.
"""

import os
import re
import json
from pathlib import Path
from datasets import Dataset
import argparse


def extract_operator_info(filename):
    """
    Extract index and operator name from filename.
    
    Args:
        filename (str): The filename (e.g., "1_Square_matrix_multiplication_.py")
        
    Returns:
        tuple: (index, operator_name)
    """
    # Remove .py extension
    name_without_ext = filename.replace('.py', '')
    
    # Split by first underscore to separate index from operator name
    parts = name_without_ext.split('_', 1)
    
    if len(parts) >= 2:
        index = int(parts[0])
        operator_name = parts[1].replace('_', ' ').strip()
        # Clean up operator name
        operator_name = re.sub(r'\s+', ' ', operator_name)
        operator_name = operator_name.strip(' .')
    else:
        # Fallback for unexpected filename format
        index = 0
        operator_name = name_without_ext.replace('_', ' ').strip()
    
    return index, operator_name


def read_python_file(filepath):
    """
    Read the content of a Python file.
    
    Args:
        filepath (str): Path to the Python file
        
    Returns:
        str: Content of the file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def extract_docstring(content):
    """
    Extract the main docstring from the Model class.
    
    Args:
        content (str): Python file content
        
    Returns:
        str: Extracted docstring or empty string
    """
    # Look for class docstring
    class_pattern = r'class Model\(.*?\):\s*"""(.*?)"""'
    match = re.search(class_pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return ""


def convert_kernelbench_to_dataset(input_dir="KernelBench/level1", output_path="kernelbench_level1_dataset"):
    """
    Convert KernelBench level1 Python files to Hugging Face dataset format.
    
    Args:
        input_dir (str): Path to the directory containing Python files
        output_path (str): Path where to save the dataset
    """
    
    # Get all Python files
    python_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")
    
    for file_path in input_path.glob("*.py"):
        python_files.append(file_path)
    
    print(f"Found {len(python_files)} Python files in {input_dir}")
    
    # Process each file
    dataset_entries = []
    
    for file_path in sorted(python_files):
        filename = file_path.name
        print(f"Processing: {filename}")
        
        # Extract index and operator name
        index, operator_name = extract_operator_info(filename)
        
        # Read file content
        code_content = read_python_file(file_path)
        
        # Extract docstring for description
        description = extract_docstring(code_content)
        
        # Create dataset entry
        entry = {
            "index": index,
            "operator_name": operator_name,
            "filename": filename,
            "code": code_content,
            "description": description,
            "file_size": len(code_content),
            "line_count": len(code_content.split('\n'))
        }
        
        dataset_entries.append(entry)
    
    # Sort by index
    dataset_entries.sort(key=lambda x: x["index"])
    
    # Create Hugging Face dataset
    dataset = Dataset.from_list(dataset_entries)
    
    # Save dataset
    print(f"Saving dataset with {len(dataset_entries)} entries to {output_path}")
    dataset.save_to_disk(output_path)
    
    # Also save as JSON for easy inspection
    json_output_path = f"{output_path}.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to: {output_path}")
    print(f"JSON version saved to: {json_output_path}")
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Total entries: {len(dataset_entries)}")
    print(f"Average code length: {sum(entry['file_size'] for entry in dataset_entries) / len(dataset_entries):.1f} characters")
    print(f"Average line count: {sum(entry['line_count'] for entry in dataset_entries) / len(dataset_entries):.1f} lines")
    
    # Show sample entries
    print(f"\nSample entries:")
    for i in range(min(3, len(dataset_entries))):
        entry = dataset_entries[i]
        print(f"  {entry['index']}: {entry['operator_name']} ({entry['filename']})")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert KernelBench level1 to Hugging Face dataset")
    parser.add_argument("--input_dir", default="KernelBench/level1", 
                       help="Input directory containing Python files")
    parser.add_argument("--output_path", default="kernelbench_level1_dataset", 
                       help="Output path for the dataset")
    
    args = parser.parse_args()
    
    try:
        dataset = convert_kernelbench_to_dataset(args.input_dir, args.output_path)
        print("Conversion completed successfully!")
        
        # Display first entry as example
        if len(dataset) > 0:
            print(f"\nExample entry (index {dataset[0]['index']}):")
            print(f"Operator: {dataset[0]['operator_name']}")
            print(f"Description: {dataset[0]['description'][:200]}...")
            print(f"Code preview: {dataset[0]['code'][:300]}...")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
