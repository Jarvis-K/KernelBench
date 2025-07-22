#!/usr/bin/env python3
"""
Example script showing how to load and use the KernelBench level1 dataset.
"""

from datasets import Dataset
import json

def load_dataset_example():
    """Demonstrate how to load and inspect the dataset."""
    
    # Method 1: Load from Hugging Face dataset format
    print("Loading dataset from Hugging Face format...")
    dataset = Dataset.load_from_disk("kernelbench_level1_dataset")
    
    print(f"Dataset loaded with {len(dataset)} entries")
    print(f"Dataset features: {dataset.features}")
    
    # Print first few entries
    print("\nFirst 3 entries:")
    for i in range(3):
        entry = dataset[i]
        print(f"\nEntry {i+1}:")
        print(f"  Index: {entry['index']}")
        print(f"  Operator: {entry['operator_name']}")
        print(f"  Filename: {entry['filename']}")
        print(f"  Description: {entry['description']}")
        print(f"  File size: {entry['file_size']} characters")
        print(f"  Line count: {entry['line_count']} lines")
        print(f"  Code preview: {entry['code'][:100]}...")
    
    # Method 2: Load from JSON format
    print("\n" + "="*50)
    print("Loading dataset from JSON format...")
    with open("kernelbench_level1_dataset.json", 'r') as f:
        json_data = json.load(f)
    
    print(f"JSON data loaded with {len(json_data)} entries")
    
    # Find specific operators
    print("\nSearching for convolution operators:")
    conv_ops = [entry for entry in json_data if 'conv' in entry['operator_name'].lower()]
    print(f"Found {len(conv_ops)} convolution operators:")
    for op in conv_ops[:5]:  # Show first 5
        print(f"  {op['index']}: {op['operator_name']}")
    
    # Statistics
    print(f"\nDataset statistics:")
    total_lines = sum(entry['line_count'] for entry in json_data)
    total_chars = sum(entry['file_size'] for entry in json_data)
    print(f"  Total lines of code: {total_lines}")
    print(f"  Total characters: {total_chars}")
    print(f"  Average lines per operator: {total_lines/len(json_data):.1f}")
    print(f"  Average characters per operator: {total_chars/len(json_data):.1f}")
    
    # Show operator types distribution
    print(f"\nOperator types (sample):")
    operator_types = {}
    for entry in json_data:
        # Extract first word as operator type
        op_type = entry['operator_name'].split()[0].lower()
        operator_types[op_type] = operator_types.get(op_type, 0) + 1
    
    # Show top 10 operator types
    for op_type, count in sorted(operator_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {op_type}: {count}")

if __name__ == "__main__":
    load_dataset_example()
