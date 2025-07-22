#!/usr/bin/env python3
"""
Simple test script to verify dataset integrity and completeness.
"""

from datasets import Dataset
import json
import os

def test_dataset():
    """Test the generated dataset for completeness and correctness."""
    
    print("Testing KernelBench Level1 Dataset...")
    
    # Test 1: Check if files exist
    assert os.path.exists("kernelbench_level1_dataset"), "HF dataset directory not found"
    assert os.path.exists("kernelbench_level1_dataset.json"), "JSON file not found"
    print("✓ Output files exist")
    
    # Test 2: Load and check HF dataset
    dataset = Dataset.load_from_disk("kernelbench_level1_dataset")
    assert len(dataset) == 100, f"Expected 100 entries, got {len(dataset)}"
    print("✓ HF dataset has correct number of entries")
    
    # Test 3: Load and check JSON dataset
    with open("kernelbench_level1_dataset.json", 'r') as f:
        json_data = json.load(f)
    assert len(json_data) == 100, f"Expected 100 entries in JSON, got {len(json_data)}"
    print("✓ JSON dataset has correct number of entries")
    
    # Test 4: Check data consistency between formats
    for i in range(len(dataset)):
        hf_entry = dataset[i]
        json_entry = json_data[i]
        
        assert hf_entry['index'] == json_entry['index'], f"Index mismatch at position {i}"
        assert hf_entry['operator_name'] == json_entry['operator_name'], f"Operator name mismatch at position {i}"
        assert hf_entry['code'] == json_entry['code'], f"Code mismatch at position {i}"
    print("✓ Data consistency between HF and JSON formats")
    
    # Test 5: Check index completeness (1-100)
    indices = [entry['index'] for entry in json_data]
    expected_indices = list(range(1, 101))
    assert sorted(indices) == expected_indices, "Missing or duplicate indices"
    print("✓ All indices 1-100 are present")
    
    # Test 6: Check required fields
    required_fields = ['index', 'operator_name', 'filename', 'code', 'description', 'file_size', 'line_count']
    for field in required_fields:
        assert field in dataset.features, f"Missing field: {field}"
    print("✓ All required fields present")
    
    # Test 7: Check data quality
    for entry in json_data:
        assert entry['code'].strip(), f"Empty code for entry {entry['index']}"
        assert entry['operator_name'].strip(), f"Empty operator name for entry {entry['index']}"
        assert entry['filename'].endswith('.py'), f"Invalid filename for entry {entry['index']}"
        assert entry['file_size'] > 0, f"Invalid file size for entry {entry['index']}"
        assert entry['line_count'] > 0, f"Invalid line count for entry {entry['index']}"
    print("✓ Data quality checks passed")
    
    # Test 8: Check some specific entries
    # Find matrix multiplication entry
    matrix_mult = next((e for e in json_data if e['index'] == 1), None)
    assert matrix_mult is not None, "Matrix multiplication entry not found"
    assert 'torch.matmul' in matrix_mult['code'], "Expected torch.matmul in matrix multiplication code"
    print("✓ Specific entry content verification passed")
    
    print(f"\n🎉 All tests passed! Dataset is ready to use.")
    print(f"   - {len(dataset)} operators successfully converted")
    print(f"   - Average code length: {sum(e['file_size'] for e in json_data) / len(json_data):.1f} characters")
    print(f"   - Total lines of code: {sum(e['line_count'] for e in json_data)}")

if __name__ == "__main__":
    test_dataset()
