#!/usr/bin/env python3
"""
Script to work with KernelBench dataset in Parquet format.
"""

import pandas as pd
import pyarrow.parquet as pq
import json
from datasets import Dataset

def convert_to_parquet():
    """Convert the dataset to Parquet format with optimizations."""
    
    print("Converting KernelBench dataset to Parquet format...")
    
    # Load from JSON for more control
    with open("kernelbench_level1_dataset.json", 'r') as f:
        data = json.load(f)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Optimize data types
    df['index'] = df['index'].astype('int16')  # Smaller integer type
    df['file_size'] = df['file_size'].astype('int32')
    df['line_count'] = df['line_count'].astype('int16')
    
    # Save to Parquet with compression
    df.to_parquet('kernelbench_level1_optimized.parquet', 
                  compression='snappy',
                  index=False)
    
    print(f"✓ Parquet file created: kernelbench_level1_optimized.parquet")
    print(f"✓ File size: {pd.io.common.file_size('kernelbench_level1_optimized.parquet')} bytes")
    print(f"✓ Rows: {len(df)}, Columns: {len(df.columns)}")

def analyze_parquet():
    """Analyze the Parquet file and show statistics."""
    
    print("Analyzing Parquet dataset...")
    
    # Load Parquet file
    df = pd.read_parquet('kernelbench_level1_optimized.parquet')
    
    print(f"\nDataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print(f"\nColumn Info:")
    print(df.dtypes)
    
    print(f"\nStatistics:")
    print(df[['index', 'file_size', 'line_count']].describe())
    
    print(f"\nOperator Types (top 10):")
    # Extract operator types
    df['op_type'] = df['operator_name'].str.split().str[0].str.lower()
    print(df['op_type'].value_counts().head(10))
    
    print(f"\nLargest operators by code size:")
    print(df.nlargest(5, 'file_size')[['index', 'operator_name', 'file_size']])

def query_parquet():
    """Demonstrate querying the Parquet file."""
    
    print("Querying Parquet dataset...")
    
    df = pd.read_parquet('kernelbench_level1_optimized.parquet')
    
    # Query 1: Find all convolution operators
    conv_ops = df[df['operator_name'].str.contains('conv', case=False)]
    print(f"\nConvolution operators: {len(conv_ops)}")
    print(conv_ops[['index', 'operator_name']].head())
    
    # Query 2: Find operators with more than 50 lines
    large_ops = df[df['line_count'] > 50]
    print(f"\nOperators with >50 lines: {len(large_ops)}")
    print(large_ops[['index', 'operator_name', 'line_count']])
    
    # Query 3: Matrix operations
    matrix_ops = df[df['operator_name'].str.contains('matrix|matmul', case=False)]
    print(f"\nMatrix operations: {len(matrix_ops)}")
    print(matrix_ops[['index', 'operator_name']].head())

def export_filtered_data():
    """Export filtered subsets of the data."""
    
    print("Exporting filtered datasets...")
    
    df = pd.read_parquet('kernelbench_level1_optimized.parquet')
    
    # Export convolution operators only
    conv_ops = df[df['operator_name'].str.contains('conv', case=False)]
    conv_ops.to_parquet('kernelbench_convolutions.parquet', index=False)
    print(f"✓ Convolution operators exported: {len(conv_ops)} entries")
    
    # Export activation functions
    activations = df[df['operator_name'].str.contains('relu|sigmoid|tanh|gelu|swish', case=False)]
    activations.to_parquet('kernelbench_activations.parquet', index=False)
    print(f"✓ Activation functions exported: {len(activations)} entries")
    
    # Export matrix operations
    matrix_ops = df[df['operator_name'].str.contains('matrix|matmul', case=False)]
    matrix_ops.to_parquet('kernelbench_matrix_ops.parquet', index=False)
    print(f"✓ Matrix operations exported: {len(matrix_ops)} entries")

def compare_formats():
    """Compare file sizes of different formats."""
    
    import os
    
    print("Comparing file sizes across formats...")
    
    formats = {
        'JSON': 'kernelbench_level1_dataset.json',
        'Parquet (optimized)': 'kernelbench_level1_optimized.parquet',
        'Parquet (default)': 'kernelbench_level1.parquet'
    }
    
    for format_name, filename in formats.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  {format_name}: {size:,} bytes ({size/1024:.1f} KB)")

def load_back_to_huggingface():
    """Load Parquet data back into Hugging Face Dataset."""
    
    print("Loading Parquet back to Hugging Face Dataset...")
    
    # Load from Parquet
    df = pd.read_parquet('kernelbench_level1_optimized.parquet')
    
    # Convert back to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    print(f"✓ Dataset loaded: {len(dataset)} entries")
    print(f"✓ Features: {list(dataset.features.keys())}")
    
    # Save as new HF dataset
    dataset.save_to_disk("kernelbench_from_parquet")
    print("✓ Saved as: kernelbench_from_parquet/")
    
    return dataset

def main():
    """Run all operations."""
    
    print("=== KernelBench Parquet Operations ===\n")
    
    # Convert to optimized Parquet
    convert_to_parquet()
    print("\n" + "="*50)
    
    # Analyze the data
    analyze_parquet()
    print("\n" + "="*50)
    
    # Query examples
    query_parquet()
    print("\n" + "="*50)
    
    # Export filtered data
    export_filtered_data()
    print("\n" + "="*50)
    
    # Compare formats
    compare_formats()
    print("\n" + "="*50)
    
    # Load back to HF
    dataset = load_back_to_huggingface()
    
    print("\n🎉 All Parquet operations completed!")

if __name__ == "__main__":
    main()
