#!/usr/bin/env python3
"""
Simple example showing how to use the Parquet dataset.
"""

import pandas as pd
from datasets import Dataset

def basic_usage():
    """Basic usage examples with Parquet format."""
    
    print("=== Using KernelBench Dataset in Parquet Format ===\n")
    
    # Method 1: Using pandas (most common)
    print("1. Loading with pandas:")
    df = pd.read_parquet('kernelbench_level1_optimized.parquet')
    print(f"   Loaded {len(df)} entries")
    print(f"   Columns: {list(df.columns)}")
    
    # Method 2: Using Hugging Face datasets
    print("\n2. Loading with Hugging Face datasets:")
    dataset = Dataset.from_parquet('kernelbench_level1_optimized.parquet')
    print(f"   Loaded {len(dataset)} entries")
    print(f"   Features: {list(dataset.features.keys())}")
    
    # Example queries
    print("\n3. Example queries:")
    
    # Find ReLU operator
    relu_ops = df[df['operator_name'].str.contains('ReLU', case=False)]
    print(f"   ReLU operators: {len(relu_ops)}")
    for _, op in relu_ops.iterrows():
        print(f"     {op['index']}: {op['operator_name']}")
    
    # Find largest operators
    print(f"\n   Top 3 largest operators by code size:")
    top_large = df.nlargest(3, 'file_size')
    for _, op in top_large.iterrows():
        print(f"     {op['index']}: {op['operator_name']} ({op['file_size']} chars)")
    
    # Code analysis
    print(f"\n4. Code analysis:")
    print(f"   Total lines of code: {df['line_count'].sum():,}")
    print(f"   Average file size: {df['file_size'].mean():.1f} characters")
    print(f"   Operators with 'torch.nn' imports: {df['code'].str.contains('torch.nn').sum()}")

def load_specific_categories():
    """Load specific categories of operators."""
    
    print("\n=== Loading Specific Categories ===\n")
    
    # Load convolution operators
    conv_df = pd.read_parquet('kernelbench_convolutions.parquet')
    print(f"Convolution operators: {len(conv_df)}")
    print(f"  Types: {conv_df['operator_name'].str.extract(r'conv (\w+)')[0].value_counts().head()}")
    
    # Load activation functions
    act_df = pd.read_parquet('kernelbench_activations.parquet')
    print(f"\nActivation functions: {len(act_df)}")
    print(f"  Functions: {act_df['operator_name'].tolist()}")
    
    # Load matrix operations
    matrix_df = pd.read_parquet('kernelbench_matrix_ops.parquet')
    print(f"\nMatrix operations: {len(matrix_df)}")
    print(f"  Operations: {matrix_df['operator_name'].tolist()[:5]}...")

def performance_comparison():
    """Compare loading performance."""
    
    import time
    
    print("\n=== Performance Comparison ===\n")
    
    # Time Parquet loading
    start = time.time()
    df_parquet = pd.read_parquet('kernelbench_level1_optimized.parquet')
    parquet_time = time.time() - start
    
    # Time JSON loading
    start = time.time()
    df_json = pd.read_json('kernelbench_level1_dataset.json')
    json_time = time.time() - start
    
    print(f"Loading times:")
    print(f"  Parquet: {parquet_time:.4f} seconds")
    print(f"  JSON:    {json_time:.4f} seconds")
    print(f"  Speedup: {json_time/parquet_time:.1f}x faster")
    
    # Memory usage
    print(f"\nMemory usage:")
    print(f"  Parquet: {df_parquet.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"  JSON:    {df_json.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    basic_usage()
    load_specific_categories()
    performance_comparison()
    
    print("\n🎉 Parquet usage examples completed!")
