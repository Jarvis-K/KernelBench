#!/usr/bin/env python3
"""
Script to convert KernelBench dataset to different formats.
"""

import json
import argparse
from datasets import Dataset
from pathlib import Path

def convert_to_verilog(dataset_path="kernelbench_level1_dataset.json", output_dir="verilog_output"):
    """Convert PyTorch operators to Verilog modules."""
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Converting {len(data)} operators to Verilog...")
    
    for entry in data:
        verilog_code = pytorch_to_verilog(entry)
        
        # Save Verilog file
        verilog_filename = f"{entry['index']:03d}_{entry['operator_name'].replace(' ', '_')}.v"
        verilog_path = Path(output_dir) / verilog_filename
        
        with open(verilog_path, 'w') as f:
            f.write(verilog_code)
        
        print(f"Generated: {verilog_filename}")

def pytorch_to_verilog(entry):
    """Convert a PyTorch operator to Verilog module."""
    
    op_name = entry['operator_name'].replace(' ', '_').replace('-', '_')
    index = entry['index']
    
    # Basic Verilog template
    verilog_template = f'''//
// Verilog module for: {entry['operator_name']}
// Generated from: {entry['filename']}
// Description: {entry['description']}
//

module {op_name}_module_{index:03d} (
    input clk,
    input rst_n,
    input valid_in,
    output valid_out,
    // Add specific ports based on operator type
    {generate_ports(entry)}
);

    // Module implementation would go here
    // This is a template - actual implementation depends on the operator
    
    {generate_logic(entry)}

endmodule
'''
    
    return verilog_template

def generate_ports(entry):
    """Generate Verilog ports based on operator type."""
    
    op_name = entry['operator_name'].lower()
    
    if 'matrix' in op_name or 'matmul' in op_name:
        return '''input [31:0] data_a,
    input [31:0] data_b,
    output [31:0] result'''
    
    elif 'conv' in op_name:
        return '''input [31:0] input_data,
    input [31:0] weight_data,
    output [31:0] output_data'''
    
    elif any(act in op_name for act in ['relu', 'sigmoid', 'tanh', 'gelu']):
        return '''input [31:0] input_data,
    output [31:0] output_data'''
    
    else:
        return '''input [31:0] input_data,
    output [31:0] output_data'''

def generate_logic(entry):
    """Generate basic Verilog logic based on operator type."""
    
    op_name = entry['operator_name'].lower()
    
    if 'relu' in op_name:
        return '''    // ReLU implementation
    assign output_data = (input_data[31] == 1'b1) ? 32'b0 : input_data;
    assign valid_out = valid_in;'''
    
    elif 'matrix' in op_name:
        return '''    // Matrix multiplication placeholder
    // Actual implementation would require systolic array or similar
    assign result = data_a + data_b; // Simplified
    assign valid_out = valid_in;'''
    
    else:
        return '''    // Generic operator implementation
    assign output_data = input_data; // Placeholder
    assign valid_out = valid_in;'''

def convert_to_parquet(dataset_path="kernelbench_level1_dataset", output_path="kernelbench_level1.parquet"):
    """Convert dataset to Parquet format."""
    
    dataset = Dataset.load_from_disk(dataset_path)
    dataset.to_parquet(output_path)
    print(f"Dataset converted to Parquet: {output_path}")

def convert_to_csv(dataset_path="kernelbench_level1_dataset.json", output_path="kernelbench_level1.csv"):
    """Convert dataset to CSV format."""
    
    import pandas as pd
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset converted to CSV: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert KernelBench dataset to different formats")
    parser.add_argument("--format", choices=["verilog", "parquet", "csv"], required=True,
                       help="Target format for conversion")
    parser.add_argument("--input", default="kernelbench_level1_dataset.json",
                       help="Input dataset path")
    parser.add_argument("--output", help="Output path (optional)")
    
    args = parser.parse_args()
    
    if args.format == "verilog":
        output = args.output or "verilog_output"
        convert_to_verilog(args.input, output)
    elif args.format == "parquet":
        output = args.output or "kernelbench_level1.parquet"
        convert_to_parquet(args.input.replace('.json', ''), output)
    elif args.format == "csv":
        output = args.output or "kernelbench_level1.csv"
        convert_to_csv(args.input, output)

if __name__ == "__main__":
    main()
