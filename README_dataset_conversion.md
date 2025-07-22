# KernelBench Level1 to Hugging Face Dataset Conversion

This script converts KernelBench level1 Python files into Hugging Face dataset format, with each Python file becoming one data entry containing the index, operator name, and full code.

## Files

- `convert_kernelbench_to_dataset.py`: Main conversion script
- `load_dataset_example.py`: Example script showing how to load and use the dataset
- `README_dataset_conversion.md`: This documentation file

## Features

The conversion script extracts the following information from each Python file:

- **index**: Extracted from filename (e.g., "1" from "1_Square_matrix_multiplication_.py")
- **operator_name**: Cleaned operator name (e.g., "Square matrix multiplication")
- **filename**: Original filename
- **code**: Complete Python code content
- **description**: Extracted docstring from the Model class
- **file_size**: Number of characters in the code
- **line_count**: Number of lines in the code

## Usage

### Prerequisites

Install required dependencies:

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install datasets
```

### Basic Usage

```bash
# Convert with default settings
python convert_kernelbench_to_dataset.py

# Convert with custom paths
python convert_kernelbench_to_dataset.py --input_dir KernelBench/level1 --output_path my_dataset
```

### Command Line Arguments

- `--input_dir`: Input directory containing Python files (default: "KernelBench/level1")
- `--output_path`: Output path for the dataset (default: "kernelbench_level1_dataset")

### Output Files

The script generates two output formats:

1. **Hugging Face Dataset Format**: `kernelbench_level1_dataset/`
   - Optimized binary format for fast loading
   - Can be loaded with `Dataset.load_from_disk()`

2. **JSON Format**: `kernelbench_level1_dataset.json`
   - Human-readable format
   - Easy to inspect and process with standard tools

## Loading the Dataset

### Using Hugging Face Datasets

```python
from datasets import Dataset

# Load the dataset
dataset = Dataset.load_from_disk("kernelbench_level1_dataset")

# Access entries
print(f"Dataset has {len(dataset)} entries")
first_entry = dataset[0]
print(f"Operator: {first_entry['operator_name']}")
print(f"Code: {first_entry['code']}")
```

### Using JSON Format

```python
import json

# Load JSON data
with open("kernelbench_level1_dataset.json", 'r') as f:
    data = json.load(f)

# Access entries
for entry in data:
    print(f"Index {entry['index']}: {entry['operator_name']}")
```

## Dataset Statistics

- **Total entries**: 100 operators
- **Average code length**: ~1,273 characters per operator
- **Average line count**: ~40 lines per operator
- **Operator types**: Includes matrix operations, convolutions, activations, normalization, pooling, reductions, and loss functions

## Example Output Structure

Each dataset entry contains:

```json
{
  "index": 1,
  "operator_name": "Square matrix multiplication",
  "filename": "1_Square_matrix_multiplication_.py",
  "code": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    ...",
  "description": "Simple model that performs a single square matrix multiplication (C = A * B)",
  "file_size": 794,
  "line_count": 32
}
```

## Use Cases

This dataset format is useful for:

- **Code analysis**: Analyzing PyTorch operator implementations
- **Machine learning**: Training models on code patterns
- **Documentation**: Generating documentation from code
- **Benchmarking**: Performance analysis of different operators
- **Education**: Learning about different neural network operations

## Example Analysis

See `load_dataset_example.py` for examples of:

- Loading the dataset in both formats
- Searching for specific operator types
- Computing statistics
- Analyzing operator distribution

Run the example:

```bash
python load_dataset_example.py
```

## Notes

- The script automatically sorts entries by index number
- Operator names are cleaned (underscores replaced with spaces, extra spaces removed)
- Docstrings are extracted from the Model class for descriptions
- Both output formats contain identical data, choose based on your use case
