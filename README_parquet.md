# KernelBench Dataset - Parquet Format

Dataset已成功转换为Parquet格式，提供了高效的存储和快速的数据访问。

## 生成的文件

### 主要数据集文件
- `kernelbench_level1.parquet` - 标准Parquet格式 (35.2 KB)
- `kernelbench_level1_optimized.parquet` - 优化的Parquet格式 (36.4 KB)

### 分类数据集文件
- `kernelbench_convolutions.parquet` - 卷积算子 (34个算子, 28.4 KB)
- `kernelbench_activations.parquet` - 激活函数 (9个算子, 9.5 KB)
- `kernelbench_matrix_ops.parquet` - 矩阵运算 (18个算子, 12.2 KB)

### 相关脚本
- `parquet_operations_fixed.py` - 完整的Parquet操作脚本
- `use_parquet_example.py` - 简单使用示例
- `convert_dataset_format.py` - 格式转换脚本

## Parquet格式优势

1. **存储效率**: 比JSON格式小约80% (186KB → 36KB)
2. **加载速度**: 比JSON格式快1.7倍
3. **列式存储**: 支持高效的列查询和过滤
4. **压缩**: 使用Snappy压缩算法
5. **类型优化**: 使用更小的数据类型 (int16, int32)

## 使用方法

### 基本加载

```python
import pandas as pd
from datasets import Dataset

# 使用pandas加载
df = pd.read_parquet('kernelbench_level1_optimized.parquet')

# 使用Hugging Face datasets加载
dataset = Dataset.from_parquet('kernelbench_level1_optimized.parquet')
```

### 查询示例

```python
# 查找卷积算子
conv_ops = df[df['operator_name'].str.contains('conv', case=False)]

# 查找大型算子 (>50行代码)
large_ops = df[df['line_count'] > 50]

# 查找特定算子
relu_ops = df[df['operator_name'] == 'ReLU']
```

### 加载特定类别

```python
# 只加载卷积算子
conv_df = pd.read_parquet('kernelbench_convolutions.parquet')

# 只加载激活函数
act_df = pd.read_parquet('kernelbench_activations.parquet')

# 只加载矩阵运算
matrix_df = pd.read_parquet('kernelbench_matrix_ops.parquet')
```

## 数据统计

- **总算子数**: 100个
- **平均代码长度**: 1,273.2字符
- **平均代码行数**: 40.2行
- **总代码行数**: 4,017行

### 算子分布
- 卷积算子: 34个 (34%)
- 矩阵运算: 18个 (18%)
- 激活函数: 9个 (9%)
- 其他: 39个 (39%)

### 卷积算子细分
- 转置卷积: 17个
- 标准卷积: 11个
- 深度卷积: 5个
- 点卷积: 1个

## 性能对比

| 格式 | 文件大小 | 加载时间 | 内存使用 |
|------|----------|----------|----------|
| JSON | 181.9 KB | 0.0033s | 0.18 MB |
| Parquet | 36.4 KB | 0.0019s | 0.18 MB |
| **提升** | **80%减少** | **1.7x更快** | **相同** |

## 运行脚本

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行完整的Parquet操作
python parquet_operations_fixed.py

# 运行使用示例
python use_parquet_example.py

# 转换其他格式
python convert_dataset_format.py --format parquet
```

## 数据结构

每个条目包含以下字段：

```json
{
  "index": 1,                    // int16
  "operator_name": "ReLU",       // string
  "filename": "19_ReLU.py",      // string
  "code": "import torch...",     // string (完整代码)
  "description": "ReLU...",      // string (docstring)
  "file_size": 705,              // int32 (字符数)
  "line_count": 31               // int16 (行数)
}
```

## 适用场景

1. **数据分析**: 使用pandas进行高效分析
2. **机器学习**: 作为训练数据使用
3. **代码搜索**: 快速查找特定算子
4. **性能基准**: 算子性能分析
5. **研究**: 神经网络算子研究

## 注意事项

- Parquet文件需要pandas和pyarrow库支持
- 推荐使用优化版本 (`kernelbench_level1_optimized.parquet`)
- 分类文件适合特定领域的分析
- 可以轻松转换回Hugging Face Dataset格式

转换完成！Dataset现在以高效的Parquet格式提供，支持快速加载和分析。
