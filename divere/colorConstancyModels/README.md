# Color Constancy Models Integration

这个目录包含了与DiVERE项目集成的颜色恒常性模型。

## Deep White Balance Integration

### 概述

Deep White Balance是一个基于深度学习的自动白平衡算法，来自CVPR 2020论文"Deep White-Balance Editing"。我们将其集成到DiVERE项目中，提供了一个基于学习的自动校色功能。

### 文件结构

```
colorConstancyModels/
├── Deep_White_Balance-master/     # 原始Deep White Balance项目
│   ├── PyTorch/                   # PyTorch实现
│   │   ├── models/                # 预训练模型
│   │   ├── arch/                  # 网络架构
│   │   └── utilities/             # 工具函数
│   └── ...
├── utils/                         # 我们的工具函数
│   ├── __init__.py
│   └── gain_calculator.py         # RGB增益计算工具
├── deep_wb_wrapper.py             # Deep White Balance包装器
└── README.md                      # 本文件
```

### 功能特性

1. **calculate_auto_gain_learning_based**: 新增的基于学习的自动校色函数
2. **RGB增益计算**: 多种方法计算输入输出图像间的RGB增益
3. **光源估计**: 从增益值估计光源颜色
4. **错误处理**: 优雅处理模型不可用的情况

### 使用方法

#### 1. 基本使用

```python
from divere.core.grading_engine import ColorGradingEngine
from divere.core.data_types import ImageData

# 创建引擎
engine = ColorGradingEngine()

# 准备图像
image = ImageData(your_image_array)

# 使用基于学习的自动校色
result = engine.calculate_auto_gain_learning_based(image)
gains = result[:3]  # RGB增益
illuminant = result[3:]  # 光源RGB
```

#### 2. 与现有功能对比

```python
# 传统方法
traditional_result = engine.calculate_auto_gain(image)

# 基于学习的方法
learning_result = engine.calculate_auto_gain_learning_based(image)
```

### 依赖要求

1. **PyTorch**: 用于运行深度学习模型
2. **torchvision**: 图像变换工具
3. **scikit-learn**: 用于多项式映射计算
4. **PIL**: 图像处理
5. **numpy**: 数值计算

### 模型文件

需要以下预训练模型文件之一：

- `models/net_awb.pth`: 单独的AWB模型
- `models/net.pth`: 完整的模型（包含AWB、Tungsten、Shade）

### 安装步骤

1. 确保已安装所有依赖：
```bash
pip install torch torchvision scikit-learn pillow numpy
```

2. 下载预训练模型到 `Deep_White_Balance-master/PyTorch/models/` 目录

3. 测试集成：
```bash
python test_learning_based_cc.py
```

### 技术细节

#### RGB增益计算方法

1. **多项式映射 (polynomial)**: 使用多项式回归计算输入输出图像间的映射关系
2. **简单比率 (simple_ratio)**: 基于通道平均值的简单比率
3. **对数比率 (log_ratio)**: 基于对数空间的比率计算

#### 光源估计

从RGB增益值估计光源颜色：
- 增益的逆数即为光源颜色
- 以绿色通道为参考进行归一化

### 错误处理

如果Deep White Balance模型不可用，函数会：
1. 打印警告信息
2. 返回默认值 `(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)`
3. 不会中断程序执行

### 性能考虑

- 深度学习模型需要GPU加速以获得最佳性能
- 图像会被缩放到最大656像素进行处理
- 处理时间取决于图像大小和硬件配置

### 引用

如果使用此功能，请引用原始论文：

```bibtex
@inproceedings{afifi2020deepWB,
  title={Deep White-Balance Editing},
  author={Afifi, Mahmoud and Brown, Michael S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
``` 