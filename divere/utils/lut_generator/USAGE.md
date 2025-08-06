# DiVERE LUT生成器 - 快速使用指南

## 快速开始

### 导入模块

```python
# 方式1：从utils包导入
from divere.utils import generate_pipeline_lut, generate_curve_lut

# 方式2：直接从lut_generator包导入
from divere.utils.lut_generator import LUT3DGenerator, DiVERELUTInterface
```

### 基本使用

```python
# 生成3D LUT
from divere.utils.lut_generator import LUT3DGenerator

generator = LUT3DGenerator(size=32)
def my_transform(rgb):
    return rgb * 1.2  # 简单的亮度调整

lut = generator.generate_lut_from_transform(my_transform)
generator.save_cube(lut, "my_lut.cube", "My 3D LUT")
```

### 生成曲线LUT

```python
from divere.utils.lut_generator import generate_curve_lut

curves = {
    'R': [(0.0, 0.0), (0.5, 0.4), (1.0, 1.0)],
    'G': [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
    'B': [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
}

generate_curve_lut(curves, "curves.cube", 1024)
```

### 生成Pipeline LUT

```python
from divere.utils.lut_generator import generate_pipeline_lut

# pipeline配置（由core提供）
pipeline_config = {
    'density_gamma': 2.0,
    'rgb_gains': (0.1, -0.05, 0.2),
    'curve_points': [(0.0, 0.0), (1.0, 1.0)]
}

# 生成3D pipeline LUT
generate_pipeline_lut(pipeline_config, "pipeline_3d.cube", "3D", 32)

# 生成1D pipeline LUT
generate_pipeline_lut(pipeline_config, "pipeline_1d.cube", "1D", 1024)
```

## 文件结构

```
divere/utils/lut_generator/
├── __init__.py          # 包导出
├── core.py              # 核心LUT生成器
├── interface.py         # DiVERE专用接口
├── example.py           # 使用示例
├── README.md           # 详细文档
└── USAGE.md            # 快速使用指南
```

## 主要功能

- ✅ 3D LUT生成（支持自定义变换函数）
- ✅ 1D LUT生成（支持曲线控制点）
- ✅ CUBE文件格式读写
- ✅ Pipeline LUT生成（接口隔离）
- ✅ 单位LUT生成
- ✅ 标准CUBE格式输出

## 注意事项

1. 3D LUT大小建议：16-64
2. 1D LUT大小建议：256-4096
3. 支持标准CUBE文件格式
4. Pipeline隔离，不暴露具体实现细节 