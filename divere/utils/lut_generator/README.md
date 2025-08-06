# DiVERE LUT生成器

DiVERE的LUT生成器是一个相对独立的工具模块，提供3D LUT和1D LUT的生成功能，支持标准CUBE格式输出。

## 功能特性

- 🎨 **3D LUT生成**：支持从变换函数生成3D LUT
- 📈 **1D LUT生成**：支持从曲线控制点生成1D LUT
- 💾 **CUBE格式支持**：标准CUBE文件格式的读写
- 🔧 **易用接口**：为DiVERE提供简单的调用接口
- 🚫 **Pipeline隔离**：接口层不暴露pipeline的具体实现细节

## 架构设计

### 核心组件

1. **LUT3DGenerator**：3D LUT生成器
2. **LUT1DGenerator**：1D LUT生成器
3. **LUTManager**：LUT管理器，提供统一接口
4. **DiVERELUTInterface**：DiVERE专用接口，不暴露pipeline细节

### 设计原则

- **独立性**：utils模块相对独立，不依赖core的具体实现
- **接口简洁**：为DiVERE提供简单的调用接口
- **Pipeline隔离**：pipeline的具体实现细节只在core中体现
- **标准兼容**：支持标准的CUBE文件格式

## 使用方法

### 基本使用

```python
from divere.utils.lut_generator import LUT3DGenerator, LUT1DGenerator
from divere.utils.lut_generator import DiVERELUTInterface

# 创建3D LUT生成器
generator_3d = LUT3DGenerator(size=32)

# 定义变换函数
def my_transform(rgb):
    # 你的变换逻辑
    return rgb * 1.2  # 简单的亮度调整

# 生成3D LUT
lut_3d = generator_3d.generate_lut_from_transform(my_transform)

# 保存为CUBE文件
generator_3d.save_cube(lut_3d, "my_lut_3d.cube", "My 3D LUT")
```

### 1D LUT生成

```python
# 创建1D LUT生成器
generator_1d = LUT1DGenerator(size=1024)

# 定义曲线
curves = {
    'R': [(0.0, 0.0), (0.5, 0.4), (1.0, 1.0)],
    'G': [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
    'B': [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
}

# 生成1D LUT
lut_1d = generator_1d.generate_lut_from_curves(curves)

# 保存为CUBE文件
generator_1d.save_cube(lut_1d, "my_lut_1d.cube", "My 1D LUT")
```

### DiVERE接口使用

```python
# 创建DiVERE LUT接口
interface = DiVERELUTInterface()

# 生成pipeline LUT（pipeline配置由core提供）
pipeline_config = {
    'density_gamma': 2.0,
    'rgb_gains': (0.1, -0.05, 0.2),
    'curve_points': [(0.0, 0.0), (1.0, 1.0)]
}

# 生成3D pipeline LUT
success = interface.generate_pipeline_lut(
    pipeline_config, "pipeline_3d.cube", "3D", 32
)

# 生成1D pipeline LUT
success = interface.generate_pipeline_lut(
    pipeline_config, "pipeline_1d.cube", "1D", 1024
)
```

### 便捷函数

```python
from divere.utils.lut_generator import (
    generate_pipeline_lut,
    generate_curve_lut,
    generate_identity_lut
)

# 生成pipeline LUT
generate_pipeline_lut(pipeline_config, "output.cube", "3D", 32)

# 生成曲线LUT
curves = {'R': [(0,0), (1,1)], 'G': [(0,0), (1,1)], 'B': [(0,0), (1,1)]}
generate_curve_lut(curves, "curves.cube", 1024)

# 生成单位LUT
generate_identity_lut("identity.cube", "3D", 16)
```

## 文件格式

### CUBE文件格式

生成的CUBE文件遵循标准格式：

```
# LUT标题
LUT_3D_SIZE 32
DOMAIN_MIN 0.0 0.0 0.0
DOMAIN_MAX 1.0 1.0 1.0

0.000000 0.000000 0.000000
0.032258 0.000000 0.000000
...
```

### 支持的功能

- ✅ 3D LUT生成和保存
- ✅ 1D LUT生成和保存
- ✅ CUBE文件格式读写
- ✅ 曲线插值
- ✅ 变换函数支持
- ✅ 单位LUT生成

## 与DiVERE的集成

### 接口设计

LUT生成器通过`DiVERELUTInterface`为DiVERE提供接口：

1. **generate_pipeline_lut()**：生成完整的pipeline LUT
2. **generate_curve_lut()**：生成曲线LUT
3. **generate_identity_lut()**：生成单位LUT
4. **load_lut()**：加载LUT文件

### Pipeline隔离

- 接口层不关心pipeline的具体实现
- pipeline配置由core模块提供
- 变换逻辑在core中实现，接口层只负责LUT生成

### 使用示例

```python
# 在DiVERE的core模块中
def create_pipeline_transform(params):
    """创建pipeline变换函数"""
    def transform(rgb):
        # 实现具体的pipeline逻辑
        # 密度反相、校正矩阵、RGB增益、密度曲线等
        return processed_rgb
    return transform

# 在utils中调用
from divere.utils.lut_generator import generate_pipeline_lut

# 生成LUT
transform_func = create_pipeline_transform(params)
generate_pipeline_lut(transform_func, "output.cube", "3D", 32)
```

## 扩展性

### 自定义变换函数

```python
def custom_transform(rgb):
    """自定义变换函数"""
    # 实现你的变换逻辑
    return transformed_rgb

# 使用自定义变换生成LUT
generator = LUT3DGenerator(32)
lut = generator.generate_lut_from_transform(custom_transform)
```

### 自定义曲线

```python
# 定义复杂的曲线
complex_curves = {
    'R': [(0.0, 0.0), (0.2, 0.1), (0.8, 0.9), (1.0, 1.0)],
    'G': [(0.0, 0.0), (0.3, 0.2), (0.7, 0.8), (1.0, 1.0)],
    'B': [(0.0, 0.0), (0.1, 0.05), (0.9, 0.95), (1.0, 1.0)]
}

# 生成LUT
generator = LUT1DGenerator(1024)
lut = generator.generate_lut_from_curves(complex_curves)
```

## 注意事项

1. **LUT大小**：3D LUT大小建议在16-64之间，1D LUT建议在256-4096之间
2. **性能考虑**：大尺寸LUT生成时间较长，建议根据需求选择合适大小
3. **内存使用**：3D LUT内存占用为size³×3×4字节（float32）
4. **文件大小**：CUBE文件大小约为LUT数据大小+头部信息

## 示例代码

完整的使用示例请参考 `example.py` 文件。 