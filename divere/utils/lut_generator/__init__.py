"""
DiVERE LUT生成器包
提供3D LUT和1D LUT的生成功能，支持标准CUBE格式输出
"""

from .core import (
    LUT3DGenerator,
    LUT1DGenerator,
    LUTManager,
    create_3d_lut,
    create_1d_lut,
    save_lut_to_file
)

from .interface import (
    DiVERELUTInterface,
    generate_pipeline_lut,
    generate_curve_lut,
    generate_identity_lut
)

__all__ = [
    # 核心LUT生成器
    "LUT3DGenerator",
    "LUT1DGenerator",
    "LUTManager",
    
    # 便捷函数
    "create_3d_lut",
    "create_1d_lut", 
    "save_lut_to_file",
    
    # DiVERE接口
    "DiVERELUTInterface",
    "generate_pipeline_lut",
    "generate_curve_lut",
    "generate_identity_lut"
]

# 版本信息
__version__ = "1.0.0"
__author__ = "V7"
__email__ = "vanadis@yeah.net" 