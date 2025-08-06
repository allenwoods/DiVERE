"""
LUT生成器使用示例
演示如何使用DiVERE的LUT生成器
"""

import numpy as np
from pathlib import Path

from .core import LUT3DGenerator, LUT1DGenerator, LUTManager
from .interface import DiVERELUTInterface, generate_pipeline_lut, generate_curve_lut


def example_3d_lut():
    """3D LUT生成示例"""
    print("=== 3D LUT生成示例 ===")
    
    # 创建3D LUT生成器
    generator = LUT3DGenerator(size=16)
    
    # 定义变换函数（例如：增加对比度）
    def contrast_transform(rgb):
        # 简单的对比度调整
        contrast = 1.2
        rgb = (rgb - 0.5) * contrast + 0.5
        return np.clip(rgb, 0, 1)
    
    # 生成LUT
    lut = generator.generate_lut_from_transform(contrast_transform)
    print(f"生成的3D LUT形状: {lut.shape}")
    
    # 保存为CUBE文件
    output_path = "example_contrast_3d.cube"
    success = generator.save_cube(lut, output_path, "DiVERE Contrast 3D LUT")
    print(f"保存3D LUT: {'成功' if success else '失败'}")
    
    return lut


def example_1d_lut():
    """1D LUT生成示例"""
    print("\n=== 1D LUT生成示例 ===")
    
    # 创建1D LUT生成器
    generator = LUT1DGenerator(size=256)
    
    # 定义曲线（例如：S型曲线）
    curves = {
        'R': [(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)],
        'G': [(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)],
        'B': [(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)]
    }
    
    # 生成LUT
    lut = generator.generate_lut_from_curves(curves)
    print(f"生成的1D LUT形状: {lut.shape}")
    
    # 保存为CUBE文件
    output_path = "example_s_curve_1d.cube"
    success = generator.save_cube(lut, output_path, "DiVERE S-Curve 1D LUT")
    print(f"保存1D LUT: {'成功' if success else '失败'}")
    
    return lut


def example_pipeline_lut():
    """Pipeline LUT生成示例"""
    print("\n=== Pipeline LUT生成示例 ===")
    
    # 创建DiVERE LUT接口
    interface = DiVERELUTInterface()
    
    # 模拟pipeline配置（实际使用时由core提供）
    pipeline_config = {
        'density_gamma': 2.0,
        'density_gain': 1.0,
        'rgb_gains': (0.1, -0.05, 0.2),
        'curve_points': [(0.0, 0.0), (0.5, 0.4), (1.0, 1.0)],
        'correction_matrix': np.eye(3)
    }
    
    # 生成3D pipeline LUT
    output_path_3d = "example_pipeline_3d.cube"
    success_3d = interface.generate_pipeline_lut(
        pipeline_config, output_path_3d, "3D", 32
    )
    print(f"生成3D Pipeline LUT: {'成功' if success_3d else '失败'}")
    
    # 生成1D pipeline LUT
    output_path_1d = "example_pipeline_1d.cube"
    success_1d = interface.generate_pipeline_lut(
        pipeline_config, output_path_1d, "1D", 1024
    )
    print(f"生成1D Pipeline LUT: {'成功' if success_1d else '失败'}")


def example_curve_lut():
    """曲线LUT生成示例"""
    print("\n=== 曲线LUT生成示例 ===")
    
    # 定义复杂的曲线
    curves = {
        'R': [(0.0, 0.0), (0.2, 0.1), (0.5, 0.4), (0.8, 0.9), (1.0, 1.0)],
        'G': [(0.0, 0.0), (0.3, 0.2), (0.6, 0.5), (0.9, 0.95), (1.0, 1.0)],
        'B': [(0.0, 0.0), (0.1, 0.05), (0.4, 0.3), (0.7, 0.8), (1.0, 1.0)]
    }
    
    # 使用便捷函数生成LUT
    output_path = "example_complex_curves_1d.cube"
    success = generate_curve_lut(curves, output_path, 512)
    print(f"生成复杂曲线LUT: {'成功' if success else '失败'}")


def example_identity_lut():
    """单位LUT生成示例"""
    print("\n=== 单位LUT生成示例 ===")
    
    # 生成3D单位LUT
    success_3d = generate_identity_lut("identity_3d.cube", "3D", 16)
    print(f"生成3D单位LUT: {'成功' if success_3d else '失败'}")
    
    # 生成1D单位LUT
    success_1d = generate_identity_lut("identity_1d.cube", "1D", 256)
    print(f"生成1D单位LUT: {'成功' if success_1d else '失败'}")


def example_load_lut():
    """LUT加载示例"""
    print("\n=== LUT加载示例 ===")
    
    interface = DiVERELUTInterface()
    
    # 尝试加载之前生成的LUT
    test_files = [
        "example_contrast_3d.cube",
        "example_s_curve_1d.cube",
        "example_pipeline_3d.cube"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            lut_info = interface.load_lut(file_path)
            if lut_info:
                print(f"成功加载 {file_path}: {lut_info['type']} LUT, 大小: {lut_info['size']}")
            else:
                print(f"加载 {file_path} 失败")
        else:
            print(f"文件 {file_path} 不存在")


def run_all_examples():
    """运行所有示例"""
    print("DiVERE LUT生成器示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_3d_lut()
        example_1d_lut()
        example_pipeline_lut()
        example_curve_lut()
        example_identity_lut()
        example_load_lut()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("生成的文件:")
        
        # 列出生成的文件
        for file_path in Path(".").glob("*.cube"):
            print(f"  - {file_path}")
            
    except Exception as e:
        print(f"运行示例时出错: {e}")


if __name__ == "__main__":
    run_all_examples() 