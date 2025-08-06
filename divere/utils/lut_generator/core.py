"""
LUT生成器
提供3D LUT和1D LUT的生成功能，支持标准CUBE格式输出
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import json


class LUT3DGenerator:
    """3D LUT生成器"""
    
    def __init__(self, size: int = 32):
        """
        初始化3D LUT生成器
        
        Args:
            size: LUT大小，默认为32x32x32
        """
        self.size = size
        self._validate_size()
        
    def _validate_size(self):
        """验证LUT大小"""
        if self.size < 2 or self.size > 256:
            raise ValueError("LUT大小必须在2到256之间")
        if not isinstance(self.size, int):
            raise ValueError("LUT大小必须是整数")
    
    def generate_identity_lut(self) -> np.ndarray:
        """
        生成单位LUT（无变换）
        
        Returns:
            3D LUT数组，形状为(size, size, size, 3)
        """
        # 创建输入坐标网格
        r, g, b = np.meshgrid(
            np.linspace(0, 1, self.size),
            np.linspace(0, 1, self.size),
            np.linspace(0, 1, self.size),
            indexing='ij'
        )
        
        # 单位变换：输出 = 输入
        lut = np.stack([r, g, b], axis=-1)
        return lut
    
    def generate_lut_from_transform(self, transform_func) -> np.ndarray:
        """
        从变换函数生成LUT
        
        Args:
            transform_func: 变换函数，接受RGB输入，返回RGB输出
            
        Returns:
            3D LUT数组
        """
        # 创建输入坐标网格
        r, g, b = np.meshgrid(
            np.linspace(0, 1, self.size),
            np.linspace(0, 1, self.size),
            np.linspace(0, 1, self.size),
            indexing='ij'
        )
        
        # 重塑为2D数组以便批量处理
        input_rgb = np.stack([r.flatten(), g.flatten(), b.flatten()], axis=1)
        
        # 应用变换
        output_rgb = transform_func(input_rgb)
        
        # 重塑回3D
        lut = output_rgb.reshape(self.size, self.size, self.size, 3)
        return lut
    
    def save_cube(self, lut: np.ndarray, filepath: str, title: str = "DiVERE Generated LUT") -> bool:
        """
        保存为CUBE格式文件
        
        Args:
            lut: 3D LUT数组
            filepath: 输出文件路径
            title: LUT标题
            
        Returns:
            是否保存成功
        """
        try:
            with open(filepath, 'w') as f:
                # 写入头部信息
                f.write(f"# {title}\n")
                f.write(f"LUT_3D_SIZE {self.size}\n")
                f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
                f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
                f.write("\n")
                
                # 写入LUT数据
                for b in range(self.size):
                    for g in range(self.size):
                        for r in range(self.size):
                            rgb = lut[r, g, b]
                            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
            
            return True
        except Exception as e:
            print(f"保存CUBE文件失败: {e}")
            return False
    
    def load_cube(self, filepath: str) -> Optional[np.ndarray]:
        """
        从CUBE文件加载LUT
        
        Args:
            filepath: CUBE文件路径
            
        Returns:
            3D LUT数组，如果加载失败返回None
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # 解析头部信息
            size = None
            domain_min = [0.0, 0.0, 0.0]
            domain_max = [1.0, 1.0, 1.0]
            
            data_start = 0
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif line.startswith('LUT_3D_SIZE'):
                    size = int(line.split()[1])
                elif line.startswith('DOMAIN_MIN'):
                    domain_min = [float(x) for x in line.split()[1:]]
                elif line.startswith('DOMAIN_MAX'):
                    domain_max = [float(x) for x in line.split()[1:]]
                elif not line:
                    data_start = i + 1
                    break
            
            if size is None:
                raise ValueError("无法解析LUT大小")
            
            # 读取LUT数据
            lut_data = []
            for line in lines[data_start:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    rgb = [float(x) for x in line.split()[:3]]
                    lut_data.append(rgb)
            
            # 重塑为3D数组
            lut = np.array(lut_data).reshape(size, size, size, 3)
            return lut
            
        except Exception as e:
            print(f"加载CUBE文件失败: {e}")
            return None


class LUT1DGenerator:
    """1D LUT生成器"""
    
    def __init__(self, size: int = 1024):
        """
        初始化1D LUT生成器
        
        Args:
            size: LUT大小，默认为1024
        """
        self.size = size
        self._validate_size()
    
    def _validate_size(self):
        """验证LUT大小"""
        if self.size < 2 or self.size > 65536:
            raise ValueError("1D LUT大小必须在2到65536之间")
        if not isinstance(self.size, int):
            raise ValueError("LUT大小必须是整数")
    
    def generate_identity_lut(self) -> np.ndarray:
        """
        生成单位1D LUT
        
        Returns:
            1D LUT数组，形状为(size, 3)
        """
        x = np.linspace(0, 1, self.size)
        lut = np.column_stack([x, x, x])  # R=G=B=x
        return lut
    
    def generate_lut_from_curves(self, curves: Dict[str, List[Tuple[float, float]]]) -> np.ndarray:
        """
        从曲线生成1D LUT
        
        Args:
            curves: 曲线字典，键为'R', 'G', 'B'，值为控制点列表
            
        Returns:
            1D LUT数组
        """
        x = np.linspace(0, 1, self.size)
        lut = np.zeros((self.size, 3))
        
        for i, channel in enumerate(['R', 'G', 'B']):
            if channel in curves:
                lut[:, i] = self._interpolate_curve(x, curves[channel])
            else:
                lut[:, i] = x  # 默认线性
        
        return lut
    
    def _interpolate_curve(self, x: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        插值曲线
        
        Args:
            x: 输入值数组
            points: 控制点列表
            
        Returns:
            插值后的值数组
        """
        if len(points) < 2:
            return x
        
        # 排序控制点
        sorted_points = sorted(points, key=lambda p: p[0])
        x_points = np.array([p[0] for p in sorted_points])
        y_points = np.array([p[1] for p in sorted_points])
        
        # 线性插值
        return np.interp(x, x_points, y_points)
    
    def save_cube(self, lut: np.ndarray, filepath: str, title: str = "DiVERE Generated 1D LUT") -> bool:
        """
        保存为CUBE格式文件（1D LUT）
        
        Args:
            lut: 1D LUT数组
            filepath: 输出文件路径
            title: LUT标题
            
        Returns:
            是否保存成功
        """
        try:
            with open(filepath, 'w') as f:
                # 写入头部信息
                f.write(f"# {title}\n")
                f.write(f"LUT_1D_SIZE {self.size}\n")
                f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
                f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
                f.write("\n")
                
                # 写入LUT数据
                for i in range(self.size):
                    rgb = lut[i]
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
            
            return True
        except Exception as e:
            print(f"保存1D CUBE文件失败: {e}")
            return False


class LUTManager:
    """LUT管理器 - 提供统一的接口"""
    
    def __init__(self):
        """初始化LUT管理器"""
        self._3d_generator = LUT3DGenerator()
        self._1d_generator = LUT1DGenerator()
    
    def generate_3d_lut(self, transform_func, size: int = 32, title: str = "DiVERE 3D LUT") -> Dict[str, Any]:
        """
        生成3D LUT
        
        Args:
            transform_func: 变换函数
            size: LUT大小
            title: LUT标题
            
        Returns:
            包含LUT数据和元数据的字典
        """
        generator = LUT3DGenerator(size)
        lut = generator.generate_lut_from_transform(transform_func)
        
        return {
            'type': '3D',
            'size': size,
            'data': lut,
            'title': title,
            'generator': generator
        }
    
    def generate_1d_lut(self, curves: Dict[str, List[Tuple[float, float]]], 
                       size: int = 1024, title: str = "DiVERE 1D LUT") -> Dict[str, Any]:
        """
        生成1D LUT
        
        Args:
            curves: 曲线字典
            size: LUT大小
            title: LUT标题
            
        Returns:
            包含LUT数据和元数据的字典
        """
        generator = LUT1DGenerator(size)
        lut = generator.generate_lut_from_curves(curves)
        
        return {
            'type': '1D',
            'size': size,
            'data': lut,
            'title': title,
            'curves': curves,
            'generator': generator
        }
    
    def save_lut(self, lut_info: Dict[str, Any], filepath: str) -> bool:
        """
        保存LUT到文件
        
        Args:
            lut_info: LUT信息字典
            filepath: 输出文件路径
            
        Returns:
            是否保存成功
        """
        try:
            if lut_info['type'] == '3D':
                return lut_info['generator'].save_cube(
                    lut_info['data'], filepath, lut_info['title']
                )
            elif lut_info['type'] == '1D':
                return lut_info['generator'].save_cube(
                    lut_info['data'], filepath, lut_info['title']
                )
            else:
                raise ValueError(f"不支持的LUT类型: {lut_info['type']}")
        except Exception as e:
            print(f"保存LUT失败: {e}")
            return False
    
    def load_lut(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载LUT
        
        Args:
            filepath: LUT文件路径
            
        Returns:
            LUT信息字典，如果加载失败返回None
        """
        try:
            # 尝试作为3D LUT加载
            lut = self._3d_generator.load_cube(filepath)
            if lut is not None:
                return {
                    'type': '3D',
                    'size': lut.shape[0],
                    'data': lut,
                    'title': f"Loaded 3D LUT from {Path(filepath).name}",
                    'generator': LUT3DGenerator(lut.shape[0])
                }
            
            # 尝试作为1D LUT加载
            # 这里需要实现1D LUT的加载逻辑
            return None
            
        except Exception as e:
            print(f"加载LUT失败: {e}")
            return None


# 便捷函数
def create_3d_lut(transform_func, size: int = 32, title: str = "DiVERE 3D LUT") -> Dict[str, Any]:
    """创建3D LUT的便捷函数"""
    manager = LUTManager()
    return manager.generate_3d_lut(transform_func, size, title)


def create_1d_lut(curves: Dict[str, List[Tuple[float, float]]], 
                 size: int = 1024, title: str = "DiVERE 1D LUT") -> Dict[str, Any]:
    """创建1D LUT的便捷函数"""
    manager = LUTManager()
    return manager.generate_1d_lut(curves, size, title)


def save_lut_to_file(lut_info: Dict[str, Any], filepath: str) -> bool:
    """保存LUT到文件的便捷函数"""
    manager = LUTManager()
    return manager.save_lut(lut_info, filepath) 