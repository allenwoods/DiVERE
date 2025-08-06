"""
LUT接口
为DiVERE提供简单的LUT生成接口，不暴露pipeline细节
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

from .core import LUTManager, create_3d_lut, create_1d_lut, save_lut_to_file


class DiVERELUTInterface:
    """DiVERE LUT接口 - 提供简单的LUT生成功能"""
    
    def __init__(self):
        """初始化接口"""
        self._manager = LUTManager()
    
    def generate_pipeline_lut(self, pipeline_config: Dict[str, Any], 
                            output_path: str, 
                            lut_type: str = "3D",
                            size: int = 32) -> bool:
        """
        生成校色pipeline的LUT
        
        Args:
            pipeline_config: pipeline配置（由core提供，接口不关心具体内容）
            output_path: 输出文件路径
            lut_type: LUT类型，"3D"或"1D"
            size: LUT大小
            
        Returns:
            是否生成成功
        """
        try:
            if lut_type == "3D":
                # 创建变换函数（这里不关心pipeline的具体实现）
                transform_func = self._create_transform_from_config(pipeline_config)
                lut_info = self._manager.generate_3d_lut(
                    transform_func, size, "DiVERE Pipeline 3D LUT"
                )
            elif lut_type == "1D":
                # 从配置中提取曲线信息
                curves = self._extract_curves_from_config(pipeline_config)
                lut_info = self._manager.generate_1d_lut(
                    curves, size, "DiVERE Pipeline 1D LUT"
                )
            else:
                raise ValueError(f"不支持的LUT类型: {lut_type}")
            
            # 保存LUT
            return self._manager.save_lut(lut_info, output_path)
            
        except Exception as e:
            print(f"生成pipeline LUT失败: {e}")
            return False
    
    def generate_curve_lut(self, curves: Dict[str, List[Tuple[float, float]]], 
                          output_path: str, 
                          size: int = 1024) -> bool:
        """
        生成曲线LUT
        
        Args:
            curves: 曲线字典，键为'R', 'G', 'B'，值为控制点列表
            output_path: 输出文件路径
            size: LUT大小
            
        Returns:
            是否生成成功
        """
        try:
            lut_info = self._manager.generate_1d_lut(
                curves, size, "DiVERE Curve 1D LUT"
            )
            return self._manager.save_lut(lut_info, output_path)
        except Exception as e:
            print(f"生成曲线LUT失败: {e}")
            return False
    
    def generate_identity_lut(self, output_path: str, 
                            lut_type: str = "3D", 
                            size: int = 32) -> bool:
        """
        生成单位LUT
        
        Args:
            output_path: 输出文件路径
            lut_type: LUT类型，"3D"或"1D"
            size: LUT大小
            
        Returns:
            是否生成成功
        """
        try:
            if lut_type == "3D":
                # 创建单位变换函数
                def identity_transform(rgb):
                    return rgb
                
                lut_info = self._manager.generate_3d_lut(
                    identity_transform, size, "DiVERE Identity 3D LUT"
                )
            elif lut_type == "1D":
                # 创建单位曲线
                curves = {
                    'R': [(0.0, 0.0), (1.0, 1.0)],
                    'G': [(0.0, 0.0), (1.0, 1.0)],
                    'B': [(0.0, 0.0), (1.0, 1.0)]
                }
                lut_info = self._manager.generate_1d_lut(
                    curves, size, "DiVERE Identity 1D LUT"
                )
            else:
                raise ValueError(f"不支持的LUT类型: {lut_type}")
            
            return self._manager.save_lut(lut_info, output_path)
            
        except Exception as e:
            print(f"生成单位LUT失败: {e}")
            return False
    
    def load_lut(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        加载LUT文件
        
        Args:
            filepath: LUT文件路径
            
        Returns:
            LUT信息字典，如果加载失败返回None
        """
        return self._manager.load_lut(filepath)
    
    def _create_transform_from_config(self, config: Dict[str, Any]):
        """
        从配置创建变换函数
        这个方法是一个占位符，实际的变换逻辑由core提供
        """
        # 这里返回一个简单的单位变换
        # 实际的实现应该由core模块提供具体的变换逻辑
        def transform(rgb):
            # 这里应该根据config中的参数实现具体的变换
            # 但接口层不关心具体实现
            return rgb
        
        return transform
    
    def _extract_curves_from_config(self, config: Dict[str, Any]) -> Dict[str, List[Tuple[float, float]]]:
        """
        从配置中提取曲线信息
        """
        curves = {}
        
        # 提取RGB主曲线
        if 'curve_points' in config:
            curves['R'] = config['curve_points']
            curves['G'] = config['curve_points']
            curves['B'] = config['curve_points']
        
        # 提取单通道曲线
        if 'curve_points_r' in config:
            curves['R'] = config['curve_points_r']
        if 'curve_points_g' in config:
            curves['G'] = config['curve_points_g']
        if 'curve_points_b' in config:
            curves['B'] = config['curve_points_b']
        
        return curves


# 便捷函数
def generate_pipeline_lut(pipeline_config: Dict[str, Any], 
                         output_path: str, 
                         lut_type: str = "3D",
                         size: int = 32) -> bool:
    """生成pipeline LUT的便捷函数"""
    interface = DiVERELUTInterface()
    return interface.generate_pipeline_lut(pipeline_config, output_path, lut_type, size)


def generate_curve_lut(curves: Dict[str, List[Tuple[float, float]]], 
                      output_path: str, 
                      size: int = 1024) -> bool:
    """生成曲线LUT的便捷函数"""
    interface = DiVERELUTInterface()
    return interface.generate_curve_lut(curves, output_path, size)


def generate_identity_lut(output_path: str, 
                         lut_type: str = "3D", 
                         size: int = 32) -> bool:
    """生成单位LUT的便捷函数"""
    interface = DiVERELUTInterface()
    return interface.generate_identity_lut(output_path, lut_type, size) 