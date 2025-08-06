"""
图像管理模块
负责图像的导入、代理生成、缓存管理
"""

import os
import hashlib
from typing import Optional, Dict, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import imageio

from .data_types import ImageData


class ImageManager:
    """图像管理器"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._proxy_cache: Dict[str, ImageData] = {}
        self._max_cache_size = 10  # 最大缓存图像数量
        
    def load_image(self, file_path: str) -> ImageData:
        """加载图像文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {file_path}")
        
        # 尝试使用OpenCV加载
        try:
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV无法加载图像")
            
            # OpenCV使用BGR格式，转换为RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 转换为float32并归一化到[0,1]
            if image.dtype != np.float32:
                original_dtype = image.dtype
                image = image.astype(np.float32)
                
                # 根据原始数据类型进行正确的归一化
                if original_dtype == np.uint8:
                    # 8bit图像: 0-255 -> 0-1
                    image /= 255.0
                elif original_dtype == np.uint16:
                    # 16bit图像: 0-65535 -> 0-1
                    image /= 65535.0
                elif original_dtype == np.uint32:
                    # 32bit图像: 0-4294967295 -> 0-1
                    image /= 4294967295.0
                elif original_dtype == np.int16:
                    # 16bit有符号: -32768-32767 -> 0-1
                    image = (image + 32768) / 65535.0
                elif original_dtype == np.int32:
                    # 32bit有符号: -2147483648-2147483647 -> 0-1
                    image = (image + 2147483648) / 4294967295.0
                elif image.max() > 1.0:
                    # 其他情况，使用最大值归一化
                    image /= image.max()
            
        except Exception as e:
            # 如果OpenCV失败，尝试使用PIL
            try:
                pil_image = Image.open(file_path)
                original_mode = pil_image.mode
                image = np.array(pil_image, dtype=np.float32)
                
                # 根据PIL模式进行正确的归一化
                if original_mode in ['L', 'RGB', 'RGBA']:
                    # 8bit图像
                    if image.max() > 1.0:
                        image /= 255.0
                elif original_mode in ['I', 'I;16']:
                    # 16bit图像
                    if image.max() > 1.0:
                        image /= 65535.0
                elif original_mode in ['F']:
                    # 32bit浮点图像，通常已经是0-1范围
                    pass
                elif image.max() > 1.0:
                    # 其他情况，使用最大值归一化
                    image /= image.max()
                
                # 处理灰度图像
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]
                    
            except Exception as e2:
                raise RuntimeError(f"无法加载图像文件: {e}, {e2}")
        
        # 创建ImageData对象（默认Film_KodakRGB_Linear，色彩空间将由用户手动选择）
        image_data = ImageData(
            array=image,
            color_space="Film_KodakRGB_Linear",  # 默认Film_KodakRGB_Linear
            file_path=str(file_path),
            is_proxy=False,
            proxy_scale=1.0
        )
        
        return image_data
    
    def generate_proxy(self, image: ImageData, max_size: Tuple[int, int] = (1920, 1080)) -> ImageData:
        """生成代理图像"""
        # 移除这个检查，允许对代理图像进行进一步缩放
        # if image.is_proxy:
        #     return image
        
        # 计算缩放比例
        h, w = image.height, image.width
        max_w, max_h = max_size
        
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h, 1.0)  # 不放大图像
        
        if scale >= 1.0:
            # 图像已经足够小，直接返回
            return image
        
        # 计算新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 使用OpenCV进行高质量缩放
        proxy_array = cv2.resize(
            image.array, 
            (new_w, new_h), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # 创建代理ImageData
        proxy_data = ImageData(
            array=proxy_array,
            color_space=image.color_space,
            icc_profile=image.icc_profile,
            metadata=image.metadata,
            file_path=image.file_path,
            is_proxy=True,
            proxy_scale=scale
        )
        
        return proxy_data
    
    def get_cached_proxy(self, image_id: str) -> Optional[ImageData]:
        """获取缓存的代理图像"""
        return self._proxy_cache.get(image_id)
    
    def cache_proxy(self, image_id: str, proxy: ImageData):
        """缓存代理图像"""
        # 简单的LRU缓存实现
        if len(self._proxy_cache) >= self._max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._proxy_cache))
            del self._proxy_cache[oldest_key]
        
        self._proxy_cache[image_id] = proxy
    
    def clear_cache(self):
        """清空缓存"""
        self._proxy_cache.clear()
    
    def get_image_id(self, file_path: str) -> str:
        """生成图像唯一标识符"""
        # 使用文件路径和修改时间的哈希作为ID
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def save_image(self, image_data: ImageData, output_path: str, quality: int = 95, bit_depth: int = 8):
        """保存图像"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保图像数据在[0,1]范围内
        image_array = np.clip(image_data.array, 0, 1)
        
        # 根据位深度和文件格式选择保存类型
        ext = output_path.suffix.lower()
        
        if bit_depth == 16 and ext in ['.png', '.tiff', '.tif']:
            # 16bit保存
            image_array = (image_array * 65535).astype(np.uint16)
        else:
            # 8bit保存
            image_array = (image_array * 255).astype(np.uint8)
        
        # 根据文件扩展名选择保存格式
        ext = output_path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg']:
            # JPEG格式
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # 转换为BGR用于OpenCV
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(output_path), image_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        elif ext in ['.png']:
            # PNG格式
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image)
            else:
                cv2.imwrite(str(output_path), image_array)
        
        elif ext in ['.tiff', '.tif']:
            # TIFF格式
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image)
            else:
                cv2.imwrite(str(output_path), image_array)
        
        else:
            # 默认使用PIL保存
            pil_image = Image.fromarray(image_array)
            pil_image.save(output_path, quality=quality)
    
    def get_supported_formats(self) -> list:
        """获取支持的图像格式"""
        return [
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', 
            '.bmp', '.webp', '.exr', '.hdr'
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = Path(file_path).suffix.lower()
        return ext in self.get_supported_formats() 