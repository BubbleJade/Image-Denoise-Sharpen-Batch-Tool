"""
图像处理器核心模块
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from .filters import apply_denoise, apply_sharpen
from .utils import generate_output_filename


class ImageProcessor:
    """图像处理器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化图像处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.denoise_config = config.get('denoise', {})
        self.sharpen_config = config.get('sharpen', {})
        self.output_config = config.get('output', {})
        self.processing_config = config.get('processing', {})
        self.verbose = config.get('verbose', True)
        
    def process_image(self, image_path: str, output_dir: str) -> Optional[str]:
        """
        处理单张图像
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径，失败返回None
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 无法读取图像: {image_path}")
                return None
            
            original_shape = image.shape
            if self.verbose:
                print(f"📷 处理: {Path(image_path).name} ({original_shape[1]}x{original_shape[0]})")
            
            # 调整大小（可选）
            if self.processing_config.get('resize', False):
                image = self._resize_image(image)
            
            # 降噪处理
            image = self._apply_denoise(image)
            
            # 锐化处理
            if self.sharpen_config.get('enabled', True):
                image = self._apply_sharpen(image)
            
            # 生成输出文件名
            output_path = generate_output_filename(
                image_path,
                output_dir,
                self.output_config.get('prefix', 'processed_'),
                self.output_config.get('format', 'png'),
                self.output_config.get('keep_original_name', True)
            )
            
            # 保存图像
            success = self._save_image(image, output_path)
            
            if success:
                if self.verbose:
                    print(f"✅ 保存成功: {Path(output_path).name}")
                return output_path
            else:
                print(f"❌ 保存失败: {output_path}")
                return None
                
        except Exception as e:
            print(f"❌ 处理图像时出错 {image_path}: {str(e)}")
            return None
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            
        Returns:
            调整后的图像
        """
        max_width = self.processing_config.get('max_width', 1920)
        max_height = self.processing_config.get('max_height', 1080)
        preserve_aspect = self.processing_config.get('preserve_aspect_ratio', True)
        
        h, w = image.shape[:2]
        
        if w <= max_width and h <= max_height:
            return image
        
        if preserve_aspect:
            # 保持宽高比
            scale = min(max_width / w, max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w = max_width
            new_h = max_height
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if self.verbose:
            print(f"  🔄 调整大小: {w}x{h} -> {new_w}x{new_h}")
        
        return resized
    
    def _apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        应用降噪
        
        Args:
            image: 输入图像
            
        Returns:
            降噪后的图像
        """
        mode = self.denoise_config.get('mode', 'bilateral')
        params = self.denoise_config.get(mode, {})
        
        if self.verbose:
            print(f"  🔧 降噪模式: {mode}")
        
        try:
            denoised = apply_denoise(image, mode, params)
            return denoised
        except Exception as e:
            print(f"  ⚠️ 降噪失败，使用原图: {str(e)}")
            return image
    
    def _apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        应用锐化
        
        Args:
            image: 输入图像
            
        Returns:
            锐化后的图像
        """
        mode = self.sharpen_config.get('mode', 'unsharp_mask')
        params = self.sharpen_config.get(mode, {})
        
        if self.verbose:
            print(f"  ✨ 锐化模式: {mode}")
        
        try:
            sharpened = apply_sharpen(image, mode, params)
            return sharpened
        except Exception as e:
            print(f"  ⚠️ 锐化失败，使用降噪后的图: {str(e)}")
            return image
    
    def _save_image(self, image: np.ndarray, output_path: str) -> bool:
        """
        保存图像
        
        Args:
            image: 要保存的图像
            output_path: 输出路径
            
        Returns:
            是否成功保存
        """
        output_format = self.output_config.get('format', 'png').lower()
        quality = self.output_config.get('quality', 95)
        
        # 设置保存参数
        if output_format in ['jpg', 'jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif output_format == 'png':
            # PNG压缩级别 (0-9, 9为最高压缩)
            compression = max(0, min(9, int((100 - quality) / 10)))
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif output_format == 'webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        else:
            params = []
        
        try:
            success = cv2.imwrite(output_path, image, params)
            return success
        except Exception as e:
            print(f"  ⚠️ 保存图像失败: {str(e)}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'denoise_mode': self.denoise_config.get('mode'),
            'sharpen_enabled': self.sharpen_config.get('enabled'),
            'sharpen_mode': self.sharpen_config.get('mode') if self.sharpen_config.get('enabled') else None,
            'output_format': self.output_config.get('format'),
            'resize_enabled': self.processing_config.get('resize', False)
        }