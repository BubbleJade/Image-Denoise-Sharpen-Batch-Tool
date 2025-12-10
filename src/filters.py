"""
图像滤波器模块 - 降噪和锐化
"""

import cv2
import numpy as np
from typing import Tuple, List


class DenoiseFilter:
    """降噪滤波器类"""
    
    @staticmethod
    def gaussian_denoise(image: np.ndarray, 
                        kernel_size: Tuple[int, int] = (5, 5), 
                        sigma: float = 1.0) -> np.ndarray:
        """
        高斯降噪
        
        Args:
            image: 输入图像
            kernel_size: 核大小 (必须是奇数)
            sigma: 标准差
            
        Returns:
            降噪后的图像
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def mean_denoise(image: np.ndarray, 
                    kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        均值降噪
        
        Args:
            image: 输入图像
            kernel_size: 核大小
            
        Returns:
            降噪后的图像
        """
        return cv2.blur(image, kernel_size)
    
    @staticmethod
    def median_denoise(image: np.ndarray, 
                      kernel_size: int = 5) -> np.ndarray:
        """
        中值降噪 (对椒盐噪声效果好)
        
        Args:
            image: 输入图像
            kernel_size: 核大小 (必须是奇数)
            
        Returns:
            降噪后的图像
        """
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_denoise(image: np.ndarray, 
                         d: int = 9,
                         sigma_color: float = 75,
                         sigma_space: float = 75) -> np.ndarray:
        """
        双边滤波降噪 (保边降噪，效果好)
        
        Args:
            image: 输入图像
            d: 像素邻域直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差
            
        Returns:
            降噪后的图像
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def nlmeans_denoise(image: np.ndarray,
                       h: float = 10,
                       template_window_size: int = 7,
                       search_window_size: int = 21) -> np.ndarray:
        """
        非局部均值降噪 (效果最好但较慢)
        
        Args:
            image: 输入图像
            h: 滤波强度
            template_window_size: 模板窗口大小
            search_window_size: 搜索窗口大小
            
        Returns:
            降噪后的图像
        """
        if len(image.shape) == 3:
            # 彩色图像
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, h, 
                template_window_size, search_window_size
            )
        else:
            # 灰度图像
            return cv2.fastNlMeansDenoising(
                image, None, h, 
                template_window_size, search_window_size
            )


class SharpenFilter:
    """锐化滤波器类"""
    
    @staticmethod
    def laplacian_sharpen(image: np.ndarray, 
                         strength: float = 0.5) -> np.ndarray:
        """
        拉普拉斯锐化
        
        Args:
            image: 输入图像
            strength: 锐化强度 (0.0-2.0)
            
        Returns:
            锐化后的图像
        """
        # 转换为浮点型
        img_float = image.astype(np.float32)
        
        # 应用拉普拉斯算子
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
        
        # 锐化
        sharpened = img_float - strength * laplacian
        
        # 裁剪并转换回uint8
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    @staticmethod
    def unsharp_mask(image: np.ndarray,
                    sigma: float = 1.0,
                    strength: float = 1.5,
                    threshold: int = 0) -> np.ndarray:
        """
        反锐化掩模 (Unsharp Mask) - 专业锐化方法
        
        Args:
            image: 输入图像
            sigma: 高斯模糊标准差
            strength: 锐化强度
            threshold: 阈值（低于此值的像素不锐化）
            
        Returns:
            锐化后的图像
        """
        # 创建模糊版本
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # 计算锐化掩模
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        
        # 应用阈值
        if threshold > 0:
            low_contrast_mask = np.abs(image - blurred) < threshold
            sharpened = np.where(low_contrast_mask, image, sharpened)
        
        return sharpened
    
    @staticmethod
    def custom_kernel_sharpen(image: np.ndarray, 
                            kernel: np.ndarray = None) -> np.ndarray:
        """
        自定义卷积核锐化
        
        Args:
            image: 输入图像
            kernel: 自定义卷积核
            
        Returns:
            锐化后的图像
        """
        if kernel is None:
            # 默认锐化核
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
        
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def adaptive_sharpen(image: np.ndarray, 
                        strength: float = 1.0) -> np.ndarray:
        """
        自适应锐化（基于边缘检测）
        
        Args:
            image: 输入图像
            strength: 锐化强度
            
        Returns:
            锐化后的图像
        """
        # 转换为灰度图用于边缘检测
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        
        # 在边缘区域进行锐化
        sharpened = SharpenFilter.unsharp_mask(image, sigma=1.0, strength=strength)
        
        # 创建边缘掩模
        if len(image.shape) == 3:
            edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            edge_mask = edges
        
        # 在边缘区域应用锐化
        edge_mask = edge_mask.astype(np.float32) / 255.0
        result = (sharpened * edge_mask + image * (1 - edge_mask)).astype(np.uint8)
        
        return result


def apply_denoise(image: np.ndarray, mode: str, params: dict) -> np.ndarray:
    """
    应用降噪滤波
    
    Args:
        image: 输入图像
        mode: 降噪模式
        params: 参数字典
        
    Returns:
        降噪后的图像
    """
    if mode == 'gaussian':
        return DenoiseFilter.gaussian_denoise(
            image,
            tuple(params.get('kernel_size', [5, 5])),
            params.get('sigma', 1.0)
        )
    elif mode == 'mean':
        return DenoiseFilter.mean_denoise(
            image,
            tuple(params.get('kernel_size', [5, 5]))
        )
    elif mode == 'median':
        return DenoiseFilter.median_denoise(
            image,
            params.get('kernel_size', 5)
        )
    elif mode == 'bilateral':
        return DenoiseFilter.bilateral_denoise(
            image,
            params.get('d', 9),
            params.get('sigma_color', 75),
            params.get('sigma_space', 75)
        )
    elif mode == 'nlmeans':
        return DenoiseFilter.nlmeans_denoise(
            image,
            params.get('h', 10),
            params.get('template_window_size', 7),
            params.get('search_window_size', 21)
        )
    else:
        raise ValueError(f"不支持的降噪模式: {mode}")


def apply_sharpen(image: np.ndarray, mode: str, params: dict) -> np.ndarray:
    """
    应用锐化滤波
    
    Args:
        image: 输入图像
        mode: 锐化模式
        params: 参数字典
        
    Returns:
        锐化后的图像
    """
    if mode == 'laplacian':
        return SharpenFilter.laplacian_sharpen(
            image,
            params.get('strength', 0.5)
        )
    elif mode == 'unsharp_mask':
        return SharpenFilter.unsharp_mask(
            image,
            params.get('sigma', 1.0),
            params.get('strength', 1.5),
            params.get('threshold', 0)
        )
    elif mode == 'custom':
        kernel = params.get('kernel', None)
        if kernel is not None:
            kernel = np.array(kernel, dtype=np.float32)
        return SharpenFilter.custom_kernel_sharpen(image, kernel)
    elif mode == 'adaptive':
        return SharpenFilter.adaptive_sharpen(
            image,
            params.get('strength', 1.0)
        )
    else:
        raise ValueError(f"不支持的锐化模式: {mode}")