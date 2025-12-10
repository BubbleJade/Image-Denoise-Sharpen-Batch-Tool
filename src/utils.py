"""
工具函数模块
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"警告: 配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"错误: 配置文件解析失败 - {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'paths': {
            'input_dir': 'input',
            'output_dir': 'output'
        },
        'denoise': {
            'mode': 'bilateral',
            'bilateral': {'d': 9, 'sigma_color': 75, 'sigma_space': 75}
        },
        'sharpen': {
            'enabled': True,
            'mode': 'unsharp_mask',
            'unsharp_mask': {'sigma': 1.0, 'strength': 1.5, 'threshold': 0}
        },
        'output': {
            'format': 'png',
            'quality': 95,
            'prefix': 'processed_',
            'keep_original_name': True
        },
        'processing': {
            'resize': False,
            'max_width': 1920,
            'max_height': 1080,
            'preserve_aspect_ratio': True
        },
        'verbose': True
    }


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_image_files(directory: str) -> List[str]:
    """
    获取目录中的所有图片文件
    
    Args:
        directory: 目录路径
        
    Returns:
        图片文件路径列表
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    if not os.path.exists(directory):
        print(f"警告: 目录 {directory} 不存在")
        return image_files
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_formats:
                image_files.append(file_path)
    
    return sorted(image_files)


def generate_output_filename(input_path: str, output_dir: str, 
                            prefix: str = "processed_", 
                            output_format: str = "png",
                            keep_original_name: bool = True) -> str:
    """
    生成输出文件名
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        prefix: 文件名前缀
        output_format: 输出格式
        keep_original_name: 是否保留原文件名
        
    Returns:
        输出文件完整路径
    """
    basename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(basename)[0]
    
    if keep_original_name:
        output_name = f"{prefix}{name_without_ext}.{output_format}"
    else:
        output_name = f"{prefix}{os.path.splitext(basename)[0]}.{output_format}"
    
    return os.path.join(output_dir, output_name)


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    打印配置摘要
    
    Args:
        config: 配置字典
    """
    print("\n" + "="*60)
    print("🎨 图像处理配置摘要")
    print("="*60)
    
    print(f"\n📂 路径配置:")
    print(f"  输入目录: {config['paths']['input_dir']}")
    print(f"  输出目录: {config['paths']['output_dir']}")
    
    print(f"\n🔧 降噪配置:")
    print(f"  模式: {config['denoise']['mode']}")
    
    print(f"\n✨ 锐化配置:")
    print(f"  启用: {config['sharpen']['enabled']}")
    if config['sharpen']['enabled']:
        print(f"  模式: {config['sharpen']['mode']}")
    
    print(f"\n💾 输出配置:")
    print(f"  格式: {config['output']['format']}")
    print(f"  质量: {config['output']['quality']}")
    print(f"  前缀: {config['output']['prefix']}")
    
    print("="*60 + "\n")