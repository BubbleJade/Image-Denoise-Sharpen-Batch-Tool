#!/usr/bin/env python3
"""
Image Denoise & Sharpen Batch Tool
图像批量降噪与清晰度增强工具

主程序入口
"""

import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

from src.processor import ImageProcessor
from src.utils import (
    load_config, 
    ensure_dir, 
    get_image_files, 
    print_config_summary
)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎨 图像批量降噪与清晰度增强工具")
    print("   Image Denoise & Sharpen Batch Tool")
    print("="*60 + "\n")
    
    # 加载配置
    config = load_config("config.yaml")
    
    # 打印配置摘要
    if config.get('verbose', True):
        print_config_summary(config)
    
    # 获取输入输出路径
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 获取所有图片文件
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print(f"❌ 在目录 '{input_dir}' 中未找到图片文件")
        print(f"   支持的格式: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        print(f"\n💡 请将图片放入 '{input_dir}' 目录后重试")
        return
    
    print(f"📂 找到 {len(image_files)} 张图片\n")
    
    # 创建图像处理器
    processor = ImageProcessor(config)
    
    # 处理统计
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    # 批量处理图片
    print("🚀 开始处理...\n")
    
    for image_path in tqdm(image_files, desc="处理进度", unit="张"):
        output_path = processor.process_image(image_path, output_dir)
        
        if output_path:
            success_count += 1
        else:
            fail_count += 1
        
        if config.get('verbose', True):
            print()  # 添加空行分隔
    
    # 计算处理时间
    elapsed_time = time.time() - start_time
    
    # 打印统计信息
    print("\n" + "="*60)
    print("📊 处理完成！统计信息:")
    print("="*60)
    print(f"✅ 成功: {success_count} 张")
    print(f"❌ 失败: {fail_count} 张")
    print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
    if success_count > 0:
        print(f"⚡ 平均速度: {elapsed_time/success_count:.2f} 秒/张")
    print(f"📁 输出目录: {output_dir}")
    print("="*60 + "\n")
    
    # 处理器统计
    stats = processor.get_processing_stats()
    print("🔧 处理参数:")
    print(f"  降噪模式: {stats['denoise_mode']}")
    if stats['sharpen_enabled']:
        print(f"  锐化模式: {stats['sharpen_mode']}")
    else:
        print(f"  锐化: 未启用")
    print(f"  输出格式: {stats['output_format']}")
    print("="*60 + "\n")


def check_dependencies():
    """检查依赖是否已安装"""
    try:
        import cv2
        import numpy
        import yaml
        import tqdm
        from PIL import Image
    except ImportError as e:
        print(f"\n❌ 缺少依赖包: {e.name}")
        print("\n请运行以下命令安装依赖:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    # 检查依赖
    check_dependencies()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n程序已执行完毕，按 Enter 键退出...")
    input()