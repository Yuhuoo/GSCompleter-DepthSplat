import cv2
import os
import re
import argparse
import numpy as np
from pathlib import Path

def numerical_sort(value):
    """提取文件名中的数字进行自然排序"""
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def add_label_to_image(img, label, bar_height=40):
    """在图像下方添加黑色条纹并写入标签文字"""
    h, w, c = img.shape
    # 1. 创建一个黑色条纹区域
    bar = np.zeros((bar_height, w, c), dtype=np.uint8)
    
    # 2. 将图像和条纹垂直拼接
    img_with_bar = cv2.vconcat([img, bar])
    
    # 3. 设置文字参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255) # 白色
    
    # 获取文字大小以居中
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h + (bar_height + text_size[1]) // 2 - 5 # 在黑条区域内居中
    
    cv2.putText(img_with_bar, label, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img_with_bar

def create_comparison_video(folder1, folder2, video_name, fps=24, label1="DepthSplat", label2="Ours"):
    """读取两个目录，打上标签，左右拼接并合成视频"""
    imgs1 = sorted([img for img in os.listdir(folder1) if img.endswith((".png", ".jpg", ".jpeg"))], key=numerical_sort)
    imgs2 = sorted([img for img in os.listdir(folder2) if img.endswith((".png", ".jpg", ".jpeg"))], key=numerical_sort)
    
    common_names = sorted(list(set(imgs1) & set(imgs2)), key=numerical_sort)
    
    if not common_names:
        print(f"  [跳过] 目录间没有共同文件。")
        return False

    # 读取第一张图确定基础尺寸
    test_img = cv2.imread(os.path.join(folder1, common_names[0]))
    if test_img is None: return False
    h, w, _ = test_img.shape
    
    # 预设下方黑条的高度
    bar_h = int(h * 0.1) # 动态高度，设为原图高度的 10%
    
    # 视频的分辨率：高度 = 原图 + 黑条；宽度 = 两张图宽度之和
    size = (w * 2, h + bar_h)
    
    Path(video_name).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    for name in common_names:
        img1 = cv2.imread(os.path.join(folder1, name))
        img2 = cv2.imread(os.path.join(folder2, name))
        
        if img1 is None or img2 is None: continue
        
        # 统一尺寸（以第一张图为准）
        if img1.shape[:2] != (h, w): img1 = cv2.resize(img1, (w, h))
        if img2.shape[:2] != (h, w): img2 = cv2.resize(img2, (w, h))
            
        # 分别添加标签
        img1_labeled = add_label_to_image(img1, label1, bar_height=bar_h)
        img2_labeled = add_label_to_image(img2, label2, bar_height=bar_h)
            
        # 左右拼接
        combined_frame = cv2.hconcat([img1_labeled, img2_labeled])
        video.write(combined_frame)

    video.release()
    print(f"  [成功] 拼接视频已保存: {video_name}")
    return True

def get_args():
    parser = argparse.ArgumentParser(description="批量左右拼接图片并添加文字标签")
    parser.add_argument("--dir1", type=str, required=True, help="左侧目录 (DepthSplat)")
    parser.add_argument("--dir2", type=str, required=True, help="右侧目录 (Ours)")
    parser.add_argument("--output_dir", type=str, default="output-video/comparison", help="输出目录")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sub_folder", type=str, default="color")
    parser.add_argument("--include_scenes", type=str, nargs='+', default=[])
    # 允许自定义标签内容
    parser.add_argument("--label1", type=str, default="DepthSplat")
    parser.add_argument("--label2", type=str, default="Ours")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    base1, base2, out_root = Path(args.dir1), Path(args.dir2), Path(args.output_dir)
    
    scenes1 = {d.name for d in base1.iterdir() if d.is_dir()}
    scenes2 = {d.name for d in base2.iterdir() if d.is_dir()}
    common_scenes = sorted(list(scenes1 & scenes2))
    target_scenes = set(args.include_scenes)

    for scene_id in common_scenes:
        if target_scenes and scene_id not in target_scenes:
            continue
            
        folder1 = base1 / scene_id / args.sub_folder
        folder2 = base2 / scene_id / args.sub_folder
        save_path = out_root / f"{scene_id}_compare.mp4"
        
        if folder1.exists() and folder2.exists():
            print(f"正在处理: {scene_id}")
            create_comparison_video(str(folder1), str(folder2), str(save_path), args.fps, args.label1, args.label2)

    cv2.destroyAllWindows()