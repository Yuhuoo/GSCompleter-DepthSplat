import cv2
import os
import re
import argparse
from pathlib import Path

def numerical_sort(value):
    """提取文件名中的数字进行自然排序 (1, 2, ... 10)"""
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def images_to_video(image_folder, video_name, fps=24):
    """将文件夹下的图片合成视频"""
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    
    if not images:
        print(f"  [跳过] '{image_folder}' 中没有图片。")
        return False

    images.sort(key=numerical_sort)
    
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        return False
        
    height, width, _ = frame.shape
    size = (width, height)
    
    # 确保输出视频的父目录存在
    Path(video_name).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    for image in images:
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path)
        if img is None: continue
        
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, size)
        video.write(img)

    video.release()
    print(f"  [成功] 已保存: {video_name} ({len(images)} 帧)")
    return True

def get_args():
    parser = argparse.ArgumentParser(description="批量将场景图片合成视频脚本")
    
    # 基础输入路径
    parser.add_argument("--base_dir", type=str, 
                        default="outputs/4.10/ablation-study/ours-all/dl3dv_ctx2_tgt1_n30/videos-frames",
                        help="包含各场景文件夹的根目录")
    
    # 输出保存路径
    parser.add_argument("--output_dir", type=str, 
                        default="output-video/dl3dv_ctx2_tgt1_n30",
                        help="视频输出的根目录")
    
    # 帧率
    parser.add_argument("--fps", type=int, default=24, help="视频帧率 (默认: 24)")
    
    # 内部文件夹名称 (比如你的结构里是 'color')
    parser.add_argument("--sub_folder", type=str, default="color", help="图片所在的子文件夹名")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    base_path = Path(args.base_dir)
    output_root = Path(args.output_dir)
    
    if not base_path.exists():
        print(f"错误：基础目录不存在 -> {base_path}")
    else:
        # 获取所有场景子目录
        scene_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        print(f">>> 发现 {len(scene_dirs)} 个场景，准备合成视频 (FPS: {args.fps})...")

        for scene_path in scene_dirs:
            scene_id = scene_path.name
            img_folder = scene_path / args.sub_folder
            
            # 构造输出文件名：output_root / scene_id.mp4
            save_path = output_root / f"{scene_id}.mp4"
            
            if img_folder.exists():
                print(f"正在处理场景: {scene_id}")
                images_to_video(str(img_folder), str(save_path), args.fps)
            else:
                print(f"  [跳过] {scene_id} 下未发现 '{args.sub_folder}' 文件夹")

        cv2.destroyAllWindows()
        print("\n>>> 所有任务处理完毕。")