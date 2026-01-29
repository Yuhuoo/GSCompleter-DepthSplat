"""
Generate the evaluation index for DL3DV-10K dataset.

python -m src.scripts.generate_evaluation_index --data_dir=datasets/DL3DV-10K-480/dl3dv_benchmark/test --num_context_views=36 --num_target_views=8 --view_selection_num=100 --type dl3dv

python -m src.scripts.generate_evaluation_index --data_dir=datasets/re10k/test --num_context_views=36 --num_target_views=8 --view_selection_num=200 --type re10k
"""

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import json
import argparse
import os
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

from ..dataset.view_sampler.view_sampler_bounded_v2 import farthest_point_sample

# Declare as global variable
max_gap = 0

def convert_poses(
    poses: Float[Tensor, "batch 18"],
) -> tuple[
    Float[Tensor, "batch 4 4"],  # extrinsics
    Float[Tensor, "batch 3 3"],  # intrinsics
]:
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def partition_list(lst, n_bins):
    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than 0")
    if len(lst) < n_bins:
        print("len(lst):", len(lst))
        print("n_bins:", n_bins)
        # raise ValueError("Number of bins cannot exceed the length of the list")
        return None

    bin_size = len(lst) // n_bins
    borders = [lst[0]]  # First border is always the first index
    for i in range(1, n_bins):
        border_index = min(
            i * bin_size, len(lst) - 1
        )  # Ensure last bin doesn't exceed list length
        borders.append(lst[border_index])
    borders.append(lst[-1])  # Last border is always the last index
    return borders


def find_train_and_test_index(
    chunk_path,
    scene_name=None,
    num_input_views=2,    # 对应你想要的 context 数量
    num_predict_views=1,  # 对应你想要的 target 数量
    view_selection_num=-1,
):
    chunk = torch.load(chunk_path)
    out_dict = OrderedDict()
    
    for example in chunk:
        cur_scene_name = example["key"]
        if scene_name is not None and cur_scene_name != scene_name:
            continue

        extrinsics, _ = convert_poses(example["cameras"])
        n_views = extrinsics.shape[0]
        
        # 确定搜索范围
        n_views = min(n_views, int(view_selection_num)) if view_selection_num != -1 else n_views

        # 1. 采样“最具代表性/最偏”的视角作为【预测目标】
        # 对应原代码逻辑：FPS 选出的其实是你的补全目标
        predict_indices = sorted(
            farthest_point_sample(
                extrinsics[:n_views, :3, -1].unsqueeze(0), 
                num_predict_views
            ).squeeze(0).tolist()
        )

        # 2. 从剩余视角中通过分桶选出【输入视角】
        # 对应原代码逻辑：从非 FPS 点中选出支撑点
        remaining_indices = [x for x in range(n_views) if x not in predict_indices]
        input_indices = partition_list(remaining_indices, num_input_views)
        
        if input_indices is None:
            continue
            
        # 仅取需要的数量，保持与原代码 partition_list[:n_tgt] 一致
        input_indices = input_indices[:num_input_views]

        # 3. 最终输出：保持 JSON 格式为 {"context": [输入], "target": [目标]}
        # 这样写逻辑清晰，且结果与你之前的 JSON 完全相同
        out_dict[cur_scene_name] = {
            "context": input_indices,   # 这里的 input 对应你之前的 index_target
            "target": predict_indices   # 这里的 predict 对应你之前的 index_context
        }

    return out_dict


def generate_index_file(args):
    n_ctx = args.num_context_views
    n_tgt = args.num_target_views

    out_dir = f"assets/{args.type}_extrapolation-test"
    os.makedirs(out_dir, exist_ok=True)
    
    # data_dir = "datasets/DL3DV-10K-480/dl3dv_benchmark/test"
    data_dir = args.data_dir
    chunk_paths = sorted(glob(os.path.join(data_dir, "*.torch")))  # [:2]
    out_dict_all = OrderedDict()
    for chunk_path in tqdm(chunk_paths):
        out_dict = find_train_and_test_index(
            chunk_path,
            num_input_views=args.num_target_views,   # 传入 2
            num_predict_views=args.num_context_views, # 传入 1
            view_selection_num=args.view_selection_num,
        )
        out_dict_all.update(out_dict)

    out_name = f"{args.type}_ctx_{n_ctx}v_tgt_{n_tgt}v"
    if args.view_selection_num != -1:
        out_name = f"{out_name}_n{int(args.view_selection_num)}"
    out_path = os.path.join(out_dir, f"{out_name}.json")

    with open(out_path, "w") as f:
        json.dump(out_dict_all, f)

    print(f"Save index to {out_path}.")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="root dir of the test data")
    parser.add_argument("--type", type=str, help="dataset type, such as re10k/dl3dv")
    parser.add_argument(
        "--num_target_views", type=int, default=-1, help="num of target views"
    )
    parser.add_argument(
        "--num_context_views", type=int, default=5, help="num of context views"
    )
    parser.add_argument(
        "--view_selection_num",
        type=int,
        default=-1,
        help="test ratio; set to 150 for N=150",
    )

    args = parser.parse_args()

    generate_index_file(args)
    print("max_gap:", max_gap)



"""
python -m src.scripts.generate_evaluation_dl3dv_index --data_dir=datasets/dl3dv/test --num_context_views=1 --num_target_views=2 --view_selection_num=4 --type dl3dv


python -m src.scripts.generate_evaluation_dl3dv_index --data_dir=datasets/dl3dv/test --num_context_views=1 --num_target_views=2 --view_selection_num=8 --type dl3dv


python -m src.scripts.generate_evaluation_dl3dv_index --data_dir=datasets/dl3dv/test --num_context_views=1 --num_target_views=2 --view_selection_num=10 --type dl3dv
"""