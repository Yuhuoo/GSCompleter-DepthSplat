#!/bin/bash

# 定义数组
index_paths=(

    # ood
    "assets/dl3dv_ood/dl3dv_ctx_1v_tgt_2v_n10.json"
    "assets/dl3dv_ood/dl3dv_ctx_1v_tgt_2v_n20.json"
    "assets/dl3dv_ood/dl3dv_ctx_1v_tgt_2v_n30.json"
    "assets/dl3dv_ood/dl3dv_ctx_1v_tgt_2v_n40.json"


    # # old
    # "assets/dl3dv_start_0_distance_10_ctx_2v_tgt_4v.json"
    # "assets/dl3dv_start_0_distance_50_ctx_2v_video_0_50.json"
    # "assets/dl3dv_start_0_distance_50_ctx_4v_video_0_50.json"
    # "assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_2v_tgt_4v_n100.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_2v_tgt_4v_n150.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_2v_tgt_4v_n200.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_3v_tgt_56v.json"

    # # MVSplat360
    # "assets/dl3dv_evaluation/dl3dv_ctx_4v_tgt_56v.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_5v_tgt_56v.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_5v_tgt_56v_n150.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_6v_tgt_56v.json"
    # "assets/dl3dv_evaluation/dl3dv_ctx_7v_tgt_56v.json"
)

output_dirs=(

    # ood
    "new_experiments/12.20/hard-hole-mask-expand_1/dl3dv_ctx_1v_tgt_2v_n10"
    "new_experiments/12.20/hard-hole-mask-expand_1/dl3dv_ctx_1v_tgt_2v_n20"
    "new_experiments/12.20/hard-hole-mask-expand_1/dl3dv_ctx_1v_tgt_2v_n30"
    "new_experiments/12.20/hard-hole-mask-expand_1/dl3dv_ctx_1v_tgt_2v_n40"

    # # old
    # "experiments/12.15/ours/dl3dv_start_0_distance_10_ctx_2v_tgt_4v"
    # "experiments/12.15/ours/dl3dv_start_0_distance_50_ctx_2v_video_0_50"
    # "experiments/12.15/ours/dl3dv_start_0_distance_50_ctx_4v_video_0_50"
    # "experiments/12.15/ours/dl3dv_start_0_distance_50_ctx_6v_video_0_50"
    # "experiments/12.15/ours/dl3dv_ctx_2v_tgt_4v_n100"
    # "experiments/12.15/ours/dl3dv_ctx_2v_tgt_4v_n150"
    # "experiments/12.15/ours/dl3dv_ctx_2v_tgt_4v_n200"
    # "experiments/12.15/ours/mvsplat360/dl3dv_ctx_3v_tgt_56v"

    # # MVSplat360
    # "experiments/12.11/ours/mvsplat360/dl3dv_ctx_4v_tgt_56v"
    # "experiments/12.10/ours/mvsplat360/dl3dv_ctx_5v_tgt_56v"
    # "experiments/12.10/ours/mvsplat360/dl3dv_ctx_5v_tgt_56v_n150"
    # "experiments/12.10/ours/mvsplat360/dl3dv_ctx_6v_tgt_56v"
    # "experiments/12.10/ours/mvsplat360/dl3dv_ctx_7v_tgt_56v"
)

# gpus=(0)
# gpus=(0 1 2 3 4 5 6 7)
gpus=(0 1 2 3)

# 确保数组长度匹配
if [ ${#index_paths[@]} -ne ${#output_dirs[@]} ]; then
    echo "数组长度不匹配!"
    exit 1
fi

# 并行执行所有任务
echo "启动 ${#index_paths[@]} 个任务..."
for i in "${!index_paths[@]}"; do
    gpu=${gpus[$i]}
    index_path="${index_paths[$i]}"
    output_dir="${output_dirs[$i]}"
    
    echo "任务 $((i+1))/${#index_paths[@]}: GPU $gpu"
    echo "  index_path: $index_path"
    echo "  output_dir: $output_dir"
    
    CUDA_VISIBLE_DEVICES=$gpu python -m src.infer \
        +experiment=dl3dv \
        dataset/view_sampler=evaluation \
        dataset.view_sampler.num_context_views=6 \
        dataset.view_sampler.index_path="$index_path" \
        model.encoder.num_scales=2 \
        model.encoder.upsample_factor=4 \
        model.encoder.lowest_feature_resolution=8 \
        model.encoder.monodepth_vit_type=vitb \
        checkpointing.pretrained_model="pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth" \
        output_dir="$output_dir" \
        test.save_image=true \
        test.save_gt_image=true \
        test.using_refinement=true &
    
    pids[$i]=$!
done

# 等待所有任务完成
echo "等待所有任务完成..."
for pid in ${pids[@]}; do
    wait $pid
done

echo "所有任务完成!"