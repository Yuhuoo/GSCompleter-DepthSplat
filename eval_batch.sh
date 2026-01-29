#!/bin/bash

# 定义数组
index_paths=(
    # eval-extrapolation
    "assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json"
    "assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n20.json"
    "assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n30.json"
    "assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n40.json"
)

output_dirs=(
    # eval-extrapolation
    "outputs/1.23-1/batch-eval/dl3dv_ctx2_tgt1_n10"
    "outputs/1.23-1/batch-eval/dl3dv_ctx2_tgt1_n20"
    "outputs/1.23-1/batch-eval/dl3dv_ctx2_tgt1_n30"
    "outputs/1.23-1/batch-eval/dl3dv_ctx2_tgt1_n40"
)

# gpus=(0)
# gpus=(0 1 2 3 4 5 6 7)
gpus=(0 1)

# 确保数组长度匹配
if [ ${#index_paths[@]} -ne ${#output_dirs[@]} ]; then
    echo "数组长度不匹配!"
    exit 1
fi

# 获取GPU总数
num_gpus=${#gpus[@]}

# 并行执行所有任务
echo "启动 ${#index_paths[@]} 个任务，使用 $num_gpus 个 GPU 循环分配..."

for i in "${!index_paths[@]}"; do
    # --- 修改开始: 循环计算 GPU 索引 ---
    gpu_idx=$((i % num_gpus))
    gpu=${gpus[$gpu_idx]}
    # --- 修改结束 ---

    index_path="${index_paths[$i]}"
    output_dir="${output_dirs[$i]}"
    
    echo "任务 $((i+1))/${#index_paths[@]}: 分配到 GPU $gpu"
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
        test.using_refinement=true \
        test.save_image=true \
        test.save_gt_image=true \
        test.render_context=true \
        test.using_stereo_anchor=true \
        test.using_depth_alignment=true \
        test.using_rc_register=true \
        test.using_mv_refinement=true \
        test.using_target_align=false \
        hydra/job_logging=stdout \
        hydra.output_subdir=null &
        # test.using_refinement=true \
    pids[$i]=$!
    
    # 可选: 稍微等待几秒，防止并发启动时的冲突
    sleep 1
done

# 等待所有任务完成
echo "等待所有任务完成..."
for pid in ${pids[@]}; do
    wait $pid
done

echo "所有任务完成!"