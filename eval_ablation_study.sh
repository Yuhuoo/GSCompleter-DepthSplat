#!/bin/bash

# 1. 基础公共参数（不随实验改变的部分）
COMMON_ARGS="+experiment=dl3dv mode=test \
    dataset.roots=[datasets/dl3dv] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=6 \
    dataset.view_sampler.index_path=assets/dl3dv_ood/dl3dv_ctx_1v_tgt_2v_n10.json \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 \
    model.encoder.monodepth_vit_type=vitb \
    checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
    test.using_refinement=true \
    test.save_image=true \
    test.save_gt_image=true \
    test.render_context=true"

# 基础输出路径
BASE_OUT="new_experiments/1.6/pe-field-4"
SUB_DIR="dl3dv_ctx_1v_tgt_2v_n10"

# 2. 定义实验配置数组
# 格式: "GPU_ID | 实验名 | using_stereo_anchor | using_depth_alignment | using_rc_register | using_mv_refinement"
experiments=(
    "0 | ours-all         | true  | true  | true  | true"
    "0 | ours-wo-sa       | false | true  | true  | true"
    "0 | ours-wo-da       | true  | false | true  | true"
    "1 | ours-wo-da-wo-rc | true  | false | false | true"
    "1 | ours-wo-rc       | true  | true  | false | true"
    "1 | ours-wo-mv       | true  | true  | true  | false"
)

echo "开始并行实验任务..."

# 3. 循环遍历并启动后台进程
for exp in "${experiments[@]}"; do
    # 解析数组行（使用 IFS 分隔符）
    IFS='|' read -r gpu name sa da rc mv <<< "$exp"
    
    # 去除首尾空格
    gpu=$(echo $gpu | xargs)
    name=$(echo $name | xargs)
    sa=$(echo $sa | xargs)
    da=$(echo $da | xargs)
    rc=$(echo $rc | xargs)
    mv=$(echo $mv | xargs)

    echo "[GPU $gpu] 启动实验: $name"

    # 执行命令
    CUDA_VISIBLE_DEVICES=$gpu python -m src.infer $COMMON_ARGS \
        output_dir=$BASE_OUT/$name/$SUB_DIR \
        test.using_stereo_anchor=$sa \
        test.using_depth_alignment=$da \
        test.using_rc_register=$rc \
        test.using_mv_refinement=$mv &
done

# 4. 等待所有任务完成
wait
echo "所有实验已并行运行完毕。"