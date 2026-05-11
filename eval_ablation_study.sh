#!/bin/bash

# 1. 基础公共参数
COMMON_ARGS="+experiment=dl3dv mode=test \
    dataset.roots=[datasets/dl3dv] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=6 \
    dataset.view_sampler.index_path=assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 \
    model.encoder.monodepth_vit_type=vitb \
    checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
    test.save_virtual_video=false \
    test.save_image=true \
    test.save_gt_image=true \
    test.render_context=true \
    test.save_colmap=false"

# 基础输出路径
BASE_OUT="outputs/5.11/visible_seams_addressing"
SUB_DIR="dl3dv_ctx2_tgt1_n10_wo_dilation"

# 2. 定义实验配置数组
# 格式: "GPU_ID | 实验名 | using_refinement | using_stereo_anchor | using_depth_alignment | using_rc_register | using_mv_refinement | using_target_align | angle_deg"
experiments=(
    # "0 | baseline         | false | false | false | false | false | false"
    # "1 | ours-all         | true  | true  | true  | true  | true  | false"
    # "2 | ours-wo-sa       | true  | false | true  | true  | true  | false"
    # "3 | ours-wo-da       | true  | true  | false | true  | true  | false"
    # "4 | ours-wo-da-wo-rc | true  | true  | false | false | true  | false"
    # "5 | ours-wo-rc       | true  | true  | true  | false | true  | false"
    # "6 | ours-wo-mv       | true  | true  | true  | true  | false | false"
    # "7 | ours-wo-all       | true  | false  | false | false | false  | false"
    # "7 | ours-tgt-align   | true  | true  | true  | true  | true | true"
    # "0 | ours-30   | true  | true  | true  | true  | true  | false | 30.0"
    "0 | ours-45   | true  | true  | true  | true  | true  | false | 45.0"
    # "0 | ours-90   | true  | true  | true  | true  | true  | false | 90.0"
)

echo "开始并行实验任务..."

# 3. 循环遍历并启动后台进程
for exp in "${experiments[@]}"; do
    # 解析数组行（使用 IFS 分隔符）
    IFS='|' read -r gpu name re sa da rc mv ta ad <<< "$exp"
    
    # 去除首尾空格
    gpu=$(echo $gpu | xargs)
    name=$(echo $name | xargs)
    re=$(echo $re | xargs)
    sa=$(echo $sa | xargs)
    da=$(echo $da | xargs)
    rc=$(echo $rc | xargs)
    mv=$(echo $mv | xargs)
    ta=$(echo $ta | xargs)
    ad=$(echo $ad | xargs)

    echo "[GPU $gpu] 启动实验: $name"

    # 执行命令
    CUDA_VISIBLE_DEVICES=$gpu python -m src.infer $COMMON_ARGS \
        output_dir=$BASE_OUT/$name/$SUB_DIR \
        test.using_refinement=$re \
        test.using_stereo_anchor=$sa \
        test.using_depth_alignment=$da \
        test.using_rc_register=$rc \
        test.using_mv_refinement=$mv \
        test.using_target_align=$ta \
        test.angle_deg=$ad \
        hydra/job_logging=stdout \
        hydra.output_subdir=null &
done

# 4. 等待所有任务完成
wait
echo "所有实验已并行运行完毕。"