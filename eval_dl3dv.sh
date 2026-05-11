# # Table 7 of depthsplat paper
# CUDA_VISIBLE_DEVICES=5 python -m src.infer +experiment=dl3dv \
# mode=test \
# dataset.roots=[datasets/dl3dv] \
# dataset/view_sampler=evaluation \
# dataset.view_sampler.num_context_views=6 \
# dataset.view_sampler.index_path=assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n30.json \
# model.encoder.num_scales=2 \
# model.encoder.upsample_factor=4 \
# model.encoder.lowest_feature_resolution=8 \
# model.encoder.monodepth_vit_type=vitb \
# checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
# output_dir=outputs/1.10/evaluation_index_extrapolation/dl3dv_ctx2_tgt1_n30 \
# test.save_image=true \
# test.save_gt_image=true \


CUDA_VISIBLE_DEVICES=0 python -m src.infer +experiment=dl3dv \
mode=test \
dataset.roots=[datasets/dl3dv] \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
output_dir=outputs/4.10/test-video/dl3dv_ctx2_tgt1_n10_w_ours \
test.save_virtual_video=true \
test.using_refinement=true \
test.save_image=true \
test.save_gt_image=true \
test.render_context=true \
test.using_stereo_anchor=true \
test.using_depth_alignment=true \
test.using_rc_register=true \
test.using_mv_refinement=true \
test.using_target_align=false \
# test.save_gaussian=true \
# test.save_gt_image=true \
# test.save_gaussian=true \
# test.save_input_images=true \
# test.save_gaussian=true
# test.using_refinement=true \