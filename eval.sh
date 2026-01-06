# Table 7 of depthsplat paper
CUDA_VISIBLE_DEVICES=0 python -m src.infer +experiment=dl3dv \
mode=test \
dataset.roots=[datasets/dl3dv] \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=6 \
dataset.view_sampler.index_path=assets/dl3dv_ood/dl3dv_ctx_1v_tgt_2v_n10.json \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
output_dir=new_experiments/1.6/pe-field-4/ours-test-time/dl3dv_ctx_1v_tgt_2v_n10 \
test.using_refinement=true \
test.save_image=true \
test.save_gt_image=true \
test.render_context=true \
test.using_stereo_anchor=true \
test.using_depth_alignment=true \
test.using_rc_register=true \
test.using_mv_refinement=true \
# test.save_gaussian=true \
# test.save_gt_image=true \
# test.save_gaussian=true \
# test.save_input_images=true \
# test.save_gaussian=true
# test.using_refinement=true \