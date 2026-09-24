
# acid_ctx2_tgt1_n30
CUDA_VISIBLE_DEVICES=0 python -m src.infer +experiment=re10k \
mode=test \
dataset.roots=[datasets/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_extrapolation/acid/acid_ctx2_tgt1_n30.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
output_dir=outputs/1.25-1.24/evaluation_index_extrapolation/acid_ctx2_tgt1_n30 \
test.save_image=true \
test.save_gt_image=true \
test.render_context=true \


# acid_ctx2_tgt1_n30_w_ours
CUDA_VISIBLE_DEVICES=0 python -m src.infer +experiment=re10k \
mode=test \
dataset.roots=[datasets/acid] \
dataset.view_sampler.index_path=assets/evaluation_index_extrapolation/acid/acid_ctx2_tgt1_n30.json \
dataset/view_sampler=evaluation \
dataset.view_sampler.num_context_views=2 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitl \
checkpointing.pretrained_model=pretrained/depthsplat-gs-large-re10k-256x256-view2-e0f0f27a.pth \
output_dir=outputs/1.25-1.24/evaluation_index_extrapolation/acid_ctx2_tgt1_n30_w_ours \
test.using_refinement=true \
test.using_gscompleter=true \
test.save_image=true \
test.save_gt_image=true \
test.render_context=true \
test.using_stereo_anchor=true \
test.using_depth_alignment=true \
test.using_rc_register=true \
test.using_mv_refinement=true \
