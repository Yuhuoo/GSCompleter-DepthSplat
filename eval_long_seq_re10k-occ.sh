
CUDA_VISIBLE_DEVICES=1 python -m src.infer +experiment=re10k \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/long_seq/evaluation_index_fixed/re10k/re10k_ctx_0_20_tgt_every_20.json \
dataset.test_chunk_interval=1 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth \
output_dir=outputs/1.16/long_seq/re10k_ctx_0_20_tgt_every_20-occ \
test.using_refinement=true \
test.save_image=true \
test.save_gt_image=true \
test.render_context=true \
test.using_stereo_anchor=true \
test.using_depth_alignment=true \
test.using_rc_register=true \
test.using_mv_refinement=true \