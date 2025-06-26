DATA=/nas/shared/pjlab_lingjun_landmarks/yumulin_group/nerf_data/mipnerf360/
SCENE=flowers
CKPT=/cpfs04/user/liaozimu/ckpt/models/

export CUDA_VISIBLE_DEVICES=0
# nsys profile \
#     --stats=true \
#     --force-overwrite=true \
#     --trace=cuda,nvtx,cublas,cudnn,osrt \
#     --cuda-memory-usage=true \
#     -o eval_report \
python render_back.py \
    -s ${DATA}/${SCENE}/ \
    -m ${CKPT}/${SCENE}/ \
    --iteration 1919810 \
    --white_background \
    --skip_train