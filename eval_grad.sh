DATA=/nas/shared/pjlab_lingjun_landmarks/yumulin_group/nerf_data/mipnerf360/
SCENE=flowers
CKPT=./output

export CUDA_VISIBLE_DEVICES=0
# nsys profile \
#     --stats=true \
#     --force-overwrite=true \
#     --trace=cuda,nvtx,cublas,cudnn,osrt \
#     --cuda-memory-usage=true \
#     -o eval_report \
python render.py \
    -s ${DATA}/${SCENE}/ \
    -m ${CKPT}/${SCENE}/ \
    --iteration 114514 \
    --white_background \
    --skip_train