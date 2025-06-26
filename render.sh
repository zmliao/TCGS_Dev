export CUDA_VISIBLE_DEVICES=1

DATA=/nas/shared/pjlab_lingjun_landmarks/yumulin_group/nerf_data/tandt
SCENE=truck
# CKPT=/cpfs04/user/liaozimu/ckpt/models/

# nsys profile \
#     --stats=true \
#     --force-overwrite=true \
#     --trace=cuda,nvtx,cublas,cudnn,osrt \
#     --cuda-memory-usage=true \
#     -o baseline-report \
python render.py \
    -s ${DATA}/${SCENE}/ \
    -m ./output/${SCENE} \
    --eval \

        # --skip_train