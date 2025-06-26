export CUDA_VISIBLE_DEVICES=1

DATA=/nas/shared/pjlab_lingjun_landmarks/yumulin_group/nerf_data/tandt
SCENE=truck
# CKPT=/cpfs04/user/liaozimu/ckpt/models/

nsys profile \
    --stats=true \
    --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
    --cuda-memory-usage=true \
    -o baseline-report \
    python train_dash.py \
        -s ${DATA}/${SCENE}/ \
        -m ./output/${SCENE} \
        --optimizer_type sparse_adam \
        --densify_mode freq --resolution_mode freq --densify_until_iter 27000 \
        --disable_viewer \
        --eval \
        --max_n_gaussian 199681\
        --iteration 30000
        # --skip_train