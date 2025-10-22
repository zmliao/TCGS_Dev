export CUDA_VISIBLE_DEVICES=0

DATA=/mnt/shared-storage-user/ailab-sys/liaozimu/data/tandt
SCENE=truck

python train_dash.py \
    -s ${DATA}/${SCENE}/ \
    -m ./output/${SCENE} \
    --optimizer_type sparse_adam \
    --densify_mode freq --resolution_mode freq --densify_until_iter 27000 \
    --disable_viewer \
    --eval \
    --max_n_gaussian 1996811\
    --iteration 30000