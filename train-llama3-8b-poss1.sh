saveckpt=checkpoints/llama3-8b-instruct/poss-1
mkdir -p $saveckpt

# Please follow EAGLE to generate training data, and fill the path below
prep=/PATH/to/training-data/llama3-8b/sharegpt_0_67999_mufp16
BasePath=/PATH/to/your/model-cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a
restore_from=/PATH/to/eagle/e.g./neurips24645/llama3-8b-instruct-eagle-reproduce

# Layers = forward_num_total / position_per_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m --mixed_precision=bf16 train.main_poss \
    --basepath $BasePath \
    --tmpdir $prep \
    --cpdir $saveckpt \
    --ckpt_path $restore_from \
    --configpath train/EAGLE-LLaMA3-Instruct-8B \
    --epoch 20 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --lr 3e-5 \
    --forward_num_total 6 \
    --position_per_layer 1
