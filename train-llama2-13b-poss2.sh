saveckpt=checkpoints/llama2-13b-chat/poss-2
mkdir -p $saveckpt

# Please follow EAGLE to generate training data, and fill the path below
prep=/PATH/to/training-data/llama2-13b/sharegpt_0_67999_mufp16
BasePath=/PATH/to/your/model-cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8
restore_from=/PATH/to/eagle/e.g./yuhuili/EAGLE-llama2-chat-13B

# Layers = forward_num_total / position_per_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m --mixed_precision=bf16 train.main_poss \
    --basepath $BasePath \
    --tmpdir $prep \
    --cpdir $saveckpt \
    --ckpt_path $restore_from \
    --configpath train/llama_2_chat_13B_config.json \
    --epoch 20 \
    --bs 2 \
    --topk 10 \
    --topk_w 0 \
    --lr 3e-5 \
    --forward_num_total 6 \
    --position_per_layer 2
