#!/bin/bash
echo ">>>M3-RKD Training Distillation on OGB Dataset molbace..."
    
#Dataset to run experiments on
DATASET_NAME="ogbg-molbace"  # Change this to your desired dataset (e.g., "ogbg-molbace")

# Teacher configuration (make sure this matches the teacher model you trained)
TEACHER_MODEL="GIN"
TEA_LAYERS=
TEA_HIDDEN_DIM=128
TEACHER_PATH="weight\ogbg-molbace_best_model_run4.pth"

# Student configuration
STUDENT_MODEL="MLP"
STU_LAYERS=3
STU_HIDDEN_DIM=128

# Common training arguments
BATCH_SIZE=32
EPOCHS=1000
LEARNING_RATE=0.001
declare -a weight_grid=(
    "1.0 1.0 0.01 0.1 1 0.1 0.1  2.0 8"
)

#declare -a weight_grid=(
#     "1.0 1.0 1.0 1.0 1 0.1 0.1 2.0 8"
#     "1.0 1.0 1.0 0.1 1 0.1 0.1  2.0 8"
#     "1.0 1.0 1.0 0.01 1 0.1 0.1 2.0 8"
#     "1.0 1.0 0.1 1.0 1 0.1 0.1 2.0 8"
#     "1.0 1.0 0.1 0.1 1 0.1 0.1 2.0 8"
#     "1.0 1.0 0.1 0.01 1 0.1 0.1 2.0 8"
#     "1.0 1.0 0.01 1.0 1 0.1 0.1 2.0 8"
#     "1.0 1.0 0.01 0.1 1 0.1 0.1 2.0 8"
#     "1.0 1.0 0.01 0.01 1 0.1 0.1 2.0 8"
#     "1.0 0.1 1.0 1.0 1 0.1 0.1 2.0 8"
#     "1.0 0.1 1.0 0.1 1 0.1 0.1 2.0 8"
#     "1.0 0.1 1.0 0.01 1 0.1 0.1 2.0 8"
#     "1.0 0.1 0.1 1.0 1 0.1 0.1 2.0 8"
#     "1.0 0.1 0.1 0.1 1 0.1 0.1 2.0 8"
#     "1.0 0.1 0.1 0.01 1 0.1 0.1 2.0 8"
#     "1.0 0.1 0.01 1.0 1 0.1 0.1 2.0 8"
#     "1.0 0.1 0.01 0.1 1 0.1 0.1 2.0 8"
#     "1.0 0.1 0.01 0.01 1 0.1 0.1 2.0 8"
#     "1.0 0.01 1.0 1.0 1 0.1 0.1 2.0 8"
#     "1.0 0.01 1.0 0.1 1 0.1 0.1 2.0 8"
#     "1.0 0.01 1.0 0.01 1 0.1 0.1 2.0 8"
#     "1.0 0.01 0.1 1.0 1 0.1 0.1 2.0 8"
#     "1.0 0.01 0.1 0.1 1 0.1 0.1 2.0 8"
#     "1.0 0.01 0.1 0.01 1 0.1 0.1 2.0 8"
#     "1.0 0.01 0.01 1.0 1 0.1 0.1 2.0 8"
#     "1.0 0.01 0.01 0.1 1 0.1 0.1 2.0 8"
#     "1.0 0.01 0.01 0.01 1 0.1 0.1 2.0 8"
# )

echo "Starting Knowledge Distillation experiments on $DATASET_NAME..."
echo "Teacher: $TEACHER_MODEL (Layers: $TEA_LAYERS, Hidden: $TEA_HIDDEN_DIM)"
echo "Student: $STUDENT_MODEL (Layers: $STU_LAYERS, Hidden: $STU_HIDDEN_DIM)"
echo "==============================================================="

for weights in "${weight_grid[@]}"; do
    # Read weights
    # read -r kl core sub topk <<< "$weights"
    read -r kl core sub topk kl_t sub_t topk_t core_t k<<< "$weights"
    #read -r kl_t <<< "$weights"
    echo "Running Experiment - KL: $kl, Core: $core, SubG: $sub, TopK: $topk, KL_T: $kl_t, SubG_T: $sub_t, TopK_T: $topk_t, Core_T: $core_t, k: $k"
    #echo "Running Experiment - KL_T: $kl_t"
    
    /home/ubuntu/anaconda3/envs/Graph/bin/python train_ogbdatasets_kd.py \
        --dataset_name $DATASET_NAME \
        --teacher_model $TEACHER_MODEL \
        --tea_layers $TEA_LAYERS \
        --tea_hidden_dim $TEA_HIDDEN_DIM \
        --teacher_path $TEACHER_PATH \
        --student_model $STUDENT_MODEL \
        --stu_layers $STU_LAYERS \
        --stu_hidden_dim $STU_HIDDEN_DIM \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --kl_weight $kl \
        --weight_decay 0.0005 \
        --patience 50 \
        --core_nce_weight $core \
        --subgraph_weight $sub \
        --topk_weight $topk \
        --kl_temperature $kl_t \
        --sub_temperature $sub_t \
        --rank_temperature $topk_t \
        --core_temperature $core_t \
        --k $k \
        --n_clusters 5 \
        --device 1 \
        --add_core \
        --add_spectral
        
    echo "Experiment finished. Moving to next..."
    echo "---------------------------------------------------------------"
done
        # --kl_weight $kl \
        # --core_nce_weight $core \
        # --subgraph_weight $sub \
        # --topk_weight $topk \
echo "All experiments completed!"
