export DATASET_PATH=${HOME}/dlrm_dataset/numpy_contiguous_shuffled_output_dataset_dir/
export NUM_EPOCHS=1
export NUM_TRAIN_BATCHES=2000
export TF32_MODE=1 # 1: TF32, 0: FP32
export TF32_TUNING_MODE=0 # 1: Tuning, 0: Normal
export TF32_TUNABLE_OPS_DIR=${TUNABLE_OPS_DIR:-${HOME}/dlrm_libs/tunable_ops}
export TRAINING_SCRIPT=dlrm_main.py # Train: dlrm_main.py, Profile: dlrm_main_profile.py

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 <RANK> <NODE_IP> <MASTER_IP> <MASTER_PORT> <WORLD_SIZE> <BATCH_SIZE>"
  exit 1
fi

RANK="$1"
NODE_IP="$2"
MASTER_IP="$3"
MASTER_PORT="$4"
WORLD_SIZE="$5"
BATCH_SIZE="$6"

echo "Rank: $RANK"
echo "Node IP: $NODE_IP"
echo "Master IP: $MASTER_IP"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Batch Size: $BATCH_SIZE"

### Standard ENV_VARS (Miami/Cairo gfx1250 variant -- 4 GPU, AINIC ionic) ###
source /root/DLRM_optimization/utils/DLRM_set_env_variables_gfx1250.sh

### Local ENV_VARS ###
WORKDIR=./training/recommendation_v2/torchrec_dlrm
export PATH=$PATH:$WORKDIR
export PYTHONPATH=$PYTHONPATH:$WORKDIR

# NOTE: nproc_per_node=4 and WORLD_SIZE/4 (was 8/8 on the old GT cluster) -- Miami/Cairo
# compute trays are 4x gfx1250 GPUs each, not 8.
if [[ $TF32_MODE -eq 1 ]]; then

  if [[ $TF32_TUNING_MODE -eq 1 ]]; then
    echo "TF32 Tuning Mode"
    chmod -R 777 ${TF32_TUNABLE_OPS_DIR}
    export NUM_TRAIN_BATCHES=1
  fi

  export HIPBLASLT_ALLOW_TF32=1
  export PYTORCH_TUNABLEOP_ENABLED=1
  export PYTORCH_TUNABLEOP_VERBOSE=1
  export PYTORCH_TUNABLEOP_FILENAME=${TF32_TUNABLE_OPS_DIR}/tunable_ops_result.csv

  echo "TF32 Training Mode"
  torchrun --nnodes $(($WORLD_SIZE / 4)) --node_rank=$RANK --nproc_per_node 4 --master_addr=$MASTER_IP --master_port=$MASTER_PORT --role trainer ${WORKDIR}/${TRAINING_SCRIPT} -- \
  --embedding_dim 128 \
  --dense_arch_layer_sizes 512,256,128 \
  --in_memory_binary_criteo_path $DATASET_PATH \
  --over_arch_layer_sizes 1024,1024,512,256,1 \
  --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
  --allow_tf32 \
  --epochs $NUM_EPOCHS \
  --pin_memory \
  --mmap_mode \
  --batch_size $BATCH_SIZE \
  --interaction_type=dcn \
  --dcn_num_layers=3 \
  --dcn_low_rank_dim=512 \
  --adagrad \
  --limit_train_batches $NUM_TRAIN_BATCHES \
  --limit_val_batches 8926 \
  --limit_test_batches 8926 \
  --learning_rate 0.005 \
  --print_progress \
  --multi_hot_distribution_type uniform \
  --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1

else

  echo "FP32 Training Mode"
  export TORCH_BLAS_PREFER_HIPBLASLT=0
  torchrun --nnodes $(($WORLD_SIZE / 4)) --node_rank=$RANK --nproc_per_node 4 --master_addr=$MASTER_IP --master_port=$MASTER_PORT --role trainer ${WORKDIR}/${TRAINING_SCRIPT} -- \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --in_memory_binary_criteo_path $DATASET_PATH \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --epochs $NUM_EPOCHS \
    --pin_memory \
    --mmap_mode \
    --batch_size $BATCH_SIZE \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --limit_train_batches $NUM_TRAIN_BATCHES \
    --limit_val_batches 8926 \
    --limit_test_batches 8926 \
    --learning_rate 0.005 \
    --print_progress \
    --multi_hot_distribution_type uniform \
    --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1

fi
