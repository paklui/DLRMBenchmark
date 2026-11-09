RANK="$1"
NODE_IP="$2"
MASTER_IP="$3"
MASTER_PORT="$4"
WORLD_SIZE="$5"
BATCH_SIZE="$6"
CWD="$7"
CONTAINER_NAME="$8"
LOG_FILE="$9"

# Reset container (preserves filesystem state -- iopath fix etc. survive stop/start, unlike rm+run)
sudo docker stop "$CONTAINER_NAME"
sudo docker start "$CONTAINER_NAME"

# Run DLRM eval
sudo docker exec -i "$CONTAINER_NAME" bash -c "
source /opt/venv/bin/activate
cd $CWD
./DLRM_local_launch_gfx1250_smoke2gpu.sh \"$RANK\" \"$NODE_IP\" \"$MASTER_IP\" \"$MASTER_PORT\" \"$WORLD_SIZE\"  \"$BATCH_SIZE\"
if [[ \"$MASTER_IP\" == \"$NODE_IP\" ]]; then
    python ./utils/process_training_log.py --world_size $WORLD_SIZE --batch_size $BATCH_SIZE --log_file $LOG_FILE
fi
"
