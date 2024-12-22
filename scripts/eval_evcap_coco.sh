SHELL_FOLDER=/workspace/EVCap/scripts

cd $SHELL_FOLDER/..
EXP_NAME=$1
DEVICE=$2
COCO_OUT_PATH=results/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER

COCO_LOG_FILE="$LOG_FOLDER/COCO_${TIME_START}.log"

python -u eval_evcap.py \
--device cuda:$DEVICE \
--name_of_datasets coco \
--path_of_val_datasets dataset_coco_test_split.json \
--image_folder annotations/coco/val2014/ \
--out_path=$COCO_OUT_PATH \
|& tee -a  ${COCO_LOG_FILE}

echo "==========================COCO EVAL================================"
python evaluation/cocoeval.py --result_file_path $COCO_OUT_PATH/coco*.json |& tee -a  ${COCO_LOG_FILE}