#!/bin/bash

PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints

TRAIN_DIR=/tmp/custom-models/inception_v3

# DATASET_DIR=/tmp/flowers

read -p "Enter path to your dataset : " dataset_dir 

echo "Your path to dataset is $dataset_dir"

all_images=($(ls -lR $dataset_dir/*/*.jpg | wc -l))

read -p "Specify train percentage (reminder test) : " train_percentage

train_data=$(python3 -c "print(int($all_images*$train_percentage/100))")

test_data=$((all_images-train_data))

echo "Total $all_images images $train_data train $test_data test."

read -p "How many category : " num_category

echo "Which model you want to train?
1.Inception V3
2.Inception V4
3.ResNet V1 152
4.ResNet V2 152
5.Inception-ResNet-v2
6.VGG 19
7.MobileNet v1 1.0_224"

read -p "Enter a number : " train_model 

model_url=""

model_file_name=""

zip_file_name=""

model_name=""

trainable_scopes=""

echo "train model $train_model"

if [ $train_model -eq 1 ]; then
	model_url="http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
	model_file_name="inception_v3.ckpt"
	zip_file_name="inception_v3_2016_08_28.tar.gz"
	model_name="inception_v3"
	trainable_scopes="InceptionV3/Logits,InceptionV3/AuxLogits"
elif [ $train_model -eq 2 ]; then
	model_url="http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
	model_file_name="inception_v4.ckpt"
	zip_file_name="inception_v4_2016_09_09.tar.gz"
	model_name="inception_v4"
	trainable_scopes="InceptionV4/Logits,InceptionV4/AuxLogits"
elif [ $train_model -eq 3 ]; then
	model_url="http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz"
	model_file_name="resnet_v1_152.ckpt"
	zip_file_name="resnet_v1_152_2016_08_28.tar.gz"
	model_name="resnet_v1_152"
	trainable_scopes="resnet_v1_152/logits"
fi

echo "model $model_file_name link : $model_url train scope $trainable_scopes zip files name $zip_file_name"

read -p "Please specify number of step : " number_of_steps

read -p "Please specify batch size : " batch_size

echo "Number of steps : $number_of_steps , batch size : $batch_size"
	
#if false; then
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/${model_file_name} ]; then
  wget ${model_url}
  tar -xvf ${zip_file_name}
  mv ${model_file_name} ${PRETRAINED_CHECKPOINT_DIR}/${model_file_name}
  rm ${zip_file_name}
fi

python download_and_convert_data.py \
	--dataset_name=custom \
	--dataset_dir=${dataset_dir} \
	--num_validation=${test_data} \
	--num_shards=${num_category} 


python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=custom \
  --dataset_split_name=train \
  --dataset_dir=${dataset_dir} \
  --train_num=${train_data} \
  --validation_num=${test_data} \
  --num_of_classes=${num_category} \
  --model_name=${model_name}\
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${model_file_name} \
  --checkpoint_exclude_scopes=${trainable_scopes} \
  --trainable_scopes=${trainable_scopes} \
  --max_number_of_steps=${number_of_steps}\
  --batch_size=${batch_size} \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=custom
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${model_name}
#fi
