#!/bin/bash

# NOTE: you need to first run the hyperparameter tuning script, and then set the "BASE_HP_PATH" variable to run the corresponding training runs.
# If you're using ray, the path would look similar to, e.g., '~/ray_results/tune_hyperparameters_2024-04-18_19-20-06'

############################################
# EXP1: ResNet-18 on CIFAR-10
############################################

TUNING_SEEDS=( 31190 77678 71333 17094 48490 79157 62135 82014 76133 8588 79377 1725 61503 33994 72192 30714 25132 93998 61638 70906 )
python main.py --description "res18-c10-tune-april" --experiment-type 'tuning' --batch-size 128 --num-epochs 30 --num-trials 10 --model 'resnet18' --dataset 'cifar10' --seed-list "${TUNING_SEEDS[@]}";

BASE_HP_PATH='SET_THIS_TO_THE_PATH_OF_THE_HP_TUNE_RESULTS'
TRAINING_SEEDS=( 68417 54285 39770 47923 37961 22896 92424 92303 )

# baseline
python main.py --description "res18-c10-train-april-baseline" --experiment-type 'training' --batch-size 128 --weight-initialization 'random' --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

# hp-final
python main.py --description "res18-c10-train-april-hp-final" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-final' --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

python main.py --description "res18-c10-train-april-hp-epoch-2" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 2 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c10-train-april-hp-epoch-5" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 5 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c10-train-april-hp-epoch-10" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 10 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c10-train-april-hp-epoch-15" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 15 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c10-train-april-hp-epoch-20" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 20 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c10-train-april-hp-epoch-25" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 25 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar10' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

############################################
# EXP2: ResNet-18 on CIFAR-100
############################################

TUNING_SEEDS=( 31190 77678 71333 17094 48490 79157 62135 82014 76133 8588 79377 1725 61503 33994 72192 30714 25132 93998 61638 70906 )
python main.py --description "res18-c100-tune-april-256-redo" --experiment-type 'tuning' --batch-size 256 --num-epochs 40 --num-trials 10 --model 'resnet18' --dataset 'cifar100' --seed-list "${TUNING_SEEDS[@]}";

BASE_HP_PATH='SET_THIS_TO_THE_PATH_OF_THE_HP_TUNE_RESULTS'
TRAINING_SEEDS=( 68417 54285 39770 47923 37961 22896 92424 92303 )

# baseline
python main.py --description "res18-c100-train-april-baseline-256-redo" --experiment-type 'training' --batch-size 256 --weight-initialization 'random' --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

# hp-final
python main.py --description "res18-c100-train-april-hp-final-256-redo" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-final' --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

python main.py --description "res18-c100-train-april-hp-epoch-5-256-redo" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 5 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c100-train-april-hp-epoch-10-256-redo" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 10 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c100-train-april-hp-epoch-15-256-redo" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 15 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c100-train-april-hp-epoch-30-256-redo" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 30 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-c100-train-april-hp-epoch-35-256-redo" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 35 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

############################################
# EXP3: ResNet-18 on Tiny ImageNet
############################################

TUNING_SEEDS=( 31190 77678 71333 17094 48490 79157 62135 82014 76133 8588 79377 1725 61503 33994 72192 30714 25132 93998 61638 70906 )
python main.py --description "res18-tinyimagenet-tune-april" --experiment-type 'tuning' --batch-size 256 --num-epochs 40 --num-trials 12 --model 'resnet18' --dataset 'tiny-imagenet' --seed-list "${TUNING_SEEDS[@]}";

BASE_HP_PATH='SET_THIS_TO_THE_PATH_OF_THE_HP_TUNE_RESULTS'
TRAINING_SEEDS=( 68417 54285 39770 47923 37961 22896 92424 92303 )

# baseline
python main.py --description "res18-tinyimagenet-train-baseline" --experiment-type 'training' --batch-size 256 --weight-initialization 'random' --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'tiny-imagenet' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

# hp-final
python main.py --description "res18-tinyimagenet-train" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-final' --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'tiny-imagenet' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

python main.py --description "res18-tinyimagenet-train-hp-epoch-5" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 5 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'tiny-imagenet' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-tinyimagenet-train-hp-epoch-10" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 10 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'tiny-imagenet' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-tinyimagenet-train-hp-epoch-15" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 15 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'tiny-imagenet' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-tinyimagenet-train-hp-epoch-30" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 30 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'tiny-imagenet' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res18-tinyimagenet-train-hp-epoch-35" --experiment-type 'training' --batch-size 256 --weight-initialization 'hp-epoch' --hp-epoch 35 --num-epochs 200 --num-trials 8 --model 'resnet18' --dataset 'tiny-imagenet' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

############################################
# EXP4: ResNet-152 on CIFAR-100
############################################

TUNING_SEEDS=( 31190 77678 71333 17094 48490 79157 62135 82014 76133 8588 79377 1725 61503 33994 72192 30714 25132 93998 61638 70906 )
python main.py --description "res152-c100-tune-april" --experiment-type 'tuning' --batch-size 128 --num-epochs 80 --num-trials 10 --model 'resnet152' --dataset 'cifar100' --seed-list "${TUNING_SEEDS[@]}";

BASE_HP_PATH='SET_THIS_TO_THE_PATH_OF_THE_HP_TUNE_RESULTS'
TRAINING_SEEDS=( 68417 54285 39770 47923 )

# baseline
python main.py --description "res152-c100-train-200-april-baseline" --experiment-type 'training' --batch-size 128 --weight-initialization 'random' --num-epochs 200 --num-trials 4 --model 'resnet152' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

# hp-final
python main.py --description "res152-c100-train-200-april-hp-final" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-final' --num-epochs 200 --num-trials 4 --model 'resnet152' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

python main.py --description "res152-c100-train-200-april-hp-epoch-65" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 65 --num-epochs 200 --num-trials 4 --model 'resnet152' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "res152-c100-train-200-april-hp-epoch-75" --experiment-type 'training' --batch-size 128 --weight-initialization 'hp-epoch' --hp-epoch 75 --num-epochs 200 --num-trials 4 --model 'resnet152' --dataset 'cifar100' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

############################################
# EXP5: InceptionV3 on Food-101
############################################

TUNING_SEEDS=( 31190 77678 71333 17094 48490 79157 62135 82014 76133 8588 79377 1725 61503 33994 72192 30714 25132 93998 61638 70906 )
python main.py --description "inception-food tuning 20 epochs" --experiment-type 'tuning' --batch-size 64 --num-epochs 20 --num-trials 12 --model 'inception_v3' --dataset 'food101' --seed-list "${TUNING_SEEDS[@]}";

BASE_HP_PATH='SET_THIS_TO_THE_PATH_OF_THE_HP_TUNE_RESULTS'
TRAINING_SEEDS=( 68417 54285 39770 47923 37961 22896 92424 92303 )

# baseline
python main.py --description "inception-food" --experiment-type 'training' --batch-size 64 --weight-initialization 'random' --num-epochs 50 --num-trials 8 --model 'inception_v3' --dataset 'food101' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

# hp-final
python main.py --description "inception-food" --experiment-type 'training' --batch-size 64 --weight-initialization 'hp-final' --num-epochs 50 --num-trials 8 --model 'inception_v3' --dataset 'food101' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";

python main.py --description "inception-food" --experiment-type 'training' --batch-size 64 --weight-initialization 'hp-epoch' --hp-epoch 5 --num-epochs 50 --num-trials 8 --model 'inception_v3' --dataset 'food101' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "inception-food" --experiment-type 'training' --batch-size 64 --weight-initialization 'hp-epoch' --hp-epoch 10 --num-epochs 50 --num-trials 8 --model 'inception_v3' --dataset 'food101' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
python main.py --description "inception-food" --experiment-type 'training' --batch-size 64 --weight-initialization 'hp-epoch' --hp-epoch 15 --num-epochs 50 --num-trials 8 --model 'inception_v3' --dataset 'food101' --base-hp-experiment-path $BASE_HP_PATH --seed-list "${TRAINING_SEEDS[@]}";
