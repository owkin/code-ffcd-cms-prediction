#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python cms_prediction/cli/cms_classification_crossval.py \
    --dataset_path cms_prediction/.data/DummyDataset \
    --max_tiles 1000 \
    --preload_items \
    --n_splits 3 \
    --n_repeats 2 \
    --seed 42 \
    --model abmil \
    --activation relu \
    --hidden_layers "50,10" \
    --d_model_attention 8 \
    --n_top 5 \
    --n_bottom 5 \
    --loss kl \
    --lr 0.001 \
    --n_epochs 10 \
    --batch_size 32 \
    --device cpu \
    --conf_thresholds "0.0,0.5" \
    --output_dir .trained-model
