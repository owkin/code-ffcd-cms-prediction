#!/bin/bash

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python cms_prediction/cli/cms_classification_external_val.py \
    --device cpu \
    --model_path .trained-model \
    --which_checkpoint last \
    --dataset_path cms_prediction/.data/DummyDataset \
    --preload_items \
    --batch_size 32 \
    --conf_thresholds "0.0,0.5" \
    --bootstraping_confidence 0.95 \
    --n_bootstrap_samples 200 \
    --output_dir .model-evaluation
#   --fold_number 3
