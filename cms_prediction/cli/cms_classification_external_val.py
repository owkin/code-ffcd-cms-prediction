# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evalute a CMS prediction model on an external dataset."""

import argparse
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.special import softmax
from typing import Optional, Literal

from cms_prediction.data.dataset import CmsDataset
from cms_prediction.training_utils import (
    log_string, 
    get_model, 
    pad_collate_fn_dict,
    evaluate_predictions,
    bootstrap_predictions
)


def predict_one_epoch(
    model: torch.nn.Module,
    model_arch: Literal["chowder", "abmil", "meanpool"],
    val_dataloader: DataLoader,
    device: Optional[str] = "cuda",
) -> (np.ndarray, np.ndarray):
    """Run predictions over the dataset for a given model."""
    model.eval()
    predictions_val = []
    labels_val = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Gather data
            features = batch["features"]
            masks = batch["tiles_mask"]
            labels = torch.stack(
                (
                    batch["RF.CMS1"],
                    batch["RF.CMS2"],
                    batch["RF.CMS3"],
                    batch["RF.CMS4"],
                ),
                dim=1,
            )
            # Apply transformations and send to GPU
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            # Pass to model
            if model_arch == "meanpool":
                logits = model(features, masks)
            elif model_arch == "chowder":
                logits, scores = model(features, masks)
            elif model_arch == "abmil":
                logits, attention_scores = model(features, masks)
            # Monitoring
            predictions_val.append(logits.detach().cpu().numpy())
            labels_val.append(labels.detach().cpu().numpy())

    predictions_val = np.vstack(predictions_val)
    labels_val = np.vstack(labels_val)
    return predictions_val, labels_val


def main(args):
    """Evaluate a CMS prediction model on a dataset."""
    # Prepare output folder
    OUTPUT_DIR = Path(args.output_dir)
    if OUTPUT_DIR.exists():
        OUTPUT_DIR = OUTPUT_DIR.parent / (
            OUTPUT_DIR.name + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
    OUTPUT_DIR.mkdir(parents=True)
    
    with open(OUTPUT_DIR / "config.json", "a") as config_save:
        json.dump(args.__dict__, config_save, indent=2)
    logs = open(OUTPUT_DIR / "log_evaluation.txt", "a")

    # Retrieve training config
    training_config_path = Path(args.model_path) / "config.json"
    with open(training_config_path) as config_file:
        config = json.load(config_file)
        training_config = argparse.Namespace()
        vars(training_config).update(config)
        print(training_config)

    # Device
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    log_string(f"Device: {device}", logs)

    ## Load dataset
    dataset_eval = CmsDataset(
        base_path=args.dataset_path,  # str or pathlib.Path
        max_tiles=training_config.max_tiles,
        preload_items=args.preload_items,  # bool
    )

    ## Dataloader
    dataloader_eval = DataLoader(
        dataset=dataset_eval,
        shuffle=True,
        num_workers=0,
        batch_size=args.batch_size,
        drop_last=False,
        collate_fn=pad_collate_fn_dict,
    )

    # Model
    model = get_model(
        model_arch=training_config.model,
        activation_type=training_config.activation,
        in_features=training_config.in_features,
        out_features=4,
        hidden_layers=[int(x) for x in training_config.hidden_layers.split(",")],
        n_top=training_config.n_top,
        n_bottom=training_config.n_bottom,
        d_model_attention=training_config.d_model_attention,
    )

    model.to(device)

    # Thresholds for metrics
    thresholds = [float(x) for x in args.conf_thresholds.split(",")]

    # Single predictions if single model
    if args.fold_number is not None:
        checkpoint_path = args.model_path / str(args.fold_number) / f"checkpoint_{args.which_checkpoint}.tar"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions_val, labels_val = predict_one_epoch(
            model, training_config.model, dataloader_eval, device
        )
    else:  # Average predictions over all folds
        nb_models = training_config.n_splits * training_config.n_repeats
        predictions_val_ = []
        paths = [
            (
                Path(args.model_path) /
                str(i) /
                f"checkpoint_{args.which_checkpoint}.tar"
            )
            for i in range(nb_models)
        ]
        for i, checkpoint_path in enumerate(paths):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            predictions_val, labels_val = predict_one_epoch(
                model, training_config.model, dataloader_eval, device
            )
            predictions_val_.append(predictions_val)
            for threshold in thresholds:
                confident_labels = np.nonzero(np.max(labels_val, axis=1) > threshold)
                selected_preds = predictions_val[confident_labels]
                selected_labels = labels_val[confident_labels]
                s_metrics = evaluate_predictions(
                    softmax(selected_preds, axis=1), selected_labels
                )
                metrics = s_metrics.to_dict()
                # Log metrics
                for key in metrics:
                    log_string(f"{key} {i} ({threshold}): {metrics[key]}", logs)
        predictions_val = np.array(predictions_val_)
        predictions_val = softmax(predictions_val, axis=2)
        predictions_val = np.mean(predictions_val, axis=0)

    # Compute metrics
    for threshold in thresholds:
        # Select confident enough classifications for metrics
        confident_labels = np.nonzero(np.max(labels_val, axis=1) > threshold)
        selected_preds = predictions_val[confident_labels]
        selected_labels = labels_val[confident_labels]

        s_metrics = evaluate_predictions(selected_preds, selected_labels)
        metrics = s_metrics.to_dict()
        df_bs_metrics = bootstrap_predictions(
            selected_preds,
            selected_labels,
            conf_threshold=threshold,
            bootstraping_confidence=args.bootstraping_confidence,
            n_bootstrap=args.n_bootstrap_samples
        )
        # Log metrics
        for key in metrics:
            # log_string(f"{key} ensembled ({threshold}): {metrics[key]}", logs)
            log_string(
                f"{key} ensembled ({threshold}): {metrics[key]} [{df_bs_metrics.loc[key, 'low']}, {df_bs_metrics.loc[key, 'high']}]",
                logs,
            )
        # store
        s_metrics.to_csv(OUTPUT_DIR / f"evaluation_metrics_{threshold}.csv")
        df_bs_metrics.to_csv(OUTPUT_DIR / f"bootstrap_evaluation_metrics_{threshold}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Device
    parser.add_argument("--device", type=str, default="auto", help="Training device (cpu, cuda). Auto = cuda if available.")
    # Model
    parser.add_argument("--model_path", type=str, help="Disk location where the models are dumped after the cross-validation phase.")
    parser.add_argument("--fold_number", type=int, default=None, help="If specified, evaluates the i-th fold instead of ensembling over all folds.")
    parser.add_argument("--which_checkpoint", type=str, default="best_0.5", help="Checkpoint to evaluate - either 'last' or 'best_<SELECTED_THRESHOLD>'")
    # Data
    parser.add_argument("--dataset_path", type=str, help="Location of the evaluation dataset on disk")
    parser.add_argument("--preload_items", action="store_true", help="Preload dataset, including features, in memory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    # Evaluation
    parser.add_argument("--conf_thresholds", type=str, default="0.0,0.5", help="For metrics computations: exclusion thresholds (sample not included if max(RF.CMS) < threshold)")
    parser.add_argument("--bootstraping_confidence", type=float, default=0.95, help=" width of the bootstrapped confidence interval (2.5%% - 97.5%% for 0.95 for instance)")
    parser.add_argument("--n_bootstrap_samples", type=int, default=1000, help="Number of bootstrapped samples to compute CI.")
    # Saving
    parser.add_argument("--output_dir", type=str, help="Output directory, for logs and metrics.")
    args = parser.parse_args()
    main(args)
