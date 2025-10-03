# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train a CMS prediction model with cross-validation."""

import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
from typing import Literal, Optional

import numpy as np
from scipy.special import softmax
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.model_selection._split import _RepeatedSplits
import torch
from torch.utils.data import DataLoader, Subset

from cms_prediction.data.dataset import CmsDataset
from cms_prediction.training_utils import (
    log_string,
    get_model,
    evaluate_predictions,
    pad_collate_fn_dict,
    save_list,
)


def generate_train_val_splits(
    n_splits: int,
    n_repeats: int,
    seed: int,
    dataset: CmsDataset,
    group_patients: bool = True,
):
    """Generate train/val splits for the cross-validation procedure.

    Splits are stratified by dominant CMS.

    Parameters
    ----------
    n_splits : int
        Number of splits in the cross-validation setup.
    n_repeats : int
        Number of random repetitions of the whole cross-validation procedure
    seed : int
        Set random seed
    dataset : CmsDataset
        CmsDataset dataset instance used for training
    group_patients : bool, optional
        Groups slides by patients (ensures all slides from a given patient are in the same fold), by default True

    Returns
    -------
    splits, cms_counts, mean_continuous_cms_labels
        `splits` contains the indices of split elements.
        `cms_counts` is a np.ndarray[int] countinnumber of samples per dominant CMS.
        `mean_continuous_cms_labels` is a np.ndarray[float] with mean CMS calls.
        `cms_counts` and `mean_continuous_cms_labels` are required for certain loss functions.
    """    
    # Retrieve stratified elements
    categorial_cms_labels = []
    continuous_cms_labels = []
    patients = []
    for item in dataset:
        scores = [item[f"RF.CMS{i}"] for i in range(1, 5)]
        patient = item["patient"]
        label = np.argmax(scores)
        categorial_cms_labels.append(label)
        continuous_cms_labels.append(scores)
        patients.append(patient)
    # Generate splits
    if group_patients:
        fold_generator = _RepeatedSplits(
            StratifiedGroupKFold,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed,
        )
        splits = fold_generator.split(
            range(len(categorial_cms_labels)), y=categorial_cms_labels, groups=patients
        )
    else:
        fold_generator = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        splits = fold_generator.split(
            range(len(categorial_cms_labels)), y=categorial_cms_labels
        )
    _, cms_counts = np.unique(categorial_cms_labels, return_counts=True)
    mean_continuous_cms_labels = np.mean(continuous_cms_labels, axis=0)
    return splits, cms_counts, mean_continuous_cms_labels


def get_criterion(
    loss_type: Literal["wce", "wsce", "ce", "kl"], 
    counts: np.ndarray[int], 
    mean_continuous_cms_labels: np.ndarray[float],
):
    """Instantiate the loss function.
    
    wce: weighted cross-entropy
    wsce: score-weighted cross-entropy
    ce: cross-entropy
    kl: Kullback-Leibler divergence
    """
    if loss_type == "wce":
        assert len(counts) == 4, f"Counts is of length {len(counts)}. Do you have at least 1 sample for each CMS class ?"
        weight = torch.tensor(counts.sum() / counts / 4, dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
    elif loss_type == "wsce":
        assert len(mean_continuous_cms_labels) == 4
        weight = torch.tensor(1 / mean_continuous_cms_labels, dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
    elif loss_type == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == "kl":
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
    else:
        raise NotImplementedError("loss_type should be within wce, wsce, ce, kl")
    return criterion


def train_one_epoch(
    model: torch.nn.Module,
    model_arch: Literal["chowder", "abmil", "meanpool"],
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    loss_type: Literal["wce", "wsce", "ce", "kl"], 
    device: Optional[str] = "cuda",
):
    """Train the model for one epoch."""
    model.train()

    losses_train = []
    predictions_train = []
    labels_train = []

    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        # Gather data
        features = batch["features"]
        masks = batch["tiles_mask"]
        labels = torch.stack([batch[f"RF.CMS{i}"] for i in range(1, 5)], dim=1)
        # Apply transformations and send to device
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
        # Compute loss
        logits_log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
        if loss_type == "kl":
            loss = criterion(logits_log_softmax, labels)
        else:
            loss = criterion(logits, labels)
        # Optimize
        loss.backward()
        optimizer.step()
        # Monitoring
        losses_train.append(loss.item())
        predictions_train.append(logits_log_softmax.detach().cpu().numpy())
        labels_train.append(labels.detach().cpu().numpy())

    predictions_train = np.vstack(predictions_train)
    labels_train = np.vstack(labels_train)
    return losses_train, predictions_train, labels_train


def validate_one_epoch(
    model: torch.nn.Module,
    model_arch: Literal["chowder", "abmil", "meanpool"],
    val_dataloader: DataLoader,
    criterion: torch.nn.Module,
    loss_type: Literal["wce", "wsce", "ce", "kl"], 
    device: Optional[str] = "cuda",
):
    """Evaluate the model for one epoch."""
    model.eval()
    losses_val = []
    predictions_val = []
    labels_val = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Gather data
            features = batch["features"]
            masks = batch["tiles_mask"]
            labels = torch.stack([batch[f"RF.CMS{i}"] for i in range(1, 5)], dim=1)
            # Apply transformations and send to device
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
            # Compute loss
            logits_log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
            if loss_type == "kl":
                loss = criterion(logits_log_softmax, labels)
            else:
                loss = criterion(logits, labels)
            # Monitoring
            losses_val.append(loss.item())
            predictions_val.append(logits_log_softmax.detach().cpu().numpy())
            labels_val.append(labels.detach().cpu().numpy())
    predictions_val = np.vstack(predictions_val)
    labels_val = np.vstack(labels_val)
    return losses_val, predictions_val, labels_val


def main(args):
    """Train a CMS prediction model."""
    # Prepare output folder
    OUTPUT_DIR = Path(args.output_dir)
    if OUTPUT_DIR.exists():
        OUTPUT_DIR = OUTPUT_DIR.parent / (
            OUTPUT_DIR.name + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )

    OUTPUT_DIR.mkdir(parents=True)
    logs = open(OUTPUT_DIR / "log_train.txt", "a")

    # Device
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    log_string(f"Device: {device}", logs)

    # Dataset
    dataset = CmsDataset(
        base_path=args.dataset_path,  # str or pathlib.Path
        max_tiles=args.max_tiles,  # can be None
        preload_items=args.preload_items,  # bool
    )

    in_features = int(dataset[0]["features"].shape[1])

    with open(OUTPUT_DIR / "config.json", "a") as config_save:
        json.dump({"in_features": in_features, **args.__dict__}, config_save, indent=2)

    # cross-validation splits
    splits, cms_counts, mean_continuous_cms_labels = generate_train_val_splits(
        args.n_splits, args.n_repeats, args.seed, dataset
    )

    # 
    thresholds = [float(x) for x in args.conf_thresholds.split(",")]
    all_best_mauc_val = [[] for _ in thresholds]
    all_last_mauc_val = [[] for _ in thresholds]

    for i, (train_idx, val_idx) in enumerate(splits):
        # Initiate split
        repeat, split = (i // args.n_splits) + 1, (i % args.n_splits) + 1
        log_string(
            f">>> Training number {i} - Repeat: {repeat}/{args.n_repeats}, Split: {split}/{args.n_splits} <<<",
            logs,
        )

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        log_string(
            f"len(train_set): {len(train_set)}, len(val_set): {len(val_set)}",
            logs,
        )

        train_slides_idx = [item["slide"] for item in train_set]
        val_slides_idx = [item["slide"] for item in val_set]

        model_save_path = OUTPUT_DIR / str(i)
        model_save_path.mkdir()

        save_list(train_idx, os.path.join(model_save_path, "train_idx.txt"))
        save_list(val_idx, os.path.join(model_save_path, "val_idx.txt"))
        save_list(
            train_slides_idx, os.path.join(model_save_path, "train_slides_idx.txt")
        )
        save_list(val_slides_idx, os.path.join(model_save_path, "val_slides_idx.txt"))

        # Dataloaders
        train_dataloader = DataLoader(
            dataset=train_set,
            shuffle=True,
            num_workers=0,
            batch_size=args.batch_size,
            drop_last=True,
            collate_fn=pad_collate_fn_dict,
        )

        val_dataloader = DataLoader(
            dataset=val_set,
            shuffle=False,
            num_workers=0,
            batch_size=args.batch_size,
            drop_last=False,
            collate_fn=pad_collate_fn_dict,
        )
        log_string(
            f"len(train_dataloader): {len(train_dataloader)}, len(val_dataloader): {len(val_dataloader)}",
            logs,
        )

        # Model
        model = get_model(
            model_arch=args.model,
            activation_type=args.activation,
            in_features=in_features,
            out_features=4,
            hidden_layers=[int(x) for x in args.hidden_layers.split(",")],
            n_top=args.n_top,
            n_bottom=args.n_bottom,
            d_model_attention=args.d_model_attention,
        )
        model.to(device)

        # Optimizer, criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = get_criterion(args.loss, cms_counts, mean_continuous_cms_labels).to(device)

        # Store metric of interest
        best_mauc_val = [0.0 for _ in thresholds]
        last_mauc_val = [0.0 for _ in thresholds]

        for epoch in range(args.n_epochs):
            log_string(f"** Epoch: {epoch + 1} **", logs)

            # Training
            log_string("Training...", logs)
            losses_train, preds_train, labels_train = train_one_epoch(
                model,
                args.model,
                train_dataloader,
                optimizer,
                criterion,
                args.loss,
                device,
            )

            # Validation
            log_string("Validation...", logs)
            losses_val, preds_val, labels_val = validate_one_epoch(
                model, args.model, val_dataloader, criterion, args.loss, device
            )

            # softmax
            preds_train = softmax(preds_train, axis=1)
            preds_val = softmax(preds_val, axis=1)

            # Gather epoch loss
            log_string(f"Mean train loss: {np.mean(losses_train)}", logs)
            log_string(f"Mean eval loss: {np.mean(losses_val)}", logs)

            # Gather metrics
            for thresh_idx, threshold in enumerate(thresholds):
                log_string(f"- Threshold: {threshold} -", logs)
                s_metrics_train = evaluate_predictions(
                    preds_train, labels_train, conf_threshold=threshold
                )
                s_metrics_val = evaluate_predictions(
                    preds_val, labels_val, conf_threshold=threshold
                )
                metrics_train = s_metrics_train.to_dict()
                metrics_val = s_metrics_val.to_dict()
                # Log metrics
                for key in metrics_train:
                    log_string(f"train_{key}: {metrics_train[key]}", logs)
                for key in metrics_val:
                    log_string(f"val_{key}: {metrics_val[key]}", logs)
                # If best metric so far, store model
                if metrics_val["mauc"] > best_mauc_val[thresh_idx]:
                    save_dict = {
                        "epoch": epoch,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_state_dict": model.state_dict(),
                    }
                    torch.save(
                        save_dict,
                        model_save_path / f"checkpoint_best_{threshold}.tar",
                    )
                    s_metrics_train.to_csv(model_save_path / f"metrics_train_best_{threshold}.csv")
                    s_metrics_val.to_csv(model_save_path / f"metrics_val_best_{threshold}.csv")
                    best_mauc_val[thresh_idx] = metrics_val["mauc"]
                last_mauc_val[thresh_idx] = metrics_val["mauc"]
                if epoch == (args.n_epochs - 1):  # last epoch
                    s_metrics_train.to_csv(model_save_path / f"metrics_train_last_{threshold}.csv")
                    s_metrics_val.to_csv(model_save_path / f"metrics_val_last_{threshold}.csv")

        # Save last model
        save_dict = {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
        }
        torch.save(save_dict, model_save_path / "checkpoint_last.tar")

        for thresh_idx, threshold in enumerate(thresholds):
            all_best_mauc_val[thresh_idx].append(best_mauc_val[thresh_idx])
            all_last_mauc_val[thresh_idx].append(last_mauc_val[thresh_idx])

    # Display results
    for thresh_idx, threshold in enumerate(thresholds):
        mean_best_val_auc, std_best_val_auc = (
            np.mean(all_best_mauc_val[thresh_idx]),
            np.std(all_best_mauc_val[thresh_idx]),
        )
        mean_last_val_auc, std_last_val_auc = (
            np.mean(all_last_mauc_val[thresh_idx]),
            np.std(all_last_mauc_val[thresh_idx]),
        )
        log_string(
            f"Best mAUC ({threshold}): {mean_best_val_auc} +/- {std_best_val_auc}",
            logs,
        )
        log_string(
            f"Last mAUC ({threshold}): {mean_last_val_auc} +/- {std_last_val_auc}",
            logs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    parser.add_argument("--dataset_path", type=str, help="Location of the training dataset on disk")
    parser.add_argument("--max_tiles", type=int, default=8_000, help="Maximum number of tiles per WSI")
    parser.add_argument("--preload_items", action="store_true", help="Preload dataset, including features, in memory")
    # Split
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of cross-validation repetitions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Model
    parser.add_argument("--model", type=str, default="abmil", help="Model architecture (either meanpool; chowder or abmil)")
    parser.add_argument("--activation", type=str, default="relu", help="Neuron activation (either relu or sigmoid)")
    parser.add_argument("--hidden_layers", default="200,100", help="Hidden layers dimension fof the MLP head")
    parser.add_argument("--d_model_attention", type=int, default=32, help="[ABMIL] Dimension of attention scores.")  # ABMIL specific
    parser.add_argument("--n_top", type=int, default=100, help="[Chowder] number of highest selected tile scores")  # Chowder specific
    parser.add_argument("--n_bottom", type=int, default=100, help="[Chowder] number of lowest selected tile scores")  # Chowder specific
    # Training
    parser.add_argument("--loss", type=str, default="kl", help="Loss (among wce, wsce, ce, kl)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Training device (cpu, cuda). Auto = cuda if available.")
    # Evaluation
    parser.add_argument("--conf_thresholds", type=str, default="0.0,0.5", help="For metrics computations: exclusion thresholds (sample not included if max(RF.CMS) < threshold)")
    # Saving
    parser.add_argument("--output_dir", type=str, help="Output directory, for logs and metrics")
    args = parser.parse_args()
    main(args)
