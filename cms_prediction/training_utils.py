# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training utilities."""

from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import kl_div
from torch.utils.data.dataloader import default_collate
from typing import Any, Optional, Literal, Union

from cms_prediction.models.abmil import ABMIL
from cms_prediction.models.chowder import Chowder
from cms_prediction.models.meanpool import MeanPool


def log_string(out_str, file):
    """Add string to log."""
    file.write(out_str + "\n")
    file.flush()
    print(out_str)


def get_model(
    model_arch: Literal["chowder", "abmil", "meanpool"],
    activation_type: Literal["relu", "sigmoid"],
    in_features: int,
    out_features: int,
    hidden_layers: Optional[list[int]],
    n_top: Optional[int] = None,  # Chower-specific
    n_bottom: Optional[int] = None,  # Chower-specific
    d_model_attention: Optional[int] = None,  # ABMIL-specific
) -> torch.nn.Module:
    """Initialize model architecture."""
    if activation_type == "relu":
        activation = torch.nn.ReLU()
    elif activation_type == "sigmoid":
        activation = torch.nn.Sigmoid()
    else:
        raise NotImplementedError("Activation not supported")

    if hidden_layers is None:
        hidden_layers = hidden_layers

    if model_arch == "meanpool":
        model = MeanPool(
            in_features=in_features,
            out_features=out_features,
            hidden=hidden_layers,
            activation=activation,
        )
    elif model_arch == "chowder":
        model = Chowder(
            in_features=in_features,
            out_features=out_features,
            n_top=n_top,
            n_bottom=n_bottom,
            mlp_hidden=hidden_layers,
            mlp_activation=activation,
        )
    elif model_arch == "abmil":
        model = ABMIL(
            in_features=in_features,
            out_features=out_features,
            d_model_attention=d_model_attention,
            mlp_hidden=hidden_layers,
            mlp_activation=activation,
        )
    else:
        raise NotImplementedError("Model not supported")
    return model


def evaluate_predictions(
    preds: np.ndarray, 
    labels: np.ndarray, 
    conf_threshold: float = 0.0
) -> pd.Series:
    """Evaluate predictions of the model against labels.
    
    Parameters
    ----------
    preds: np.ndarray
        CMS Predictions - numpy array of size (N, 4)
    labels: np.ndarray
        Ground treuth CMS calls - numpy array of size (N, 4)
    conf_threshold: float
        Between 0 and 1. Excludes from the computation samples whose maximum ground 
        truth CMS call is less than `conf_threshold`

    Returns
    -------
    metrics: dict
        Accuracy, One-vs-all AUCs and macroaverage AUC (mAUC)
    """
    metrics = {}
    # Select confident enough classifications for metrics
    confident_labels = np.nonzero(np.max(labels, axis=1) > conf_threshold)
    selected_preds = preds[confident_labels]
    selected_labels = labels[confident_labels]
    # Gather scores and metrics
    max_label = np.argmax(selected_labels, axis=1)
    max_pred = np.argmax(selected_preds, axis=1)
    # KL-divergence
    pointwise_kl = kl_div(selected_labels,selected_preds)
    metrics["kl_div"] = float(np.sum(pointwise_kl) / pointwise_kl.shape[0])
    # Accuracy
    metrics["accuracy"] = accuracy_score(max_label, max_pred)
    # One-vs-all AUCs
    ova_aucs = []
    for cms in range(4):
        one_vs_all_label = np.array(max_label == cms, int)
        metrics[f"one_vs_all_aucs.CMS{cms+1}"] = roc_auc_score(one_vs_all_label, selected_preds[:, cms])
        ova_aucs.append(metrics[f"one_vs_all_aucs.CMS{cms+1}"] )
    # mAUC
    metrics["mauc"] = float(np.mean(ova_aucs))
    return pd.Series(metrics)


def bootstrap_predictions(
    preds: np.ndarray, 
    labels: np.ndarray, 
    conf_threshold: float = 0.0,
    bootstraping_confidence: float = 0.95, 
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute predictions metrics with confidence intervals using bootstrapping.
      
    Parameters
    ----------
    preds: np.ndarray
        CMS Predictions - numpy array of size (N, 4)
    labels: np.ndarray
        Ground treuth CMS calls - numpy array of size (N, 4)
    conf_threshold: float
        Between 0 and 1. Excludes from the computation samples whose maximum ground 
        truth CMS call is less than `conf_threshold`
    bootstraping_confidence: float
        width of the bootstrapped confidence interval (2.5% - 97.5% for 0.95 for instance)
    n_bootstrap: int
        number of bootstrapped samples

    Returns
    -------
    metrics: dict
        Accuracy, KL, One-vs-all AUCs and macroaverage AUC (mAUC)
    """
    # Select confident enough classifications for metrics
    confident_labels = np.nonzero(np.max(labels, axis=1) > conf_threshold)
    selected_preds = preds[confident_labels]
    selected_labels = labels[confident_labels]
    # compute metrics
    all_bootstrap_metrics = []
    for i in range(n_bootstrap):
        bootstrap_indices = np.random.randint(
            len(selected_preds), size=len(selected_preds)
        )
        bootstrap_predictions = selected_preds[bootstrap_indices]
        bootstrap_labels = selected_labels[bootstrap_indices]
        bootstrap_metrics = evaluate_predictions(
            bootstrap_predictions, bootstrap_labels, conf_threshold=0
        )
        all_bootstrap_metrics.append(bootstrap_metrics)

    df_bootstrap_metrics = pd.DataFrame(all_bootstrap_metrics)
    quantiles = [0.5 - bootstraping_confidence * 0.5, 0.5, 0.5 + bootstraping_confidence * 0.5]
    col_names = ["low", "median", "high"]

    bs_metrics = df_bootstrap_metrics.quantile(quantiles).T.set_axis(col_names, axis=1)

    return bs_metrics


def pad_collate_fn_dict(
    batch: list[dict[str, Any]],
    batch_first: bool = True,
    max_len: Optional[int] = None,
    padding_entries=["level", "coords", "features"],
    mask_name="tiles_mask",
) -> dict[str, Any]:
    """Pad and collate batches within data loaders."""
    # Assumes that all paddings will use the same mask
    # (i.e. original sequences lengths are the same)
    # Separate padded sequences and other sequences
    sequences = []
    others = []
    for sample in batch:
        sequence = {
            key: torch.as_tensor(sample[key])
            for key in sample.keys()
            if key in padding_entries
        }
        other = {
            key: sample[key] for key in sample.keys() if key not in padding_entries
        }
        sequences.append(sequence)
        others.append(other)

    # Find max len
    if max_len is None:
        max_len = max([len(f[padding_entries[0]]) for f in sequences])

    # Find padding dims
    padded_dims = {}
    for key in padding_entries:
        trailing_dims = sequences[0][key].size()[1:]
        if batch_first:
            padded_dims[key] = (len(sequences), max_len) + trailing_dims
        else:
            padded_dims[key] = (max_len, len(sequences)) + trailing_dims

    if batch_first:
        masks_dims = (len(sequences), max_len, 1)
    else:
        masks_dims = (max_len, len(sequences), 1)

    # Pad sequences
    sequences_dict = {}
    for key in padding_entries:
        padded_sequences = sequences[0][key].data.new(*padded_dims[key]).fill_(0.0)
        for i, tensor in enumerate([sequence[key] for sequence in sequences]):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                padded_sequences[i, :length, ...] = tensor[:max_len, ...]
            else:
                padded_sequences[:length, i, ...] = tensor[:max_len, ...]
        sequences_dict[key] = padded_sequences

    # Create mask
    masks = torch.ones(*masks_dims, dtype=torch.bool)
    for i, tensor in enumerate([sequence[key] for sequence in sequences]):
        length = tensor.size(0)
        if batch_first:
            masks[i, :length, ...] = False
        else:
            masks[:length, i, ...] = False
    masks_dict = {mask_name: masks}

    # Batch other members of the dictionary
    others = default_collate(others)

    return {**sequences_dict, **masks_dict, **others}


def save_list(list_to_save: list, path: Union[str, Path]):
    """Save a list into a file (one line per item)."""
    with open(path, "w", encoding="utf-8") as file:
        for item in list_to_save:
            # write each item on a new line
            file.write(f"{item}\n")
