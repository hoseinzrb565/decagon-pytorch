from typing import Literal, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from torch import Tensor
from torch.optim import Adam, AdamW
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from decagon.model import DecagonLinkPredictor


def loss_fn(pos_logits: Tensor, neg_logits: Tensor) -> Tensor:
    # pos_score: [E], neg_score: [E]
    assert pos_logits.ndim == 1 and neg_logits.ndim == 1
    assert pos_logits.size(0) == neg_logits.size(0)

    pos_labels = torch.ones_like(pos_logits)  # [E]
    neg_labels = torch.zeros_like(neg_logits)  # [E]
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)  # []
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)  # []
    loss = pos_loss + neg_loss  # []

    return loss


# TODO: Use better metrics (e.g. ranking metrics). AUROC and AUPRC require the number of true/false positives/negatives to be known, which is not suited for incomplete knowledge graphs.
# TODO: Add AP@K.
class EvalResult(NamedTuple):
    auroc: float
    auprc: float


def eval_fn(pos_logits: Tensor, neg_logits: Tensor) -> EvalResult:
    # pos_score: [E], neg_score: [E]
    assert pos_logits.ndim == 1 and neg_logits.ndim == 1
    assert pos_logits.size(0) == neg_logits.size(0)

    pos_labels = torch.ones_like(pos_logits, dtype=torch.int)  # [E]
    neg_labels = torch.zeros_like(neg_logits, dtype=torch.int)  # [E]

    logits = torch.cat([pos_logits, neg_logits], dim=0)  # [E]
    labels = torch.cat([pos_labels, neg_labels], dim=0)  # [E]

    auroc: Tensor = BinaryAUROC().forward(logits, labels)  # []
    auprc: Tensor = BinaryAveragePrecision().forward(logits, labels)  # []

    return EvalResult(auroc=auroc.item(), auprc=auprc.item())


def train(
    train_pos_g: DGLGraph,
    train_neg_g: DGLGraph,
    val_pos_g: DGLGraph,
    val_neg_g: DGLGraph,
    device: Literal["cuda", "cpu"],
    dims: list[int],
    dropout_ps: list[float],
    epochs: int,
    optimizer_name: Literal["adam", "adamw"],
    lr: float,
) -> None:
    model = DecagonLinkPredictor(train_pos_g, dims, dropout_ps).to(device)
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unrecognized optimizer {optimizer_name}.")

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # *Train*
        model.train()

        # Forward
        c_etype2pos_logits, c_etype2neg_logits = model.forward(train_pos_g, train_neg_g)
        n_edges: int = train_pos_g.num_edges() - train_pos_g.num_edges("pdi")
        c_etype2loss: dict[tuple[str, str, str], Tensor] = {}
        for c_etype in train_pos_g.canonical_etypes:  # type: ignore
            importance_weight: float = train_pos_g.num_edges(c_etype) / n_edges

            pos_logits = c_etype2pos_logits[c_etype]  # [E]
            neg_logits = c_etype2neg_logits[c_etype]  # [E]
            loss = loss_fn(pos_logits, neg_logits)  # []
            c_etype2loss[c_etype] = loss * importance_weight  # []
        loss = torch.sum(torch.stack(list(c_etype2loss.values())))  # []

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1} Training Loss: {loss.item():.4f}")

        # *Eval*
        if (epoch + 1) % 10 != 0:
            continue

        with torch.no_grad():
            model.eval()
            c_etype2pos_logits, c_etype2neg_logits = model.forward(val_pos_g, val_neg_g)
            c_etype2eval_result: dict[tuple[str, str, str], EvalResult] = {}
            for c_etype in train_pos_g.canonical_etypes:  # type: ignore
                pos_logits = c_etype2pos_logits[c_etype]  # [E]
                neg_logits = c_etype2neg_logits[c_etype]  # [E]
                c_etype2eval_result[c_etype] = eval_fn(pos_logits, neg_logits)

        # TODO: Better printing
        mean_ddi_auroc = np.mean(
            [
                eval_result.auroc
                for c_etype, eval_result in c_etype2eval_result.items()
                if c_etype[0] == c_etype[2] == "drug"
            ]
        ).item()
        mean_ddi_auprc = np.mean(
            [
                eval_result.auprc
                for c_etype, eval_result in c_etype2eval_result.items()
                if c_etype[0] == c_etype[2] == "drug"
            ]
        ).item()
        print(f"\nAUROC: {mean_ddi_auroc:.4f} AUPRC: {mean_ddi_auprc:.4f}\n")

        # TODO: Checkpointing


# TODO: Standalone eval function for test set
