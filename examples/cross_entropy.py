"""
Cross Entropy Loss Example
==========================

This example demonstrates how to implement a cross entropy loss function using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import LONG_INT_TYPE
from helion._testing import run_example
import helion.language as hl

# %%
# Cross Entropy Kernel
# --------------------


# %%
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def cross_entropy(
    logits: torch.Tensor,  # [N, V] input logits
    labels: torch.Tensor,  # [N] target labels
) -> torch.Tensor:
    """
    Computes the cross entropy loss between logits and target labels.

    Implements the cross entropy loss function commonly used in classification tasks.
    The function computes the log softmax of the logits and then calculates the negative
    log likelihood of the true labels.

    Args:
        logits: Input logits tensor of shape [N, V] where N is batch size and V is vocabulary size
        labels: Target labels tensor of shape [N] containing class indices

    Returns:
        A scalar tensor containing the mean cross entropy loss
    """
    n, v = logits.shape
    # fp32 accumulation: the loss is a log-sum-exp, so every intermediate is
    # kept in fp32 rather than round-tripped through the input dtype.
    losses = torch.zeros([n], dtype=torch.float32, device=logits.device)

    # Flatten logits once at the beginning
    logits_flat = logits.view(-1)

    for tile_n in hl.tile(n):
        # Get data for this tile
        labels_tile = labels[tile_n]  # [tile_size]
        base_indices_tile = tile_n.index * v  # [tile_size]

        # Compute the actual flat indices by adding the label offset
        flat_indices = base_indices_tile + labels_tile

        # Load the logits at the target indices
        logits_at_target = hl.load(logits_flat, [flat_indices]).to(torch.float32)

        # Compute log_softmax for numerical stability.
        # Load the full rows for this tile, upcasting ONCE.  Every intermediate
        # below then stays fp32; the previous version left the row in the input
        # dtype and round-tripped ``exp``'s fp32 result back through bf16 before
        # accumulating, which costs both accuracy and ~13% of the instructions.
        logits_rows = logits[tile_n, :].to(torch.float32)  # [tile_size, V]

        # Compute log-sum-exp, shifted by the row max.  The shift is what makes
        # this unconditionally stable -- the largest summand is exp(0) == 1, so
        # no logit magnitude can overflow -- and it is also why there are TWO
        # sweeps over ``logits``: the summand depends on the max, so the max must
        # be fully reduced before the sum can start.  A single-sweep form needs
        # either a fixed shift (which is NOT stable: measured Inf at logit scale
        # 1e2 when shifting by the target logit) or the online (max, sum)
        # recurrence, whose per-element rescale costs a second exp -- measured
        # more expensive than the second sweep on this backend.
        max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
        shifted = logits_rows - max_logits
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))

        # Cross entropy loss: log_sum_exp - logit_at_target
        losses[tile_n] = log_sum_exp - logits_at_target

    return losses.mean()


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the cross entropy kernel verification.
    """
    batch_size, seq_len, vocab_size = 8, 2048, 131072
    n = batch_size * seq_len
    logits = torch.randn(n, vocab_size, device=DEVICE, dtype=torch.float32)
    labels = torch.randint(0, vocab_size, (n,), device=DEVICE, dtype=LONG_INT_TYPE)

    run_example(
        cross_entropy,
        torch.nn.functional.cross_entropy,
        (logits, labels),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    main()
