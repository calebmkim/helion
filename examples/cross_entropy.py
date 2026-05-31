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
    losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)

    # Flatten logits once at the beginning
    logits_flat = logits.view(-1)

    for tile_n in hl.tile(n):
        # Get data for this tile
        labels_tile = labels[tile_n]  # [tile_size]
        base_indices_tile = tile_n.index * v  # [tile_size]

        # Compute the actual flat indices by adding the label offset
        flat_indices = base_indices_tile + labels_tile

        # Load the logits at the target indices
        logits_at_target = hl.load(logits_flat, [flat_indices])

        # Compute log_softmax for numerical stability
        # Load the full rows for this tile
        logits_rows = logits[tile_n, :]  # [tile_size, V]

        # Compute log-sum-exp
        max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
        shifted = logits_rows - max_logits
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))

        # Cross entropy loss: log_sum_exp - logit_at_target
        losses[tile_n] = log_sum_exp - logits_at_target

    return losses.mean()


# %%
# Online (single-pass) Cross Entropy Kernel
# -----------------------------------------


# %%
@helion.kernel()
def cross_entropy_online(
    logits: torch.Tensor,  # [N, V] input logits
    labels: torch.Tensor,  # [N] target labels
) -> torch.Tensor:
    """
    Computes the cross entropy loss with a SINGLE online (flash) pass over V.

    Unlike :func:`cross_entropy` (which reduces the row twice: an ``amax`` pass and
    a separate ``exp``-``sum`` pass), this variant streams each row of logits
    exactly ONCE, maintaining a running max ``mi`` and a running normaliser
    ``di = sum(exp(x - mi))`` with the flash/online rescale update. The
    log-sum-exp is then ``mi + log(di)`` and the per-row loss is
    ``logsumexp - logits[row, label]``. The wide row is therefore read only once
    (no second exp-sum pass re-reading the logits), which closes the wide-vocab
    gap where the two-pass kernel re-streams the row.

    Result is identical to ``torch.nn.functional.cross_entropy(logits, labels)``
    with the default mean reduction.

    Args:
        logits: Input logits tensor of shape [N, V].
        labels: Target labels tensor of shape [N] containing class indices.

    Returns:
        A scalar tensor containing the mean cross entropy loss.
    """
    n, v = logits.shape
    losses = torch.zeros([n], dtype=torch.float32, device=logits.device)

    # Flatten logits once so the label logit can be gathered by flat index.
    logits_flat = logits.view(-1)

    block_size_n = hl.register_block_size(n)
    block_size_v = hl.register_block_size(v)
    neg_inf = float("-inf")

    for tile_n in hl.tile(n, block_size=block_size_n):
        # Gather the logit at each row's target label (single scalar load per row).
        labels_tile = labels[tile_n]
        flat_indices = tile_n.index * v + labels_tile
        logits_at_target = hl.load(logits_flat, [flat_indices]).to(torch.float32)

        # Running online state per row. Init max = -inf so the very first chunk's
        # max always wins (handles the -inf/empty-tile start correctly) and the
        # rescale exp(mi - mi_next) of the initial state is exp(-inf - finite) = 0.
        mi = hl.full([tile_n], neg_inf, dtype=torch.float32)
        di = hl.zeros([tile_n], dtype=torch.float32)

        for tile_v in hl.tile(v, block_size=block_size_v):
            values = logits[tile_n, tile_v].to(torch.float32)
            # Mask out-of-bounds vocab lanes to -inf so they neither inflate the
            # running max nor contribute to the exp-sum (exp(-inf - m) == 0). This
            # is the masked-index idiom (welford), required at non-pow2 / non-divisor
            # V where the last tile is padded.
            col_mask = tile_v.index[None, :] < v
            values = torch.where(col_mask, values, neg_inf)

            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next

        log_sum_exp = mi + torch.log(di)
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
