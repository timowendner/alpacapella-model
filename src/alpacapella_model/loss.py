import torch
import torch.nn.functional as F

class ShiftTolerantBeatDownbeatLoss(torch.nn.Module):
    def __init__(self, beat_weight: float = 1, downbeat_weight: float = 1, tolerance: int = 3):
        super().__init__()
        self.beat_loss = ShiftTolerantBCELoss(pos_weight=beat_weight, tolerance=tolerance)
        self.downbeat_loss = ShiftTolerantBCELoss(pos_weight=downbeat_weight, tolerance=tolerance)
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None):
        beat_logits, downbeat_logits = preds[..., 0], preds[..., 1]
        beat_targets = (targets > 0).float()
        downbeat_targets = (targets == 2).float()
        
        beat_loss = self.beat_loss(beat_logits + downbeat_logits, beat_targets, mask)
        downbeat_loss = self.downbeat_loss(downbeat_logits, downbeat_targets, mask)
        
        return beat_loss + downbeat_loss

class ShiftTolerantBCELoss(torch.nn.Module):
    """
    BCE loss variant for sequence labeling that tolerates small shifts between
    predictions and targets. This is accomplished by max-pooling the
    predictions with a given tolerance and a stride of 1, so the gradient for a
    positive label affects the largest prediction in a window around it.
    Expects predictions to be given as logits, and accepts an optional mask
    with zeros indicating the entries to ignore. Note that the edges of the
    sequence will not receive a gradient, as it is assumed to be unknown
    whether there is a nearby positive annotation.

    Args:
        pos_weight (float): Weight for positive examples compared to negative
            examples (default: 1)
        tolerance (int): Tolerated shift in time steps in each direction
            (default: 3)
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 3):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        
        # spread preds and crop targets to match
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # ignore around the positive targets
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:  # consider padding and no-downbeat mask
            look_at = look_at * self.crop(mask, factor=2)
        # compute loss
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )
