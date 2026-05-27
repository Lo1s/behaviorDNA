"""
pipeline.sequences
==================
Convert raw event JSONs into fixed-width tensor sequences for the LSTM
autoencoder. Two submodules:

- ``preprocessing``: session JSON → ``(N, 8)`` numpy array; fit/apply
  per-feature mean / std on the training fold only (no leakage).
- ``dataset``: ``EventSequenceDataset`` — PyTorch Dataset that chunks one
  or more session tensors into fixed-length overlapping windows.

The 8 feature channels per event (constant order across the module):

    0: dt_ms          log1p of ms since previous event
    1: dx_norm        mouse delta x / norm_factor   (0 for non-mouse)
    2: dy_norm        mouse delta y / norm_factor
    3: is_mouse_move
    4: is_mouse_click_press   (button-down only)
    5: is_mouse_scroll
    6: is_key_press
    7: is_key_release
"""

from pipeline.sequences.dataset import EventSequenceDataset
from pipeline.sequences.preprocessing import (
    EVENT_FEATURE_DIM,
    apply_normalizer,
    fit_normalizer,
    session_to_event_tensor,
)

__all__ = [
    "EVENT_FEATURE_DIM",
    "EventSequenceDataset",
    "apply_normalizer",
    "fit_normalizer",
    "session_to_event_tensor",
]
