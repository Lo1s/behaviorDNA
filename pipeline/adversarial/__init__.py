"""
pipeline.adversarial
====================
Synthetic cheat generation and detection benchmarking.

The point of this module is to ground BehaviorDNA's anomaly detectors in
something measurable: given a known cheating pattern (aimbot, triggerbot, macro)
injected into a real session, do the detectors catch it?

Two sub-modules:

- ``bot_generator``: takes a legit session and overlays one of three cheat
  signatures, producing a labelled hybrid session (still valid as a
  BehaviorDNA event JSON).
- ``benchmark``: runs every available detector against a labelled mix of
  legit + synthetic-cheat sessions and reports ROC / PR / detection-rate.

See ``docs/ADVERSARIAL.md`` for the methodology write-up and
``notebooks/10_adversarial_bots.ipynb`` for the step-by-step tutorial.
"""

from pipeline.adversarial.bot_generator import (
    AimbotGenerator,
    MacroGenerator,
    TriggerbotGenerator,
    inject_cheat,
)

__all__ = [
    "AimbotGenerator",
    "TriggerbotGenerator",
    "MacroGenerator",
    "inject_cheat",
]
