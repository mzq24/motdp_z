"""
Route B agent with improved ego localization and standard split-forward inference.

This keeps the localizer-based GPS filtering from
`route_b_constructed_localizer_b2d_agent.py`, while restoring the normal
Route B inference path via `predict_action()`.
"""

from route_b_b2d_agent import MOTAgent as StandardRouteBAgent
from route_b_constructed_localizer_b2d_agent import (
    RouteBConstructedLocalizerAgent as _LocalizerBase,
)


def get_entry_point():
    return 'RouteBLocalizerAgent'


class RouteBLocalizerAgent(_LocalizerBase):
    """Standard Route B inference + localizer target pose filtering."""

    def resolve_checkpoint_path(self):
        return StandardRouteBAgent.resolve_checkpoint_path(self)

    def _predict_dp_action(self, dp_obs_dict):
        return StandardRouteBAgent._predict_dp_action(self, dp_obs_dict)
