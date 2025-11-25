"""
Multe: Multichoice Logit Estimation

A Python library for estimating discrete choice models where agents can select
either single alternatives or unordered pairs of alternatives.
"""

from .model import MultichoiceLogit
from .simulate import parse_choices, simulate_choices, simulate_data

__all__ = [
    "MultichoiceLogit",
    "parse_choices",
    "simulate_choices",
    "simulate_data",
]
