"""Namespace shim so ``import rby1.control`` works when running from source."""

import sys
from importlib import import_module

control = import_module("control")
sys.modules.setdefault(__name__ + ".control", control)

__all__ = ["control"]
