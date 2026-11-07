"""Compatibility namespace. New applications use multigpubeamsearch."""
import importlib as _importlib
import sys as _sys
from multigpubeamsearch import *
from multigpubeamsearch import __all__

for _name in ['backend', 'bootstrap', 'build', 'contracts', 'dispatch', 'errors', 'hamming', 'models', 'options', 'preparation', 'results', 'sources']:
    _module = _importlib.import_module("multigpubeamsearch." + _name)
    _sys.modules[__name__ + "." + _name] = _module
    globals()[_name] = _module
