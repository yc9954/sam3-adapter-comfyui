import sys
import os

# Ensure SAM-Adapter-PyTorch is importable regardless of CWD
_HERE   = os.path.dirname(os.path.abspath(__file__))
_SAM_DIR = os.path.join(os.path.dirname(_HERE), "SAM-Adapter-PyTorch")
if _SAM_DIR not in sys.path:
    sys.path.insert(0, _SAM_DIR)

from .sam3_node import SAM3AdapterNode

NODE_CLASS_MAPPINGS = {
    "SAM3Adapter": SAM3AdapterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Adapter": "SAM3 Adapter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
