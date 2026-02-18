# ComfyUI entry point when this repo is git-cloned directly into custom_nodes/.
# The actual node lives in comfyui_node/; this file just re-exports the mappings
# so ComfyUI can find them at the repo root.
from comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
