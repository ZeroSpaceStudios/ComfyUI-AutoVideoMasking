from .vlm_sam3_bridge import NODE_CLASS_MAPPINGS as _VLM_MAP, NODE_DISPLAY_NAME_MAPPINGS as _VLM_NAMES
from .samhera_utils import NODE_CLASS_MAPPINGS as _UTILS_MAP, NODE_DISPLAY_NAME_MAPPINGS as _UTILS_NAMES

NODE_CLASS_MAPPINGS = {**_VLM_MAP, **_UTILS_MAP}
NODE_DISPLAY_NAME_MAPPINGS = {**_VLM_NAMES, **_UTILS_NAMES}

# ---------------------------------------------------------------------------
# Backward-compat aliases: old SAMhera*/VLM* class IDs → new AVM* names
# These let saved workflows load without migration.  The old names are
# hidden from the "Add Node" menu via DEPRECATED display names.
# ---------------------------------------------------------------------------

_DEPRECATED_ALIASES = {
    # old ID                              → new ID
    "SAMheraAPIKey":                       "AVMAPIConfig",
    "SAMheraCropByBox":                    "AVMCropByBox",
    "SAMheraPasteBackMask":                "AVMPasteBackMask",
    "SAMheraAddFramePrompt":               "AVMAddFramePrompt",
    "SAMheraAutoLayer":                    "AVMAutoLayer",
    "SAMheraLayerPropagate":               "AVMLayerPropagate",
    "SAMheraMultiFrameAutoLayer":          "AVMMultiFrameAutoLayer",
    "SAMheraMultiFrameLayerPropagate":     "AVMMultiFrameLayerPropagate",
    "SAMheraReferenceMatch":               "VLMReferenceMatch",
    "SAMheraLayerSelector":                "AVMLayerSelector",
    "SAMheraAddFramePromptBundle":         "AVMAddFramePromptBundle",
    "SAMheraUnpackBundle":                 "AVMUnpackBundle",
    "SAMheraAutoCrop":                     "VLMAutoCrop",
    "SAMheraReload":                       "AVMReload",
}

_deprecated_found = []
for old_id, new_id in _DEPRECATED_ALIASES.items():
    if new_id in NODE_CLASS_MAPPINGS:
        NODE_CLASS_MAPPINGS[old_id] = NODE_CLASS_MAPPINGS[new_id]
        NODE_DISPLAY_NAME_MAPPINGS[old_id] = f"(deprecated) {NODE_DISPLAY_NAME_MAPPINGS.get(new_id, new_id)}"
        _deprecated_found.append(old_id)

if _deprecated_found:
    print(f"[AVM] Registered {len(_deprecated_found)} deprecated aliases for backward compatibility.")
    print(f"[AVM] Run 'python migrate_workflows.py <workflow.json>' to update saved workflows.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
