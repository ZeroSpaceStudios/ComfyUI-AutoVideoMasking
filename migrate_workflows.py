#!/usr/bin/env python3
"""
Migrate saved ComfyUI workflow JSON files from old SAMhera* class IDs
to the new AVM* names.

Usage:
    python migrate_workflows.py workflow1.json [workflow2.json ...]

A backup of each file is saved as <name>.json.bak before overwriting.
"""

import json
import shutil
import sys

RENAMES = {
    "SAMheraAPIKey":                   "AVMAPIConfig",
    "SAMheraCropByBox":                "AVMCropByBox",
    "SAMheraPasteBackMask":            "AVMPasteBackMask",
    "SAMheraAddFramePrompt":           "AVMAddFramePrompt",
    "SAMheraAutoLayer":                "AVMAutoLayer",
    "SAMheraLayerPropagate":           "AVMLayerPropagate",
    "SAMheraMultiFrameAutoLayer":      "AVMMultiFrameAutoLayer",
    "SAMheraMultiFrameLayerPropagate": "AVMMultiFrameLayerPropagate",
    "SAMheraReferenceMatch":           "VLMReferenceMatch",
    "SAMheraLayerSelector":            "AVMLayerSelector",
    "SAMheraAddFramePromptBundle":     "AVMAddFramePromptBundle",
    "SAMheraUnpackBundle":             "AVMUnpackBundle",
    "SAMheraAutoCrop":                 "VLMAutoCrop",
    "SAMheraReload":                   "AVMReload",
}

# The custom wire type was also renamed
TYPE_RENAMES = {
    "SAMHERA_API": "AVM_API",
}


def migrate(path: str) -> None:
    text = open(path, encoding="utf-8").read()
    original = text

    for old, new in {**RENAMES, **TYPE_RENAMES}.items():
        text = text.replace(old, new)

    if text == original:
        print(f"  {path}: no changes needed")
        return

    shutil.copy2(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(text)
    print(f"  {path}: migrated (backup saved as {path}.bak)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python migrate_workflows.py <workflow.json> [...]")
        sys.exit(1)

    print("AVM workflow migration")
    for p in sys.argv[1:]:
        migrate(p)
    print("Done.")
