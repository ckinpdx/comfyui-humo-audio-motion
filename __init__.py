"""
HuMo Audio-Driven Body Motion
Boosts attention in early transformer blocks to enable audio-driven body motion
Based on discovery that LoRA-enhanced attention allows audio to drive body movement
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__version__ = "1.0.0"

print("\n" + "="*70)
print("🎵 HuMo Audio-Driven Body Motion")
print("="*70)
print(f"Version: {__version__}")
print("Loaded: HuMo Audio Attention Boost")
print("Category: HuMo Audio/Motion")
print("="*70 + "\n")