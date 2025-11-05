"""
nodes.py - HuMo Audio-Driven Body Motion v3

Maximum granularity version with independent Q/K/V/O controls for each attention type.
Each of the 12 components can be adjusted independently.
"""

import torch
import copy


class HuMoAudioAttentionBoostV3:
    """
    Maximum granularity attention boosting with independent Q/K/V/O controls.
    
    12 independent sliders:
    - Audio Cross-Attention: audio_q, audio_k, audio_v, audio_o
    - Text Cross-Attention: cross_q, cross_k, cross_v, cross_o  
    - Self-Attention: self_q, self_k, self_v, self_o
    
    Each can boost (>1.0) or attenuate (<1.0) independently.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        block_presets = [
            "custom",
            "early_0-10_body",
            "early_mid_0-20_full_body", 
            "mid_10-25_gestures",
            "most_0-30_aggressive",
            "all_0-39_maximum",
        ]
        
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                
                # ===== AUDIO CROSS-ATTENTION =====
                "enable_audio_cross_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable audio cross-attention boosting (direct audio → motion)"
                }),
                "audio_blocks": (block_presets, {
                    "default": "early_0-10_body"
                }),
                "audio_q_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Audio Query boost (what motion to look for)"
                }),
                "audio_k_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Audio Key boost (what audio features are available)"
                }),
                "audio_v_boost": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Audio Value boost (motion content from audio)"
                }),
                "audio_o_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Audio Output boost (final audio-driven motion)"
                }),
                
                # ===== TEXT CROSS-ATTENTION =====
                "enable_cross_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable text cross-attention boosting (shared latent with audio)"
                }),
                "cross_blocks": (block_presets, {
                    "default": "early_0-10_body"
                }),
                "cross_q_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Text Query boost"
                }),
                "cross_k_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Text Key boost"
                }),
                "cross_v_boost": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Text Value boost"
                }),
                "cross_o_boost": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Text Output boost"
                }),
                
                # ===== SELF-ATTENTION =====
                "enable_self_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable self-attention boosting (spatial motion coherence)"
                }),
                "self_blocks": (block_presets, {
                    "default": "early_0-10_body"
                }),
                "self_q_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Self Query boost"
                }),
                "self_k_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Self Key boost"
                }),
                "self_v_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Self Value boost"
                }),
                "self_o_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Self Output boost"
                }),
            },
            "optional": {
                "audio_custom_range": ("STRING", {
                    "default": "0-10",
                    "multiline": False,
                    "tooltip": "Custom block range for audio (e.g. '0-10' or '0-5,15-20')"
                }),
                "cross_custom_range": ("STRING", {
                    "default": "0-10",
                    "multiline": False,
                    "tooltip": "Custom block range for cross-attn"
                }),
                "self_custom_range": ("STRING", {
                    "default": "0-10",
                    "multiline": False,
                    "tooltip": "Custom block range for self-attn"
                }),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "boost_attention"
    CATEGORY = "HuMo Audio/Motion"
    DESCRIPTION = "Maximum granularity audio-driven motion with per-component controls"
    
    def parse_block_range(self, preset, custom_range=None):
        """Parse block range selection"""
        if preset == "custom" and custom_range:
            blocks = []
            for part in custom_range.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    blocks.extend(range(start, end + 1))
                else:
                    blocks.append(int(part))
            return sorted(set(blocks))
        
        ranges = {
            "early_0-10_body": list(range(0, 11)),
            "early_mid_0-20_full_body": list(range(0, 21)),
            "mid_10-25_gestures": list(range(10, 26)),
            "most_0-30_aggressive": list(range(0, 31)),
            "all_0-39_maximum": list(range(0, 40)),
        }
        
        return ranges.get(preset, list(range(0, 11)))
    
    def create_boost_hook(self, boost_factor, attn_type, block_idx, component):
        """
        Create a hook that boosts attention weights or projections.
        
        Args:
            boost_factor: Multiplication factor (can be <1.0 for attenuation)
            attn_type: 'audio', 'cross', or 'self'
            block_idx: Block number
            component: 'q', 'k', 'v', or 'o'
        """
        def hook_fn(module, input, output):
            """Boost the output"""
            if output is None:
                return output
            
            # Handle tuple outputs
            if isinstance(output, tuple):
                boosted = output[0] * boost_factor
                return (boosted,) + output[1:]
            else:
                return output * boost_factor
        
        return hook_fn
    
    def boost_attention(self, model, 
                       # Audio controls
                       enable_audio_cross_attn, audio_blocks,
                       audio_q_boost, audio_k_boost, audio_v_boost, audio_o_boost,
                       # Cross controls  
                       enable_cross_attn, cross_blocks,
                       cross_q_boost, cross_k_boost, cross_v_boost, cross_o_boost,
                       # Self controls
                       enable_self_attn, self_blocks,
                       self_q_boost, self_k_boost, self_v_boost, self_o_boost,
                       # Optional
                       audio_custom_range=None, cross_custom_range=None, self_custom_range=None):
        """Apply attention boosting to model"""
        
        # Clean up old hooks first
        if hasattr(model, '_attention_boost_hooks'):
            print(f"🧹 Removing {len(model._attention_boost_hooks)} old hooks")
            for hook in model._attention_boost_hooks:
                hook.remove()
            model._attention_boost_hooks = []
        
        # Parse block ranges
        audio_block_indices = self.parse_block_range(audio_blocks, audio_custom_range) if enable_audio_cross_attn else []
        cross_block_indices = self.parse_block_range(cross_blocks, cross_custom_range) if enable_cross_attn else []
        self_block_indices = self.parse_block_range(self_blocks, self_custom_range) if enable_self_attn else []
        
        # Print configuration
        print(f"\n{'='*80}")
        print(f"🎵 HuMo Audio Attention Boost v3 - Granular Control")
        print(f"{'='*80}")
        
        if enable_audio_cross_attn:
            print(f"\n🎯 AUDIO CROSS-ATTENTION (Direct Audio → Motion)")
            print(f"   Blocks: {min(audio_block_indices)}-{max(audio_block_indices)} ({len(audio_block_indices)} blocks)")
            print(f"   Q: {audio_q_boost}x | K: {audio_k_boost}x | V: {audio_v_boost}x | O: {audio_o_boost}x")
        
        if enable_cross_attn:
            print(f"\n📝 TEXT CROSS-ATTENTION (Shared Latent Space)")
            print(f"   Blocks: {min(cross_block_indices)}-{max(cross_block_indices)} ({len(cross_block_indices)} blocks)")
            print(f"   Q: {cross_q_boost}x | K: {cross_k_boost}x | V: {cross_v_boost}x | O: {cross_o_boost}x")
        
        if enable_self_attn:
            print(f"\n🔄 SELF-ATTENTION (Spatial Motion Coherence)")
            print(f"   Blocks: {min(self_block_indices)}-{max(self_block_indices)} ({len(self_block_indices)} blocks)")
            print(f"   Q: {self_q_boost}x | K: {self_k_boost}x | V: {self_v_boost}x | O: {self_o_boost}x")
        
        # Access the model
        diffusion_model = model.model.diffusion_model
        
        # Store hook handles
        hook_handles = []
        patch_count = 0
        
        # Register hooks for each attention type
        all_blocks = set(audio_block_indices) | set(cross_block_indices) | set(self_block_indices)
        
        for block_idx in sorted(all_blocks):
            if block_idx >= len(diffusion_model.blocks):
                print(f"⚠️  Warning: Block {block_idx} doesn't exist (max: {len(diffusion_model.blocks)-1})")
                continue
                
            block = diffusion_model.blocks[block_idx]
            
            # ===== Audio Cross-Attention =====
            if block_idx in audio_block_indices:
                if hasattr(block, 'audio_cross_attn_wrapper'):
                    audio_attn = block.audio_cross_attn_wrapper.audio_cross_attn
                    
                    if audio_q_boost != 1.0 and hasattr(audio_attn, 'q'):
                        hook = audio_attn.q.register_forward_hook(
                            self.create_boost_hook(audio_q_boost, 'audio', block_idx, 'q')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                    
                    if audio_k_boost != 1.0 and hasattr(audio_attn, 'k'):
                        hook = audio_attn.k.register_forward_hook(
                            self.create_boost_hook(audio_k_boost, 'audio', block_idx, 'k')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                    
                    if audio_v_boost != 1.0 and hasattr(audio_attn, 'v'):
                        hook = audio_attn.v.register_forward_hook(
                            self.create_boost_hook(audio_v_boost, 'audio', block_idx, 'v')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                    
                    if audio_o_boost != 1.0 and hasattr(audio_attn, 'o'):
                        hook = audio_attn.o.register_forward_hook(
                            self.create_boost_hook(audio_o_boost, 'audio', block_idx, 'o')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                    
                    if patch_count > 0:
                        print(f"  ✓ Patched block {block_idx}.audio_cross_attn")
                else:
                    print(f"  ⚠️  Block {block_idx} has no audio_cross_attn_wrapper")
            
            # ===== Text Cross-Attention =====
            if block_idx in cross_block_indices:
                if hasattr(block, 'cross_attn'):
                    cross_attn = block.cross_attn
                    
                    block_patches = 0
                    
                    if cross_q_boost != 1.0 and hasattr(cross_attn, 'q'):
                        hook = cross_attn.q.register_forward_hook(
                            self.create_boost_hook(cross_q_boost, 'cross', block_idx, 'q')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if cross_k_boost != 1.0 and hasattr(cross_attn, 'k'):
                        hook = cross_attn.k.register_forward_hook(
                            self.create_boost_hook(cross_k_boost, 'cross', block_idx, 'k')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if cross_v_boost != 1.0 and hasattr(cross_attn, 'v'):
                        hook = cross_attn.v.register_forward_hook(
                            self.create_boost_hook(cross_v_boost, 'cross', block_idx, 'v')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if cross_o_boost != 1.0 and hasattr(cross_attn, 'o'):
                        hook = cross_attn.o.register_forward_hook(
                            self.create_boost_hook(cross_o_boost, 'cross', block_idx, 'o')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if block_patches > 0:
                        print(f"  ✓ Patched block {block_idx}.cross_attn")
            
            # ===== Self-Attention =====
            if block_idx in self_block_indices:
                if hasattr(block, 'self_attn'):
                    self_attn = block.self_attn
                    
                    block_patches = 0
                    
                    if self_q_boost != 1.0 and hasattr(self_attn, 'q'):
                        hook = self_attn.q.register_forward_hook(
                            self.create_boost_hook(self_q_boost, 'self', block_idx, 'q')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if self_k_boost != 1.0 and hasattr(self_attn, 'k'):
                        hook = self_attn.k.register_forward_hook(
                            self.create_boost_hook(self_k_boost, 'self', block_idx, 'k')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if self_v_boost != 1.0 and hasattr(self_attn, 'v'):
                        hook = self_attn.v.register_forward_hook(
                            self.create_boost_hook(self_v_boost, 'self', block_idx, 'v')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if self_o_boost != 1.0 and hasattr(self_attn, 'o'):
                        hook = self_attn.o.register_forward_hook(
                            self.create_boost_hook(self_o_boost, 'self', block_idx, 'o')
                        )
                        hook_handles.append(hook)
                        patch_count += 1
                        block_patches += 1
                    
                    if block_patches > 0:
                        print(f"  ✓ Patched block {block_idx}.self_attn")
        
        print(f"\n✅ Successfully registered {patch_count} attention boost hooks")
        print(f"   Hooks will remain active until model is unloaded")
        print(f"{'='*80}\n")
        
        # Store hook handles on the model so they persist
        if not hasattr(model, '_attention_boost_hooks'):
            model._attention_boost_hooks = []
        model._attention_boost_hooks.extend(hook_handles)
        
        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "HuMoAudioAttentionBoostV3": HuMoAudioAttentionBoostV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuMoAudioAttentionBoostV3": "HuMo Audio Attention Boost v3",
}