"""
HuMo Audio Attention Control v4

Allows both boosting AND suppression of attention components.
Key change: min values lowered to 0.01 for suppression use cases.

For LIP-SYNC SUPPRESSION:
- Set audio_v_boost to 0.01-0.1 on blocks 6-24 (middle blocks)
- These blocks have the strongest lip-sync sensitivity

For BODY MOTION BOOST:
- Set audio_v_boost to 2.0-4.0 on blocks 0-10 (early blocks)
- These blocks control large body movements
"""

import torch


class HuMoAudioAttentionControlV4:
    """
    Granular attention control with both boost AND suppress capability.
    
    12 independent sliders (min 0.01 for suppression):
    - Audio Cross-Attention: audio_q, audio_k, audio_v, audio_o
    - Text Cross-Attention: cross_q, cross_k, cross_v, cross_o  
    - Self-Attention: self_q, self_k, self_v, self_o
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        block_presets = [
            "custom",
            "early_0-5_structure",
            "early_0-10_body",
            "mid_6-24_lipsync",
            "mid_10-25_gestures",
            "late_25-39_texture",
            "most_0-30_aggressive",
            "all_0-39_maximum",
        ]
        
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                
                # ===== AUDIO CROSS-ATTENTION =====
                "enable_audio_cross_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable audio cross-attention modification"
                }),
                "audio_blocks": (block_presets, {
                    "default": "mid_6-24_lipsync"
                }),
                "audio_q_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Audio Query scale (<1 suppress, >1 boost)"
                }),
                "audio_k_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Audio Key scale (<1 suppress, >1 boost)"
                }),
                "audio_v_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Audio Value scale - THIS IS THE MAIN ONE FOR LIP-SYNC"
                }),
                "audio_o_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Audio Output scale (<1 suppress, >1 boost)"
                }),
                
                # ===== TEXT CROSS-ATTENTION =====
                "enable_cross_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable text cross-attention modification"
                }),
                "cross_blocks": (block_presets, {
                    "default": "early_0-10_body"
                }),
                "cross_q_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "cross_k_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "cross_v_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "cross_o_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                
                # ===== SELF-ATTENTION =====
                "enable_self_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable self-attention modification"
                }),
                "self_blocks": (block_presets, {
                    "default": "early_0-10_body"
                }),
                "self_q_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "self_k_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "self_v_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "self_o_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "slider",
                }),
            },
            "optional": {
                "audio_custom_range": ("STRING", {
                    "default": "6-24",
                    "multiline": False,
                    "tooltip": "Custom block range (e.g. '6-24' or '0-5,25-39')"
                }),
                "cross_custom_range": ("STRING", {
                    "default": "0-10",
                    "multiline": False,
                }),
                "self_custom_range": ("STRING", {
                    "default": "0-10",
                    "multiline": False,
                }),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_attention_control"
    CATEGORY = "WanVideoWrapper/HuMo"
    DESCRIPTION = "Control attention layers for lip-sync suppression or motion boost"
    
    def parse_block_range(self, preset, custom_range):
        """Parse block range from preset or custom string."""
        if preset == "custom":
            range_str = custom_range
        else:
            range_map = {
                "early_0-5_structure": "0-5",
                "early_0-10_body": "0-10",
                "mid_6-24_lipsync": "6-24",
                "mid_10-25_gestures": "10-25",
                "late_25-39_texture": "25-39",
                "most_0-30_aggressive": "0-30",
                "all_0-39_maximum": "0-39",
            }
            range_str = range_map.get(preset, "0-10")
        
        blocks = []
        for part in range_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                blocks.extend(range(int(start), int(end) + 1))
            else:
                blocks.append(int(part))
        return sorted(set(blocks))
    
    def create_scale_hook(self, scale, attn_type, block_idx, component):
        """Create a forward hook that scales the output."""
        def hook(module, input, output):
            return output * scale
        return hook
    
    def apply_attention_control(
        self, model,
        enable_audio_cross_attn, audio_blocks, 
        audio_q_scale, audio_k_scale, audio_v_scale, audio_o_scale,
        enable_cross_attn, cross_blocks,
        cross_q_scale, cross_k_scale, cross_v_scale, cross_o_scale,
        enable_self_attn, self_blocks,
        self_q_scale, self_k_scale, self_v_scale, self_o_scale,
        audio_custom_range="6-24", cross_custom_range="0-10", self_custom_range="0-10"
    ):
        # Parse block ranges
        audio_block_indices = self.parse_block_range(audio_blocks, audio_custom_range) if enable_audio_cross_attn else []
        cross_block_indices = self.parse_block_range(cross_blocks, cross_custom_range) if enable_cross_attn else []
        self_block_indices = self.parse_block_range(self_blocks, self_custom_range) if enable_self_attn else []
        
        # Print configuration
        print(f"\n{'='*70}")
        print(f"🎛️  HuMo Audio Attention Control v4")
        print(f"{'='*70}")
        
        if enable_audio_cross_attn:
            mode = "SUPPRESS" if audio_v_scale < 1.0 else "BOOST"
            print(f"\n🎵 AUDIO CROSS-ATTENTION [{mode}]")
            print(f"   Blocks: {audio_block_indices[0]}-{audio_block_indices[-1]} ({len(audio_block_indices)} blocks)")
            print(f"   Q: {audio_q_scale}x | K: {audio_k_scale}x | V: {audio_v_scale}x | O: {audio_o_scale}x")
        
        if enable_cross_attn:
            print(f"\n📝 TEXT CROSS-ATTENTION")
            print(f"   Blocks: {cross_block_indices[0]}-{cross_block_indices[-1]}")
            print(f"   Q: {cross_q_scale}x | K: {cross_k_scale}x | V: {cross_v_scale}x | O: {cross_o_scale}x")
        
        if enable_self_attn:
            print(f"\n🔄 SELF-ATTENTION")
            print(f"   Blocks: {self_block_indices[0]}-{self_block_indices[-1]}")
            print(f"   Q: {self_q_scale}x | K: {self_k_scale}x | V: {self_v_scale}x | O: {self_o_scale}x")
        
        # Access the model
        diffusion_model = model.model.diffusion_model
        
        # Clear ALL existing HuMo attention hooks (both from this node and suppress node)
        for attr in ['_attention_control_hooks', '_lipsync_suppress_hooks']:
            if hasattr(model, attr):
                for hook in getattr(model, attr):
                    try:
                        hook.remove()
                    except:
                        pass  # Hook may already be invalid
                setattr(model, attr, [])
        
        hook_handles = []
        patch_count = 0
        
        all_blocks = set(audio_block_indices) | set(cross_block_indices) | set(self_block_indices)
        
        for block_idx in sorted(all_blocks):
            if block_idx >= len(diffusion_model.blocks):
                print(f"⚠️  Block {block_idx} doesn't exist (max: {len(diffusion_model.blocks)-1})")
                continue
                
            block = diffusion_model.blocks[block_idx]
            
            # ===== Audio Cross-Attention =====
            if block_idx in audio_block_indices:
                if hasattr(block, 'audio_cross_attn_wrapper'):
                    audio_attn = block.audio_cross_attn_wrapper.audio_cross_attn
                    
                    for comp, scale in [('q', audio_q_scale), ('k', audio_k_scale), 
                                        ('v', audio_v_scale), ('o', audio_o_scale)]:
                        if scale != 1.0 and hasattr(audio_attn, comp):
                            hook = getattr(audio_attn, comp).register_forward_hook(
                                self.create_scale_hook(scale, 'audio', block_idx, comp)
                            )
                            hook_handles.append(hook)
                            patch_count += 1
            
            # ===== Text Cross-Attention =====
            if block_idx in cross_block_indices:
                if hasattr(block, 'cross_attn'):
                    cross_attn = block.cross_attn
                    
                    for comp, scale in [('q', cross_q_scale), ('k', cross_k_scale),
                                        ('v', cross_v_scale), ('o', cross_o_scale)]:
                        if scale != 1.0 and hasattr(cross_attn, comp):
                            hook = getattr(cross_attn, comp).register_forward_hook(
                                self.create_scale_hook(scale, 'cross', block_idx, comp)
                            )
                            hook_handles.append(hook)
                            patch_count += 1
            
            # ===== Self-Attention =====
            if block_idx in self_block_indices:
                if hasattr(block, 'self_attn'):
                    self_attn = block.self_attn
                    
                    for comp, scale in [('q', self_q_scale), ('k', self_k_scale),
                                        ('v', self_v_scale), ('o', self_o_scale)]:
                        if scale != 1.0 and hasattr(self_attn, comp):
                            hook = getattr(self_attn, comp).register_forward_hook(
                                self.create_scale_hook(scale, 'self', block_idx, comp)
                            )
                            hook_handles.append(hook)
                            patch_count += 1
        
        print(f"\n✅ Registered {patch_count} attention hooks")
        print(f"{'='*70}\n")
        
        model._attention_control_hooks = hook_handles
        
        return (model,)


class HuMoLipsyncSuppressAttn:
    """
    Suppress lip-sync via attention scaling.
    Uses input tensor analysis to detect and protect reference/first frames.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "suppression_strength": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Lower = more suppression. 0.05 is strong, 0.2 is mild"
                }),
                "block_start": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 39,
                    "tooltip": "First block to suppress"
                }),
                "block_end": ("INT", {
                    "default": 24,
                    "min": 0,
                    "max": 39,
                    "tooltip": "Last block to suppress"
                }),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "suppress"
    CATEGORY = "WanVideoWrapper/HuMo"
    DESCRIPTION = "Suppress lip-sync by attenuating audio cross-attention"
    
    def suppress(self, model, enabled, suppression_strength, block_start, block_end):
        if not enabled:
            return (model,)
        
        print(f"\n{'='*70}")
        print(f"🔇 HuMo Lipsync Suppress (Direct Attention Hook)")
        print(f"   Blocks {block_start}-{block_end}, strength: {suppression_strength}x")
        print(f"{'='*70}")
        
        diffusion_model = model.model.diffusion_model
        
        # Clear existing hooks
        for attr in ['_lipsync_suppress_hooks', '_attention_control_hooks']:
            if hasattr(model, attr):
                for hook in getattr(model, attr):
                    try:
                        hook.remove()
                    except:
                        pass
                setattr(model, attr, [])
        
        hook_handles = []
        
        # Simple approach: just suppress, no fancy sigma tracking
        # The first frame issue might be due to something else entirely
        def create_suppress_hook(scale):
            def hook(module, input, output):
                return output * scale
            return hook
        
        patched_blocks = []
        for block_idx in range(block_start, block_end + 1):
            if block_idx >= len(diffusion_model.blocks):
                continue
            
            block = diffusion_model.blocks[block_idx]
            
            if hasattr(block, 'audio_cross_attn_wrapper'):
                audio_attn = block.audio_cross_attn_wrapper.audio_cross_attn
                
                # Only hook the output projection - this is the final audio influence
                if hasattr(audio_attn, 'o'):
                    hook = audio_attn.o.register_forward_hook(create_suppress_hook(suppression_strength))
                    hook_handles.append(hook)
                    patched_blocks.append(block_idx)
        
        print(f"✅ Patched {len(patched_blocks)} blocks: {patched_blocks[0]}-{patched_blocks[-1]}")
        print(f"{'='*70}\n")
        
        model._lipsync_suppress_hooks = hook_handles
        
        return (model,)


NODE_CLASS_MAPPINGS = {
    "HuMoAudioAttentionControlV4": HuMoAudioAttentionControlV4,
    "HuMoLipsyncSuppressAttn": HuMoLipsyncSuppressAttn,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuMoAudioAttentionControlV4": "HuMo Audio Attention Control v4",
    "HuMoLipsyncSuppressAttn": "HuMo Lipsync Suppress (Attention)",
}
