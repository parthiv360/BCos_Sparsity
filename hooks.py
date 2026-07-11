import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Callable

class Hooks:
    """
    A class to manage hooks for a model.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles = []

    def _save_hook(self, layer_name: str) -> Callable:
        """
         Hook function that saves the output of the layer to the activations dictionary
        """
        def hook(module: nn.Module, input: Any, output: Any):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[layer_name] = output.detach().clone()
        return hook
    
    def register_save_hook(self, target_module: nn.Module, hook_name: str):
        """
        Register hooks to save the outputs of specified module.
        """
        if target_module is None:
            raise ValueError(f"Target module not found for hook: {hook_name}")
        handle = target_module.register_forward_hook(self._save_hook(layer_name=hook_name))
        self.hook_handles.append(handle)
        print(f"[*] Save hook registered: {hook_name}")

    def _patch_hook(self, patch_tensor: torch.Tensor) -> Callable:
        """
        Hook function that replaces the output of the layer with the patch_tensor
        """
        
        def hook(module: nn.Module, input: Any, output: Any):
            if isinstance(output, tuple):
                # replace only the first element (hidden_states) and keep the rest.
                hidden_states = output[0]
                assert hidden_states.shape == patch_tensor.shape, f"Shape mismatch: {hidden_states.shape} vs {patch_tensor.shape}"
                return (patch_tensor.clone(),) + output[1:]
            else:
                assert output.shape == patch_tensor.shape, f"Shape mismatch: {output.shape} vs {patch_tensor.shape}"
                return patch_tensor.clone()
        return hook
    
    def register_patch_hook(self, target_module: nn.Module, patch_tensor: torch.Tensor,hook_name: str = "patch"):
        """
        Register hooks to replace the outputs of specified module with patch_tensor.
        """
        if target_module is None:
            raise ValueError(f"Target module not found for hook: {hook_name}")

        handle = target_module.register_forward_hook(self._patch_hook(patch_tensor))
        self.hook_handles.append(handle)
        print(f"[*] Patch hook registered: {hook_name}")

    def register_all_blocks(self):
        """
        Register hooks for all blocks in the model.
        """
        for i, block in enumerate(self.model.transformer.h):
            self.register_save_hook(block, hook_name=f"block_{i}")
    
    def register_all_attentions(self):
        """
        Register hooks for all attention layers in the model.
        """
        for i, block in enumerate(self.model.transformer.h):
            self.register_save_hook(block.attn, hook_name=f"block_{i}.attn")
    
    def register_all_mlps(self):
        """
        Register hooks for all MLP layers in the model.
        """
        for i, block in enumerate(self.model.transformer.h):
            self.register_save_hook(block.mlp, hook_name=f"block_{i}.mlp")

    def clear_cache(self):
        """
        Clear the activations cache keeping the registered hooks intact.
        """
        self.activations.clear()
        print("[*] Activations cache cleared.")
    
    def remove_hooks(self):
        """
        Remove all registered hooks.
        """
        total = len(self.hook_handles)

        for handle in self.hook_handles:
            handle.remove()
        
        self.hook_handles.clear()
        self.activations.clear()

        if total > 0:
            print(f"[*] Removed {total} hooks and activations.")

    def print_activation_cache(self):
        """
        Print all cached activations.
        """
        print("\nActivation Cache")
        print("-" * 40)

        for name, tensor in self.activations.items():

            print(
                f"{name}"
                f" Shape={tuple(tensor.shape)}" # shape is [batch_size, seq_len, hidden_dim]
                f" Mean={tensor.mean():.4f}"
                f" Std={tensor.std():.4f}"
            )