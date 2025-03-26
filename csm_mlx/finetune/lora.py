"""LoRA (Low-Rank Adaptation) implementation for MLX models."""

import math
import os
from typing import Dict, List, Optional, Union, Any, Tuple

import mlx.core as mx
import mlx.nn as nn
from safetensors.numpy import load_file as load_safetensors


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        bias: bool = False,
        original_module: Optional[nn.Linear] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original weights, frozen during training
        if original_module is not None:
            self.weight = original_module.weight
            # Handle bias correctly based on the original module's parameters
            self.bias = getattr(original_module, "bias", None)
        else:
            self.weight = mx.zeros((out_features, in_features))
            self.bias = mx.zeros(out_features) if bias else None
        
        # LoRA matrices
        self.lora_A = mx.random.normal(
            (rank, in_features), scale=1.0 / math.sqrt(in_features)
        )
        self.lora_B = mx.zeros((out_features, rank))
        
        # Note: MLX doesn't use requires_grad attribute
        # Parameters are tracked automatically
            
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA adaptation."""
        # Original forward
        y = mx.matmul(x, self.weight.T)
        if self.bias is not None:
            y = y + self.bias
            
        # LoRA adaptation
        lora_output = mx.matmul(x, (self.lora_B @ self.lora_A).T) * self.scaling
        return y + lora_output
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into the original weight matrix."""
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.weight = self.weight + delta_weight


def add_lora_to_linear(
    module: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
) -> LoRALinear:
    """Replace a nn.Linear module with a LoRA-adapted version."""
    # Check if the module has a bias by inspecting its parameters
    has_bias = False
    try:
        # Check if the module has a bias parameter
        params = module.parameters()
        has_bias = len(params) > 1  # Typically weight and bias
    except:
        # If parameter inspection fails, assume no bias
        has_bias = False
    
    lora_layer = LoRALinear(
        in_features=module.weight.shape[1],
        out_features=module.weight.shape[0],
        rank=rank,
        alpha=alpha,
        bias=has_bias,
        original_module=module,
    )
    return lora_layer


def find_linear_layers(
    model: nn.Module,
    exclude_modules: Optional[List[str]] = None,
) -> Dict[str, nn.Linear]:
    """Find all Linear layers in a model.
    
    Args:
        model: Model to search
        exclude_modules: List of module names to exclude
        
    Returns:
        Dictionary mapping of parameter names to Linear layers
    """
    exclude_modules = exclude_modules or []
    linear_layers = {}
    
    for name, module in model.named_modules():
        if any(excluded in name for excluded in exclude_modules):
            continue
            
        if isinstance(module, nn.Linear):
            linear_layers[name] = module
            
    return linear_layers


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Apply LoRA to specific linear layers in a model.
    
    Args:
        model: Model to modify
        rank: Rank of LoRA matrices
        alpha: Scaling factor
        target_modules: Specific module names to target (if None, target all Linear layers)
        exclude_modules: Module names to exclude
        
    Returns:
        Model with LoRA applied to specified layers
    """
    linear_layers = find_linear_layers(model, exclude_modules)
    
    # Filter to target modules if specified
    if target_modules:
        linear_layers = {
            name: module 
            for name, module in linear_layers.items()
            if any(target in name for target in target_modules)
        }
    
    # Replace with LoRA layers
    for name, module in linear_layers.items():
        path = name.split('.')
        parent = model
        
        # Navigate to parent module
        for part in path[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # Replace the module
        if path[-1].isdigit():
            parent[int(path[-1])] = add_lora_to_linear(module, rank, alpha)
        else:
            setattr(parent, path[-1], add_lora_to_linear(module, rank, alpha))
    
    return model


def get_lora_params(model: nn.Module) -> List[mx.array]:
    """Get all LoRA parameters from a model."""
    lora_params = []
    
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend([module.lora_A, module.lora_B])
            
    return lora_params


def trainable_params(
    model: nn.Module,
    include_lora_only: bool = True
) -> Dict[str, mx.array]:
    """Get trainable parameters from a model.
    
    Args:
        model: Model to extract parameters from
        include_lora_only: If True, only include LoRA parameters
        
    Returns:
        Dictionary of trainable parameters
    """
    if include_lora_only:
        params = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                params[f"{name}.lora_A"] = module.lora_A
                params[f"{name}.lora_B"] = module.lora_B
        return params
    else:
        # In MLX, we need to manually filter which parameters to train
        # We'll exclude weights and biases from LoRA modules as they should be frozen
        all_params = model.parameters()
        trainable_params = {}
        
        for name, param in all_params.items():
            # Skip original weights in LoRA modules
            if any(name.startswith(f"{lora_name}.weight") or name.startswith(f"{lora_name}.bias") 
                   for lora_name, mod in model.named_modules() if isinstance(mod, LoRALinear)):
                continue
            trainable_params[name] = param
            
        return trainable_params


def load_lora_weights(
    model: nn.Module, 
    checkpoint_path: str,
    load_embeddings: bool = False,
) -> nn.Module:
    """Load LoRA weights and optionally embedding weights from a checkpoint file.
    
    Args:
        model: Model with LoRA layers
        checkpoint_path: Path to checkpoint file
        load_embeddings: Whether to load embedding weights if present
        
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Weights file not found: {checkpoint_path}")
        
    # Check file size as a basic validation
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1000:  # Less than 1KB is suspiciously small
        print(f"Warning: File size ({file_size} bytes) is suspiciously small for a weights file")
    
    # Get all LoRA modules
    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules[name] = module
    
    # Try different loading methods in sequence
    loading_methods = [
        # Method 1: Try direct MLX loading (weights method)
        lambda: (
            print("Trying MLX weights loading..."),
            nn.load_weights(model, checkpoint_path)
        ),
        
        # Method 2: Try direct MLX state dict loading
        lambda: (
            print("Trying MLX state dict loading..."),
            mx.load(checkpoint_path)
        ),
        
        # Method 3: Try safetensors loading
        lambda: (
            print("Trying safetensors loading..."),
            load_safetensors(checkpoint_path)
        ),
    ]
    
    # Try each loading method in sequence
    all_weights = None
    last_error = None
    
    for method in loading_methods:
        try:
            _, all_weights = method()
            if all_weights:  # If we got any weights, break the loop
                break
        except Exception as e:
            print(f"Loading method failed: {str(e)}")
            last_error = e
            continue
    
    # If all methods failed, raise the last error
    if all_weights is None:
        raise RuntimeError(f"All loading methods failed. Last error: {str(last_error)}")
        
    # Check if we got any weights
    if not all_weights:
        raise ValueError(f"No weights found in file: {checkpoint_path}")
        
    print(f"Successfully loaded {len(all_weights)} parameters")
    
    # Separate LoRA weights and embedding weights
    lora_weights = {k: v for k, v in all_weights.items() if ".lora_A" in k or ".lora_B" in k}
    embedding_weights = {k: v for k, v in all_weights.items() 
                        if "text_embeddings" in k or "audio_embeddings" in k}
    
    print(f"Found {len(lora_weights)} LoRA parameters and {len(embedding_weights)} embedding parameters")
    
    # Load LoRA weights into modules
    for key, value in lora_weights.items():
        module_name = key.rsplit(".", 1)[0]
        param_name = key.rsplit(".", 1)[1]
        
        if module_name in lora_modules:
            # Convert to MLX array if needed
            if not isinstance(value, mx.array):
                value = mx.array(value)
                
            setattr(lora_modules[module_name], param_name, value)
            print(f"Loaded {param_name} for {module_name} with shape {value.shape}")
    
    # If no LoRA parameters found, warn the user
    if not lora_weights:
        print("Warning: No LoRA parameters found in the checkpoint")
    
    # Load embedding weights if requested
    if load_embeddings and embedding_weights:
        all_params = model.parameters()
        for key, value in embedding_weights.items():
            if key in all_params:
                # Convert to MLX array if needed
                if not isinstance(value, mx.array):
                    value = mx.array(value)
                
                # Find the parameter in the model and update it
                path = key.split('.')
                current = model
                
                # Navigate to the parameter's parent module
                for part in path[:-1]:
                    if part.isdigit():
                        current = current[int(part)]
                    else:
                        current = getattr(current, part)
                
                # Update the parameter
                if path[-1].isdigit():
                    current[int(path[-1])] = value
                else:
                    setattr(current, path[-1], value)
                    
                print(f"Loaded embedding parameter {key} with shape {value.shape}")
    
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the original weights.
    
    This is useful for inference, as it removes the LoRA overhead.
    Note: Directly trained embedding weights (if using --train-embeddings)
    are already in their final form and don't need merging.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Model with merged weights (LoRA removed)
    """
    # Find all LoRA modules
    lora_modules_merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Merge weights
            module.merge_weights()
            lora_modules_merged += 1
    
    if lora_modules_merged > 0:
        print(f"Merged weights for {lora_modules_merged} LoRA modules")
    else:
        print("No LoRA modules found to merge")
    
    # Check for embedding parameters
    embedding_params = 0
    all_params = model.parameters()
    for name in all_params:
        if "text_embeddings" in name or "audio_embeddings" in name:
            embedding_params += 1
    
    if embedding_params > 0:
        print(f"Found {embedding_params} directly trained embedding parameters (already applied)")
    
    # Now replace LoRA modules with standard Linear modules
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            path = name.split('.')
            parent = model
            
            # Navigate to parent module
            for part in path[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            # Create a new Linear layer with merged weights
            has_bias = module.bias is not None
            linear = nn.Linear(
                module.in_features, 
                module.out_features,
                bias=has_bias
            )
            linear.weight = module.weight
            if has_bias:
                linear.bias = module.bias
            
            # Replace the LoRA module
            if path[-1].isdigit():
                parent[int(path[-1])] = linear
            else:
                setattr(parent, path[-1], linear)
    
    return model 