"""
Utility functions for YOLOv10 model
"""

import torch
from pathlib import Path

def check_model_params(model, model_name="YOLOv10"):
    """
    Diagnose model parameters for potential issues.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for printing
    
    Returns:
        bool: True if no issues were found
    """
    issues_found = False
    
    print(f"==== {model_name} Parameter Diagnosis ====")
    
    # Check for nan or inf values in parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"WARNING: NaN values found in {name}")
            issues_found = True
        if torch.isinf(param).any():
            print(f"WARNING: Inf values found in {name}")
            issues_found = True
            
        # Check for extreme values
        with torch.no_grad():
            max_val = param.abs().max().item()
            if max_val > 1e3:
                print(f"CAUTION: Extreme values found in {name}, max abs = {max_val:.2e}")
                issues_found = True
    
    if not issues_found:
        print("No issues found in model parameters.")
    
    return not issues_found

def diagnose_nan_loss(loss_dict, inputs, targets, model):
    """
    Diagnose the source of NaN loss.
    
    Args:
        loss_dict: Dictionary of loss components
        inputs: Model inputs
        targets: Ground truth targets
        model: PyTorch model
    
    Returns:
        str: Diagnosis message
    """
    nan_components = []
    
    # Check loss components
    for name, value in loss_dict.items():
        if torch.isnan(value):
            nan_components.append(name)
    
    # Check input values
    input_has_nan = torch.isnan(inputs).any()
    input_has_inf = torch.isinf(inputs).any()
    
    # Check targets
    target_has_nan = False
    target_has_inf = False
    
    for t in targets:
        for k, v in t.items():
            if torch.isnan(v).any():
                target_has_nan = True
            if torch.isinf(v).any():
                target_has_inf = True
    
    # Check model parameters
    param_has_nan = False
    param_has_inf = False
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            param_has_nan = True
        if torch.isinf(param).any():
            param_has_inf = True
    
    # Generate diagnosis
    diagnosis = []
    if nan_components:
        diagnosis.append(f"NaN found in the following loss components: {', '.join(nan_components)}")
    if input_has_nan:
        diagnosis.append("NaN values found in model inputs")
    if input_has_inf:
        diagnosis.append("Inf values found in model inputs")
    if target_has_nan:
        diagnosis.append("NaN values found in targets")
    if target_has_inf:
        diagnosis.append("Inf values found in targets")
    if param_has_nan:
        diagnosis.append("NaN values found in model parameters")
    if param_has_inf:
        diagnosis.append("Inf values found in model parameters")
    
    if not diagnosis:
        diagnosis.append("No obvious issues found. The NaN could be from intermediate calculations.")
    
    return "\n".join(diagnosis)

def extract_model_from_checkpoint(checkpoint_path, output_path=None, variant='b'):
    """
    Extract only the model weights from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Path to save the extracted model weights 
        variant: Model variant (for filename if output_path is not specified)
        
    Returns:
        Path to the saved model weights
    """
    checkpoint_path = Path(checkpoint_path)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get model state dict from checkpoint
        model_state_dict = None
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            # Check if the checkpoint is already a state dict
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                model_state_dict = checkpoint
                
        if model_state_dict is None:
            print(f"No model weights found in {checkpoint_path}")
            return None
            
        # Check if variant info is available in the checkpoint
        detected_variant = None
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            if 'variant' in checkpoint['config']:
                detected_variant = checkpoint['config']['variant']
        elif 'variant' in checkpoint:
            detected_variant = checkpoint['variant']
            
        if detected_variant and detected_variant != variant:
            print(f"Warning: Checkpoint is for YOLOv10{detected_variant} but requested variant is YOLOv10{variant}")
            print(f"Using variant {variant} for output filename, but weights are for variant {detected_variant}")
            
        # Determine output path if not specified
        if output_path is None:
            output_dir = checkpoint_path.parent
            output_path = output_dir / f"YOLOv10{variant}_weights.pt"
        
        # Save the model weights
        torch.save(model_state_dict, output_path)
        print(f"Model weights extracted and saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error extracting model from checkpoint: {e}")
        return None 