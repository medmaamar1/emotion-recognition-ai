"""
Enhanced Grad-CAM and XAI utilities for model interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Args:
        model: PyTorch model
        target_layer: Target layer for Grad-CAM
        use_cuda: Whether to use CUDA (default: True)
    """
    def __init__(self, model, target_layer, use_cuda=True):
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
        if self.use_cuda:
            self.model.cuda()
    
    def save_activation(self, module, input, output):
        """Save activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients."""
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x: Input image tensor
            target_class: Target class (default: None, uses predicted class)
        
        Returns:
            Heatmap and predicted class
        """
        self.model.eval()
        
        if self.use_cuda:
            x = x.cuda()
        
        # Forward pass
        output = self.model(x)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate heatmap
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        return heatmap, target_class
    
    def visualize(self, heatmap, original_image, save_path='gradcam.png'):
        """
        Visualize Grad-CAM heatmap.
        
        Args:
            heatmap: Grad-CAM heatmap
            original_image: Original image (numpy array or tensor)
            save_path: Path to save visualization (default: 'gradcam.png')
        
        Returns:
            Superimposed image
        """
        # Convert tensor to numpy if needed
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = original_image * std + mean
            original_image = np.clip(original_image, 0, 1)
            original_image = (original_image * 255).astype(np.uint8)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        # Save
        cv2.imwrite(save_path, superimposed)
        
        return superimposed


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ for better localization.
    
    Args:
        model: PyTorch model
        target_layer: Target layer for Grad-CAM++
        use_cuda: Whether to use CUDA (default: True)
    """
    def __call__(self, x, target_class=None):
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            x: Input image tensor
            target_class: Target class (default: None)
        
        Returns:
            Heatmap and predicted class
        """
        self.model.eval()
        
        if self.use_cuda:
            x = x.cuda()
        
        # Forward pass
        output = self.model(x)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Compute Grad-CAM++ weights
        alpha = np.power(gradients, 2)
        weights = np.sum(alpha, axis=(1, 2))
        weights = np.maximum(weights, 1e-10)
        
        # Normalize gradients
        gradients = gradients / weights[:, np.newaxis, np.newaxis]
        
        # Compute weights
        weights = np.sum(gradients * activations, axis=(1, 2))
        
        # Generate heatmap
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        return heatmap, target_class


class LayerGradCAM:
    """
    Layer-wise Grad-CAM for multi-scale visualization.
    
    Args:
        model: PyTorch model
        target_layers: List of target layers
        use_cuda: Whether to use CUDA (default: True)
    """
    def __init__(self, model, target_layers, use_cuda=True):
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.grad_cams = []
        for layer in target_layers:
            grad_cam = GradCAM(model, layer, use_cuda)
            self.grad_cams.append(grad_cam)
    
    def __call__(self, x, target_class=None):
        """
        Generate multi-scale Grad-CAM heatmaps.
        
        Args:
            x: Input image tensor
            target_class: Target class (default: None)
        
        Returns:
            List of heatmaps
        """
        heatmaps = []
        for grad_cam in self.grad_cams:
            heatmap, _ = grad_cam(x, target_class)
            heatmaps.append(heatmap)
        
        return heatmaps
    
    def visualize_multi_scale(self, heatmaps, original_image, save_path='layer_gradcam.png'):
        """
        Visualize multi-scale Grad-CAM.
        
        Args:
            heatmaps: List of heatmaps
            original_image: Original image
            save_path: Path to save visualization
        
        Returns:
            Figure
        """
        # Convert tensor to numpy if needed
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = original_image * std + mean
            original_image = np.clip(original_image, 0, 1)
            original_image = (original_image * 255).astype(np.uint8)
        
        # Create subplots
        n_layers = len(heatmaps)
        fig, axes = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmaps
        for i, heatmap in enumerate(heatmaps):
            # Resize heatmap
            heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            
            # Superimpose
            superimposed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
            
            axes[i + 1].imshow(superimposed)
            axes[i + 1].set_title(f'Layer {i + 1}')
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig


class ScoreCAM:
    """
    Score-CAM: Gradient-free class activation mapping.
    
    Args:
        model: PyTorch model
        target_layer: Target layer for Score-CAM
        use_cuda: Whether to use CUDA (default: True)
    """
    def __init__(self, model, target_layer, use_cuda=True):
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        
        if self.use_cuda:
            self.model.cuda()
    
    def save_activation(self, module, input, output):
        """Save activations."""
        self.activations = output.detach()
    
    def __call__(self, x, target_class=None):
        """
        Generate Score-CAM heatmap.
        
        Args:
            x: Input image tensor
            target_class: Target class (default: None)
        
        Returns:
            Heatmap and predicted class
        """
        self.model.eval()
        
        if self.use_cuda:
            x = x.cuda()
        
        # Get activations
        activations = self.activations.cpu().data.numpy()[0]
        
        # Create masks for each activation map
        num_maps = activations.shape[0]
        heatmaps = []
        
        for i in range(num_maps):
            # Create mask from activation map
            mask = activations[i:i+1]
            mask = torch.from_numpy(mask).float()
            
            if self.use_cuda:
                mask = mask.cuda()
            
            # Resize mask to input size
            mask = F.interpolate(mask.unsqueeze(0), size=x.shape[2:], mode='bilinear', align_corners=False)
            mask = mask.squeeze(0)
            
            # Apply mask to input
            masked_input = x * mask
            
            # Forward pass with masked input
            output = self.model(masked_input)
            
            # Get score for target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            score = output[0, target_class].item()
            heatmaps.append(score * activations[i])
        
        # Combine heatmaps
        heatmap = np.sum(heatmaps, axis=0)
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        return heatmap, target_class


class IntegratedGradients:
    """
    Integrated Gradients for attribution.
    
    Args:
        model: PyTorch model
        use_cuda: Whether to use CUDA (default: True)
        n_steps: Number of integration steps (default: 50)
    """
    def __init__(self, model, use_cuda=True, n_steps=50):
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.n_steps = n_steps
    
    def __call__(self, x, target_class=None):
        """
        Generate integrated gradients attribution.
        
        Args:
            x: Input image tensor
            target_class: Target class (default: None)
        
        Returns:
            Attribution map
        """
        self.model.eval()
        
        if self.use_cuda:
            x = x.cuda()
        
        # Get baseline (black image)
        baseline = torch.zeros_like(x)
        
        if self.use_cuda:
            baseline = baseline.cuda()
        
        # Get prediction for target class
        output = self.model(x)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Compute integrated gradients
        alphas = torch.linspace(0, 1, self.n_steps)
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad = True
            
            if self.use_cuda:
                interpolated = interpolated.cuda()
            
            # Forward pass
            output = self.model(interpolated)
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward(retain_graph=True)
            
            # Get gradient
            gradient = interpolated.grad.data.cpu().numpy()[0]
            gradients.append(gradient)
        
        # Average gradients
        gradients = np.array(gradients)
        avg_gradients = np.mean(gradients, axis=0)
        
        # Compute attribution
        attribution = (x.cpu().numpy()[0] - baseline.cpu().numpy()[0]) * avg_gradients
        
        # Sum across channels
        attribution = np.sum(attribution, axis=0)
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)
        
        return attribution


def generate_xai_report(model, image, target_layers, save_dir='./xai_report'):
    """
    Generate comprehensive XAI report.
    
    Args:
        model: PyTorch model
        image: Input image tensor
        target_layers: List of target layers
        save_dir: Directory to save report (default: './xai_report')
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Create XAI methods
    grad_cam = GradCAM(model, target_layers[0])
    grad_cam_plus = GradCAMPlusPlus(model, target_layers[0])
    layer_grad_cam = LayerGradCAM(model, target_layers)
    score_cam = ScoreCAM(model, target_layers[0])
    integrated_grad = IntegratedGradients(model)
    
    # Generate heatmaps
    print("Generating Grad-CAM...")
    heatmap, pred_class = grad_cam(image)
    grad_cam.visualize(heatmap, image, 
                       save_path=os.path.join(save_dir, 'gradcam.png'))
    
    print("Generating Grad-CAM++...")
    heatmap_plus, _ = grad_cam_plus(image)
    grad_cam.visualize(heatmap_plus, image, 
                       save_path=os.path.join(save_dir, 'gradcam_plus.png'))
    
    print("Generating Layer-wise Grad-CAM...")
    heatmaps = layer_grad_cam(image)
    layer_grad_cam.visualize_multi_scale(heatmaps, image,
                                        save_path=os.path.join(save_dir, 'layer_gradcam.png'))
    
    print("Generating Score-CAM...")
    heatmap_score, _ = score_cam(image)
    grad_cam.visualize(heatmap_score, image,
                       save_path=os.path.join(save_dir, 'scorecam.png'))
    
    print("Generating Integrated Gradients...")
    attribution = integrated_grad(image)
    
    # Visualize integrated gradients
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = image
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attribution
    axes[1].imshow(attribution, cmap='hot')
    axes[1].set_title('Integrated Gradients')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'integrated_gradients.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"XAI report saved to: {save_dir}")


if __name__ == "__main__":
    # Test XAI utilities
    print("Testing enhanced XAI utilities...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 14)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = DummyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy image
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Test GradCAM
    print("\nTesting GradCAM...")
    grad_cam = GradCAM(model, model.conv1, use_cuda=False)
    heatmap, pred_class = grad_cam(dummy_image)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Predicted class: {pred_class}")
    
    # Test GradCAMPlusPlus
    print("\nTesting GradCAMPlusPlus...")
    grad_cam_plus = GradCAMPlusPlus(model, model.conv1, use_cuda=False)
    heatmap_plus, _ = grad_cam_plus(dummy_image)
    print(f"Heatmap shape: {heatmap_plus.shape}")
    
    # Test LayerGradCAM
    print("\nTesting LayerGradCAM...")
    layer_grad_cam = LayerGradCAM(model, [model.conv1, model.conv2], use_cuda=False)
    heatmaps = layer_grad_cam(dummy_image)
    print(f"Number of heatmaps: {len(heatmaps)}")
    print(f"Heatmap shapes: {[h.shape for h in heatmaps]}")
    
    # Test ScoreCAM
    print("\nTesting ScoreCAM...")
    score_cam = ScoreCAM(model, model.conv1, use_cuda=False)
    heatmap_score, _ = score_cam(dummy_image)
    print(f"Heatmap shape: {heatmap_score.shape}")
    
    # Test IntegratedGradients
    print("\nTesting IntegratedGradients...")
    integrated_grad = IntegratedGradients(model, use_cuda=False, n_steps=10)
    attribution = integrated_grad(dummy_image)
    print(f"Attribution shape: {attribution.shape}")
    
    print("\nAll enhanced XAI utilities tested successfully!")
