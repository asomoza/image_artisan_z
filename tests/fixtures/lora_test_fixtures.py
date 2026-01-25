"""Test fixtures for LoRA spatial masking tests."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class DummyLoRALayer(nn.Module):
    """Dummy LoRA layer for testing patching.

    Simulates a simplified LoRA adapter with down-projection (lora_A)
    and up-projection (lora_B) matrices.
    """

    def __init__(self, in_features: int = 768, out_features: int = 768, rank: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = 1.0

        # Initialize with small values for predictable test outputs
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.normal_(self.lora_B.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate LoRA: down-project -> up-project."""
        return self.lora_B(self.lora_A(x)) * self.scaling


class DummyTransformerWithLoRA(nn.Module):
    """Dummy transformer with LoRA adapters for testing.

    Simulates a simplified transformer model with multiple LoRA-equipped
    layers, mimicking PEFT's adapter structure.
    """

    def __init__(self, num_layers: int = 4, hidden_dim: int = 768, rank: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Create LoRA layers for each transformer layer
        self.layers = nn.ModuleList([
            DummyLoRALayer(hidden_dim, hidden_dim, rank)
            for _ in range(num_layers)
        ])

        # Simulate PEFT's adapter tracking
        self.peft_config = {}
        self._adapter_names: List[str] = []
        self._active_adapters: List[str] = []

    def set_adapters(self, adapter_names: List[str], adapter_weights) -> None:
        """Simulate PEFT's set_adapters method.

        Args:
            adapter_names: List of adapter names to activate
            adapter_weights: Weights for the adapters (can be single value or list)
        """
        self._active_adapters = adapter_names
        # In real PEFT, this would activate adapters with specified weights

    def get_lora_layers(self, adapter_name: str) -> Dict[str, DummyLoRALayer]:
        """Get all LoRA layers for an adapter.

        Args:
            adapter_name: Name of the adapter (not used in dummy, but kept for API)

        Returns:
            Dict mapping layer names to layer modules
        """
        return {
            f"layer_{i}": layer
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            hidden_states: Input tensor [B, N, D]

        Returns:
            Output tensor [B, N, D] after all LoRA layers
        """
        x = hidden_states
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return x


def create_dummy_spatial_mask(
    height: int = 64,
    width: int = 64,
    painted_region: str = "left_half"
) -> torch.Tensor:
    """Create test spatial mask with specific patterns.

    Args:
        height: Mask height in latent space
        width: Mask width in latent space
        painted_region: Pattern - one of:
            - "left_half": Left half is 1.0, right half is 0.0
            - "right_half": Right half is 1.0, left half is 0.0
            - "top_half": Top half is 1.0, bottom half is 0.0
            - "bottom_half": Bottom half is 1.0, top half is 0.0
            - "center": Center quadrant is 1.0, edges are 0.0
            - "edges": Edges are 1.0, center is 0.0
            - "all_ones": Entire mask is 1.0
            - "all_zeros": Entire mask is 0.0
            - "checkerboard": Alternating 1.0 and 0.0 in checkerboard pattern
            - "gradient_h": Horizontal gradient from 0.0 to 1.0
            - "gradient_v": Vertical gradient from 0.0 to 1.0

    Returns:
        Mask tensor [1, 1, H, W] with values 0.0 to 1.0
    """
    mask = torch.zeros(1, 1, height, width)

    if painted_region == "left_half":
        mask[:, :, :, :width // 2] = 1.0
    elif painted_region == "right_half":
        mask[:, :, :, width // 2:] = 1.0
    elif painted_region == "top_half":
        mask[:, :, :height // 2, :] = 1.0
    elif painted_region == "bottom_half":
        mask[:, :, height // 2:, :] = 1.0
    elif painted_region == "center":
        h_start, h_end = height // 4, 3 * height // 4
        w_start, w_end = width // 4, 3 * width // 4
        mask[:, :, h_start:h_end, w_start:w_end] = 1.0
    elif painted_region == "edges":
        mask[:, :, :, :] = 1.0
        h_start, h_end = height // 4, 3 * height // 4
        w_start, w_end = width // 4, 3 * width // 4
        mask[:, :, h_start:h_end, w_start:w_end] = 0.0
    elif painted_region == "all_ones":
        mask[:, :, :, :] = 1.0
    elif painted_region == "all_zeros":
        pass  # Already zeros
    elif painted_region == "checkerboard":
        for i in range(height):
            for j in range(width):
                if (i + j) % 2 == 0:
                    mask[:, :, i, j] = 1.0
    elif painted_region == "gradient_h":
        for j in range(width):
            mask[:, :, :, j] = j / (width - 1)
    elif painted_region == "gradient_v":
        for i in range(height):
            mask[:, :, i, :] = i / (height - 1)

    return mask


def create_dummy_hidden_states(
    batch_size: int = 1,
    seq_len: int = 4096,
    hidden_dim: int = 768,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create dummy hidden states for LoRA forward pass.

    Args:
        batch_size: Batch size
        seq_len: Sequence length (H*W for images, e.g., 64*64=4096)
        hidden_dim: Hidden dimension
        device: Target device (default: CPU)
        dtype: Data type

    Returns:
        Hidden states tensor [B, N, D]
    """
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_dim,
        device=device, dtype=dtype
    )
    return hidden_states


def create_multi_resolution_hidden_states(
    batch_size: int = 1,
    resolutions: List[Tuple[int, int]] = None,
    hidden_dim: int = 768
) -> Dict[str, torch.Tensor]:
    """Create hidden states at multiple resolutions.

    This simulates different transformer layers operating at different
    spatial resolutions (like in a multi-scale architecture).

    Args:
        batch_size: Batch size
        resolutions: List of (H, W) tuples. Default: [(16, 16), (32, 32), (64, 64)]
        hidden_dim: Hidden dimension

    Returns:
        Dict mapping resolution key to hidden states tensor
    """
    if resolutions is None:
        resolutions = [(16, 16), (32, 32), (64, 64)]

    result = {}
    for h, w in resolutions:
        key = f"{h}x{w}"
        seq_len = h * w
        result[key] = torch.randn(batch_size, seq_len, hidden_dim)

    return result


class MockPEFTModel:
    """Mock PEFT model for testing adapter operations.

    Simulates the structure of a PEFT-wrapped model with named modules
    following PEFT's naming conventions.
    """

    def __init__(self, base_transformer: DummyTransformerWithLoRA):
        self.base_model = base_transformer
        self._loaded_adapters: Dict[str, Dict[str, nn.Module]] = {}

    def load_adapter(self, adapter_name: str, path: str = "") -> None:
        """Simulate loading an adapter.

        Args:
            adapter_name: Name for the adapter
            path: Path to adapter files (unused in mock)
        """
        # Create LoRA layers for this adapter
        self._loaded_adapters[adapter_name] = {
            f"base_model.model.layers.{i}.lora_{adapter_name}": DummyLoRALayer()
            for i in range(self.base_model.num_layers)
        }

    def named_modules(self):
        """Yield named modules including LoRA layers."""
        for adapter_name, layers in self._loaded_adapters.items():
            for name, module in layers.items():
                yield name, module

    def set_adapters(self, adapter_names: List[str], weights) -> None:
        """Set active adapters with weights."""
        self.base_model.set_adapters(adapter_names, weights)


def verify_spatial_masking(
    output: torch.Tensor,
    mask: torch.Tensor,
    spatial_dims: Tuple[int, int],
    tolerance: float = 0.01
) -> Tuple[bool, str]:
    """Verify that spatial masking was applied correctly.

    Args:
        output: Output tensor [B, N, D] from masked forward pass
        mask: Original mask [1, 1, H, W] used for masking
        spatial_dims: (H, W) spatial dimensions
        tolerance: Tolerance for numerical comparison

    Returns:
        Tuple of (passed: bool, message: str)
    """
    H, W = spatial_dims
    B, N, D = output.shape

    if N != H * W:
        return False, f"Sequence length {N} doesn't match spatial dims {H}x{W}={H * W}"

    # Reshape output to spatial: [B, N, D] -> [B, H, W, D]
    spatial_output = output.view(B, H, W, D)

    # Resize mask to match spatial dims if needed
    if mask.shape[-2:] != (H, W):
        mask = torch.nn.functional.interpolate(
            mask, size=(H, W), mode='bilinear', align_corners=False
        )

    # Check that masked regions (mask=0) have zero output
    mask_flat = mask.view(1, H, W, 1).expand(B, H, W, D)
    masked_regions = spatial_output[mask_flat < 0.5]
    if masked_regions.numel() > 0:
        max_in_masked = masked_regions.abs().max().item()
        if max_in_masked > tolerance:
            return False, f"Found non-zero values ({max_in_masked}) in masked regions"

    # Check that unmasked regions (mask=1) have non-zero output (probabilistically)
    unmasked_regions = spatial_output[mask_flat > 0.5]
    if unmasked_regions.numel() > 0:
        if torch.all(unmasked_regions.abs() < tolerance):
            return False, "All values in unmasked regions are zero"

    return True, "Spatial masking verified correctly"
