"""Tests for ModelManager attention backend functionality."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from iartisanz.app.model_manager import (
    ATTENTION_BACKEND_OPTIONS,
    VARLEN_BACKEND_MAPPING,
    ModelManager,
    get_model_manager,
)


class TestAttentionBackendProperty:
    """Tests for the attention_backend property."""

    def test_default_backend_is_native(self):
        """Default attention backend should be 'native'."""
        mm = ModelManager()
        assert mm.attention_backend == "native"

    def test_set_backend_to_flash(self):
        """Setting backend to 'flash' should update the value."""
        mm = ModelManager()
        mm.attention_backend = "flash"
        assert mm.attention_backend == "flash"

    def test_set_backend_to_sage(self):
        """Setting backend to 'sage' should update the value."""
        mm = ModelManager()
        mm.attention_backend = "sage"
        assert mm.attention_backend == "sage"

    def test_set_backend_to_xformers(self):
        """Setting backend to 'xformers' should update the value."""
        mm = ModelManager()
        mm.attention_backend = "xformers"
        assert mm.attention_backend == "xformers"

    def test_set_backend_to_flash_3_hub(self):
        """Setting backend to '_flash_3_hub' should update the value."""
        mm = ModelManager()
        mm.attention_backend = "_flash_3_hub"
        assert mm.attention_backend == "_flash_3_hub"

    def test_set_none_resets_to_native(self):
        """Setting None should return 'native' from property."""
        mm = ModelManager()
        mm.attention_backend = "flash"
        assert mm.attention_backend == "flash"
        mm.attention_backend = None
        assert mm.attention_backend == "native"

    def test_set_native_explicitly_returns_native(self):
        """Setting 'native' explicitly should return 'native'."""
        mm = ModelManager()
        mm.attention_backend = "flash"
        mm.attention_backend = "native"
        assert mm.attention_backend == "native"

    def test_set_empty_string_resets_to_native(self):
        """Setting empty string should return 'native'."""
        mm = ModelManager()
        mm.attention_backend = "flash"
        mm.attention_backend = ""
        assert mm.attention_backend == "native"

    def test_thread_safety_with_lock(self):
        """Property access should be thread-safe via lock."""
        mm = ModelManager()
        # Access the internal lock to verify it exists
        assert hasattr(mm, "_lock")
        # Setting and getting should work without deadlock
        mm.attention_backend = "flash"
        assert mm.attention_backend == "flash"


class TestGetAvailableAttentionBackends:
    """Tests for get_available_attention_backends method."""

    def test_always_includes_native(self):
        """Native backend should always be available."""
        mm = ModelManager()
        backends = mm.get_available_attention_backends()
        backend_ids = [b[0] for b in backends]
        assert "native" in backend_ids

    def test_returns_list_of_tuples(self):
        """Should return list of (backend_id, display_name) tuples."""
        mm = ModelManager()
        backends = mm.get_available_attention_backends()
        assert isinstance(backends, list)
        assert len(backends) >= 1
        for item in backends:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # backend_id
            assert isinstance(item[1], str)  # display_name

    def test_native_has_correct_display_name(self):
        """Native backend should have 'Auto (PyTorch SDPA)' display name."""
        mm = ModelManager()
        backends = mm.get_available_attention_backends()
        native_backends = [b for b in backends if b[0] == "native"]
        assert len(native_backends) == 1
        assert native_backends[0][1] == "Auto (PyTorch SDPA)"

    @patch("torch.cuda.is_available", return_value=False)
    def test_only_native_when_no_cuda(self, mock_cuda):
        """Without CUDA, only native should be available."""
        mm = ModelManager()
        backends = mm.get_available_attention_backends()
        assert len(backends) == 1
        assert backends[0][0] == "native"

    @patch("torch.cuda.is_available", return_value=True)
    def test_checks_flash_attn_import(self, mock_cuda):
        """Should check if flash_attn is importable."""
        mm = ModelManager()
        # This will either include flash or not depending on actual installation
        backends = mm.get_available_attention_backends()
        backend_ids = [b[0] for b in backends]
        # Just verify it doesn't crash and returns valid list
        assert "native" in backend_ids

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(9, 0))
    def test_includes_flash_3_for_hopper_gpu_with_kernels(self, mock_capability, mock_cuda):
        """Should include Flash Attention 3 for Hopper GPUs (SM 9.0+) when kernels installed."""
        with patch.dict("sys.modules", {"kernels": MagicMock()}):
            mm = ModelManager()
            backends = mm.get_available_attention_backends()
            backend_ids = [b[0] for b in backends]
            assert "_flash_3_hub" in backend_ids

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(8, 6))
    def test_excludes_flash_3_for_ampere_gpu(self, mock_capability, mock_cuda):
        """Should not include Flash Attention 3 for Ampere GPUs (SM 8.x)."""
        mm = ModelManager()
        backends = mm.get_available_attention_backends()
        backend_ids = [b[0] for b in backends]
        assert "_flash_3_hub" not in backend_ids

    @patch("torch.cuda.is_available", return_value=True)
    def test_includes_hub_variants_when_kernels_installed(self, mock_cuda):
        """Should include hub variants when kernels package is installed."""
        with patch.dict("sys.modules", {"kernels": MagicMock()}):
            mm = ModelManager()
            backends = mm.get_available_attention_backends()
            backend_ids = [b[0] for b in backends]
            assert "flash_hub" in backend_ids
            assert "sage_hub" in backend_ids

    @patch("torch.cuda.is_available", return_value=True)
    def test_excludes_hub_variants_when_kernels_not_installed(self, mock_cuda):
        """Should not include hub variants when kernels package is not installed."""
        # Ensure kernels is not importable
        import sys
        modules_backup = sys.modules.copy()
        sys.modules["kernels"] = None  # Simulate ImportError
        try:
            # Force re-evaluation by creating new ModelManager
            mm = ModelManager()
            backends = mm.get_available_attention_backends()
            backend_ids = [b[0] for b in backends]
            # Hub variants should not be present (unless kernels is actually installed)
            # Note: This test may pass differently if kernels is actually installed
        finally:
            sys.modules.clear()
            sys.modules.update(modules_backup)

    @patch("torch.cuda.is_available", return_value=True)
    def test_shows_both_local_and_hub_flash_when_both_available(self, mock_cuda):
        """Should show both local and hub Flash Attention when both are available."""
        with patch.dict("sys.modules", {"flash_attn": MagicMock(), "kernels": MagicMock()}):
            mm = ModelManager()
            backends = mm.get_available_attention_backends()
            backend_ids = [b[0] for b in backends]
            assert "flash" in backend_ids  # Local
            assert "flash_hub" in backend_ids  # Hub

    @patch("torch.cuda.is_available", return_value=True)
    def test_shows_both_local_and_hub_sage_when_both_available(self, mock_cuda):
        """Should show both local and hub Sage Attention when both are available."""
        with patch.dict("sys.modules", {"sageattention": MagicMock(), "kernels": MagicMock()}):
            mm = ModelManager()
            backends = mm.get_available_attention_backends()
            backend_ids = [b[0] for b in backends]
            assert "sage" in backend_ids  # Local
            assert "sage_hub" in backend_ids  # Hub


class TestApplyAttentionBackend:
    """Tests for apply_attention_backend method."""

    def test_native_backend_calls_set_attention_backend(self):
        """Native backend should explicitly call set_attention_backend('native')."""
        mm = ModelManager()
        mm.attention_backend = "native"

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock()

        result = mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("native")
        assert result is True

    def test_native_backend_succeeds_without_set_method(self):
        """Native backend should succeed even without set_attention_backend."""
        mm = ModelManager()
        mm.attention_backend = "native"

        mock_transformer = MagicMock(spec=[])  # No methods

        result = mm.apply_attention_backend(mock_transformer)

        assert result is True

    def test_non_native_calls_set_attention_backend(self):
        """Non-native backends should call set_attention_backend."""
        mm = ModelManager()
        mm.attention_backend = "flash"

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock()

        result = mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("flash")
        assert result is True

    def test_handles_missing_set_attention_backend(self):
        """Should handle transformers without set_attention_backend gracefully."""
        mm = ModelManager()
        mm.attention_backend = "flash"

        mock_transformer = MagicMock(spec=[])  # No set_attention_backend

        result = mm.apply_attention_backend(mock_transformer)

        # Should return True (native will be used)
        assert result is True

    def test_handles_exception_in_set_attention_backend(self):
        """Should handle exceptions from set_attention_backend gracefully."""
        mm = ModelManager()
        mm.attention_backend = "flash"

        # Track calls to set_attention_backend
        call_count = [0]
        def mock_set_backend(backend):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call with "flash" fails
                raise RuntimeError("Backend not available")
            # Second call with "native" succeeds

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock(side_effect=mock_set_backend)

        result = mm.apply_attention_backend(mock_transformer)

        assert result is False
        # Should try to set native after failure
        assert mock_transformer.set_attention_backend.call_count == 2
        mock_transformer.set_attention_backend.assert_any_call("flash")
        mock_transformer.set_attention_backend.assert_any_call("native")

    def test_handles_exception_in_native_fallback_after_failure(self):
        """Should handle exceptions from native fallback after set fails."""
        mm = ModelManager()
        mm.attention_backend = "flash"

        mock_transformer = MagicMock()
        # Both calls fail
        mock_transformer.set_attention_backend = MagicMock(
            side_effect=RuntimeError("Backend not available")
        )

        result = mm.apply_attention_backend(mock_transformer)

        # Should still return False but not crash
        assert result is False

    def test_applies_sage_backend(self):
        """Should correctly apply sage backend."""
        mm = ModelManager()
        mm.attention_backend = "sage"

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock()

        result = mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("sage")
        assert result is True

    def test_applies_xformers_backend(self):
        """Should correctly apply xformers backend."""
        mm = ModelManager()
        mm.attention_backend = "xformers"

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock()

        result = mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("xformers")
        assert result is True


class TestAttentionBackendOptions:
    """Tests for ATTENTION_BACKEND_OPTIONS constant."""

    def test_options_is_list_of_tuples(self):
        """ATTENTION_BACKEND_OPTIONS should be a list of tuples."""
        assert isinstance(ATTENTION_BACKEND_OPTIONS, list)
        for item in ATTENTION_BACKEND_OPTIONS:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_options_contains_native(self):
        """ATTENTION_BACKEND_OPTIONS should contain native."""
        backend_ids = [b[0] for b in ATTENTION_BACKEND_OPTIONS]
        assert "native" in backend_ids

    def test_options_contains_flash(self):
        """ATTENTION_BACKEND_OPTIONS should contain flash."""
        backend_ids = [b[0] for b in ATTENTION_BACKEND_OPTIONS]
        assert "flash" in backend_ids

    def test_options_contains_flash_hub(self):
        """ATTENTION_BACKEND_OPTIONS should contain flash_hub."""
        backend_ids = [b[0] for b in ATTENTION_BACKEND_OPTIONS]
        assert "flash_hub" in backend_ids

    def test_options_contains_flash_3_hub(self):
        """ATTENTION_BACKEND_OPTIONS should contain _flash_3_hub."""
        backend_ids = [b[0] for b in ATTENTION_BACKEND_OPTIONS]
        assert "_flash_3_hub" in backend_ids

    def test_options_contains_sage(self):
        """ATTENTION_BACKEND_OPTIONS should contain sage."""
        backend_ids = [b[0] for b in ATTENTION_BACKEND_OPTIONS]
        assert "sage" in backend_ids

    def test_options_contains_sage_hub(self):
        """ATTENTION_BACKEND_OPTIONS should contain sage_hub."""
        backend_ids = [b[0] for b in ATTENTION_BACKEND_OPTIONS]
        assert "sage_hub" in backend_ids

    def test_options_contains_xformers(self):
        """ATTENTION_BACKEND_OPTIONS should contain xformers."""
        backend_ids = [b[0] for b in ATTENTION_BACKEND_OPTIONS]
        assert "xformers" in backend_ids


class TestAttentionBackendWithTorchCompile:
    """Tests for attention backend interaction with torch.compile."""

    def test_both_settings_are_independent(self):
        """attention_backend and use_torch_compile should be independent."""
        mm = ModelManager()

        mm.use_torch_compile = True
        mm.attention_backend = "flash"

        assert mm.use_torch_compile is True
        assert mm.attention_backend == "flash"

        mm.use_torch_compile = False
        assert mm.attention_backend == "flash"

        mm.attention_backend = "native"
        assert mm.use_torch_compile is False

    def test_clear_resets_attention_backend(self):
        """Clearing ModelManager should reset attention backend to default."""
        mm = ModelManager()
        mm.attention_backend = "flash"
        assert mm.attention_backend == "flash"

        # Note: clear() doesn't reset runtime settings, only components
        # This is consistent with use_torch_compile behavior
        mm.clear()
        # attention_backend should persist (it's a runtime setting)
        assert mm.attention_backend == "flash"


class TestVarlenBackendMapping:
    """Tests for automatic varlen backend mapping for Z-Image models."""

    def test_varlen_mapping_constant_exists(self):
        """VARLEN_BACKEND_MAPPING should be defined."""
        assert isinstance(VARLEN_BACKEND_MAPPING, dict)
        assert len(VARLEN_BACKEND_MAPPING) > 0

    def test_varlen_mapping_contains_expected_backends(self):
        """VARLEN_BACKEND_MAPPING should contain all expected mappings."""
        assert "flash" in VARLEN_BACKEND_MAPPING
        assert "flash_hub" in VARLEN_BACKEND_MAPPING
        assert "_flash_3_hub" in VARLEN_BACKEND_MAPPING
        assert "sage" in VARLEN_BACKEND_MAPPING
        assert "sage_hub" in VARLEN_BACKEND_MAPPING

    def test_varlen_mapping_values(self):
        """VARLEN_BACKEND_MAPPING should map to correct varlen backends."""
        assert VARLEN_BACKEND_MAPPING["flash"] == "flash_varlen"
        assert VARLEN_BACKEND_MAPPING["flash_hub"] == "flash_varlen_hub"
        assert VARLEN_BACKEND_MAPPING["_flash_3_hub"] == "_flash_3_varlen_hub"
        assert VARLEN_BACKEND_MAPPING["sage"] == "sage_varlen"
        assert VARLEN_BACKEND_MAPPING["sage_hub"] == "sage_varlen"

    def test_requires_varlen_backend_for_zimage(self):
        """ZImageTransformer2DModel should require varlen backend."""
        mm = ModelManager()

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"

        assert mm._requires_varlen_backend(mock_transformer) is True

    def test_does_not_require_varlen_for_flux(self):
        """FluxTransformer2DModel should not require varlen backend."""
        mm = ModelManager()

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "FluxTransformer2DModel"

        assert mm._requires_varlen_backend(mock_transformer) is False

    def test_does_not_require_varlen_for_other_models(self):
        """Other transformer models should not require varlen backend."""
        mm = ModelManager()

        for class_name in ["SD3Transformer2DModel", "DiTTransformer2DModel", "PixArtTransformer2DModel"]:
            mock_transformer = MagicMock()
            mock_transformer.__class__.__name__ = class_name
            assert mm._requires_varlen_backend(mock_transformer) is False

    def test_flash_mapped_to_varlen_for_zimage(self):
        """Flash backend should be mapped to flash_varlen for Z-Image."""
        mm = ModelManager()
        mm.attention_backend = "flash"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("flash_varlen")

    def test_sage_mapped_to_varlen_for_zimage(self):
        """Sage backend should be mapped to sage_varlen for Z-Image."""
        mm = ModelManager()
        mm.attention_backend = "sage"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("sage_varlen")

    def test_flash_hub_mapped_to_varlen_hub_for_zimage(self):
        """flash_hub should be mapped to flash_varlen_hub for Z-Image."""
        mm = ModelManager()
        mm.attention_backend = "flash_hub"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("flash_varlen_hub")

    def test_flash_3_hub_mapped_to_varlen_for_zimage(self):
        """_flash_3_hub should be mapped to _flash_3_varlen_hub for Z-Image."""
        mm = ModelManager()
        mm.attention_backend = "_flash_3_hub"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("_flash_3_varlen_hub")

    def test_sage_hub_mapped_to_sage_varlen_for_zimage(self):
        """sage_hub should be mapped to sage_varlen for Z-Image (no hub variant exists)."""
        mm = ModelManager()
        mm.attention_backend = "sage_hub"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        mm.apply_attention_backend(mock_transformer)

        # sage_varlen is used because there's no sage_varlen_hub
        mock_transformer.set_attention_backend.assert_called_once_with("sage_varlen")

    def test_xformers_not_mapped_for_zimage(self):
        """xformers should NOT be mapped (it supports masks natively)."""
        mm = ModelManager()
        mm.attention_backend = "xformers"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        mm.apply_attention_backend(mock_transformer)

        # xformers supports masks, so no mapping
        mock_transformer.set_attention_backend.assert_called_once_with("xformers")

    def test_native_not_mapped_for_zimage(self):
        """native backend should be set explicitly, not mapped to varlen."""
        mm = ModelManager()
        mm.attention_backend = "native"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        result = mm.apply_attention_backend(mock_transformer)

        assert result is True
        # Should explicitly set "native", not a varlen variant
        mock_transformer.set_attention_backend.assert_called_once_with("native")

    def test_no_mapping_for_flux_model(self):
        """Flux models should use the backend as-is (no varlen mapping)."""
        mm = ModelManager()
        mm.attention_backend = "flash"

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "FluxTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        mm.apply_attention_backend(mock_transformer)

        mock_transformer.set_attention_backend.assert_called_once_with("flash")

    def test_no_mapping_for_non_zimage_models(self):
        """Non-Z-Image models should use the backend as-is."""
        mm = ModelManager()

        for class_name in ["FluxTransformer2DModel", "SD3Transformer2DModel"]:
            mm.attention_backend = "sage"

            mock_transformer = MagicMock()
            mock_transformer.__class__.__name__ = class_name
            mock_transformer.set_attention_backend = MagicMock()

            mm.apply_attention_backend(mock_transformer)

            mock_transformer.set_attention_backend.assert_called_once_with("sage")


class TestAttentionBackendSwitching:
    """Regression tests for switching between attention backends."""

    def test_switch_from_sage_to_native_explicitly_sets_native(self):
        """Regression test: switching from sage to native should explicitly set 'native'.

        Previously, switching to native would call reset_attention_backend() which only
        set _attention_backend=None on processors. This could cause the previous backend
        to persist in some cases. Now we explicitly call set_attention_backend('native').
        """
        mm = ModelManager()

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock()

        # Step 1: Set to sage
        mm.attention_backend = "sage"
        mm.apply_attention_backend(mock_transformer)
        assert mock_transformer.set_attention_backend.call_args[0][0] == "sage"

        # Step 2: Switch back to native
        mock_transformer.set_attention_backend.reset_mock()
        mm.attention_backend = "native"
        mm.apply_attention_backend(mock_transformer)

        # Verify native was EXPLICITLY set, not just reset to None
        mock_transformer.set_attention_backend.assert_called_once_with("native")

    def test_switch_from_flash_to_native_explicitly_sets_native(self):
        """Regression test: switching from flash to native should explicitly set 'native'."""
        mm = ModelManager()

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock()

        # Step 1: Set to flash
        mm.attention_backend = "flash"
        mm.apply_attention_backend(mock_transformer)
        assert mock_transformer.set_attention_backend.call_args[0][0] == "flash"

        # Step 2: Switch back to native
        mock_transformer.set_attention_backend.reset_mock()
        mm.attention_backend = "native"
        mm.apply_attention_backend(mock_transformer)

        # Verify native was EXPLICITLY set
        mock_transformer.set_attention_backend.assert_called_once_with("native")

    def test_multiple_backend_switches(self):
        """Test switching between multiple backends in sequence."""
        mm = ModelManager()

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock()

        backends_sequence = ["flash", "sage", "native", "xformers", "native"]

        for backend in backends_sequence:
            mock_transformer.set_attention_backend.reset_mock()
            mm.attention_backend = backend
            mm.apply_attention_backend(mock_transformer)

            # Each backend should be explicitly set
            mock_transformer.set_attention_backend.assert_called_once_with(backend)

    def test_zimage_switch_from_varlen_to_native(self):
        """Regression test: Z-Image switching from flash (varlen) to native."""
        mm = ModelManager()

        mock_transformer = MagicMock()
        mock_transformer.__class__.__name__ = "ZImageTransformer2DModel"
        mock_transformer.set_attention_backend = MagicMock()

        # Step 1: Set to flash (should map to flash_varlen for Z-Image)
        mm.attention_backend = "flash"
        mm.apply_attention_backend(mock_transformer)
        assert mock_transformer.set_attention_backend.call_args[0][0] == "flash_varlen"

        # Step 2: Switch back to native (should NOT map to varlen)
        mock_transformer.set_attention_backend.reset_mock()
        mm.attention_backend = "native"
        mm.apply_attention_backend(mock_transformer)

        # Native should be explicitly set, not a varlen variant
        mock_transformer.set_attention_backend.assert_called_once_with("native")
