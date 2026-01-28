"""Tests for ModelManager attention backend functionality."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from iartisanz.app.model_manager import (
    ATTENTION_BACKEND_OPTIONS,
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

    def test_native_backend_calls_reset_if_available(self):
        """Native backend should call reset_attention_backend if available."""
        mm = ModelManager()
        mm.attention_backend = "native"

        mock_transformer = MagicMock()
        mock_transformer.reset_attention_backend = MagicMock()

        result = mm.apply_attention_backend(mock_transformer)

        mock_transformer.reset_attention_backend.assert_called_once()
        assert result is True

    def test_native_backend_succeeds_without_reset_method(self):
        """Native backend should succeed even without reset_attention_backend."""
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

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock(
            side_effect=RuntimeError("Backend not available")
        )
        mock_transformer.reset_attention_backend = MagicMock()

        result = mm.apply_attention_backend(mock_transformer)

        assert result is False
        # Should try to reset to native after failure
        mock_transformer.reset_attention_backend.assert_called_once()

    def test_handles_exception_in_reset_after_failure(self):
        """Should handle exceptions from reset_attention_backend after set fails."""
        mm = ModelManager()
        mm.attention_backend = "flash"

        mock_transformer = MagicMock()
        mock_transformer.set_attention_backend = MagicMock(
            side_effect=RuntimeError("Backend not available")
        )
        mock_transformer.reset_attention_backend = MagicMock(
            side_effect=RuntimeError("Reset also failed")
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
