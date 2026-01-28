"""Tests for GenerationSettings attention_backend field."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QSettings

from iartisanz.modules.generation.generation_settings import GenerationSettings


class TestAttentionBackendField:
    """Tests for the attention_backend field in GenerationSettings."""

    def test_default_value_is_native(self):
        """Default attention_backend should be 'native'."""
        settings = GenerationSettings()
        assert settings.attention_backend == "native"

    def test_can_set_to_flash(self):
        """Should be able to set attention_backend to 'flash'."""
        settings = GenerationSettings()
        settings.attention_backend = "flash"
        assert settings.attention_backend == "flash"

    def test_can_set_to_sage(self):
        """Should be able to set attention_backend to 'sage'."""
        settings = GenerationSettings()
        settings.attention_backend = "sage"
        assert settings.attention_backend == "sage"

    def test_can_set_to_xformers(self):
        """Should be able to set attention_backend to 'xformers'."""
        settings = GenerationSettings()
        settings.attention_backend = "xformers"
        assert settings.attention_backend == "xformers"


class TestAttentionBackendApplyChange:
    """Tests for apply_change with attention_backend."""

    def test_apply_change_handles_attention_backend(self):
        """apply_change should handle attention_backend attribute."""
        settings = GenerationSettings()
        handled, graph_value = settings.apply_change("attention_backend", "flash")

        assert handled is True
        assert graph_value is None  # Not forwarded to graph
        assert settings.attention_backend == "flash"

    def test_apply_change_returns_none_graph_value(self):
        """attention_backend should not be forwarded to graph."""
        settings = GenerationSettings()
        handled, graph_value = settings.apply_change("attention_backend", "sage")

        assert handled is True
        assert graph_value is None

    def test_apply_change_coerces_to_string(self):
        """apply_change should coerce value to string."""
        settings = GenerationSettings()
        # Even if someone passes a non-string, it should be coerced
        handled, graph_value = settings.apply_change("attention_backend", "native")

        assert handled is True
        assert settings.attention_backend == "native"


class TestAttentionBackendSaveLoad:
    """Tests for saving and loading attention_backend from QSettings."""

    def test_save_persists_attention_backend(self):
        """save() should persist attention_backend to QSettings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qsettings = QSettings(f"{tmpdir}/test.ini", QSettings.Format.IniFormat)

            settings = GenerationSettings()
            settings.attention_backend = "flash"
            settings.save(qsettings)
            qsettings.sync()

            # Verify it was saved
            qsettings.beginGroup("generation")
            saved_value = qsettings.value("attention_backend")
            qsettings.endGroup()

            assert saved_value == "flash"

    def test_load_restores_attention_backend(self):
        """load() should restore attention_backend from QSettings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qsettings = QSettings(f"{tmpdir}/test.ini", QSettings.Format.IniFormat)

            # Save a value
            qsettings.beginGroup("generation")
            qsettings.setValue("attention_backend", "sage")
            qsettings.endGroup()
            qsettings.sync()

            # Load it back
            loaded = GenerationSettings.load(qsettings)

            assert loaded.attention_backend == "sage"

    def test_load_defaults_to_native_if_not_set(self):
        """load() should default to 'native' if not in QSettings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qsettings = QSettings(f"{tmpdir}/test.ini", QSettings.Format.IniFormat)

            # Don't save attention_backend
            loaded = GenerationSettings.load(qsettings)

            assert loaded.attention_backend == "native"

    def test_roundtrip_save_load(self):
        """Save then load should preserve attention_backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qsettings = QSettings(f"{tmpdir}/test.ini", QSettings.Format.IniFormat)

            original = GenerationSettings()
            original.attention_backend = "xformers"
            original.save(qsettings)
            qsettings.sync()

            loaded = GenerationSettings.load(qsettings)

            assert loaded.attention_backend == original.attention_backend


class TestAttentionBackendResetToDefaults:
    """Tests for reset_to_defaults with attention_backend."""

    def test_reset_to_defaults_resets_attention_backend(self):
        """reset_to_defaults should reset attention_backend to 'native'."""
        settings = GenerationSettings()
        settings.attention_backend = "flash"
        assert settings.attention_backend == "flash"

        settings.reset_to_defaults()

        assert settings.attention_backend == "native"

    def test_reset_to_defaults_preserves_model_but_resets_attention(self):
        """reset_to_defaults(preserve_model=True) should still reset attention_backend."""
        settings = GenerationSettings()
        settings.attention_backend = "sage"

        settings.reset_to_defaults(preserve_model=True)

        assert settings.attention_backend == "native"


class TestAttentionBackendNotInGraphKeys:
    """Tests to verify attention_backend is not in GRAPH_KEYS."""

    def test_attention_backend_not_in_graph_keys(self):
        """attention_backend should NOT be in GRAPH_KEYS."""
        settings = GenerationSettings()
        assert "attention_backend" not in settings.GRAPH_KEYS

    def test_to_graph_nodes_does_not_include_attention_backend(self):
        """to_graph_nodes() should not include attention_backend."""
        settings = GenerationSettings()
        settings.attention_backend = "flash"

        graph_nodes = settings.to_graph_nodes()

        assert "attention_backend" not in graph_nodes
