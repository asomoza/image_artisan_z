"""Tests for ComponentRegistry CRUD and compact storage."""
from __future__ import annotations

import json
import os
import shutil

import pytest
import torch
from safetensors.torch import save_file

from iartisanz.app.component_registry import COMPONENT_TYPES, ComponentRegistry
from iartisanz.utils.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_db(db_path: str) -> Database:
    """Create the app database with the required schema."""
    db = Database(db_path)
    db.create_table(
        "model",
        [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "root_filename TEXT",
            "filepath TEXT",
            "name TEXT",
            "version TEXT",
            "model_type INT",
            "model_format INT DEFAULT 0",
            "hash TEXT",
            "tags TEXT",
            "thumbnail TEXT",
            "triggers TEXT",
            "example TEXT",
            "deleted BOOLEAN DEFAULT 0",
        ],
    )
    db.create_table(
        "component",
        [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "component_type TEXT NOT NULL",
            "content_hash TEXT NOT NULL UNIQUE",
            "storage_path TEXT NOT NULL",
            "size_bytes INTEGER DEFAULT 0",
            "architecture TEXT",
            "config_json TEXT",
            "created_at TEXT DEFAULT (datetime('now'))",
        ],
    )
    db.create_table(
        "model_component",
        [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "model_id INTEGER NOT NULL",
            "component_type TEXT NOT NULL",
            "component_id INTEGER NOT NULL",
            "UNIQUE(model_id, component_type)",
            "FOREIGN KEY (model_id) REFERENCES model(id)",
            "FOREIGN KEY (component_id) REFERENCES component(id)",
        ],
    )
    db.create_table(
        "app_meta",
        [
            "key TEXT PRIMARY KEY",
            "value TEXT",
        ],
    )
    return db


def _insert_model(db: Database, name: str, filepath: str, model_format: int = 1) -> int:
    """Insert a model row and return its id."""
    db.insert("model", {
        "name": name,
        "filepath": filepath,
        "model_format": model_format,
        "deleted": 0,
    })
    return db.last_insert_rowid()


def _create_safetensors_component(comp_dir: str, tensors: dict[str, torch.Tensor], config: dict | None = None) -> None:
    os.makedirs(comp_dir, exist_ok=True)
    save_file(tensors, os.path.join(comp_dir, "model.safetensors"))
    if config is not None:
        with open(os.path.join(comp_dir, "config.json"), "w") as f:
            json.dump(config, f)


def _create_text_files(comp_dir: str, files: dict[str, str]) -> None:
    os.makedirs(comp_dir, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(comp_dir, name), "w") as f:
            f.write(content)


@pytest.fixture
def registry_env(tmp_path):
    """Provide a ready-to-use registry + database + paths."""
    db_path = str(tmp_path / "data" / "app.db")
    components_dir = str(tmp_path / "models" / "_components")
    db = _init_db(db_path)
    registry = ComponentRegistry(db_path, components_dir)
    yield registry, db, tmp_path
    db.disconnect()


# ---------------------------------------------------------------------------
# register_component
# ---------------------------------------------------------------------------


class TestRegisterComponent:
    def test_registers_new_component(self, registry_env):
        registry, db, tmp_path = registry_env
        comp_dir = str(tmp_path / "vae")
        _create_safetensors_component(comp_dir, {"w": torch.randn(4, 4)}, config={"_class_name": "AutoencoderKL"})

        info = registry.register_component(
            component_type="vae",
            source_path=comp_dir,
            content_hash="abc123",
        )

        assert info.id > 0
        assert info.component_type == "vae"
        assert info.content_hash == "abc123"
        assert info.storage_path == comp_dir
        assert info.size_bytes > 0
        assert info.architecture == "AutoencoderKL"

    def test_returns_existing_on_duplicate_hash(self, registry_env):
        registry, db, tmp_path = registry_env
        comp_dir = str(tmp_path / "vae")
        _create_safetensors_component(comp_dir, {"w": torch.randn(4, 4)})

        info1 = registry.register_component("vae", comp_dir, "same_hash")
        info2 = registry.register_component("vae", comp_dir, "same_hash")

        assert info1.id == info2.id

    def test_reads_config_json_automatically(self, registry_env):
        registry, db, tmp_path = registry_env
        comp_dir = str(tmp_path / "te")
        config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
        _create_safetensors_component(comp_dir, {"w": torch.randn(2, 2)}, config=config)

        info = registry.register_component("text_encoder", comp_dir, "te_hash")
        assert info.config_json is not None
        assert info.architecture == "Qwen3ForCausalLM"

    def test_explicit_architecture_wins(self, registry_env):
        registry, db, tmp_path = registry_env
        comp_dir = str(tmp_path / "vae")
        _create_safetensors_component(comp_dir, {"w": torch.randn(2, 2)}, config={"_class_name": "FromConfig"})

        info = registry.register_component(
            "vae", comp_dir, "vae_hash", architecture="Explicit"
        )
        assert info.architecture == "Explicit"


# ---------------------------------------------------------------------------
# get_component_by_hash
# ---------------------------------------------------------------------------


class TestGetComponentByHash:
    def test_found(self, registry_env):
        registry, db, tmp_path = registry_env
        comp_dir = str(tmp_path / "comp")
        _create_text_files(comp_dir, {"f.txt": "data"})
        registry.register_component("tokenizer", comp_dir, "tok_hash_1")

        result = registry.get_component_by_hash("tok_hash_1")
        assert result is not None
        assert result.content_hash == "tok_hash_1"

    def test_not_found(self, registry_env):
        registry, db, tmp_path = registry_env
        assert registry.get_component_by_hash("nonexistent") is None


# ---------------------------------------------------------------------------
# register_model_components / get_model_components / model_has_components
# ---------------------------------------------------------------------------


class TestModelComponentMapping:
    def test_register_and_get(self, registry_env):
        registry, db, tmp_path = registry_env

        model_dir = str(tmp_path / "models" / "TestModel")
        model_id = _insert_model(db, "TestModel", model_dir)

        comp_ids = {}
        for comp_type in COMPONENT_TYPES:
            comp_dir = str(tmp_path / "comps" / comp_type)
            _create_text_files(comp_dir, {"data.txt": f"{comp_type} data"})
            info = registry.register_component(comp_type, comp_dir, f"hash_{comp_type}")
            comp_ids[comp_type] = info.id

        registry.register_model_components(model_id, comp_ids)

        result = registry.get_model_components(model_id)
        assert set(result.keys()) == set(COMPONENT_TYPES)
        for comp_type in COMPONENT_TYPES:
            assert result[comp_type].content_hash == f"hash_{comp_type}"

    def test_model_has_components_true(self, registry_env):
        registry, db, tmp_path = registry_env

        model_id = _insert_model(db, "M", str(tmp_path / "m"))
        comp_ids = {}
        for comp_type in COMPONENT_TYPES:
            comp_dir = str(tmp_path / comp_type)
            _create_text_files(comp_dir, {"f": "d"})
            info = registry.register_component(comp_type, comp_dir, f"h_{comp_type}")
            comp_ids[comp_type] = info.id

        registry.register_model_components(model_id, comp_ids)
        assert registry.model_has_components(model_id) is True

    def test_model_has_components_false_partial(self, registry_env):
        registry, db, tmp_path = registry_env

        model_id = _insert_model(db, "M", str(tmp_path / "m"))
        comp_dir = str(tmp_path / "tok")
        _create_text_files(comp_dir, {"f": "d"})
        info = registry.register_component("tokenizer", comp_dir, "partial_hash")
        registry.register_model_components(model_id, {"tokenizer": info.id})

        assert registry.model_has_components(model_id) is False

    def test_model_has_components_false_none(self, registry_env):
        registry, db, tmp_path = registry_env
        model_id = _insert_model(db, "M", str(tmp_path / "m"))
        assert registry.model_has_components(model_id) is False


# ---------------------------------------------------------------------------
# find_compatible_components
# ---------------------------------------------------------------------------


class TestFindCompatibleComponents:
    def test_finds_compatible_vae(self, registry_env):
        registry, db, tmp_path = registry_env

        comp_dir = str(tmp_path / "vae")
        _create_safetensors_component(comp_dir, {"w": torch.randn(2, 2)})
        registry.register_component("vae", comp_dir, "vae_hash", architecture="AutoencoderKL")

        result = registry.find_compatible_components("ZImageTransformer2DModel")
        assert "vae" in result
        assert any(c.architecture == "AutoencoderKL" for c in result["vae"])

    def test_finds_compatible_text_encoder(self, registry_env):
        registry, db, tmp_path = registry_env

        comp_dir = str(tmp_path / "te")
        _create_safetensors_component(comp_dir, {"w": torch.randn(2, 2)})
        registry.register_component("text_encoder", comp_dir, "te_hash", architecture="Qwen3Model")

        result = registry.find_compatible_components("ZImageTransformer2DModel")
        assert "text_encoder" in result

    def test_unknown_architecture_empty(self, registry_env):
        registry, db, tmp_path = registry_env
        result = registry.find_compatible_components("UnknownArch")
        assert result == {}


# ---------------------------------------------------------------------------
# compact_shared_components
# ---------------------------------------------------------------------------


class TestCompactSharedComponents:
    def test_compacts_shared_component(self, registry_env):
        registry, db, tmp_path = registry_env

        # Create two model directories with identical VAE content
        model_a_dir = str(tmp_path / "models" / "ModelA")
        model_b_dir = str(tmp_path / "models" / "ModelB")
        vae_data = {"decoder.weight": torch.randn(4, 4)}

        vae_a = os.path.join(model_a_dir, "vae")
        vae_b = os.path.join(model_b_dir, "vae")
        _create_safetensors_component(vae_a, vae_data)
        _create_safetensors_component(vae_b, vae_data)

        model_a_id = _insert_model(db, "ModelA", model_a_dir)
        model_b_id = _insert_model(db, "ModelB", model_b_dir)

        # Register the same component for both models
        info = registry.register_component("vae", vae_a, "shared_vae_hash")
        registry.register_model_components(model_a_id, {"vae": info.id})
        registry.register_model_components(model_b_id, {"vae": info.id})

        stats = registry.compact_shared_components()

        assert stats["moved"] >= 1
        # Both model dirs should now have symlinks
        assert os.path.islink(vae_a) or os.path.islink(vae_b)

    def test_no_op_when_no_shared_components(self, registry_env):
        registry, db, tmp_path = registry_env

        model_dir = str(tmp_path / "models" / "ModelA")
        comp_dir = os.path.join(model_dir, "vae")
        _create_safetensors_component(comp_dir, {"w": torch.randn(2, 2)})

        model_id = _insert_model(db, "ModelA", model_dir)
        info = registry.register_component("vae", comp_dir, "unique_hash")
        registry.register_model_components(model_id, {"vae": info.id})

        stats = registry.compact_shared_components()
        assert stats["moved"] == 0
        assert stats["symlinked"] == 0

    def test_compact_creates_canonical_dir(self, registry_env):
        registry, db, tmp_path = registry_env

        model_a_dir = str(tmp_path / "models" / "ModelA")
        model_b_dir = str(tmp_path / "models" / "ModelB")
        tensors = {"w": torch.tensor([1.0, 2.0])}

        vae_a = os.path.join(model_a_dir, "vae")
        vae_b = os.path.join(model_b_dir, "vae")
        _create_safetensors_component(vae_a, tensors)
        _create_safetensors_component(vae_b, tensors)

        model_a_id = _insert_model(db, "ModelA", model_a_dir)
        model_b_id = _insert_model(db, "ModelB", model_b_dir)

        info = registry.register_component("vae", vae_a, "compact_hash")
        registry.register_model_components(model_a_id, {"vae": info.id})
        registry.register_model_components(model_b_id, {"vae": info.id})

        registry.compact_shared_components()

        # Canonical dir should exist under _components/
        canonical = os.path.join(registry.components_base_dir, "vae", "compact_hash")
        assert os.path.isdir(canonical)
        assert os.path.isfile(os.path.join(canonical, "model.safetensors"))

    def test_compact_idempotent(self, registry_env):
        registry, db, tmp_path = registry_env

        model_a_dir = str(tmp_path / "models" / "ModelA")
        model_b_dir = str(tmp_path / "models" / "ModelB")
        tensors = {"w": torch.tensor([42.0])}

        vae_a = os.path.join(model_a_dir, "vae")
        vae_b = os.path.join(model_b_dir, "vae")
        _create_safetensors_component(vae_a, tensors)
        _create_safetensors_component(vae_b, tensors)

        model_a_id = _insert_model(db, "ModelA", model_a_dir)
        model_b_id = _insert_model(db, "ModelB", model_b_dir)

        info = registry.register_component("vae", vae_a, "idem_hash")
        registry.register_model_components(model_a_id, {"vae": info.id})
        registry.register_model_components(model_b_id, {"vae": info.id})

        stats1 = registry.compact_shared_components()
        stats2 = registry.compact_shared_components()

        # Second run should be a no-op
        assert stats2["moved"] == 0
