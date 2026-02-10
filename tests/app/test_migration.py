"""Tests for schema migration system."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import save_file

from iartisanz.app.migration import (
    CURRENT_SCHEMA_VERSION,
    _get_meta,
    _populate_component_registry,
    _set_meta,
    run_migrations,
)
from iartisanz.utils.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeDirectories:
    models_diffusers: str


def _init_db(db_path: str) -> Database:
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


def _create_diffusers_model(model_dir: str, *, vae_tensors=None, te_tensors=None, tx_tensors=None):
    """Create a minimal diffusers model directory with all 4 components."""
    if vae_tensors is None:
        vae_tensors = {"decoder.weight": torch.randn(4, 4)}
    if te_tensors is None:
        te_tensors = {"layers.0.weight": torch.randn(4, 4)}
    if tx_tensors is None:
        tx_tensors = {"context_refiner.weight": torch.randn(4, 4)}

    # VAE
    vae_dir = os.path.join(model_dir, "vae")
    os.makedirs(vae_dir, exist_ok=True)
    save_file(vae_tensors, os.path.join(vae_dir, "model.safetensors"))
    with open(os.path.join(vae_dir, "config.json"), "w") as f:
        json.dump({"_class_name": "AutoencoderKL"}, f)

    # Text Encoder
    te_dir = os.path.join(model_dir, "text_encoder")
    os.makedirs(te_dir, exist_ok=True)
    save_file(te_tensors, os.path.join(te_dir, "model.safetensors"))
    with open(os.path.join(te_dir, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3Model"]}, f)

    # Transformer
    tx_dir = os.path.join(model_dir, "transformer")
    os.makedirs(tx_dir, exist_ok=True)
    save_file(tx_tensors, os.path.join(tx_dir, "model.safetensors"))
    with open(os.path.join(tx_dir, "config.json"), "w") as f:
        json.dump({"_class_name": "ZImageTransformer2DModel"}, f)

    # Tokenizer
    tok_dir = os.path.join(model_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    with open(os.path.join(tok_dir, "vocab.json"), "w") as f:
        json.dump({"hello": 0, "world": 1}, f)
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"model_type": "qwen2"}, f)


@pytest.fixture
def migration_env(tmp_path):
    db_path = str(tmp_path / "data" / "app.db")
    models_dir = str(tmp_path / "models")
    os.makedirs(models_dir, exist_ok=True)
    db = _init_db(db_path)
    dirs = FakeDirectories(models_diffusers=models_dir)
    yield db, dirs, tmp_path
    db.disconnect()


# ---------------------------------------------------------------------------
# _get_meta / _set_meta
# ---------------------------------------------------------------------------


class TestAppMeta:
    def test_get_meta_returns_none_for_missing_key(self, migration_env):
        db, dirs, _ = migration_env
        assert _get_meta(db, "nonexistent") is None

    def test_set_and_get_meta(self, migration_env):
        db, dirs, _ = migration_env
        _set_meta(db, "schema_version", "2")
        assert _get_meta(db, "schema_version") == "2"

    def test_set_meta_upserts(self, migration_env):
        db, dirs, _ = migration_env
        _set_meta(db, "key", "v1")
        _set_meta(db, "key", "v2")
        assert _get_meta(db, "key") == "v2"


# ---------------------------------------------------------------------------
# _populate_component_registry
# ---------------------------------------------------------------------------


class TestPopulateComponentRegistry:
    def test_populates_from_existing_model(self, migration_env):
        db, dirs, tmp_path = migration_env

        model_dir = os.path.join(dirs.models_diffusers, "TestModel")
        _create_diffusers_model(model_dir)

        model_id = db.last_insert_rowid()
        db.insert("model", {
            "name": "TestModel",
            "filepath": model_dir,
            "deleted": 0,
        })
        model_id = db.last_insert_rowid()

        _populate_component_registry(db, dirs)

        # Should have 4 components registered
        rows = db.fetch_all("SELECT COUNT(*) FROM component")
        assert rows[0][0] == 4

        # Should have 4 model_component mappings
        rows = db.fetch_all("SELECT COUNT(*) FROM model_component WHERE model_id = ?", (model_id,))
        assert rows[0][0] == 4

    def test_shared_components_get_same_id(self, migration_env):
        db, dirs, tmp_path = migration_env

        # Create two models with identical VAE and text_encoder
        shared_vae = {"decoder.weight": torch.randn(4, 4)}
        shared_te = {"layers.0.weight": torch.randn(4, 4)}

        model_a_dir = os.path.join(dirs.models_diffusers, "ModelA")
        model_b_dir = os.path.join(dirs.models_diffusers, "ModelB")

        _create_diffusers_model(model_a_dir, vae_tensors=shared_vae, te_tensors=shared_te)
        _create_diffusers_model(model_b_dir, vae_tensors=shared_vae, te_tensors=shared_te)

        db.insert("model", {"name": "ModelA", "filepath": model_a_dir, "deleted": 0})
        id_a = db.last_insert_rowid()
        db.insert("model", {"name": "ModelB", "filepath": model_b_dir, "deleted": 0})
        id_b = db.last_insert_rowid()

        _populate_component_registry(db, dirs)

        # VAE component_id should be the same for both models
        row_a = db.fetch_one(
            "SELECT component_id FROM model_component WHERE model_id = ? AND component_type = 'vae'",
            (id_a,),
        )
        row_b = db.fetch_one(
            "SELECT component_id FROM model_component WHERE model_id = ? AND component_type = 'vae'",
            (id_b,),
        )
        assert row_a[0] == row_b[0]

        # text_encoder component_id should also be the same
        row_a = db.fetch_one(
            "SELECT component_id FROM model_component WHERE model_id = ? AND component_type = 'text_encoder'",
            (id_a,),
        )
        row_b = db.fetch_one(
            "SELECT component_id FROM model_component WHERE model_id = ? AND component_type = 'text_encoder'",
            (id_b,),
        )
        assert row_a[0] == row_b[0]

    def test_skips_deleted_models(self, migration_env):
        db, dirs, tmp_path = migration_env

        model_dir = os.path.join(dirs.models_diffusers, "Deleted")
        _create_diffusers_model(model_dir)
        db.insert("model", {"name": "Deleted", "filepath": model_dir, "deleted": 1})

        _populate_component_registry(db, dirs)

        rows = db.fetch_all("SELECT COUNT(*) FROM component")
        assert rows[0][0] == 0

    def test_skips_missing_model_dir(self, migration_env):
        db, dirs, tmp_path = migration_env

        missing_dir = os.path.join(dirs.models_diffusers, "MissingModel")
        db.insert("model", {"name": "MissingModel", "filepath": missing_dir, "deleted": 0})

        _populate_component_registry(db, dirs)

        rows = db.fetch_all("SELECT COUNT(*) FROM component")
        assert rows[0][0] == 0

    def test_skips_already_registered_models(self, migration_env):
        db, dirs, tmp_path = migration_env

        model_dir = os.path.join(dirs.models_diffusers, "M")
        _create_diffusers_model(model_dir)
        db.insert("model", {"name": "M", "filepath": model_dir, "deleted": 0})

        _populate_component_registry(db, dirs)
        count_before = db.fetch_all("SELECT COUNT(*) FROM component")[0][0]

        # Second call should skip (model already has components)
        _populate_component_registry(db, dirs)
        count_after = db.fetch_all("SELECT COUNT(*) FROM component")[0][0]

        assert count_before == count_after


# ---------------------------------------------------------------------------
# run_migrations
# ---------------------------------------------------------------------------


class TestRunMigrations:
    def test_sets_schema_version(self, migration_env):
        db, dirs, tmp_path = migration_env
        run_migrations(db, dirs)
        assert _get_meta(db, "schema_version") == CURRENT_SCHEMA_VERSION

    def test_idempotent(self, migration_env):
        db, dirs, tmp_path = migration_env

        model_dir = os.path.join(dirs.models_diffusers, "M")
        _create_diffusers_model(model_dir)
        db.insert("model", {"name": "M", "filepath": model_dir, "deleted": 0})

        run_migrations(db, dirs)
        count_1 = db.fetch_all("SELECT COUNT(*) FROM component")[0][0]

        run_migrations(db, dirs)
        count_2 = db.fetch_all("SELECT COUNT(*) FROM component")[0][0]

        assert count_1 == count_2

    def test_skips_when_already_at_current_version(self, migration_env):
        db, dirs, tmp_path = migration_env
        _set_meta(db, "schema_version", CURRENT_SCHEMA_VERSION)

        model_dir = os.path.join(dirs.models_diffusers, "M")
        _create_diffusers_model(model_dir)
        db.insert("model", {"name": "M", "filepath": model_dir, "deleted": 0})

        run_migrations(db, dirs)

        # Should not have populated because version is already current
        count = db.fetch_all("SELECT COUNT(*) FROM component")[0][0]
        assert count == 0

    def test_no_models_succeeds(self, migration_env):
        db, dirs, tmp_path = migration_env
        # Should complete without error even with no models
        run_migrations(db, dirs)
        assert _get_meta(db, "schema_version") == CURRENT_SCHEMA_VERSION
