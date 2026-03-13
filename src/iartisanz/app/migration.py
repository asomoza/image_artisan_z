from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iartisanz.app.directories import DirectoriesObject
    from iartisanz.utils.database import Database


logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = "8"


def _get_meta(db: Database, key: str) -> str | None:
    try:
        row = db.fetch_one("SELECT value FROM app_meta WHERE key = ?", (key,))
        return row[0] if row else None
    except Exception:
        return None


def _set_meta(db: Database, key: str, value: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
        (key, value),
    )


def _populate_component_registry(db: Database, directories: DirectoriesObject) -> None:
    """Populate component and model_component tables from existing diffusers models."""
    from iartisanz.app.component_registry import COMPONENT_TYPES, ComponentRegistry
    from iartisanz.utils.model_utils import calculate_component_hash

    components_base_dir = os.path.join(directories.models_diffusers, "_components")
    registry = ComponentRegistry(db.db_path, components_base_dir)

    rows = db.fetch_all(
        "SELECT id, filepath, name FROM model WHERE deleted = 0"
    )

    if not rows:
        logger.info("Component registry: no diffusers models to register.")
        return

    for model_id, filepath, name in rows:
        if not os.path.isdir(filepath):
            logger.warning("Component registry: model directory missing for '%s' (%s), skipping.", name, filepath)
            continue

        if registry.model_has_components(model_id):
            logger.debug("Component registry: model '%s' already fully registered, skipping.", name)
            continue

        component_mapping: dict[str, int] = {}

        for comp_type in COMPONENT_TYPES:
            comp_dir = os.path.join(filepath, comp_type)
            if not os.path.isdir(comp_dir):
                # Directory missing — may already be deduplicated into _components/
                # or may be a legacy symlink. Try resolving from registry.
                if os.path.islink(comp_dir):
                    # Legacy symlink — resolve to real path
                    real_path = os.path.realpath(comp_dir)
                    if os.path.isdir(real_path):
                        comp_dir = real_path
                    else:
                        logger.warning(
                            "Component registry: broken symlink '%s' for model '%s', skipping.",
                            comp_type,
                            name,
                        )
                        continue
                else:
                    logger.warning(
                        "Component registry: component '%s' not found for model '%s', skipping.",
                        comp_type,
                        name,
                    )
                    continue

            try:
                content_hash = calculate_component_hash(comp_dir)
            except Exception as e:
                logger.error(
                    "Component registry: failed to hash %s/%s: %s", name, comp_type, e
                )
                continue

            comp_info = registry.register_component(
                component_type=comp_type,
                source_path=comp_dir,
                content_hash=content_hash,
            )
            component_mapping[comp_type] = comp_info.id

        if component_mapping:
            registry.register_model_components(model_id, component_mapping)
            logger.info(
                "Component registry: registered %d components for model '%s'.",
                len(component_mapping),
                name,
            )


def _compact_storage(db: Database, directories: DirectoriesObject) -> None:
    """Consolidate shared components into _components/ and remove model-local duplicates."""
    from iartisanz.app.component_registry import ComponentRegistry

    components_base_dir = os.path.join(directories.models_diffusers, "_components")
    registry = ComponentRegistry(db.db_path, components_base_dir)

    stats = registry.compact_shared_components()

    if stats["moved"] or stats["deduplicated"]:
        logger.info(
            "Compact storage: moved %d components, deduplicated %d copies, saved ~%.1f MB.",
            stats["moved"],
            stats["deduplicated"],
            stats["bytes_saved"] / (1024 * 1024),
        )
    else:
        logger.info("Compact storage: nothing to compact.")


def _backfill_component_architectures(db: Database) -> None:
    """Backfill NULL architecture on components by re-reading their config files."""
    import json

    rows = db.fetch_all(
        "SELECT id, storage_path, component_type FROM component WHERE architecture IS NULL"
    )
    if not rows:
        return

    for comp_id, storage_path, comp_type in rows:
        config_json = None
        for config_name in ("config.json", "tokenizer_config.json"):
            config_path = os.path.join(storage_path, config_name)
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_json = f.read()
                    break
                except Exception:
                    pass

        if config_json is None:
            continue

        try:
            cfg = json.loads(config_json)
            architecture = (
                cfg.get("_class_name")
                or (cfg.get("architectures") or [None])[0]
                or cfg.get("tokenizer_class")
                or cfg.get("model_type")
            )
        except Exception:
            continue

        if architecture:
            db.execute(
                "UPDATE component SET architecture = ?, config_json = ? WHERE id = ?",
                (architecture, config_json, comp_id),
            )
            logger.info("Backfilled architecture '%s' for component %d (%s).", architecture, comp_id, comp_type)


def _remove_legacy_symlinks(directories: DirectoriesObject) -> None:
    """Remove any legacy symlinks in model directories.

    Previous versions replaced shared component directories with symlinks to
    _components/. This migration removes those symlinks — the canonical data
    is safe in _components/ and the registry resolves paths from the DB.
    """
    models_dir = directories.models_diffusers
    if not os.path.isdir(models_dir):
        return

    component_subdirs = ("tokenizer", "text_encoder", "transformer", "vae")
    removed = 0

    for entry in os.listdir(models_dir):
        if entry == "_components":
            continue
        model_dir = os.path.join(models_dir, entry)
        if not os.path.isdir(model_dir):
            continue

        for comp_name in component_subdirs:
            comp_path = os.path.join(model_dir, comp_name)
            if os.path.islink(comp_path):
                os.remove(comp_path)
                removed += 1
                logger.info("Migration v5: removed legacy symlink %s", comp_path)

    if removed:
        logger.info("Migration v5: removed %d legacy symlinks.", removed)


def _consolidate_klein_model_types(db: Database) -> None:
    """Merge Klein distilled/base model types and add distilled column.

    Old types 3 (Klein 9B) and 4 (Klein Base 9B) → type 3, distinguished by distilled flag.
    Old types 5 (Klein 4B) and 6 (Klein Base 4B) → type 5, distinguished by distilled flag.
    """
    for table in ("model", "lora_model"):
        # Add distilled column (defaults to 1 = distilled)
        try:
            db.execute(f"ALTER TABLE {table} ADD COLUMN distilled INT DEFAULT 1")
        except Exception:
            logger.debug("distilled column already exists on %s", table)

        # Old type 3 (distilled) → keep type 3, distilled=1 (already default)
        # Old type 4 (base) → remap to type 3, distilled=0
        db.execute(f"UPDATE {table} SET model_type = 3, distilled = 0 WHERE model_type = 4")
        # Old type 5 (distilled) → keep type 5, distilled=1 (already default)
        # Old type 6 (base) → remap to type 5, distilled=0
        db.execute(f"UPDATE {table} SET model_type = 5, distilled = 0 WHERE model_type = 6")

    count_4 = db.fetch_one("SELECT COUNT(*) FROM model WHERE model_type = 4")
    count_6 = db.fetch_one("SELECT COUNT(*) FROM model WHERE model_type = 6")
    logger.info(
        "Klein consolidation: remaining type 4=%d, type 6=%d (should both be 0)",
        count_4[0] if count_4 else 0,
        count_6[0] if count_6 else 0,
    )


def run_migrations(db: Database, directories: DirectoriesObject) -> None:
    """Run any pending database migrations."""
    version = _get_meta(db, "schema_version")

    if version is None or version < "3":
        logger.info("Running component registry population...")
        try:
            _populate_component_registry(db, directories)
        except Exception as e:
            logger.error("Component registry population failed: %s", e, exc_info=True)

        logger.info("Running storage compaction...")
        try:
            _compact_storage(db, directories)
        except Exception as e:
            logger.error("Storage compaction failed: %s", e, exc_info=True)

    if version is None or version < "4":
        logger.info("Backfilling component architectures...")
        try:
            _backfill_component_architectures(db)
        except Exception as e:
            logger.error("Architecture backfill failed: %s", e, exc_info=True)

    if version is None or version < "5":
        logger.info("Removing legacy symlinks...")
        try:
            _remove_legacy_symlinks(directories)
        except Exception as e:
            logger.error("Legacy symlink removal failed: %s", e, exc_info=True)

    if version is not None and version < "6":
        logger.info("Adding dtype column to component table...")
        try:
            db.execute("ALTER TABLE component ADD COLUMN dtype TEXT")
        except Exception as e:
            # Column may already exist if table was freshly created
            logger.debug("dtype column migration: %s", e)

    if version is not None and version < "7":
        logger.info("Clearing invalid dtype values...")
        try:
            db.execute("UPDATE component SET dtype = NULL WHERE dtype NOT IN ('bfloat16', 'float16', 'float32', 'float64')")
        except Exception as e:
            logger.debug("dtype cleanup: %s", e)

    if version is not None and version < "8":
        logger.info("Consolidating Klein model types and adding distilled column...")
        try:
            _consolidate_klein_model_types(db)
        except Exception as e:
            logger.error("Klein model type consolidation failed: %s", e, exc_info=True)

    _set_meta(db, "schema_version", CURRENT_SCHEMA_VERSION)
    logger.info("Migration to schema v%s complete.", CURRENT_SCHEMA_VERSION)
