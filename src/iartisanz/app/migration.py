from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iartisanz.app.directories import DirectoriesObject
    from iartisanz.utils.database import Database


logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = "3"


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
        "SELECT id, filepath, name FROM model WHERE model_format = 1 AND deleted = 0"
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
                # Follow symlinks — the directory might already be a symlink
                if os.path.islink(comp_dir) and os.path.isdir(os.path.realpath(comp_dir)):
                    comp_dir = os.path.realpath(comp_dir)
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
    """Consolidate shared components into _components/ and symlink originals."""
    from iartisanz.app.component_registry import ComponentRegistry

    components_base_dir = os.path.join(directories.models_diffusers, "_components")
    registry = ComponentRegistry(db.db_path, components_base_dir)

    stats = registry.compact_shared_components()

    if stats["moved"] or stats["symlinked"]:
        logger.info(
            "Compact storage: moved %d components, created %d symlinks, saved ~%.1f MB.",
            stats["moved"],
            stats["symlinked"],
            stats["bytes_saved"] / (1024 * 1024),
        )
    else:
        logger.info("Compact storage: nothing to compact.")


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

        _set_meta(db, "schema_version", CURRENT_SCHEMA_VERSION)
        logger.info("Migration to schema v%s complete.", CURRENT_SCHEMA_VERSION)
