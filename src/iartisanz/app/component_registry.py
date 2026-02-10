from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from iartisanz.utils.database import Database


logger = logging.getLogger(__name__)

COMPONENT_TYPES = ("tokenizer", "text_encoder", "transformer", "vae")


@dataclass
class ComponentInfo:
    id: int
    component_type: str
    content_hash: str
    storage_path: str
    size_bytes: int = 0
    architecture: Optional[str] = None
    config_json: Optional[str] = None


class ComponentRegistry:
    def __init__(self, db_path: str, components_base_dir: str):
        self.db_path = db_path
        self.components_base_dir = components_base_dir

    def _db(self) -> Database:
        return Database(self.db_path)

    def register_component(
        self,
        component_type: str,
        source_path: str,
        content_hash: str,
        *,
        architecture: str | None = None,
        config_json: str | None = None,
    ) -> ComponentInfo:
        """Register a component by its hash. If the hash already exists, return the existing entry."""
        db = self._db()

        existing = self.get_component_by_hash(content_hash)
        if existing is not None:
            return existing

        size_bytes = 0
        if os.path.isdir(source_path):
            for dirpath, _dirnames, filenames in os.walk(source_path):
                for f in filenames:
                    size_bytes += os.path.getsize(os.path.join(dirpath, f))
        elif os.path.isfile(source_path):
            size_bytes = os.path.getsize(source_path)

        if config_json is None:
            # Try config.json first, then tokenizer_config.json for tokenizers
            for config_name in ("config.json", "tokenizer_config.json"):
                config_path = os.path.join(source_path, config_name)
                if os.path.isfile(config_path):
                    try:
                        with open(config_path, "r") as f:
                            config_json = f.read()
                        break
                    except Exception:
                        pass

        if architecture is None and config_json is not None:
            try:
                cfg = json.loads(config_json)
                architecture = (
                    cfg.get("_class_name")
                    or (cfg.get("architectures") or [None])[0]
                    or cfg.get("tokenizer_class")
                    or cfg.get("model_type")
                )
            except Exception:
                pass

        db.insert(
            "component",
            {
                "component_type": component_type,
                "content_hash": content_hash,
                "storage_path": source_path,
                "size_bytes": size_bytes,
                "architecture": architecture,
                "config_json": config_json,
            },
        )
        component_id = db.last_insert_rowid()

        return ComponentInfo(
            id=component_id,
            component_type=component_type,
            content_hash=content_hash,
            storage_path=source_path,
            size_bytes=size_bytes,
            architecture=architecture,
            config_json=config_json,
        )

    def get_model_components(self, model_id: int) -> dict[str, ComponentInfo]:
        """Return {component_type: ComponentInfo} for all components of a model."""
        db = self._db()
        rows = db.fetch_all(
            """
            SELECT c.id, c.component_type, c.content_hash, c.storage_path,
                   c.size_bytes, c.architecture, c.config_json
            FROM model_component mc
            JOIN component c ON mc.component_id = c.id
            WHERE mc.model_id = ?
            """,
            (model_id,),
        )
        result: dict[str, ComponentInfo] = {}
        for row in rows:
            info = ComponentInfo(
                id=row[0],
                component_type=row[1],
                content_hash=row[2],
                storage_path=row[3],
                size_bytes=row[4] or 0,
                architecture=row[5],
                config_json=row[6],
            )
            result[info.component_type] = info
        return result

    def register_model_components(self, model_id: int, components: dict[str, int]) -> None:
        """Map model → components in model_component table.

        Args:
            model_id: The model's database ID.
            components: {component_type: component_id} mapping.
        """
        db = self._db()
        for component_type, component_id in components.items():
            db.execute(
                "INSERT OR REPLACE INTO model_component (model_id, component_type, component_id) VALUES (?, ?, ?)",
                (model_id, component_type, component_id),
            )

    def find_compatible_components(self, transformer_architecture: str) -> dict[str, list[ComponentInfo]]:
        """Find existing vae/text_encoder/tokenizer compatible with a transformer architecture."""
        from iartisanz.modules.generation.component_compatibility import ARCHITECTURE_COMPATIBILITY

        compat = ARCHITECTURE_COMPATIBILITY.get(transformer_architecture)
        if compat is None:
            return {}

        db = self._db()
        result: dict[str, list[ComponentInfo]] = {}

        for comp_type in ("text_encoder", "vae", "tokenizer"):
            arch_list = compat.get(comp_type, [])
            if not arch_list:
                continue

            placeholders = ", ".join(["?"] * len(arch_list))
            rows = db.fetch_all(
                f"""
                SELECT id, component_type, content_hash, storage_path,
                       size_bytes, architecture, config_json
                FROM component
                WHERE component_type = ? AND architecture IN ({placeholders})
                """,
                (comp_type, *arch_list),
            )
            if rows:
                result[comp_type] = [
                    ComponentInfo(
                        id=r[0],
                        component_type=r[1],
                        content_hash=r[2],
                        storage_path=r[3],
                        size_bytes=r[4] or 0,
                        architecture=r[5],
                        config_json=r[6],
                    )
                    for r in rows
                ]

        return result

    def get_component_by_hash(self, content_hash: str) -> ComponentInfo | None:
        """Lookup component by hash."""
        db = self._db()
        row = db.fetch_one(
            """
            SELECT id, component_type, content_hash, storage_path,
                   size_bytes, architecture, config_json
            FROM component
            WHERE content_hash = ?
            """,
            (content_hash,),
        )
        if row is None:
            return None
        return ComponentInfo(
            id=row[0],
            component_type=row[1],
            content_hash=row[2],
            storage_path=row[3],
            size_bytes=row[4] or 0,
            architecture=row[5],
            config_json=row[6],
        )

    def model_has_components(self, model_id: int) -> bool:
        """Check if a model has all 4 component mappings."""
        db = self._db()
        row = db.fetch_one(
            "SELECT COUNT(*) FROM model_component WHERE model_id = ?",
            (model_id,),
        )
        return row is not None and row[0] >= len(COMPONENT_TYPES)

    def compact_shared_components(self) -> dict[str, int]:
        """Move shared components to _components/ and replace originals with symlinks.

        Returns dict with stats: {"moved": N, "symlinked": N, "bytes_saved": N}
        """
        import shutil

        db = self._db()
        stats = {"moved": 0, "symlinked": 0, "bytes_saved": 0}

        # Find components used by more than one model
        shared_rows = db.fetch_all(
            """
            SELECT c.id, c.component_type, c.content_hash, c.storage_path, c.size_bytes
            FROM component c
            JOIN model_component mc ON mc.component_id = c.id
            GROUP BY c.id
            HAVING COUNT(DISTINCT mc.model_id) > 1
            """
        )

        for comp_id, comp_type, content_hash, current_path, size_bytes in shared_rows:
            canonical_dir = os.path.join(self.components_base_dir, comp_type, content_hash)

            # Already in _components/?
            if os.path.realpath(current_path).startswith(os.path.realpath(self.components_base_dir)):
                # Still need to symlink any model dirs that point here
                self._symlink_model_dirs(db, comp_id, comp_type, canonical_dir)
                continue

            # Move the component to _components/
            if not os.path.exists(canonical_dir):
                os.makedirs(os.path.dirname(canonical_dir), exist_ok=True)
                if os.path.isdir(current_path) and not os.path.islink(current_path):
                    shutil.copytree(current_path, canonical_dir)
                    stats["moved"] += 1
                    logger.info("Compact: moved %s/%s to %s", comp_type, content_hash[:12], canonical_dir)
                else:
                    continue

            # Update DB to point to canonical location
            db.execute(
                "UPDATE component SET storage_path = ? WHERE id = ?",
                (canonical_dir, comp_id),
            )

            # Replace all model-local copies with symlinks
            link_count = self._symlink_model_dirs(db, comp_id, comp_type, canonical_dir)
            stats["symlinked"] += link_count
            stats["bytes_saved"] += (size_bytes or 0) * max(0, link_count)

        return stats

    def _symlink_model_dirs(self, db: Database, comp_id: int, comp_type: str, canonical_dir: str) -> int:
        """Replace model-local component dirs with symlinks to the canonical dir.

        Returns the number of symlinks created.
        """
        import shutil

        rows = db.fetch_all(
            """
            SELECT m.filepath
            FROM model_component mc
            JOIN model m ON mc.model_id = m.id
            WHERE mc.component_id = ?
            """,
            (comp_id,),
        )

        link_count = 0
        for (model_path,) in rows:
            local_dir = os.path.join(model_path, comp_type)

            # Already a symlink pointing to the right place?
            if os.path.islink(local_dir):
                if os.path.realpath(local_dir) == os.path.realpath(canonical_dir):
                    continue
                # Symlink pointing elsewhere — remove and re-create
                os.remove(local_dir)
            elif os.path.isdir(local_dir):
                # Real directory — verify it's a duplicate, then remove
                shutil.rmtree(local_dir)
                link_count += 1
                logger.info("Compact: removed duplicate %s, symlinking to %s", local_dir, canonical_dir)
            elif not os.path.exists(local_dir):
                # Nothing there, just create the symlink
                pass
            else:
                continue

            os.symlink(canonical_dir, local_dir)
            if not link_count:
                link_count += 1

        return link_count
