"""Versioned, configuration-bound artifacts for long-running training scripts."""

from __future__ import annotations

import hashlib
import json
import pickle
import gc
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ARTIFACT_SCHEMA_VERSION = 2


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_hash(value: Any) -> str:
    serialized = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ArtifactStore:
    """Store pickles in a namespace isolated by an experiment signature."""

    def __init__(self, root: str | Path, namespace: str, experiment_config: dict):
        self.namespace = namespace
        self.experiment_config = _jsonable(experiment_config)
        self.experiment_hash = stable_hash(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "namespace": namespace,
                "config": self.experiment_config,
            }
        )
        self.root = Path(root) / namespace / self.experiment_hash[:16]

    def artifact_id(self, artifact_type: str) -> str:
        return stable_hash(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "experiment_hash": self.experiment_hash,
                "artifact_type": artifact_type,
            }
        )

    def path(self, artifact_type: str) -> Path:
        return self.root / f"{artifact_type}-{self.artifact_id(artifact_type)[:16]}.pkl"

    def metadata_path(self, artifact_type: str) -> Path:
        return self.path(artifact_type).with_suffix(".metadata.json")

    def figure_path(self, artifact_type: str, suffix: str = ".eps") -> Path:
        return self.root / f"{artifact_type}-{self.artifact_id(artifact_type)[:16]}{suffix}"

    def _dependency_ids(self, dependencies: Iterable[str]) -> dict[str, str]:
        return {dependency: self.artifact_id(dependency) for dependency in dependencies}

    def _expected_metadata(self, artifact_type: str, dependencies: Iterable[str]) -> dict:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "namespace": self.namespace,
            "experiment_hash": self.experiment_hash,
            "experiment_config": self.experiment_config,
            "artifact_type": artifact_type,
            "artifact_id": self.artifact_id(artifact_type),
            "dependencies": self._dependency_ids(dependencies),
        }

    def _validate_metadata(
        self,
        metadata: dict,
        artifact_type: str,
        dependencies: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(metadata, dict):
            raise ValueError(f"Malformed metadata for {self.path(artifact_type)}")
        expected = self._expected_metadata(
            artifact_type,
            metadata.get("dependencies", {}).keys() if dependencies is None else dependencies,
        )
        if dependencies is None:
            # Dependency artifacts only need their identity checked here. Their
            # own dependency list is checked when they are used as a stage.
            expected.pop("dependencies")
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"Artifact metadata mismatch for {self.path(artifact_type)}: "
                    f"{key}={metadata.get(key)!r}, expected {expected_value!r}"
                )

    def _write_metadata(self, artifact_type: str, metadata: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_path = self.metadata_path(artifact_type)
        temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)

    def _metadata_from_existing_pickle(self, artifact_type: str) -> dict:
        """Migrate a schema-v2 pickle without reconstructing arrays on the GPU."""
        path = self.path(artifact_type)
        try:
            import jax

            cpu = next(device for device in jax.devices() if device.platform == "cpu")
            with jax.default_device(cpu), path.open("rb") as artifact_file:
                envelope = pickle.load(artifact_file)
        except (ImportError, StopIteration):
            with path.open("rb") as artifact_file:
                envelope = pickle.load(artifact_file)
        if not isinstance(envelope, dict) or "payload" not in envelope:
            raise ValueError(f"Legacy or malformed artifact rejected: {path}")
        metadata = {key: value for key, value in envelope.items() if key != "payload"}
        # Release potentially multi-gigabyte CPU arrays before inspecting the
        # next dependency during one-time sidecar migration.
        del envelope
        gc.collect()
        self._validate_metadata(metadata, artifact_type, dependencies=None)
        self._write_metadata(artifact_type, metadata)
        return metadata

    def _load_metadata(self, artifact_type: str) -> dict:
        path = self.path(artifact_type)
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata_path = self.metadata_path(artifact_type)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._validate_metadata(metadata, artifact_type, dependencies=None)
            return metadata
        return self._metadata_from_existing_pickle(artifact_type)

    def _validate_dependencies(self, dependencies: Iterable[str]) -> None:
        for dependency in dependencies:
            try:
                self._load_metadata(dependency)
            except (FileNotFoundError, OSError, pickle.PickleError, EOFError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"Missing or invalid dependency {self.path(dependency)}"
                ) from error

    def load(self, artifact_type: str, dependencies: Iterable[str] = ()) -> Any:
        dependencies = tuple(dependencies)
        path = self.path(artifact_type)
        with path.open("rb") as artifact_file:
            envelope = pickle.load(artifact_file)
        expected_dependencies = self._dependency_ids(dependencies)
        if not isinstance(envelope, dict) or "payload" not in envelope:
            raise ValueError(f"Legacy or malformed artifact rejected: {path}")
        metadata = {key: value for key, value in envelope.items() if key != "payload"}
        self._validate_metadata(metadata, artifact_type, dependencies)
        self._write_metadata(artifact_type, metadata)
        self._validate_dependencies(expected_dependencies)
        return envelope["payload"]

    def is_valid(self, artifact_type: str, dependencies: Iterable[str] = ()) -> bool:
        try:
            metadata = self._load_metadata(artifact_type)
            self._validate_metadata(metadata, artifact_type, dependencies)
            self._validate_dependencies(dependencies)
        except (FileNotFoundError, OSError, pickle.PickleError, EOFError, ValueError, RuntimeError):
            return False
        return True

    def save(self, artifact_type: str, payload: Any, dependencies: Iterable[str] = ()) -> Path:
        dependencies = tuple(dependencies)
        self._validate_dependencies(dependencies)
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self._expected_metadata(artifact_type, dependencies)
        envelope = {**metadata, "payload": payload}
        path = self.path(artifact_type)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as artifact_file:
            pickle.dump(envelope, artifact_file)
        temporary_path.replace(path)
        self._write_metadata(artifact_type, metadata)
        return path
