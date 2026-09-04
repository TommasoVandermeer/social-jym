"""Versioned, configuration-bound artifacts for long-running training scripts."""

from __future__ import annotations

import hashlib
import json
import pickle
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

    def figure_path(self, artifact_type: str, suffix: str = ".eps") -> Path:
        return self.root / f"{artifact_type}-{self.artifact_id(artifact_type)[:16]}{suffix}"

    def _dependency_ids(self, dependencies: Iterable[str]) -> dict[str, str]:
        return {dependency: self.artifact_id(dependency) for dependency in dependencies}

    def load(self, artifact_type: str, dependencies: Iterable[str] = ()) -> Any:
        dependencies = tuple(dependencies)
        path = self.path(artifact_type)
        with path.open("rb") as artifact_file:
            envelope = pickle.load(artifact_file)
        expected_dependencies = self._dependency_ids(dependencies)
        if not isinstance(envelope, dict) or "payload" not in envelope:
            raise ValueError(f"Legacy or malformed artifact rejected: {path}")
        expected = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "namespace": self.namespace,
            "experiment_hash": self.experiment_hash,
            "artifact_type": artifact_type,
            "artifact_id": self.artifact_id(artifact_type),
            "dependencies": expected_dependencies,
        }
        for key, expected_value in expected.items():
            if envelope.get(key) != expected_value:
                raise ValueError(
                    f"Artifact metadata mismatch for {path}: {key}="
                    f"{envelope.get(key)!r}, expected {expected_value!r}"
                )
        for dependency, dependency_id in expected_dependencies.items():
            dependency_path = self.path(dependency)
            try:
                with dependency_path.open("rb") as dependency_file:
                    dependency_envelope = pickle.load(dependency_file)
            except (FileNotFoundError, OSError, pickle.PickleError, EOFError) as error:
                raise ValueError(
                    f"Missing or unreadable dependency {dependency_path} for {path}"
                ) from error
            if (
                not isinstance(dependency_envelope, dict)
                or dependency_envelope.get("schema_version") != ARTIFACT_SCHEMA_VERSION
                or dependency_envelope.get("namespace") != self.namespace
                or dependency_envelope.get("experiment_hash") != self.experiment_hash
                or dependency_envelope.get("artifact_type") != dependency
                or dependency_envelope.get("artifact_id") != dependency_id
            ):
                raise ValueError(f"Invalid dependency {dependency_path} for {path}")
        return envelope["payload"]

    def is_valid(self, artifact_type: str, dependencies: Iterable[str] = ()) -> bool:
        try:
            self.load(artifact_type, dependencies)
        except (FileNotFoundError, OSError, pickle.PickleError, EOFError, ValueError):
            return False
        return True

    def save(self, artifact_type: str, payload: Any, dependencies: Iterable[str] = ()) -> Path:
        dependencies = tuple(dependencies)
        for dependency in dependencies:
            dependency_path = self.path(dependency)
            try:
                with dependency_path.open("rb") as dependency_file:
                    dependency_envelope = pickle.load(dependency_file)
            except (FileNotFoundError, OSError, pickle.PickleError, EOFError) as error:
                raise ValueError(
                    f"Cannot save {artifact_type}: dependency {dependency_path} is unreadable"
                ) from error
            if (
                not isinstance(dependency_envelope, dict)
                or dependency_envelope.get("schema_version") != ARTIFACT_SCHEMA_VERSION
                or dependency_envelope.get("namespace") != self.namespace
                or dependency_envelope.get("experiment_hash") != self.experiment_hash
                or dependency_envelope.get("artifact_type") != dependency
                or dependency_envelope.get("artifact_id") != self.artifact_id(dependency)
            ):
                raise ValueError(
                    f"Cannot save {artifact_type}: dependency {dependency_path} is invalid"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "namespace": self.namespace,
            "experiment_hash": self.experiment_hash,
            "experiment_config": self.experiment_config,
            "artifact_type": artifact_type,
            "artifact_id": self.artifact_id(artifact_type),
            "dependencies": self._dependency_ids(dependencies),
            "payload": payload,
        }
        path = self.path(artifact_type)
        with path.open("wb") as artifact_file:
            pickle.dump(envelope, artifact_file)
        return path
