import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from socialjym.utils.training_artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_round_trip_and_dependency_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory, "jessi_s2r", {"n_humans": 4})
            store.save("raw", {"value": 1})
            store.save("model", {"value": 2}, dependencies=("raw",))
            self.assertTrue(store.metadata_path("raw").is_file())
            self.assertTrue(store.metadata_path("model").is_file())
            self.assertTrue(store.is_valid("model", dependencies=("raw",)))
            self.assertEqual(store.load("model", dependencies=("raw",))["value"], 2)
            store.path("raw").unlink()
            self.assertFalse(store.is_valid("model", dependencies=("raw",)))

    def test_changed_config_uses_different_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ArtifactStore(directory, "jessi_s2r", {"n_humans": 4})
            second = ArtifactStore(directory, "jessi_s2r", {"n_humans": 5})
            self.assertNotEqual(first.path("dataset"), second.path("dataset"))

    def test_legacy_pickle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory, "jessi_s2r", {"version": 2})
            store.root.mkdir(parents=True)
            with store.path("legacy").open("wb") as artifact_file:
                pickle.dump({"old": "payload"}, artifact_file)
            self.assertFalse(store.is_valid("legacy"))

    def test_saving_a_dependent_artifact_does_not_unpickle_dependency_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory, "jessi_s2r", {"n_humans": 4})
            store.save("dataset", {"large": [1, 2, 3]})
            with patch(
                "socialjym.utils.training_artifacts.pickle.load",
                side_effect=AssertionError("dependency payload was unpickled"),
            ):
                store.save("controller", {"value": 1}, dependencies=("dataset",))
            self.assertTrue(store.is_valid("controller", dependencies=("dataset",)))

    def test_existing_schema_v2_pickle_gets_a_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory, "jessi_s2r", {"n_humans": 4})
            store.save("dataset", {"value": 1})
            store.metadata_path("dataset").unlink()
            self.assertTrue(store.is_valid("dataset"))
            self.assertTrue(store.metadata_path("dataset").is_file())


if __name__ == "__main__":
    unittest.main()
