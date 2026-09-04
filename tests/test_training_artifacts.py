import pickle
import tempfile
import unittest
from pathlib import Path

from socialjym.utils.training_artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_round_trip_and_dependency_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory, "jessi_s2r", {"n_humans": 4})
            store.save("raw", {"value": 1})
            store.save("model", {"value": 2}, dependencies=("raw",))
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


if __name__ == "__main__":
    unittest.main()
