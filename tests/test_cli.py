import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from aether_measurement.cli import main
from aether_measurement.simulation import SimulationConfig, simulate


class CLITests(unittest.TestCase):
    def test_measure_npz_and_json_roundtrip(self):
        data = simulate("signal", SimulationConfig(subjects=3, trials=24, channels=4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path / "input.npz", **{key: data[key] for key in ("covariances", "labels", "subjects")})
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["measure", "--input", str(path / "input.npz"), "--out", str(path / "report.json")])
            self.assertEqual(code, 0)
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["trials"], 72)
            self.assertNotIn("scores", report)

    def test_corrupt_input_is_readable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, wrong=np.ones(4))
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                self.assertEqual(main(["measure", "--input", str(path)]), 1)
            self.assertIn("error:", errors.getvalue())

    def test_ledger_dispatch(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["ledger", "init", "--out", directory]), 0)
            self.assertTrue((Path(directory) / "event_ledger.csv").exists())

    def test_npy_is_rejected_as_wrong_container(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.npy"
            np.save(path, np.ones(4))
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(["measure", "--input", str(path)]), 1)
            self.assertIn("NPZ archive", errors.getvalue())

    def test_corrupt_zip_is_readable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.npz"
            path.write_bytes(b"PK\x03\x04incomplete")
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(["measure", "--input", str(path)]), 1)
            self.assertIn("error:", errors.getvalue())
