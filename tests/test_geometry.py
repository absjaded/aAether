import unittest

import numpy as np

from aether_measurement.geometry import covariance_features, validate_covariances
from aether_measurement.instrument import grouped_readout, measure
from aether_measurement.simulation import MODES, SimulationConfig, simulate


class GeometryTests(unittest.TestCase):
    def test_log_whitening_known_diagonals(self):
        covs = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 4.0])])
        actual = covariance_features(covs, np.array([0, 0]))
        expected = [[np.log(0.5), 0, np.log(2 / 3)], [np.log(1.5), 0, np.log(4 / 3)]]
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_off_diagonals_preserve_frobenius_norm(self):
        # Exp(A), Exp(-A) have reference cosh(a) I, so their log-whitened
        # off-diagonal entries remain +/-a in a two-dimensional construction.
        a = 0.2
        covs = np.array([[[np.cosh(a), np.sinh(a)], [np.sinh(a), np.cosh(a)]],
                         [[np.cosh(a), -np.sinh(a)], [-np.sinh(a), np.cosh(a)]]])
        features = covariance_features(covs, np.array([0, 0]))
        np.testing.assert_allclose(features[:, 1], [np.sqrt(2) * a, -np.sqrt(2) * a])

    def test_physical_scale_invariance(self):
        covs = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 4.0])])
        np.testing.assert_allclose(covariance_features(covs, np.zeros(2)),
                                   covariance_features(covs * 1e-24, np.zeros(2)), atol=1e-13)

    def test_subject_batch_alignment(self):
        covs = np.array([np.eye(2), np.eye(2) * 10, np.eye(2) * 3, np.eye(2) * 30])
        actual = covariance_features(covs, np.array(["a", "b", "a", "b"]))
        np.testing.assert_allclose(actual[[0, 2]], actual[[1, 3]], atol=1e-14)

    def test_input_covariances_are_not_modified(self):
        covs = np.array([np.eye(3), np.eye(3) * 2])
        before = covs.copy()
        covariance_features(covs, np.zeros(2))
        np.testing.assert_array_equal(covs, before)

    def test_bad_covariances_fail(self):
        for covs in [np.zeros((0, 2, 2)), np.ones((2, 2)), np.ones((2, 2, 3)),
                     np.array([[[1, 2], [0, 1]]]), np.array([np.diag([1, 0])]),
                     np.array([np.diag([1, -1])]), np.full((1, 2, 2), np.nan),
                     np.ones((1, 2, 2), dtype=complex)]:
            with self.subTest(shape=covs.shape), self.assertRaises(ValueError):
                validate_covariances(covs)

    def test_subject_identifiers_are_validated(self):
        for groups in [np.zeros(3), np.array([np.nan, 0]), np.array(["", "s"]), np.array([None, None])]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                covariance_features(np.array([np.eye(2), np.eye(2)]), groups)


class ReadoutTests(unittest.TestCase):
    def test_interleaved_oof_predictions_and_subject_holdout(self):
        labels = np.tile([0, 1], 40)
        groups = np.repeat(np.arange(4), 20)
        report = grouped_readout((labels * 2 - 1)[:, None], labels, groups)
        self.assertEqual(report["auc"], 1)
        self.assertEqual(len(report["scores"]), len(labels))
        np.testing.assert_array_equal(np.array(report["scores"]) > 0.5, labels.astype(bool))
        for fold in report["folds"]:
            self.assertFalse(set(fold["train_subjects"]) & set(fold["test_subjects"]))
            self.assertEqual(sum(np.array(report["fold_ids"]) == fold["fold"]), fold["test_rows"])

    def test_pca_and_scaling_only_see_training_rows(self):
        from unittest.mock import patch
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        seen_pca, seen_scaler = [], []
        pca_fit, scaler_fit = PCA.fit_transform, StandardScaler.fit

        def pca_spy(instance, x, *args, **kwargs):
            seen_pca.append(len(x))
            return pca_fit(instance, x, *args, **kwargs)

        def scaler_spy(instance, x, *args, **kwargs):
            seen_scaler.append(len(x))
            return scaler_fit(instance, x, *args, **kwargs)

        with patch.object(PCA, "fit_transform", pca_spy), patch.object(StandardScaler, "fit", scaler_spy):
            grouped_readout(np.arange(80)[:, None], np.tile([0, 1], 40), np.repeat(np.arange(4), 20))
        self.assertEqual(seen_pca, [60] * 4)
        self.assertEqual(seen_scaler, [60] * 4)

    def test_constant_features_have_chance_auc(self):
        report = grouped_readout(np.ones((20, 3)), np.tile([0, 1], 10), np.repeat([0, 1], 10))
        self.assertEqual(report["auc"], 0.5)

    def test_missing_training_class_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "training fold"):
            grouped_readout(np.arange(8)[:, None], np.repeat([0, 1], 4), np.repeat([0, 1], 4))

    def test_invalid_inputs_fail(self):
        x = np.arange(8)[:, None]
        y = np.tile([0, 1], 4)
        groups = np.repeat([0, 1], 4)
        cases = [(np.full((8, 1), np.inf), y, groups, {}),
                 (x, np.zeros(8), groups, {}), (x, y, np.zeros(8), {}),
                 (x, y + 1, groups, {}), (x, y, groups, {"components": 0}),
                 (x, y, groups, {"splits": 1})]
        for args in cases:
            with self.subTest(args=args[3]), self.assertRaises(ValueError):
                grouped_readout(*args[:3], **args[3])


class SimulationTests(unittest.TestCase):
    def test_all_modes_produce_finite_spd_batches_and_repeat(self):
        config = SimulationConfig(subjects=3, trials=16, channels=4)
        for mode in MODES:
            with self.subTest(mode=mode):
                data = simulate(mode, config)
                validate_covariances(data["covariances"])
                np.testing.assert_array_equal(data["covariances"], simulate(mode, config)["covariances"])
                self.assertEqual(data["labels"].shape, (48,))
                self.assertEqual(data["subjects"].shape, (48,))

    def test_tiny_measurement_smoke(self):
        data = simulate("signal", SimulationConfig(subjects=4, trials=24, channels=4))
        report = measure(data["covariances"], data["labels"], data["subjects"])
        self.assertTrue(0 <= report["auc"] <= 1)
        self.assertEqual(report["trials"], 96)
        self.assertEqual(report["feature_dimension"], 10)
        self.assertIn("held-out subject", report["calibration"])

    def test_configuration_validation(self):
        for config in [SimulationConfig(channels=2), SimulationConfig(subjects=1),
                       SimulationConfig(violation_rate=0), SimulationConfig(response_label_rho=1),
                       SimulationConfig(signal_effect=-1), SimulationConfig(seed=-1)]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                simulate("signal", config)
