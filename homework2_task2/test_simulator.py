"""Basic checks for the Task 2 simulated bench."""

import unittest

import numpy as np

try:
    from homework2_task2.simulator import BenchConfig, SimulatedBench
except ModuleNotFoundError:  # Also support running this file from its own folder.
    from simulator import BenchConfig, SimulatedBench


class SimulatorTests(unittest.TestCase):
    def test_shapes_and_regions(self):
        bench = SimulatedBench(BenchConfig(seed=7))
        measurement = bench.measure(bench.gaussian_beam(), noisy=False)
        self.assertEqual(measurement.intensity.shape, (64, 64))
        self.assertEqual(measurement.target.extract(measurement.intensity).shape, (5, 5))
        self.assertEqual(len(measurement.backgrounds), 4)
        self.assertGreater(measurement.enhancement, 0.0)

    def test_noise_changes_repeated_measurements(self):
        bench = SimulatedBench(BenchConfig(seed=7))
        pattern = bench.gaussian_beam()
        first = bench.measure(pattern).intensity
        second = bench.measure(pattern).intensity
        self.assertFalse(np.array_equal(first, second))

    def test_amplitude_validation(self):
        bench = SimulatedBench()
        with self.assertRaises(ValueError):
            bench.measure(np.ones((16, 16)) * 1.1)


if __name__ == "__main__":
    unittest.main()
