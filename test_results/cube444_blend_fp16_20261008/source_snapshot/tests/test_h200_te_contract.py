"""CPU contract tests; real TE correctness/performance are separate GPU gates."""
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'hpc' / 'h200_te_benchmark.py'


class Contract(unittest.TestCase):
    def test_mapping_and_activation(self):
        spec = importlib.util.spec_from_file_location('te_contract', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.QKV_MAPPING['weight'], 'qkv_weight_linear')
        self.assertEqual(module.MLP_MAPPING['fc1_weight'], 'ff1_weight_linear')
        self.assertEqual(module.MLP_MAPPING['fc2_bias'], 'ff2_bias')
        self.assertEqual(module.checked_activation('relu'), 'relu')
        with self.assertRaises(ValueError):
            module.checked_activation('silu')


if __name__ == '__main__':
    unittest.main()
