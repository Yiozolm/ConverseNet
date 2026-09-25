import copy
import unittest
from audit_training_time import assert_matched, partition, step_phases


class TimingAccountingTests(unittest.TestCase):
    def test_initial_checkpoint_counted_once(self):
        result = partition(30, 5, 1, 3, 2, 15, 4)
        self.assertEqual(result['setup_excluding_initial_checkpoint'], 4)
        self.assertEqual(result['checkpoint'], 3)
        self.assertEqual(result['unattributed_process'], 2)
        self.assertEqual(sum(result.values()), 30)

    def test_inconsistent_parent_rejected(self):
        with self.assertRaises(ValueError):
            partition(10, 5, 1, 3, 2, 15, 4)

    def test_step_children_are_not_added_twice(self):
        row = dict(h2d_wall_ms=1, forward_backward_wall_ms=10,
                   finite_check=dict(wall_ms=2), optimizer=dict(wall_ms=3), training_step_wall_ms=18)
        result = step_phases(row)
        self.assertEqual(result['unattributed_step'], 2)
        self.assertEqual(sum(result.values()), 18)
        row['training_step_wall_ms'] = 1
        with self.assertRaises(ValueError):
            step_phases(row)

    def test_same_update_requires_same_input_and_state_diagnostics(self):
        row = dict(step=1, data_step=0, batch_sha256='a', samples=4, microbatches=1,
                   optimizer_steps=1, loss=.1, grad_l2_norm=.2)
        assert_matched([row], [row.copy()])
        for key in ('batch_sha256', 'loss', 'grad_l2_norm', 'step'):
            changed = copy.deepcopy(row)
            changed[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_matched([row], [changed])
        with self.assertRaises(ValueError):
            assert_matched([row, row], [row, row])


if __name__ == '__main__':
    unittest.main()
