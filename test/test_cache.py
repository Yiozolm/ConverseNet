"""Check cache invalidation, tensor identity and inference/training transitions."""
import sys
import unittest
import torch
import test_correctness as checks

if __name__ == "__main__":
    checks.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    names = ("test_cache_mutation_and_training_transition",
             "test_cache_tensor_identity_and_inference_tensors", "test_nondefault_cuda_stream")
    suite = unittest.TestSuite(checks.Correctness(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
