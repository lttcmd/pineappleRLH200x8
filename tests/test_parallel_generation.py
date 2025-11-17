import numpy as np
import pytest

try:
    import ofc_cpp as _CPP
except Exception:
    _CPP = None

from train import ParallelEpisodeGenerator, _random_episode_worker


@pytest.mark.skipif(_CPP is None, reason="C++ backend not available")
def test_parallel_generator_single_worker_matches_direct():
    total = 6
    seed = 1234
    direct_encoded, direct_offsets, direct_scores = _random_episode_worker((seed, total))
    gen = ParallelEpisodeGenerator(num_workers=1)
    encoded, offsets, scores = gen.generate_random(total, seed)
    gen.close()
    np.testing.assert_allclose(encoded, direct_encoded)
    np.testing.assert_array_equal(offsets, direct_offsets)
    np.testing.assert_allclose(scores, direct_scores)


@pytest.mark.skipif(_CPP is None, reason="C++ backend not available")
def test_parallel_generator_offsets_monotonic():
    total = 12
    seed = 9876
    gen = ParallelEpisodeGenerator(num_workers=2)
    encoded, offsets, scores = gen.generate_random(total, seed)
    gen.close()
    assert encoded.shape[0] == offsets[-1]
    assert len(offsets) == len(scores) + 1
    diffs = np.diff(offsets)
    assert np.all(diffs >= 0)
    assert np.any(diffs > 0)

