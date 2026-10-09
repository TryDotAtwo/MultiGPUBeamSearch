"""Independent numerical check of actual native lane tensor artifact."""
import json
import os
from pathlib import Path
import unittest
from tools.full_score_comparison import compare_scores, compare_graph_lane_scores


@unittest.skipUnless(os.environ.get('BEAM_LANE_SCORE_DIR'), 'actual native lane run required')
class NativeLaneScoreTests(unittest.TestCase):
    def test_all_active_lane_scores_match_independent_fp32(self):
        root = Path(os.environ['BEAM_LANE_SCORE_DIR'])
        artifact = root / 'test_results/stream1_lane_scores.json'
        self.assertTrue(artifact.is_file(), 'native probe must preserve full raw lane scores')
        actual = json.loads(artifact.read_text())
        reference = json.loads(Path(os.environ['BEAM_LANE_REFERENCE']).read_text())['scores_fp32']
        micro = int(os.environ.get('BEAM_LANE_MICRO', '1024'))
        lanes = int(os.environ.get('BEAM_LANE_COUNT', '4'))
        if 'parent_bases' in actual or 'active_counts' in actual:
            result = compare_graph_lane_scores(reference, actual, microbatch=micro, lanes=lanes)
        else:
            # Replay historical 1024x4 evidence only; a legacy artifact cannot
            # admit any newly requested profile lacking explicit row metadata.
            self.assertEqual((micro, lanes), (1024, 4))
            self.assertEqual(set(actual), {'microbatch', 'lane_count', 'scores'})
            self.assertEqual(actual['microbatch'], 1024)
            self.assertEqual(actual['lane_count'], 4)
            expected = [reference[(lane + row) % 8]
                        for lane in range(4) for row in range(1024 - lane)]
            result = compare_scores(expected, actual['scores'], atol=.05, rtol=.001,
                                    top_k=4, min_topk_overlap=.75)
        self.assertEqual(result['status'], 'pass', result)
        self.assertEqual(result['elements_compared'],
                         sum(micro - lane % micro for lane in range(lanes)) * 24)
