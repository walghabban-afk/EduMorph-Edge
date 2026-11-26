import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verify_statistics  # noqa: E402


def test_all_reported_statistics_reproduce():
    c = verify_statistics.run()
    failed = [r for r in c.rows if r["status"] == "FAIL"]
    assert not failed, failed
    assert len(c.rows) >= 100
