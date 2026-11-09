"""One generous limit for waits that only guard against hangs.

Busy hosts can stall a healthy test for many seconds; a stuck one still fails.
Set SLIPAGENT_TEST_DEADLINE (seconds) to change it.
"""

import os

TEST_DEADLINE = float(os.environ.get("SLIPAGENT_TEST_DEADLINE", "60"))
