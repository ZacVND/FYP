"""
@author: ZacVND

Exactly like Hold-out but doesn't show True and Predicted tokens, only the
predicted phrases.
"""

from ie_tools.scripts.hold_out import run

if __name__ == "__main__":
    run(demo=True)
