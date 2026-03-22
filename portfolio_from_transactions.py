#!/usr/bin/env python3
"""
Compatibility entrypoint for the documented transaction-based portfolio tool.

The main implementation lives in `portfolio_backcast.py`, but the README/docstring
examples instruct users to run `portfolio_from_transactions.py`. Keeping this thin
wrapper ensures that command continues to work and always executes the current code.
"""

from portfolio_backcast import main


if __name__ == "__main__":
    main()
