import os

# Slack webhook URL is read from an environment variable.
# On GitHub Actions, set this as a Secret named SLACK_WEBHOOK_URL.
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Your trading account size in dollars
ACCOUNT_EQUITY = 2000.0

# Fraction of account risked per trade (0.01 = 1%)
RISK_PER_TRADE_PCT = 0.01

# File that contains the list of tickers to scan
UNIVERSE_FILE = "universe.txt"

# Trade logging file
LOG_FILE = "trades.csv"

# Strategy risk mode: "conservative", "moderate", or "aggressive"
RISK_MODE = "moderate"

