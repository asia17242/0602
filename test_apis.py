import sys
import pandas as pd

print("Python version:", sys.version)
try:
    import yfinance as yf
    print("yfinance installed successfully")
except ImportError:
    print("yfinance NOT installed")

try:
    import FinMind
    print("FinMind installed successfully")
except ImportError:
    print("FinMind NOT installed")
