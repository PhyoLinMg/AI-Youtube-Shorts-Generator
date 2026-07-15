"""Entry point for the local dashboard.

Usage:
    python dashboard.py

Then open http://127.0.0.1:5050 in a browser.
"""
from shorts_generator.webapp import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)
