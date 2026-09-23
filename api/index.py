import sys
import os

# Add parent directory to sys.path so main and utils are discoverable
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from main import app

# Vercel looks for 'app' or 'handler' as the WSGI callable
handler = app
