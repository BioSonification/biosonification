"""
Production WSGI entrypoint for BioSonification web application.
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
except ImportError:
    print("python-dotenv not installed, skipping .env file")

# Import Flask app
from web.app import app

# Configure production settings
app.config['ENV'] = 'production'
app.config['DEBUG'] = False
app.config['TESTING'] = False

if __name__ == '__main__':
    from waitress import serve

    host = os.getenv("BIOSONIFICATION_HOST", "127.0.0.1")
    port = int(os.getenv("BIOSONIFICATION_PORT", "5001"))
    threads = int(os.getenv("BIOSONIFICATION_THREADS", "4"))

    print(f"Starting Waitress WSGI server")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Threads: {threads}")
    print(f"  Access: http://{host}:{port}")

    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        url_scheme='http',
        ident='BioSonification',
        # Timeouts
        channel_timeout=300,  # 5 minutes for long generations
        # Limits
        max_request_body_size=10485760,  # 10MB
    )
