import os
import sys
import subprocess

if __name__ == "__main__":
    backend_main = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'main.py'))
    print(f"Starting AI Crowd Face Surveillance Console from {backend_main}...")
    try:
        subprocess.run([sys.executable, backend_main], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")
