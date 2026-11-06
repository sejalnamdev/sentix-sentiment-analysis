"""
Lightweight Local HTTP Server for the Sentix Sentiment Dashboard.

Usage:
    python serve_dashboard.py
"""
import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUTPUT_DIR), **kwargs)


def main():
    print("=" * 60)
    print(f"🚀 Sentix Dashboard Server running at: http://localhost:{PORT}/sentiment_dashboard.html")
    print("=" * 60)
    print("Press Ctrl + C to stop the server.\n")

    try:
        webbrowser.open(f"http://localhost:{PORT}/sentiment_dashboard.html")
    except Exception:
        pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Server stopped.")


if __name__ == "__main__":
    main()
