"""Servidor estático del tablero.

    python src/dashboard/serve.py [puerto]

La API se levanta aparte, desde la raíz del repo:
    uvicorn src.api.main:app --port 8000
"""
from __future__ import annotations

import functools
import http.server
import socketserver
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PORT = 8080


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  %s\n" % (fmt % args))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    handler = functools.partial(Handler, directory=str(HERE))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Dashboard en http://127.0.0.1:{port}  (Ctrl+C para detener)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDetenido.")


if __name__ == "__main__":
    main()
