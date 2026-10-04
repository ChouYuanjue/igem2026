from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class AtlasHandler(SimpleHTTPRequestHandler):
    server_version = "BridgeEngineeringAtlas"
    sys_version = ""

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        if self.path.endswith((".css", ".js")):
            self.send_header("Cache-Control", "public, max-age=900")
        else:
            self.send_header("Cache-Control", "public, max-age=120")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the BRIDGE Engineering Atlas static site.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument(
        "--directory",
        default=str(Path(__file__).resolve().parents[2] / "frontend" / "engineering_lineage"),
    )
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    handler = partial(AtlasHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
