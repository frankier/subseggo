#!/usr/bin/env python3
"""Mock of https://api.mistral.ai/v1/audio/transcriptions for testing retries/resume.

Env:
  PORT        - listen port (default 8799)
  FAIL_FIRST  - fail the first N requests with HTTP 520 (default 0)
  FAIL_FILE   - if this file exists, always fail with 520 (default: none)

Each successful POST /v1/audio/transcriptions returns two canned segments with
chunk-local timestamps (the pipeline adds the chunk offset itself).
Logs every request to stdout.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("PORT", "8799"))
FAIL_FIRST = int(os.environ.get("FAIL_FIRST", "0"))
FAIL_FILE = os.environ.get("FAIL_FILE", "")
count = 0


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        global count
        count += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # consume multipart body
        if count <= FAIL_FIRST or (FAIL_FILE and os.path.exists(FAIL_FILE)):
            body = b"<!DOCTYPE html><html>520 origin error</html>"
            self.send_response(520)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f"req {count}: -> 520 (FAIL_FIRST/FAIL_FILE)", flush=True)
            return
        body = json.dumps({
            "model": "voxtral-small-transcribe-3",
            "text": "hello world from mock chunk.",
            "segments": [
                {"start": 0.0, "end": 10.0, "text": " hello", "speaker_id": None},
                {"start": 10.0, "end": 20.0, "text": " world from mock chunk.", "speaker_id": None},
            ],
            "usage": {"prompt_audio_seconds": 20, "prompt_tokens": 4,
                      "total_tokens": 10, "completion_tokens": 6},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"req {count}: -> 200", flush=True)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"mock mistral server on 127.0.0.1:{PORT} (FAIL_FIRST={FAIL_FIRST})", flush=True)
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
