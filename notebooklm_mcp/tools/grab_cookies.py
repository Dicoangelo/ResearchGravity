#!/usr/bin/env python3
"""NotebookLM Cookie Grabber — Zero Keychain.

Opens a tiny local server, opens your browser to NotebookLM via a redirect trick,
captures the cookies from the redirect, saves them, done.

Usage (run in Terminal):
    python3 grab_cookies.py

What happens:
1. Starts a local HTTPS server on port 8899
2. Opens notebooklm.google.com in your default browser
3. Waits for you to be on the NLM page (already logged in)
4. You paste a one-liner from the console
5. Cookies saved. Never do this again for ~30 days.
"""

import http.server
import json
import os
import ssl
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse


AUTH_STATE_DIR = Path.home() / ".ucw" / "notebooklm" / "auth_state"
COOKIE_FILE = AUTH_STATE_DIR / "cookies.txt"
PORT = 8899


class CookieHandler(http.server.BaseHTTPRequestHandler):
    """Handle cookie POST from the browser bookmarklet."""

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            cookies = data.get('cookies', '')

            if cookies and len(cookies) > 100:
                # Save cookies
                AUTH_STATE_DIR.mkdir(parents=True, exist_ok=True)
                COOKIE_FILE.write_text(cookies)
                COOKIE_FILE.chmod(0o600)

                # Count cookie pairs
                count = len([c for c in cookies.split('; ') if '=' in c])

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'ok',
                    'message': f'Saved {count} cookies to {COOKIE_FILE}'
                }).encode())

                print(f"\n  SAVED {count} cookies to {COOKIE_FILE}")
                print(f"  Cookie string: {len(cookies)} bytes")
                print(f"\n  You can close this terminal now.")

                # Shutdown server after response
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'{"status":"error","message":"No cookies received"}')
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Serve the extraction page."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(EXTRACTION_PAGE.encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


EXTRACTION_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>NotebookLM Cookie Extractor</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #0a0a0a; color: #e0e0e0; }
h1 { color: #7c3aed; }
.step { background: #1a1a2e; padding: 16px; border-radius: 8px; margin: 12px 0; border-left: 3px solid #7c3aed; }
.step-num { color: #7c3aed; font-weight: bold; font-size: 1.2em; }
code { background: #2d1b69; padding: 2px 8px; border-radius: 4px; font-size: 0.95em; }
pre { background: #1e1e3f; padding: 12px; border-radius: 8px; overflow-x: auto; cursor: pointer; }
pre:hover { background: #2a2a4f; }
.status { padding: 16px; border-radius: 8px; margin: 20px 0; text-align: center; font-size: 1.1em; }
.waiting { background: #1a1a2e; border: 1px solid #7c3aed; }
.success { background: #0d2818; border: 1px solid #22c55e; color: #22c55e; }
.error { background: #2d0a0a; border: 1px solid #ef4444; color: #ef4444; }
button { background: #7c3aed; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1em; }
button:hover { background: #6d28d9; }
</style>
</head>
<body>
<h1>NotebookLM Cookie Extractor</h1>
<p>One-time setup. Takes 30 seconds.</p>

<div class="step">
<span class="step-num">1.</span> Open <a href="https://notebooklm.google.com" target="_blank" style="color:#a78bfa">notebooklm.google.com</a> in a new tab (make sure you're logged in)
</div>

<div class="step">
<span class="step-num">2.</span> On that page, press <code>Cmd+Option+J</code> to open the Console
</div>

<div class="step">
<span class="step-num">3.</span> Paste this one-liner and press Enter:
<pre id="snippet" onclick="copySnippet()">fetch('http://localhost:""" + str(PORT) + """',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cookies:document.cookie})}).then(r=>r.json()).then(d=>console.log('Done:',d.message)).catch(e=>console.error(e))</pre>
<small style="color:#888">Click to copy</small>
</div>

<div id="status" class="status waiting">
Waiting for cookies...
</div>

<script>
function copySnippet() {
    navigator.clipboard.writeText(document.getElementById('snippet').innerText);
    document.getElementById('snippet').style.borderColor = '#22c55e';
    setTimeout(() => document.getElementById('snippet').style.borderColor = '', 1000);
}

// Poll for completion
setInterval(async () => {
    try {
        const r = await fetch('http://localhost:""" + str(PORT) + """/status');
        // If server is still running, we're waiting
    } catch(e) {
        // Server shut down = success
        document.getElementById('status').className = 'status success';
        document.getElementById('status').innerHTML = 'Cookies saved! You can close this page.';
    }
}, 2000);
</script>
</body>
</html>"""


def main():
    print(f"""
  ╔══════════════════════════════════════════╗
  ║   NotebookLM Cookie Extractor           ║
  ║   No Keychain. No DevTools digging.     ║
  ╚══════════════════════════════════════════╝

  Opening browser...

  Follow the 3 steps on the page.
  (This server auto-shuts down when done)
""")

    server = http.server.HTTPServer(('127.0.0.1', PORT), CookieHandler)

    # Open browser to our local page
    webbrowser.open(f'http://localhost:{PORT}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    # Verify
    if COOKIE_FILE.exists():
        cookies = COOKIE_FILE.read_text().strip()
        if cookies:
            print(f"\n  Cookies saved ({len(cookies)} bytes)")
            print(f"  File: {COOKIE_FILE}")
            print(f"\n  Now use 'auto_auth' or 'save_auth_tokens' in Claude Code.")
            return 0

    print("\n  No cookies captured. Try again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
