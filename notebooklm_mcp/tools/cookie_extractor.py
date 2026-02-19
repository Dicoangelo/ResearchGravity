"""Auto-extract NotebookLM cookies from Chrome cookie database.

Reads the Chrome cookie DB directly, decrypts using macOS Keychain,
and saves to the NotebookLM auth cache. Eliminates manual DevTools extraction.

Usage:
    python3 -m notebooklm_mcp.tools.cookie_extractor          # Extract + save
    python3 -m notebooklm_mcp.tools.cookie_extractor --test    # Extract + test auth
    python3 -m notebooklm_mcp.tools.cookie_extractor --print   # Just print cookie string

On first run, macOS will prompt to allow Keychain access to "Chrome Safe Storage".
Click "Always Allow" once — subsequent runs will be silent.
"""

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Domains we need cookies for
GOOGLE_DOMAINS = (".google.com", ".notebooklm.google.com", ".accounts.google.com")

# Critical auth cookies (must have these)
REQUIRED_COOKIES = {"SID", "HSID", "SSID", "SAPISID", "__Secure-1PSID"}

# All cookies to extract
COOKIE_NAMES = {
    # Core Google auth (long-lived)
    "SID", "HSID", "SSID", "APISID", "SAPISID",
    "__Secure-1PSID", "__Secure-3PSID",
    "__Secure-1PAPISID", "__Secure-3PAPISID",
    # Session cookies (rotate)
    "SIDCC", "__Secure-1PSIDCC", "__Secure-3PSIDCC",
    "__Secure-1PSIDTS", "__Secure-3PSIDTS",
    # NotebookLM-specific
    "OSID", "__Secure-OSID",
    # Misc
    "NID", "AEC", "__Secure-BUCKET", "SEARCH_SAMESITE",
    "_gcl_au", "_ga", "ADS_VISITOR_ID",
}

CHROME_COOKIE_DB = Path.home() / "Library/Application Support/Google/Chrome/Default/Cookies"
CHROME_SAFE_STORAGE_SERVICE = "Chrome Safe Storage"
CHROME_SAFE_STORAGE_ACCOUNT = "Chrome"


def _get_chrome_encryption_key() -> bytes:
    """Get Chrome's AES encryption key from macOS Keychain.

    First run will show a macOS prompt — click "Always Allow" to never be asked again.
    """
    try:
        key = subprocess.check_output(
            ["security", "find-generic-password", "-w",
             "-s", CHROME_SAFE_STORAGE_SERVICE,
             "-a", CHROME_SAFE_STORAGE_ACCOUNT],
            stderr=subprocess.PIPE,
            timeout=30,
        ).strip()
        return key
    except subprocess.CalledProcessError as e:
        if e.returncode == 128:
            raise PermissionError(
                "Keychain access denied. Run this command manually in Terminal:\n"
                f"  security find-generic-password -w -s '{CHROME_SAFE_STORAGE_SERVICE}' "
                f"-a '{CHROME_SAFE_STORAGE_ACCOUNT}'\n"
                "Click 'Always Allow' in the macOS dialog to grant permanent access."
            ) from e
        raise
    except subprocess.TimeoutExpired:
        raise PermissionError(
            "Keychain prompt timed out. Run manually in Terminal:\n"
            f"  security find-generic-password -w -s '{CHROME_SAFE_STORAGE_SERVICE}' "
            f"-a '{CHROME_SAFE_STORAGE_ACCOUNT}'"
        )


def _decrypt_cookie_value(encrypted_value: bytes, derived_key: bytes) -> str:
    """Decrypt a Chrome cookie value (v10 format on macOS)."""
    if not encrypted_value:
        return ""

    # v10 = Chrome macOS encryption
    if encrypted_value[:3] == b"v10":
        from Crypto.Cipher import AES
        iv = b" " * 16
        cipher = AES.new(derived_key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(encrypted_value[3:])
        # Remove PKCS7 padding
        padding = decrypted[-1]
        if isinstance(padding, int) and 1 <= padding <= 16:
            decrypted = decrypted[:-padding]
        return decrypted.decode("utf-8", errors="replace")

    # Unencrypted (rare)
    return encrypted_value.decode("utf-8", errors="replace")


def extract_chrome_cookies() -> dict[str, str]:
    """Extract Google/NotebookLM cookies from Chrome's cookie database.

    Returns dict of {cookie_name: cookie_value}.
    Raises PermissionError if Keychain access is denied.
    """
    if not CHROME_COOKIE_DB.exists():
        raise FileNotFoundError(f"Chrome cookie DB not found: {CHROME_COOKIE_DB}")

    # Get encryption key
    raw_key = _get_chrome_encryption_key()

    # Derive AES key (Chrome uses PBKDF2 with 'saltysalt' and 1003 iterations)
    from Crypto.Protocol.KDF import PBKDF2
    derived_key = PBKDF2(raw_key, b"saltysalt", dkLen=16, count=1003)

    # Copy DB to temp (Chrome holds a lock on it)
    tmp_path = tempfile.mktemp(suffix=".db")
    try:
        shutil.copy2(str(CHROME_COOKIE_DB), tmp_path)

        conn = sqlite3.connect(tmp_path)
        # Build WHERE clause for cookie names
        placeholders = ",".join(["?"] * len(COOKIE_NAMES))
        cursor = conn.execute(
            f"""SELECT name, encrypted_value, host_key
                FROM cookies
                WHERE host_key LIKE '%google.com'
                AND name IN ({placeholders})
                ORDER BY name""",
            list(COOKIE_NAMES),
        )

        cookies = {}
        for name, encrypted_value, host_key in cursor.fetchall():
            try:
                value = _decrypt_cookie_value(encrypted_value, derived_key)
                if value and len(value) > 1:
                    # Prefer notebooklm-specific cookies over generic ones
                    if name not in cookies or "notebooklm" in host_key:
                        cookies[name] = value
            except Exception as e:
                log.debug(f"Failed to decrypt {name}: {e}")

        conn.close()
        return cookies
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def cookies_to_header(cookies: dict[str, str]) -> str:
    """Convert cookie dict to HTTP Cookie header string."""
    return "; ".join(f"{k}={v}" for k, v in sorted(cookies.items()))


def save_cookies(cookie_string: str) -> Path:
    """Save cookie string to NotebookLM auth cache."""
    from ..config_notebooklm import NotebookLMConfig
    cache_path = NotebookLMConfig.AUTH_STATE_DIR / "cookies.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(cookie_string)
    cache_path.chmod(0o600)
    return cache_path


def auto_refresh() -> str:
    """One-call: extract Chrome cookies, save to cache, return cookie string.

    This is the function that should be called automatically when auth expires.
    """
    cookies = extract_chrome_cookies()

    # Validate we have the critical cookies
    missing = REQUIRED_COOKIES - set(cookies.keys())
    if missing:
        raise ValueError(
            f"Missing required cookies: {missing}. "
            "Make sure you're logged into Google in Chrome."
        )

    cookie_string = cookies_to_header(cookies)
    save_path = save_cookies(cookie_string)
    log.info(f"Auto-refreshed {len(cookies)} cookies → {save_path}")
    return cookie_string


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Extract NotebookLM cookies from Chrome")
    parser.add_argument("--test", action="store_true", help="Test auth after extraction")
    parser.add_argument("--print", action="store_true", dest="print_only", help="Print cookie string only")
    args = parser.parse_args()

    try:
        cookies = extract_chrome_cookies()
    except PermissionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    cookie_string = cookies_to_header(cookies)
    missing = REQUIRED_COOKIES - set(cookies.keys())

    print(f"Extracted {len(cookies)} cookies ({len(cookie_string)} bytes)")
    print(f"Keys: {sorted(cookies.keys())}")

    if missing:
        print(f"WARNING: Missing required cookies: {missing}")
        print("Make sure you're logged into Google in Chrome.")

    if args.print_only:
        print(f"\n{cookie_string}")
        return

    # Save
    save_path = save_cookies(cookie_string)
    print(f"Saved to: {save_path}")

    if args.test:
        print("\nTesting auth...")
        from ..api.client import NotebookLMAPIClient
        try:
            client = NotebookLMAPIClient(cookies=cookie_string)
            notebooks = client.list_notebooks()
            print(f"SUCCESS: Found {len(notebooks)} notebooks")
            client.close()
        except Exception as e:
            print(f"AUTH FAILED: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
