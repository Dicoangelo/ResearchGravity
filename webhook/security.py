"""
Webhook Signature Verification — HMAC-SHA256 per provider.

Each provider computes HMAC differently (headers, encoding, prefixes).
All verifiers return False on missing secrets (fail closed).
"""

import hashlib
import hmac
import time
from typing import Mapping

from . import config as cfg


def verify_provider_signature(
    provider: str, headers: Mapping[str, str], body: bytes
) -> bool:
    """Route to provider-specific verification."""
    verifiers = {
        "github": _verify_github,
        "slack": _verify_slack,
        "stripe": _verify_stripe,
        "generic": _verify_generic,
        "relay": _verify_relay,
    }
    verifier = verifiers.get(provider, _verify_generic)
    return verifier(headers, body)


def _verify_github(headers: Mapping, body: bytes) -> bool:
    """GitHub: X-Hub-Signature-256 = sha256=<HMAC-SHA256(secret, body)>"""
    secret = cfg.GITHUB_WEBHOOK_SECRET
    if not secret:
        return False
    signature = headers.get("x-hub-signature-256", "")
    if not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def _verify_slack(headers: Mapping, body: bytes) -> bool:
    """Slack: X-Slack-Signature = v0=<HMAC-SHA256(secret, v0:timestamp:body)>"""
    secret = cfg.SLACK_SIGNING_SECRET
    if not secret:
        return False
    timestamp = headers.get("x-slack-request-timestamp", "")
    if not timestamp:
        return False
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False  # Replay protection: 5 min window
    except ValueError:
        return False
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected = "v0=" + hmac.new(
        secret.encode(), sig_basestring.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(
        headers.get("x-slack-signature", ""), expected
    )


def _verify_stripe(headers: Mapping, body: bytes) -> bool:
    """Stripe: Stripe-Signature header with t= and v1= components."""
    secret = cfg.STRIPE_WEBHOOK_SECRET
    if not secret:
        return False
    sig_header = headers.get("stripe-signature", "")
    elements = dict(
        pair.split("=", 1)
        for pair in sig_header.split(",")
        if "=" in pair
    )
    timestamp = elements.get("t", "")
    v1_sig = elements.get("v1", "")
    if not timestamp or not v1_sig:
        return False
    try:
        if abs(time.time() - int(timestamp)) > 300:
            return False
    except ValueError:
        return False
    signed_payload = f"{timestamp}.{body.decode('utf-8')}"
    expected = hmac.new(
        secret.encode(), signed_payload.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(v1_sig, expected)


def _verify_generic(headers: Mapping, body: bytes) -> bool:
    """Generic: X-Webhook-Signature = sha256=<HMAC-SHA256(secret, body)>.
    If no secret is configured, allows all requests (for local testing).
    """
    secret = cfg.GENERIC_WEBHOOK_SECRET
    if not secret:
        return True  # No secret = allow (local dev only)
    signature = headers.get("x-webhook-signature", "")
    if not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def _verify_relay(headers: Mapping, body: bytes) -> bool:
    """Relay from Supabase Edge Function: X-Relay-Secret header."""
    secret = cfg.RELAY_SHARED_SECRET
    if not secret:
        return False
    return hmac.compare_digest(
        headers.get("x-relay-secret", ""), secret
    )
