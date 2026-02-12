"""Handler registry — maps provider names to handler instances."""

from typing import Dict, List, Optional

from .base import WebhookHandler

_REGISTRY: Dict[str, WebhookHandler] = {}


def register_handlers(providers: List[str]) -> None:
    """Register handlers for enabled providers."""
    from .github import GitHubHandler
    from .slack import SlackHandler
    from .generic import GenericHandler

    handler_classes: Dict[str, type] = {
        "github": GitHubHandler,
        "slack": SlackHandler,
        "generic": GenericHandler,
    }
    for name in providers:
        name = name.strip()
        cls = handler_classes.get(name)
        if cls:
            _REGISTRY[name] = cls()


def get_handler(provider: str) -> Optional[WebhookHandler]:
    """Get a registered handler by provider name."""
    return _REGISTRY.get(provider)


def list_handlers() -> Dict[str, dict]:
    """List all registered handlers with metadata."""
    return {
        name: {
            "platform": h.platform,
            "supported_events": h.supported_events(),
        }
        for name, h in _REGISTRY.items()
    }
