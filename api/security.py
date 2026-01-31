"""
API Security Module
====================

Provides JWT authentication, rate limiting, and input validation.

Components:
1. JWT Bearer token authentication
2. API key validation
3. Rate limiting (slowapi)
4. Path traversal prevention
5. Request logging
"""

import os
import re
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger("researchgravity.security")

# Try to import security dependencies
try:
    from fastapi import HTTPException, Security, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. Run: pip install PyJWT")

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATELIMIT_AVAILABLE = True
except ImportError:
    RATELIMIT_AVAILABLE = False
    logger.warning("slowapi not installed. Run: pip install slowapi")


# =============================================================================
# Configuration
# =============================================================================

# Secret key for JWT (generate a new one for production!)
# In production, use: export RG_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
SECRET_KEY = os.environ.get("RG_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# API key for service-to-service auth
API_KEY = os.environ.get("RG_API_KEY", None)

# Rate limits
RATE_LIMIT_DEFAULT = "60/minute"
RATE_LIMIT_SEARCH = "10/minute"
RATE_LIMIT_WRITE = "30/minute"


# =============================================================================
# Input Validation
# =============================================================================

# Patterns for safe input validation
SAFE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
SAFE_SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')
SAFE_PROJECT_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')


def validate_session_id(session_id: str) -> str:
    """Validate session ID to prevent path traversal attacks.

    Args:
        session_id: The session ID to validate

    Returns:
        The validated session ID

    Raises:
        HTTPException: If session ID contains invalid characters
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    if len(session_id) > 100:
        raise HTTPException(status_code=400, detail="Session ID too long")

    if not SAFE_SESSION_ID_PATTERN.match(session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID format. Only alphanumeric, underscore, and hyphen allowed."
        )

    # Additional check for path traversal attempts
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        logger.warning(f"Path traversal attempt detected: {session_id}")
        raise HTTPException(status_code=400, detail="Invalid session ID")

    return session_id


def validate_project_id(project_id: str) -> str:
    """Validate project ID."""
    if not project_id:
        return project_id

    if len(project_id) > 50:
        raise HTTPException(status_code=400, detail="Project ID too long")

    if not SAFE_PROJECT_PATTERN.match(project_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid project ID format"
        )

    return project_id


def validate_id(id_value: str, id_type: str = "ID") -> str:
    """Generic ID validation."""
    if not id_value:
        raise HTTPException(status_code=400, detail=f"{id_type} is required")

    if len(id_value) > 200:
        raise HTTPException(status_code=400, detail=f"{id_type} too long")

    if not SAFE_ID_PATTERN.match(id_value):
        raise HTTPException(status_code=400, detail=f"Invalid {id_type} format")

    return id_value


# =============================================================================
# JWT Authentication
# =============================================================================

if FASTAPI_AVAILABLE:
    security = HTTPBearer(auto_error=False)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time

    Returns:
        Encoded JWT token
    """
    if not JWT_AVAILABLE:
        raise HTTPException(status_code=500, detail="JWT not available")

    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token.

    Args:
        token: The JWT token to decode

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    if not JWT_AVAILABLE:
        raise HTTPException(status_code=500, detail="JWT not available")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


if FASTAPI_AVAILABLE:
    async def get_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
        api_key: Optional[str] = Security(api_key_header)
    ) -> Dict[str, Any]:
        """Dependency to get the current authenticated user.

        Supports both JWT Bearer token and API key authentication.

        Args:
            credentials: JWT bearer token credentials
            api_key: API key from header

        Returns:
            User data from token or API key validation

        Raises:
            HTTPException: If authentication fails
        """
        # Check API key first
        if api_key:
            if API_KEY and api_key == API_KEY:
                return {"type": "api_key", "scope": "full"}
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Check JWT
        if credentials:
            return decode_token(credentials.credentials)

        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Provide Bearer token or X-API-Key header."
        )


    async def optional_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
        api_key: Optional[str] = Security(api_key_header)
    ) -> Optional[Dict[str, Any]]:
        """Optional authentication - returns None if not authenticated."""
        try:
            return await get_current_user(credentials, api_key)
        except HTTPException:
            return None


# =============================================================================
# Rate Limiting
# =============================================================================

if RATELIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)

    def get_limiter() -> Limiter:
        """Get the rate limiter instance."""
        return limiter
else:
    limiter = None

    def get_limiter():
        return None


# =============================================================================
# Request Logging Middleware
# =============================================================================

import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar('request_id', default='')


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:8]


if FASTAPI_AVAILABLE:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
    import time

    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        """Middleware to log all requests with timing and request ID."""

        async def dispatch(self, request: Request, call_next) -> Response:
            request_id = generate_request_id()
            request_id_var.set(request_id)

            # Add request ID to request state
            request.state.request_id = request_id

            # Log request
            start_time = time.time()
            logger.info(f"[{request_id}] {request.method} {request.url.path}")

            # Process request
            try:
                response = await call_next(request)

                # Calculate duration
                duration = (time.time() - start_time) * 1000

                # Log response
                logger.info(
                    f"[{request_id}] {request.method} {request.url.path} "
                    f"-> {response.status_code} ({duration:.1f}ms)"
                )

                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id

                return response

            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(
                    f"[{request_id}] {request.method} {request.url.path} "
                    f"-> ERROR: {str(e)} ({duration:.1f}ms)"
                )
                raise


# =============================================================================
# Error Response Standardization
# =============================================================================

from pydantic import BaseModel
from typing import List


class ErrorDetail(BaseModel):
    """Standardized error detail."""
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: bool = True
    error_code: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    request_id: Optional[str] = None
    timestamp: str = None

    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.utcnow().isoformat()
        if 'request_id' not in data or data['request_id'] is None:
            data['request_id'] = request_id_var.get() or None
        super().__init__(**data)


def create_error_response(
    code: str,
    message: str,
    status_code: int = 400,
    details: Optional[List[Dict[str, Any]]] = None
) -> HTTPException:
    """Create a standardized error response.

    Args:
        code: Error code (e.g., "SESSION_NOT_FOUND")
        message: Human-readable error message
        status_code: HTTP status code
        details: Optional list of error details

    Returns:
        HTTPException with standardized error body
    """
    error_response = ErrorResponse(
        error_code=code,
        message=message,
        details=[ErrorDetail(**d) for d in details] if details else None
    )

    raise HTTPException(
        status_code=status_code,
        detail=error_response.dict()
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Validation
    "validate_session_id",
    "validate_project_id",
    "validate_id",
    # JWT
    "create_access_token",
    "decode_token",
    "get_current_user",
    "optional_auth",
    # Rate limiting
    "limiter",
    "get_limiter",
    "RATE_LIMIT_DEFAULT",
    "RATE_LIMIT_SEARCH",
    "RATE_LIMIT_WRITE",
    # Middleware
    "RequestLoggingMiddleware",
    "request_id_var",
    # Errors
    "ErrorResponse",
    "ErrorDetail",
    "create_error_response",
]
