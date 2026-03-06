"""
Structura AI - API Key Authentication System
"""
import secrets
import hashlib
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import time
from collections import defaultdict

from database import get_api_key_by_hash, get_user_by_id
from models import PLAN_CONFIG

# Security scheme
security = HTTPBearer(
    scheme_name="API Key",
    description="Bearer token authentication. Use your API key as: Bearer sk_live_xxx or Bearer sk_free_xxx"
)


def generate_api_key(plan: str = "free") -> dict:
    """Generate a secure API key with prefix."""
    raw_key = secrets.token_urlsafe(32)  # 256-bit entropy
    prefix = "sk_free_" if plan == "free" else "sk_live_"
    full_key = f"{prefix}{raw_key}"

    # Store only the hash
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_display = f"{prefix}{raw_key[:4]}...{raw_key[-4:]}"

    return {
        "full_key": full_key,       # Show ONCE to user, never store
        "key_hash": key_hash,       # Store in DB
        "key_prefix": key_display   # For display in dashboard
    }


def hash_api_key(api_key: str) -> str:
    """Hash an API key for lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# --- In-Memory Rate Limiter (MVP - replace with Redis in production) ---

class RateLimiter:
    """Simple in-memory rate limiter per API key."""

    def __init__(self):
        # {key_hash: [(timestamp, ...), ...]}
        self._minute_requests = defaultdict(list)
        self._day_requests = defaultdict(list)

    def _clean_old(self, key: str, window: str):
        """Remove expired entries."""
        now = time.time()
        if window == "minute":
            cutoff = now - 60
            self._minute_requests[key] = [
                t for t in self._minute_requests[key] if t > cutoff
            ]
        elif window == "day":
            cutoff = now - 86400
            self._day_requests[key] = [
                t for t in self._day_requests[key] if t > cutoff
            ]

    def check_rate_limit(self, key_hash: str, plan: str) -> dict:
        """
        Check if request is within rate limits.
        Returns dict with limit info and whether allowed.
        """
        config = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])
        now = time.time()

        # Clean old entries
        self._clean_old(key_hash, "minute")
        self._clean_old(key_hash, "day")

        minute_count = len(self._minute_requests[key_hash])
        day_count = len(self._day_requests[key_hash])

        minute_limit = config["rate_limit_per_minute"]
        day_limit = config["rate_limit_per_day"]

        if minute_count >= minute_limit:
            reset_time = int(self._minute_requests[key_hash][0] + 60)
            return {
                "allowed": False,
                "limit": minute_limit,
                "remaining": 0,
                "reset": reset_time,
                "reason": f"Rate limit exceeded: {minute_limit} requests per minute"
            }

        if day_count >= day_limit:
            reset_time = int(self._day_requests[key_hash][0] + 86400)
            return {
                "allowed": False,
                "limit": day_limit,
                "remaining": 0,
                "reset": reset_time,
                "reason": f"Daily limit exceeded: {day_limit} requests per day"
            }

        return {
            "allowed": True,
            "limit": minute_limit,
            "remaining": minute_limit - minute_count - 1,
            "reset": int(now + 60),
        }

    def record_request(self, key_hash: str):
        """Record a request for rate limiting."""
        now = time.time()
        self._minute_requests[key_hash].append(now)
        self._day_requests[key_hash].append(now)


# Global rate limiter instance
rate_limiter = RateLimiter()


# --- Auth Dependency ---

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """
    FastAPI dependency that validates the API key and returns user info.
    """
    api_key = credentials.credentials
    key_hash = hash_api_key(api_key)

    # Look up the key
    key_data = get_api_key_by_hash(key_hash)
    if not key_data:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "AUTH_REQUIRED",
                    "message": "Invalid or inactive API key. Get a free key at POST /v1/auth/register"
                }
            }
        )

    user_id = key_data["user_id"]
    plan = key_data["plan"]

    # Check rate limits
    rate_check = rate_limiter.check_rate_limit(key_hash, plan)
    if not rate_check["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "RATE_LIMITED",
                    "message": rate_check["reason"],
                    "details": {
                        "limit": rate_check["limit"],
                        "reset": rate_check["reset"]
                    }
                }
            },
            headers={
                "X-RateLimit-Limit": str(rate_check["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_check["reset"]),
                "Retry-After": str(rate_check["reset"] - int(time.time()))
            }
        )

    # Check credits
    credits_remaining = key_data["credits_remaining"]
    if credits_remaining <= 0:
        plan_config = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])
        if plan == "free" or plan_config["overage_rate"] == 0:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": {
                        "code": "CREDITS_EXHAUSTED",
                        "message": "No credits remaining. Upgrade your plan at /v1/subscribe",
                        "details": {
                            "credits_remaining": credits_remaining,
                            "plan": plan,
                            "upgrade_url": "/v1/subscribe"
                        }
                    }
                }
            )

    # Record for rate limiting
    rate_limiter.record_request(key_hash)

    return {
        "user_id": user_id,
        "api_key_id": key_data["id"],
        "key_hash": key_hash,
        "email": key_data["email"],
        "plan": plan,
        "credits_remaining": credits_remaining,
        "credits_monthly_limit": key_data["credits_monthly_limit"],
        "rate_limit": rate_check,
    }
