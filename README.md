"""
Structura AI - Main FastAPI Application
Turn Any Text Into Structured JSON. Instantly.

API server with all endpoints for extraction, classification, transformation,
schema management, usage tracking, and authentication.
"""
import os
import json
import uuid
import time
from datetime import datetime, date
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local imports
from models import (
    RegisterRequest, RegisterResponse,
    ExtractRequest, ExtractResponse, ValidationResult, UsageInfo,
    BatchRequest, BatchResponse, BatchResultItem, BatchSummary,
    ClassifyRequest, ClassifyResponse, ClassificationResult, SecondaryCategory,
    TransformRequest, TransformResponse,
    SchemaCreateRequest, SchemaResponse, SchemaListResponse,
    UsageResponse, AccountResponse, SubscribeResponse,
    ErrorResponse, ErrorDetail, APIKeyInfo,
    PLAN_CONFIG, CREDIT_COSTS, RequestStatus,
)
from database import (
    init_db,
    create_user, get_user_by_email, get_user_by_id,
    update_user_credits,
    store_api_key, get_user_api_keys,
    create_schema, get_schema, list_schemas, delete_schema, increment_schema_usage,
    log_request, get_usage_stats,
)
from auth import generate_api_key, get_current_user, rate_limiter
from extraction import extract_data, classify_text, transform_data


# --- App Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    yield


# --- FastAPI App ---

app = FastAPI(
    title="Structura AI",
    description=(
        "Turn Any Text Into Structured JSON. Instantly.\n\n"
        "Structura AI is a developer-first API that extracts structured data from any "
        "unstructured text -- emails, reviews, support tickets, resumes, invoices, forms "
        "-- using AI, with YOUR custom schema.\n\n"
        "## Quick Start\n"
        "1. Register for a free API key: `POST /v1/auth/register`\n"
        "2. Extract data: `POST /v1/extract` with your text + JSON schema\n"
        "3. Get clean, validated JSON back instantly\n\n"
        "## Authentication\n"
        "All API endpoints require a Bearer token:\n"
        "```\nAuthorization: Bearer sk_live_xxxxx\n```\n\n"
        "## Pricing\n"
        "- **Free**: 100 credits/month (no credit card required)\n"
        "- **Starter**: $9/mo - 1,000 credits\n"
        "- **Growth**: $29/mo - 5,000 credits\n"
        "- **Scale**: $99/mo - 25,000 credits\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response Headers Middleware ---

@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    """Add rate limit and credit headers to all responses."""
    response = await call_next(request)
    # Add custom headers
    response.headers["X-Powered-By"] = "Structura AI"
    return response


# --- Helper Functions ---

def gen_id(prefix: str = "ext") -> str:
    """Generate a short unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def get_billing_period() -> dict:
    """Get current billing period."""
    now = datetime.utcnow()
    start = date(now.year, now.month, 1)
    if now.month == 12:
        end = date(now.year + 1, 1, 1)
    else:
        end = date(now.year, now.month + 1, 1)
    from datetime import timedelta
    end = end - timedelta(days=1)
    return {"start": start.isoformat(), "end": end.isoformat()}


def check_input_length(text: str, plan: str):
    """Check if input text is within plan limits."""
    max_chars = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])["max_input_chars"]
    if len(text) > max_chars:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "TEXT_TOO_LONG",
                    "message": f"Input text ({len(text)} chars) exceeds your plan limit ({max_chars} chars)",
                    "details": {
                        "text_length": len(text),
                        "plan_limit": max_chars,
                        "plan": plan,
                    }
                }
            }
        )


def check_model_allowed(model: str, plan: str):
    """Check if the model is allowed for the plan."""
    allowed = PLAN_CONFIG.get(plan, PLAN_CONFIG["free"])["models_allowed"]
    if model not in allowed:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": "MODEL_NOT_AVAILABLE",
                    "message": f"Model '{model}' is not available on the {plan} plan. Allowed models: {allowed}",
                    "details": {"allowed_models": allowed, "upgrade_url": "/v1/subscribe"}
                }
            }
        )


def deduct_and_log(user: dict, endpoint: str, credits_cost: float, model: str = "fast",
                   input_tokens: int = 0, output_tokens: int = 0, latency_ms: int = 0):
    """Deduct credits and log the request."""
    update_user_credits(user["user_id"], credits_cost)
    log_request(
        user_id=user["user_id"],
        api_key_id=user["api_key_id"],
        endpoint=endpoint,
        credits_used=credits_cost,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


def resolve_schema(req_schema: Optional[dict], schema_id: Optional[str], user: dict) -> dict:
    """Resolve schema from inline definition or saved schema ID."""
    if schema_id:
        saved = get_schema(schema_id, user["user_id"])
        if not saved:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": "SCHEMA_NOT_FOUND",
                        "message": f"Schema '{schema_id}' not found",
                    }
                }
            )
        increment_schema_usage(schema_id)
        return json.loads(saved["schema_json"])
    elif req_schema:
        return req_schema
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "SCHEMA_INVALID",
                    "message": "Either 'schema' or 'schema_id' must be provided",
                }
            }
        )


# ============================================================================
# ROOT & INFO ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - returns info and links."""
    return {
        "name": "Structura AI",
        "tagline": "Turn Any Text Into Structured JSON. Instantly.",
        "version": "1.0.0",
        "description": (
            "Developer-first API that extracts structured data from any unstructured text "
            "using AI with YOUR custom schema."
        ),
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "register": "POST /v1/auth/register",
            "extract": "POST /v1/extract",
            "batch_extract": "POST /v1/extract/batch",
            "classify": "POST /v1/classify",
            "transform": "POST /v1/transform",
            "schemas": "GET/POST/DELETE /v1/schemas",
            "usage": "GET /v1/usage",
            "account": "GET /v1/account",
            "subscribe": "GET /v1/subscribe",
        },
        "quick_start": {
            "step_1": "POST /v1/auth/register with your email to get a free API key",
            "step_2": "Add header: Authorization: Bearer YOUR_API_KEY",
            "step_3": "POST /v1/extract with text + schema to get structured JSON",
        },
        "pricing": {
            "free": "100 credits/month - $0",
            "starter": "1,000 credits/month - $9/mo",
            "growth": "5,000 credits/month - $29/mo",
            "scale": "25,000 credits/month - $99/mo",
        },
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.post("/v1/auth/register", response_model=RegisterResponse, tags=["Authentication"])
async def register(req: RegisterRequest):
    """
    Register for a free API key.
    
    Provide your email to get a free tier API key with 100 credits/month.
    The API key will only be shown ONCE - save it immediately!
    """
    # Check if email already registered
    existing = get_user_by_email(req.email)
    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "code": "EMAIL_EXISTS",
                    "message": f"An account with email '{req.email}' already exists. "
                               "If you lost your API key, contact support.",
                }
            }
        )

    # Create user
    user = create_user(email=req.email, name=req.name)

    # Generate API key
    key_data = generate_api_key(plan="free")
    store_api_key(
        user_id=user["id"],
        key_hash=key_data["key_hash"],
        key_prefix=key_data["key_prefix"],
    )

    return RegisterResponse(
        message="Welcome to Structura AI! Save your API key - it won't be shown again.",
        api_key=key_data["full_key"],
        key_prefix=key_data["key_prefix"],
        email=user["email"],
        plan="free",
        credits=100,
    )


# ============================================================================
# EXTRACT ENDPOINT
# ============================================================================

@app.post("/v1/extract", tags=["Extraction"])
async def extract(req: ExtractRequest, user: dict = Depends(get_current_user)):
    """
    Extract structured data from unstructured text.
    
    Send text + a JSON schema, get clean validated JSON back.
    This is the core endpoint of Structura AI.
    
    **Credit cost:** 1 credit (fast model) or 3 credits (quality model)
    """
    # Validate inputs
    check_input_length(req.text, user["plan"])
    check_model_allowed(req.options.model.value, user["plan"])

    # Resolve schema
    schema = resolve_schema(req.schema_definition, req.schema_id, user)

    # Validate schema has properties
    if "properties" not in schema:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "SCHEMA_INVALID",
                    "message": "Schema must contain a 'properties' object defining fields to extract",
                }
            }
        )

    try:
        # Call extraction engine
        result = extract_data(
            text=req.text,
            schema=schema,
            model=req.options.model.value,
            confidence_scores=req.options.confidence_scores,
            strict_mode=req.options.strict_mode,
        )

        # Check for validation errors in strict mode
        validation = result["validation"]
        if req.options.strict_mode and validation["type_errors"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "code": "SCHEMA_VALIDATION_FAILED",
                        "message": "Extraction failed validation checks",
                        "details": {
                            "type_errors": validation["type_errors"],
                            "warnings": validation["warnings"],
                        }
                    }
                }
            )

        # Deduct credits and log
        credits_cost = result["usage"]["cost_credits"]
        deduct_and_log(
            user=user,
            endpoint="extract",
            credits_cost=credits_cost,
            model=req.options.model.value,
            input_tokens=result["usage"].get("input_tokens", 0),
            output_tokens=result["usage"].get("output_tokens", 0),
            latency_ms=result.get("latency_ms", 0),
        )

        response_id = gen_id("ext")
        response_data = {
            "id": response_id,
            "status": "success",
            "data": result["data"],
            "validation": {
                "all_required_present": validation["all_required_present"],
                "type_errors": validation["type_errors"],
                "warnings": validation["warnings"],
            },
            "usage": result["usage"],
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        if result.get("confidence"):
            response_data["confidence"] = result["confidence"]

        return response_data

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"error": {"code": "EXTRACTION_FAILED", "message": str(e)}})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"code": "INTERNAL_ERROR", "message": str(e)}})


# ============================================================================
# BATCH EXTRACT ENDPOINT
# ============================================================================

@app.post("/v1/extract/batch", tags=["Extraction"])
async def batch_extract(req: BatchRequest, user: dict = Depends(get_current_user)):
    """
    Batch extract structured data from multiple texts using the same schema.
    
    Process up to 100 items per batch (limit depends on plan tier).
    
    **Credit cost:** 0.8 credits per item
    """
    plan_config = PLAN_CONFIG.get(user["plan"], PLAN_CONFIG["free"])
    max_batch = plan_config["max_batch_size"]

    if len(req.items) > max_batch:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "BATCH_TOO_LARGE",
                    "message": f"Batch size ({len(req.items)}) exceeds your plan limit ({max_batch})",
                    "details": {"max_batch_size": max_batch, "plan": user["plan"]}
                }
            }
        )

    # Resolve schema
    schema = resolve_schema(req.schema_definition, req.schema_id, user)

    results = []
    succeeded = 0
    failed = 0
    total_credits = 0.0

    for item in req.items:
        try:
            check_input_length(item.text, user["plan"])
            result = extract_data(
                text=item.text,
                schema=schema,
                model=req.options.model.value,
                confidence_scores=req.options.confidence_scores,
                strict_mode=False,  # Batch uses flexible mode
            )
            results.append(BatchResultItem(
                item_id=item.id,
                status=RequestStatus.success,
                data=result["data"],
            ))
            succeeded += 1
            total_credits += CREDIT_COSTS["batch_per_item"]
        except Exception as e:
            results.append(BatchResultItem(
                item_id=item.id,
                status=RequestStatus.failed,
                error=str(e),
            ))
            failed += 1

    # Deduct credits and log
    if total_credits > 0:
        deduct_and_log(user=user, endpoint="batch", credits_cost=total_credits, model=req.options.model.value)

    return BatchResponse(
        id=gen_id("batch"),
        status="completed",
        results=results,
        summary=BatchSummary(total=len(req.items), succeeded=succeeded, failed=failed),
        usage={"total_credits": total_credits},
    )


# ============================================================================
# CLASSIFY ENDPOINT
# ============================================================================

@app.post("/v1/classify", tags=["Classification"])
async def classify(req: ClassifyRequest, user: dict = Depends(get_current_user)):
    """
    Classify text into provided categories.
    
    Lightweight classification endpoint - uses fewer credits than full extraction.
    
    **Credit cost:** 0.5 credits
    """
    check_input_length(req.text, user["plan"])

    try:
        categories_list = [{"name": c.name, "description": c.description} for c in req.categories]
        result = classify_text(
            text=req.text,
            categories=categories_list,
            multi_label=req.options.multi_label,
            include_reasoning=req.options.include_reasoning,
        )

        # Deduct credits
        deduct_and_log(
            user=user,
            endpoint="classify",
            credits_cost=CREDIT_COSTS["classify"],
            latency_ms=result.get("latency_ms", 0),
        )

        return {
            "id": gen_id("cls"),
            "status": "success",
            "classification": result["classification"],
            "secondary_categories": result["secondary_categories"],
            "usage": result["usage"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"code": "INTERNAL_ERROR", "message": str(e)}})


# ============================================================================
# TRANSFORM ENDPOINT
# ============================================================================

@app.post("/v1/transform", tags=["Transform"])
async def transform(req: TransformRequest, user: dict = Depends(get_current_user)):
    """
    Validate and transform messy structured data into clean, normalized output.
    
    Takes already-structured (but messy) data and validates/enriches/transforms it
    according to a target schema.
    
    **Credit cost:** 1 credit
    """
    try:
        result = transform_data(
            input_data=req.input_data,
            target_schema=req.target_schema,
        )

        # Deduct credits
        deduct_and_log(
            user=user,
            endpoint="transform",
            credits_cost=CREDIT_COSTS["transform"],
            latency_ms=result.get("latency_ms", 0),
        )

        return {
            "id": gen_id("tfm"),
            "status": "success",
            "data": result["data"],
            "usage": result["usage"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"code": "INTERNAL_ERROR", "message": str(e)}})


# ============================================================================
# SCHEMA ENDPOINTS (CRUD)
# ============================================================================

@app.get("/v1/schemas", tags=["Schemas"])
async def list_user_schemas(user: dict = Depends(get_current_user)):
    """List all saved schemas for the authenticated user."""
    schemas = list_schemas(user["user_id"])
    return {
        "schemas": [
            {
                "id": s["id"],
                "name": s["name"],
                "description": s["description"],
                "schema": json.loads(s["schema_json"]),
                "usage_count": s["usage_count"],
                "created_at": s["created_at"],
                "updated_at": s["updated_at"],
            }
            for s in schemas
        ],
        "total": len(schemas),
    }


@app.post("/v1/schemas", tags=["Schemas"], status_code=201)
async def create_user_schema(req: SchemaCreateRequest, user: dict = Depends(get_current_user)):
    """
    Create a saved schema template for reuse.
    
    Save a schema once, then reference it by ID in extract/batch calls.
    """
    plan_config = PLAN_CONFIG.get(user["plan"], PLAN_CONFIG["free"])
    max_schemas = plan_config["max_schemas"]

    # Check schema limit (unless unlimited)
    if max_schemas > 0:
        existing = list_schemas(user["user_id"])
        if len(existing) >= max_schemas:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "code": "SCHEMA_LIMIT_REACHED",
                        "message": f"Your {user['plan']} plan allows {max_schemas} saved schemas. Upgrade for more.",
                        "details": {"current_count": len(existing), "limit": max_schemas}
                    }
                }
            )

    try:
        schema_id = create_schema(
            user_id=user["user_id"],
            name=req.name,
            description=req.description,
            schema_json=json.dumps(req.schema_definition),
        )

        return {
            "id": schema_id,
            "name": req.name,
            "description": req.description,
            "schema": req.schema_definition,
            "message": f"Schema '{req.name}' created. Use schema_id='{schema_id}' in extract calls.",
        }

    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": {
                        "code": "SCHEMA_NAME_EXISTS",
                        "message": f"A schema named '{req.name}' already exists. Use a different name.",
                    }
                }
            )
        raise HTTPException(status_code=500, detail={"error": {"code": "INTERNAL_ERROR", "message": str(e)}})


@app.delete("/v1/schemas/{schema_id}", tags=["Schemas"])
async def delete_user_schema(schema_id: str, user: dict = Depends(get_current_user)):
    """Delete a saved schema."""
    success = delete_schema(schema_id, user["user_id"])
    if not success:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "SCHEMA_NOT_FOUND", "message": f"Schema '{schema_id}' not found"}}
        )
    return {"message": f"Schema '{schema_id}' deleted successfully"}


# ============================================================================
# USAGE ENDPOINT
# ============================================================================

@app.get("/v1/usage", tags=["Usage & Account"])
async def get_usage(user: dict = Depends(get_current_user)):
    """
    Get current usage statistics for the billing period.
    
    Shows credits used, remaining, and request counts by endpoint.
    """
    stats = get_usage_stats(user["user_id"])
    billing = get_billing_period()
    plan_config = PLAN_CONFIG.get(user["plan"], PLAN_CONFIG["free"])

    return {
        "plan": user["plan"],
        "billing_period": billing,
        "credits": {
            "included": user["credits_monthly_limit"],
            "used": round(stats["total_credits"], 2),
            "remaining": round(user["credits_remaining"], 2),
            "overage_rate_usd": plan_config["overage_rate"],
        },
        "requests": {
            "total": stats["total"],
            "extract": stats["extract_count"],
            "classify": stats["classify_count"],
            "transform": stats["transform_count"],
            "batch": stats["batch_count"],
        },
    }


# ============================================================================
# ACCOUNT ENDPOINT
# ============================================================================

@app.get("/v1/account", tags=["Usage & Account"])
async def get_account(user: dict = Depends(get_current_user)):
    """Get account information and plan details."""
    user_data = get_user_by_id(user["user_id"])
    api_keys = get_user_api_keys(user["user_id"])

    return {
        "email": user["email"],
        "plan": user["plan"],
        "credits_remaining": round(user["credits_remaining"], 2),
        "credits_monthly_limit": user["credits_monthly_limit"],
        "api_keys": [
            {
                "key_prefix": k["key_prefix"],
                "name": k["name"],
                "is_active": bool(k["is_active"]),
                "created_at": k["created_at"],
                "last_used_at": k["last_used_at"],
            }
            for k in api_keys
        ],
        "plan_details": PLAN_CONFIG.get(user["plan"], PLAN_CONFIG["free"]),
        "created_at": user_data["created_at"] if user_data else None,
    }


# ============================================================================
# SUBSCRIBE ENDPOINT
# ============================================================================

@app.get("/v1/subscribe", tags=["Billing"])
async def subscribe():
    """
    Get subscription plans and payment links.
    
    Choose your plan and pay with PayPal, ETH, or SOL.
    No credit card required for free tier.
    """
    return {
        "plans": {
            "free": {
                "price": "$0/month",
                "credits": 100,
                "features": [
                    "100 credits/month",
                    "10 requests/minute",
                    "Fast model (gpt-4o-mini)",
                    "1 saved schema",
                ],
            },
            "starter": {
                "price": "$9/month",
                "annual_price": "$86/year (save $22)",
                "credits": 1000,
                "features": [
                    "1,000 credits/month",
                    "30 requests/minute",
                    "Batch processing (25 items)",
                    "10 saved schemas",
                    "Confidence scores",
                ],
            },
            "growth": {
                "price": "$29/month",
                "annual_price": "$278/year (save $70)",
                "credits": 5000,
                "features": [
                    "5,000 credits/month",
                    "60 requests/minute",
                    "Fast + Quality models",
                    "Batch processing (50 items)",
                    "50 saved schemas",
                    "Confidence + reasoning",
                ],
            },
            "scale": {
                "price": "$99/month",
                "annual_price": "$950/year (save $238)",
                "credits": 25000,
                "features": [
                    "25,000 credits/month",
                    "120 requests/minute",
                    "All models + priority queue",
                    "Batch processing (100 items)",
                    "Unlimited saved schemas",
                    "Full analytics + export",
                ],
            },
        },
        "payment_methods": {
            "paypal": {
                "link": "https://paypal.me/icandoanythingagent",
                "instructions": "Send payment via PayPal, then email your receipt + plan choice to activate.",
            },
            "ethereum": {
                "wallet": "0xAC0320ac14498BA80295ab005f0ba0DC04760e23",
                "network": "Ethereum / Base",
                "instructions": "Send ETH equivalent, include your email in tx data/memo.",
                "approximate_prices": {
                    "starter": "~0.003 ETH/mo",
                    "growth": "~0.009 ETH/mo",
                    "scale": "~0.03 ETH/mo",
                },
            },
            "solana": {
                "wallet": "47uUqKznBDR4iph1VY6ffzycbfBWSwhkjWWYruf7VsmR",
                "instructions": "Send SOL equivalent, include your email in tx memo.",
                "approximate_prices": {
                    "starter": "~0.06 SOL/mo",
                    "growth": "~0.18 SOL/mo",
                    "scale": "~0.6 SOL/mo",
                },
            },
        },
        "instructions": (
            "1. Choose your plan above. "
            "2. Send payment via your preferred method. "
            "3. Your plan will be activated within 24 hours. "
            "Free tier is activated instantly upon registration."
        ),
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": "NOT_FOUND",
                "message": f"Endpoint '{request.url.path}' not found. See / for available endpoints.",
            }
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again.",
            }
        }
    )


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
