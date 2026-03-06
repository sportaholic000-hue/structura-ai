"""
Structura AI - Pydantic Models for Request/Response
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Any, Dict, List
from datetime import datetime
from enum import Enum


# --- Enums ---

class PlanTier(str, Enum):
    free = "free"
    starter = "starter"
    growth = "growth"
    scale = "scale"


class ModelChoice(str, Enum):
    fast = "fast"
    quality = "quality"


class RequestStatus(str, Enum):
    success = "success"
    failed = "failed"
    partial = "partial"


# --- Plan Configuration ---

PLAN_CONFIG = {
    "free": {
        "credits_monthly": 100,
        "rate_limit_per_minute": 10,
        "rate_limit_per_day": 100,
        "max_input_chars": 5000,
        "max_batch_size": 5,
        "max_schemas": 1,
        "models_allowed": ["fast"],
        "overage_rate": 0.0,
        "price_usd": 0,
    },
    "starter": {
        "credits_monthly": 1000,
        "rate_limit_per_minute": 30,
        "rate_limit_per_day": 3000,
        "max_input_chars": 15000,
        "max_batch_size": 25,
        "max_schemas": 10,
        "models_allowed": ["fast"],
        "overage_rate": 0.012,
        "price_usd": 9,
    },
    "growth": {
        "credits_monthly": 5000,
        "rate_limit_per_minute": 60,
        "rate_limit_per_day": 10000,
        "max_input_chars": 50000,
        "max_batch_size": 50,
        "max_schemas": 50,
        "models_allowed": ["fast", "quality"],
        "overage_rate": 0.008,
        "price_usd": 29,
    },
    "scale": {
        "credits_monthly": 25000,
        "rate_limit_per_minute": 120,
        "rate_limit_per_day": 50000,
        "max_input_chars": 100000,
        "max_batch_size": 100,
        "max_schemas": -1,
        "models_allowed": ["fast", "quality"],
        "overage_rate": 0.005,
        "price_usd": 99,
    },
}

CREDIT_COSTS = {
    "extract_fast": 1.0,
    "extract_quality": 3.0,
    "classify": 0.5,
    "transform": 1.0,
    "batch_per_item": 0.8,
}


# --- Auth Models ---

class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Email address for the account")
    name: Optional[str] = Field(None, description="Optional display name")


class RegisterResponse(BaseModel):
    message: str
    api_key: str = Field(..., description="Your API key - save this! It won't be shown again.")
    key_prefix: str
    email: str
    plan: str = "free"
    credits: int = 100


class APIKeyInfo(BaseModel):
    key_prefix: str
    name: str
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None


# --- Extraction Models ---

class ExtractionOptions(BaseModel):
    model: ModelChoice = Field(default=ModelChoice.fast, description="Model to use: fast or quality")
    confidence_scores: bool = Field(default=False, description="Include confidence scores per field")
    strict_mode: bool = Field(default=True, description="Fail on missing required fields vs return nulls")


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Unstructured text to extract data from")
    schema_definition: Optional[Dict[str, Any]] = Field(None, alias="schema", description="JSON Schema defining fields to extract")
    schema_id: Optional[str] = Field(None, description="ID of a saved schema to use")
    options: ExtractionOptions = Field(default_factory=ExtractionOptions)

    class Config:
        populate_by_name = True


class ValidationResult(BaseModel):
    all_required_present: bool = True
    type_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class UsageInfo(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: str = "fast"
    cost_credits: float = 1.0


class ExtractResponse(BaseModel):
    id: str
    status: RequestStatus = RequestStatus.success
    data: Dict[str, Any]
    confidence: Optional[Dict[str, float]] = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- Batch Models ---

class BatchItem(BaseModel):
    id: str = Field(..., description="Unique identifier for this item")
    text: str = Field(..., description="Text to extract from")


class BatchRequest(BaseModel):
    schema_definition: Optional[Dict[str, Any]] = Field(None, alias="schema")
    schema_id: Optional[str] = None
    items: List[BatchItem] = Field(..., max_length=100)
    options: ExtractionOptions = Field(default_factory=ExtractionOptions)

    class Config:
        populate_by_name = True


class BatchResultItem(BaseModel):
    item_id: str
    status: RequestStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchSummary(BaseModel):
    total: int
    succeeded: int
    failed: int


class BatchResponse(BaseModel):
    id: str
    status: str = "completed"
    results: List[BatchResultItem]
    summary: BatchSummary
    usage: Dict[str, Any]


# --- Classification Models ---

class CategoryDefinition(BaseModel):
    name: str = Field(..., description="Category identifier")
    description: str = Field(..., description="What this category represents")


class ClassifyOptions(BaseModel):
    multi_label: bool = Field(default=False, description="Allow multiple categories")
    include_reasoning: bool = Field(default=False, description="Include reasoning for classification")


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    categories: List[CategoryDefinition] = Field(..., min_length=2, description="List of possible categories")
    options: ClassifyOptions = Field(default_factory=ClassifyOptions)


class ClassificationResult(BaseModel):
    category: str
    confidence: float
    reasoning: Optional[str] = None


class SecondaryCategory(BaseModel):
    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    id: str
    status: RequestStatus = RequestStatus.success
    classification: ClassificationResult
    secondary_categories: List[SecondaryCategory] = Field(default_factory=list)
    usage: Dict[str, Any]


# --- Transform Models ---

class TransformRequest(BaseModel):
    input_data: Dict[str, Any] = Field(..., alias="input", description="Messy/semi-structured input data")
    target_schema: Dict[str, Any] = Field(..., description="Target JSON schema for output")

    class Config:
        populate_by_name = True


class TransformResponse(BaseModel):
    id: str
    status: RequestStatus = RequestStatus.success
    data: Dict[str, Any]
    usage: Dict[str, Any]


# --- Schema Models ---

class SchemaCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Schema template name")
    description: Optional[str] = Field(None, description="What this schema extracts")
    schema_definition: Dict[str, Any] = Field(..., alias="schema", description="JSON Schema definition")

    class Config:
        populate_by_name = True


class SchemaResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    schema_definition: Dict[str, Any] = Field(serialization_alias="schema")
    usage_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True


class SchemaListResponse(BaseModel):
    schemas: List[SchemaResponse]
    total: int


# --- Usage Models ---

class UsageResponse(BaseModel):
    plan: str
    billing_period: Dict[str, str]
    credits: Dict[str, Any]
    requests: Dict[str, int]


class AccountResponse(BaseModel):
    email: str
    plan: str
    credits_remaining: float
    credits_monthly_limit: int
    api_keys: List[APIKeyInfo]
    created_at: datetime


# --- Subscribe Models ---

class SubscribeResponse(BaseModel):
    plans: Dict[str, Any]
    payment_methods: Dict[str, Any]
    instructions: str


# --- Error Models ---

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
