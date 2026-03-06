# Structura AI

**Turn Any Text Into Structured JSON. Instantly.**

Structura AI is a developer-first API that extracts structured data from any unstructured text -- emails, reviews, support tickets, resumes, invoices, forms -- using AI, with YOUR custom schema.

## Quick Start

### 1. Get your free API key

```bash
curl -X POST https://structura-ai.onrender.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

### 2. Extract structured data

```bash
curl -X POST https://structura-ai.onrender.com/v1/extract \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hi, I am John Smith from Acme Corp. My email is john@acme.com and I need help with billing.",
    "schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "company": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "topic": {"type": "string", "enum": ["billing", "technical", "general"]}
      },
      "required": ["name", "email"]
    }
  }'
```

### 3. Get clean JSON back

```json
{
  "id": "ext_a1b2c3d4e5f6",
  "status": "success",
  "data": {
    "name": "John Smith",
    "company": "Acme Corp",
    "email": "john@acme.com",
    "topic": "billing"
  }
}
```

## API Endpoints

| Endpoint | Method | Description | Credits |
|----------|--------|-------------|---------|
| `/v1/auth/register` | POST | Get a free API key | Free |
| `/v1/extract` | POST | Extract structured data from text | 1-3 |
| `/v1/extract/batch` | POST | Batch extract (up to 100 items) | 0.8/item |
| `/v1/classify` | POST | Classify text into categories | 0.5 |
| `/v1/transform` | POST | Validate and normalize messy data | 1 |
| `/v1/schemas` | GET/POST/DELETE | Save and reuse extraction templates | Free |
| `/v1/usage` | GET | View credit usage stats | Free |
| `/v1/account` | GET | Account info and API keys | Free |
| `/v1/subscribe` | GET | Pricing plans and payment links | Free |

## Interactive Documentation

- **Swagger UI**: [/docs](https://structura-ai.onrender.com/docs)
- **ReDoc**: [/redoc](https://structura-ai.onrender.com/redoc)

## Pricing

| Plan | Price | Credits/Month | Rate Limit |
|------|-------|--------------|------------|
| Free | $0 | 100 | 10 req/min |
| Starter | $9/mo | 1,000 | 30 req/min |
| Growth | $29/mo | 5,000 | 60 req/min |
| Scale | $99/mo | 25,000 | 120 req/min |

## Tech Stack

- **Framework**: FastAPI (Python)
- **AI**: OpenAI GPT-4o-mini / GPT-4o
- **Database**: SQLite (MVP) / PostgreSQL (production)
- **Deployment**: Render (Docker)

## Local Development

```bash
# Clone the repo
git clone https://github.com/sportaholic000-hue/structura-ai.git
cd structura-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key

# Run the server
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

## Payment Methods

- **PayPal**: [paypal.me/icandoanythingagent](https://paypal.me/icandoanythingagent)
- **Ethereum (ETH/Base)**: `0xAC0320ac14498BA80295ab005f0ba0DC04760e23`
- **Solana (SOL)**: `47uUqKznBDR4iph1VY6ffzycbfBWSwhkjWWYruf7VsmR`

## License

MIT License
