# Generic AI Customer Service Agent — Setup Guide

## Overview
This is a generic AI-powered customer service agent that handles orders, appointments, and inquiries for **any business**. Simply provide a PDF or text file describing your business, and the agent automatically:
- Ingests and chunks the document
- Detects the business type
- Enriches knowledge via LLM
- Generates synthetic customer data
- Becomes your AI customer service representative

## System Requirements
- **Windows 10/11** (tested on Windows 11)
- **Python 3.11+** ([Download from python.org](https://www.python.org/downloads/))
- **API Key** for OpenAI or Google Gemini (at least one required)

## Installation (Step-by-Step)

### Step 1: Open Terminal
Open PowerShell or Command Prompt, then navigate to the project folder:
```powershell
cd "Folder path"
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment
```powershell
copy .env.example .env
```
Open `.env` in any text editor and add your API key(s):
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
OPENAI_API_KEY=your_actual_openai_api_key_here
```
Only one API key is required. If both are provided, OpenAI is used as primary with Gemini as fallback.

### Step 5: Run the Agent
```powershell
# Interactive mode (will show sample files to choose from)
python main.py

# Or specify a file directly
python main.py --pdf sample_pdfs/mario_pizza.md
python main.py --pdf sample_pdfs/bright_smile_dental.md
python main.py --pdf "C:\path\to\your_business.pdf"
```

## Running the HTTP API Server
```powershell
# Start the FastAPI server
python app/server.py

# The server runs at http://localhost:8000
# API docs available at http://localhost:8000/docs
```

### API Endpoints
| Endpoint             | Method | Description                      |
| -------------------- | ------ | -------------------------------- |
| `/health`            | GET    | Health check and business status |
| `/agent/chat`        | POST   | Send message, get AI response    |
| `/business/load_pdf` | POST   | Load a new business document     |

### Example API Call
```bash
# Load a business
curl -X POST http://localhost:8000/business/load_pdf \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "sample_pdfs/mario_pizza.md"}'

# Send a chat message
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to order a large pepperoni pizza", "user_id": 1}'
```

## Running Tests
```powershell
# Run the smoke test (tests both sample businesses)
python tests/smoke_test.py

# Results are saved to tests/test_results.json
```

## Project Structure
```
agent/
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── agent/
│   ├── agent.py               # LangGraph 7-node state machine
│   ├── tools.py               # 26 agent tools
│   └── rag_engine.py          # LLM-only RAG (no embeddings)
├── core/
│   ├── database.py            # SQLite schema (11 tables)
│   ├── llm_client.py          # OpenAI + Gemini fallback
│   └── logger.py              # Structured SQLite logging
├── processing/
│   ├── pdf_processor.py       # Intelligent chunking
│   └── knowledge_enricher.py  # LLM-based enrichment
├── auth/
│   └── auth.py                # CLI login/register (bcrypt)
├── app/
│   └── server.py              # FastAPI HTTP server
├── tests/
│   └── smoke_test.py          # Automated test script
├── sample_pdfs/
│   ├── mario_pizza.md          # Sample: Italian restaurant
│   └── bright_smile_dental.md  # Sample: Dental clinic
└── calendar/                   # Auto-generated .ics files
```

## Architecture

### Agent Pipeline (7 Nodes)
```
receive_input → detect_intents → rag_retrieval → route_tools → execute_tools → build_response → critic → END
```

| Node             | Purpose                                               |
| ---------------- | ----------------------------------------------------- |
| `receive_input`  | Load conversation history, restore conversation state |
| `detect_intents` | LLM identifies all intents in the message             |
| `rag_retrieval`  | Find relevant knowledge chunks via LLM                |
| `route_tools`    | Custom batched LLM router selects tools               |
| `execute_tools`  | Run selected tools, update state                      |
| `build_response` | Generate response using composite prompt              |
| `critic`         | Validate response against PDF rules                   |

### Tools (26 Total)
**Cart/Orders:** add_to_cart, remove_from_cart, view_cart, confirm_order
**Booking:** book_appointment, reschedule_booking, cancel_booking, check_availability
**Delivery:** set_delivery_type, validate_address
**Info:** get_pricing, get_recommendations, get_order_history, get_business_hours, get_provider_info, get_faqs, get_promotions
**Loyalty:** apply_loyalty_discount, get_loyalty_balance
**Profile:** update_customer_profile, check_family_members
**Calendar:** get_calendar
**Other:** handle_dispute, search_web, escalate_to_human, get_global_stats

## Swapping to a New Business
```powershell
# Delete the existing database
del business_agent.db

# Run with a new file
python main.py --pdf path\to\new_business.pdf
```
The agent will automatically detect the new business type, generate services, and configure itself.

## Troubleshooting
| Issue                      | Solution                                                 |
| -------------------------- | -------------------------------------------------------- |
| `No API keys found`        | Edit `.env` and add `GEMINI_API_KEY` or `OPENAI_API_KEY` |
| `Gemini 429 / quota error` | Wait 60s (auto-retry) or switch to OpenAI                |
| `File not found`           | Use full path like `C:\Users\you\Desktop\menu.pdf`       |
| `bcrypt install fails`     | Run `pip install bcrypt --no-binary bcrypt`              |
| `PyMuPDF install fails`    | Run `pip install PyMuPDF` (not `pip install fitz`)       |
| `Database locked`          | Close other scripts or delete `.db` file and restart     |
| `Empty responses`          | Check API key is valid and has credit                    |

## Supported File Formats
| Format     | Extension | Notes                                 |
| ---------- | --------- | ------------------------------------- |
| PDF        | `.pdf`    | Best for formatted business documents |
| Markdown   | `.md`     | Great for structured text with tables |
| Plain Text | `.txt`    | Simple business descriptions          |
