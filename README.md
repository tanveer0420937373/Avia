# AI Models API

Unified FastAPI backend for LLM, Vision & Image Generation using hosted APIs (OpenRouter + Fal.ai).

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file and add your API keys (see `.env.example`)
4. Run locally: `uvicorn main:app --reload`
5. Access docs at `http://localhost:8000/docs`

## Deploy on Render

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect repository
4. Add environment variables (OPENROUTER_API_KEY, FAL_API_KEY)
5. Deploy!

## API Endpoints

- `POST /chat` - Chat with LLM
- `POST /vision` - Image analysis
- `POST /generate-image` - Generate images
- `GET /models` - List recommended models

## Mobile App Integration

Use the deployed URL as your backend base URL. All endpoints return JSON.
