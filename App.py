import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Models API",
    description="Unified API for LLM, Vision & Image Generation using hosted services",
    version="1.0.0"
)

# CORS middleware - important for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production mein specific domain daalna
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FAL_API_KEY = os.getenv("FAL_API_KEY")

# Check if keys are present
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not set")
if not FAL_API_KEY:
    print("WARNING: FAL_API_KEY not set")

# Request/Response Models
class ChatRequest(BaseModel):
    prompt: str
    model: str = "meta-llama/llama-3-8b-instruct"  # default model
    max_tokens: int = 500

class ChatResponse(BaseModel):
    response: str
    model: str
    provider: str

class VisionRequest(BaseModel):
    image_url: str
    prompt: str = "Describe this image in detail"
    model: str = "google/gemini-pro-vision"

class VisionResponse(BaseModel):
    description: str
    model: str

class ImageGenRequest(BaseModel):
    prompt: str
    model: str = "fal-ai/flux/schnell"  # fast image gen model
    image_size: str = "1024x1024"

class ImageGenResponse(BaseModel):
    image_url: str
    model: str

# Health check endpoint
@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Models API"}

# 1. LLM Chat Endpoint (using OpenRouter)
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with an LLM via OpenRouter.
    Supports multiple models: llama-3, mistral, gemini, etc.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",  # Required by OpenRouter
                    "X-Title": "My AI App"
                },
                json={
                    "model": request.model,
                    "messages": [
                        {"role": "user", "content": request.prompt}
                    ],
                    "max_tokens": request.max_tokens
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            return ChatResponse(
                response=data["choices"][0]["message"]["content"],
                model=request.model,
                provider="OpenRouter"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Vision Model Endpoint (using OpenRouter's vision models)
@app.post("/vision", response_model=VisionResponse)
async def vision(request: VisionRequest):
    """
    Analyze an image using vision-language model via OpenRouter.
    Provide image URL and optional prompt.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "My AI App"
                },
                json={
                    "model": request.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": request.prompt},
                                {"type": "image_url", "image_url": {"url": request.image_url}}
                            ]
                        }
                    ],
                    "max_tokens": 500
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            return VisionResponse(
                description=data["choices"][0]["message"]["content"],
                model=request.model
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Image Generation Endpoint (using Fal.ai)
@app.post("/generate-image", response_model=ImageGenResponse)
async def generate_image(request: ImageGenRequest):
    """
    Generate an image from text prompt using Fal.ai.
    Supports multiple models: flux, stable-diffusion, etc.
    """
    if not FAL_API_KEY:
        raise HTTPException(status_code=500, detail="Fal.ai API key not configured")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"https://fal.run/{request.model}",
                headers={
                    "Authorization": f"Key {FAL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "prompt": request.prompt,
                    "image_size": request.image_size
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            # Fal.ai typically returns images in 'images' array or 'image' field
            if "images" in data and len(data["images"]) > 0:
                image_url = data["images"][0]["url"]
            elif "image" in data:
                image_url = data["image"]["url"]
            else:
                image_url = data.get("url", "")
            
            return ImageGenResponse(
                image_url=image_url,
                model=request.model
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoint: List available models (optional)
@app.get("/models")
async def list_models():
    """Return some recommended models for each task."""
    return {
        "llm": [
            "meta-llama/llama-3-8b-instruct",
            "mistralai/mistral-7b-instruct",
            "google/gemini-pro",
            "openai/gpt-3.5-turbo"
        ],
        "vision": [
            "google/gemini-pro-vision",
            "openai/gpt-4-vision-preview",
            "llava-hf/llava-13b"
        ],
        "image_generation": [
            "fal-ai/flux/schnell",
            "fal-ai/stable-diffusion-v3",
            "fal-ai/playground-v2"
        ]
}
