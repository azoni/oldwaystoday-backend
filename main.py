from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
import os
import httpx
from prompt_templates import prompt_templates

app = FastAPI()

# ✅ Add this CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8888",        # Local dev
        "https://oldwaystoday.com"     # Production site
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY is not set.")

# ✅ Add rate limiter middleware
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc):
    return JSONResponse(
        status_code=200,  # <-- make it look successful
        content={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "You've asked many questions, wise one 🌿\n\n"
                            "Let your thoughts settle, and return with clarity in a moment."
                        )
                    }
                }
            ]
        }
    )

# ✅ Health check route
@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/chat")
@limiter.limit("10/minute")  # Adjust this rate as needed
async def chat(request: Request):
    try:
        ip = request.client.host
        print("🔹 Request from:", ip)
        body = await request.json()
        messages = body.get("messages")
        mode = body.get("mode", "oldwaystoday")

        if not messages or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Invalid 'messages' format")

        # Add system prompt
        template = prompt_templates.get(mode, {})
        system_prompt = template.get("system", "")
        structured_messages = [{"role": "system", "content": system_prompt}] + messages

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": structured_messages
                }
            )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
