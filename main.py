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
from assistant_messages import error_messages
import logging
from datetime import datetime
import traceback

# Basic log setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error("🔥 Unhandled Exception:")
    logging.error("🔍 Exception Type: %s", type(exc).__name__)
    logging.error("📄 Exception Message: %s", str(exc))
    logging.error("📚 Full Traceback:\n%s", traceback.format_exc())
    
    return JSONResponse(
        status_code=200,  # looks successful to frontend
        content={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            error_messages["server_error"]
                        )
                    }
                }
            ]
        }
    )
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
                            error_messages["rate_limit"]
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
@limiter.limit("50/minute")  # Adjust this rate as needed
async def chat(request: Request):
    try:
        ip = request.client.host
        print("🔹 Request from:", ip)
        body = await request.json()
        messages = body.get("messages")
        mode = body.get("mode", "oldwaystoday")

        if not messages or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Invalid 'messages' format")
        
        # ✅ Extract user's latest message
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # ✅ Check word count
        word_count = len(user_message.split())
        print(f"🔹 User message word count: {word_count}")

        if word_count > 100:  # or whatever limit you want
            return JSONResponse(
                status_code=200,
                content={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    error_messages["too_long"]
                                )
                            }
                        }
                    ]
                }
            )

        # Add system prompt
        template = prompt_templates.get(mode, {})
        system_prompt = template.get("system", "")
        structured_messages = [{"role": "system", "content": system_prompt}] + messages
        logging.info(f"🔸 New request from {request.client.host} | Mode: {mode} | Msg: {messages[-1]['content'][:100]}")

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
        # ✅ Parse the actual response JSON
        response_json = response.json()

        # ✅ Now you can log tokens safely
        usage = response_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        logging.info(f"✅ OpenAI response tokens — Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
