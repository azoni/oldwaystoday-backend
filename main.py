from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import os
import httpx
from prompt_templates import prompt_templates
from fastapi.middleware.cors import CORSMiddleware

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
app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY is not set.")

@app.post("/chat")
async def chat(request: Request):
    try:
        print("Incoming origin:", request.headers.get("origin"))
        body = await request.json()
        messages = body.get("messages")
        mode = body.get("mode", "azoni")

        print("MODE:", mode)
        print("MESSAGES:", messages)

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
