from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import os
import httpx
from prompt_templates import prompt_templates

load_dotenv()
app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages")
        mode = body.get("mode", "azoni")

        template = prompt_templates.get(mode)
        if not template:
            raise HTTPException(status_code=400, detail="Invalid mode")

        structured_messages = [
            {"role": "system", "content": template["system"]},
            *messages
        ]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": structured_messages
                }
            )
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
