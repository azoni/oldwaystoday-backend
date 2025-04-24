from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import os
import httpx
from prompt_templates import prompt_templates

load_dotenv()
app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY is not set.")

@app.post("/chat")
async def chat(request: Request):
    try:
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
