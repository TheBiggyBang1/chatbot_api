# app.py — Merged Streamlit Chatbot + FastAPI backend

import os
import threading
import uvicorn

from dotenv import load_dotenv
from datetime import datetime, timedelta

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── ENV ───────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── JWT / AUTH SETTINGS ───────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production-long-random-string")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MIN = 30

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
users_db: dict[str, dict] = {}
bearer_scheme = HTTPBearer()

# ── SHARED LLM ────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a helpful AI assistant."

groq_llm = ChatGroq(
    model=MODEL,
    temperature=0,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Groq LLM API — AI Lab",
    description="Secured FastAPI for Groq Llama 3.3 70B consumption.",
    version="1.0.0",
)

# ── API Key store ─────────────────────────────────────────────────────────────
VALID_API_KEYS: dict[str, str] = {
    "sk-student-001": "student_a",
    "sk-student-002": "student_b",
    "sk-admin-999": "admin",
}

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    system_prompt: str = "You are a helpful AI assistant."

class ChatResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int
    authenticated_as: str

class AuthRequest(BaseModel):
    username: str
    password: str

# ── Auth helpers ──────────────────────────────────────────────────────────────
def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return VALID_API_KEYS[x_api_key]

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

def require_jwt(credentials=Depends(bearer_scheme)) -> dict:
    payload = decode_token(credentials.credentials)
    username = payload.get("sub")
    if not username or username not in users_db:
        raise HTTPException(status_code=401, detail="User not found.")
    return payload

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/ping", tags=["Public"])
async def ping():
    return {"status": "ok", "message": "Groq API is running."}

@app.post("/chat", response_model=ChatResponse, tags=["AI — Protected"])
async def chat(req: ChatRequest, username: str = Depends(require_api_key)):
    messages = [
        SystemMessage(content=req.system_prompt),
        HumanMessage(content=req.question),
    ]
    response = groq_llm.invoke(messages)
    usage = response.usage_metadata or {}
    return ChatResponse(
        answer=response.content,
        model=MODEL,
        tokens_used=usage.get("total_tokens", 0),
        authenticated_as=username,
    )

@app.post("/auth/register", status_code=201, tags=["Auth"])
async def register(req: AuthRequest):
    if req.username in users_db:
        raise HTTPException(status_code=409, detail="Username already taken.")
    users_db[req.username] = {"hashed_password": pwd_ctx.hash(req.password)}
    return {"message": f"User {req.username!r} created."}

@app.post("/auth/login", tags=["Auth"])
async def login(req: AuthRequest):
    user = users_db.get(req.username)
    if not user or not pwd_ctx.verify(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect credentials.")
    token = create_token({"sub": req.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chat2", response_model=ChatResponse, tags=["AI — Protected"])
async def chat2(req: ChatRequest, payload: dict = Depends(require_jwt)):
    username = payload["sub"]
    messages = [
        SystemMessage(content=req.system_prompt),
        HumanMessage(content=req.question),
    ]
    response = groq_llm.invoke(messages)
    usage = response.usage_metadata or {}
    return ChatResponse(
        answer=response.content,
        model=MODEL,
        tokens_used=usage.get("total_tokens", 0),
        authenticated_as=username,
    )

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_streamlit():
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not found in .env file.")
        st.stop()

    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
    st.title("AI Chatbot")
    st.caption(f"Powered by **{MODEL}** via Groq API")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = SYSTEM_PROMPT

    with st.sidebar:
        st.write("### LLM Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        max_tokens = st.slider("Max tokens", 50, 2048, 512)
        new_system = st.text_area("System prompt", value=st.session_state.system_prompt)

        if st.button("Apply"):
            st.session_state.system_prompt = new_system
            st.success("Settings applied!")

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    current_llm = ChatGroq(
                        model=MODEL,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=GROQ_API_KEY
                    )
                    lc_messages = [SystemMessage(content=st.session_state.system_prompt)]
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            lc_messages.append(HumanMessage(content=msg["content"]))
                        else:
                            lc_messages.append(AIMessage(content=msg["content"]))

                    response = current_llm.invoke(lc_messages)
                    answer = response.content
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start FastAPI in background thread, Streamlit in main thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    run_streamlit()