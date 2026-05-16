# App.py — FastAPI backend only

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ── ENV ───────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production-long-random-string")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MIN = 30
MODEL = "llama-3.3-70b-versatile"

# ── AUTH SETUP ────────────────────────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
users_db: dict[str, dict] = {}
bearer_scheme = HTTPBearer()

# ── LLM ───────────────────────────────────────────────────────────────────────
groq_llm = ChatGroq(
    model=MODEL,
    temperature=0,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

# ── FASTAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Groq LLM API — AI Lab",
    description="Secured FastAPI for Groq Llama 3.3 70B consumption.",
    version="1.0.0",
)

# ── API Keys ──────────────────────────────────────────────────────────────────
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