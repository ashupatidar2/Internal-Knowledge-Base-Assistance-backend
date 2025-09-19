import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from retriever import retrieve_top_k, answer_query

load_dotenv()

app = FastAPI(title="AI Knowledge Base Assistant")

# Enable CORS to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.post("/search")
async def search(req: SearchRequest):
    try:
        matches = await retrieve_top_k(query=req.query, top_k=req.top_k or 5)
        return JSONResponse({"matches": matches})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = await answer_query(query=req.query, top_k=req.top_k or 5)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)

# To run: uvicorn main:app --reload
