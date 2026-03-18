from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

PASSWORD = "LakeMountain4622"


class LoginBody(BaseModel):
    name: str
    password: str


@router.post("/smile-auth/login")
async def login(body: LoginBody):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if body.password != PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"ok": True, "annotator": body.name.strip()}
