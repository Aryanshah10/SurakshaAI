import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException, status, Depends
from models.schemas import UserLogin, Token, UserRole, TokenData
from utils.auth import authenticate_user, create_access_token, get_current_user

router = APIRouter(prefix="/api/auth", tags=["Auth"])

@router.post("/login", response_model=Token)
def login(credentials: UserLogin):
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    token = create_access_token({
        "sub": user["username"],
        "role": user["role"]
    })
    return Token(
        access_token=token,
        token_type="bearer",
        role=user["role"]
    )

@router.get("/me")
def get_me(current_user: TokenData = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "role": current_user.role
    }