import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
import os

from models.schemas import TokenData, UserRole

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "fallback_dev_key_change_in_prod")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ─── Hardcoded users for hackathon ──────────────────
FAKE_USERS_DB = {
    "rajesh.kumar": {
        "hashed_password": pwd_context.hash(os.getenv("OFFICER1_PASS")),
        "role": UserRole.officer,
    },
    "priya.nair": {
        "hashed_password": pwd_context.hash(os.getenv("OFFICER2_PASS")),
        "role": UserRole.officer,
    },
}


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str):
    user = FAKE_USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        return TokenData(username=username, role=UserRole(role))
    except JWTError:
        raise credentials_exception


def require_officer(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Dependency — officer-only routes."""
    if current_user.role != UserRole.officer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Officer access required"
        )
    return current_user