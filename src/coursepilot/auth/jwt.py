"""JWT token 签发与验证"""
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from coursepilot.config import settings

def create_token(user_id: str, role: str) -> tuple[str, int]:
    """签发 JWT token

    返回（token_string, expire_seconds）
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(seconds=settings.jwt_expire_seconds)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": expire,
    }
    token = jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm,
    )
    return token, settings.jwt_expire_seconds

def decode_token(token: str) -> dict | None:
    """解码并验证 JWT token。失败返回 None"""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        return None
