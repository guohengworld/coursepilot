"""密码哈希 —— 使用 bcrypt（passlib 封装）"""
from passlib.context import CryptContext

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """对明文密码进行 bcrypt 哈希"""
    return _pwd_ctx.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码是否匹配哈希值"""
    return _pwd_ctx.verify(plain, hashed)
