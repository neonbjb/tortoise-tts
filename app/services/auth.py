import base64
import os
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, APIKeyHeader

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Secret key to encode and decode JWT tokens
DEFAULT_SECRET_KEY = str(os.getenv("DEFAULT_SECRET_KEY", "fek3kz9xzlsndSuczhgjds0vndi"))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24*60

security = HTTPBasic()
api_key_header = APIKeyHeader(
    name="Authorization", auto_error=False)


def verify_user(username: str, password: str):
    user = os.getenv("DEFAULT_USERNAME")
    if user and secrets.compare_digest(os.getenv("DEFAULT_PASSWORD"), password):
        return True
    return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, DEFAULT_SECRET_KEY, algorithm=ALGORITHM)
    print(f"Encoded JWT: {encoded_jwt}")
    return encoded_jwt

async def get_current_user(
    request: Request,  # Access the full request to manually check headers
    authorization: Optional[str] = Security(api_key_header)
):
    logger.debug("Attempting to authenticate user...")
    # First, try to authenticate using Basic Authentication
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Basic "):
        print("Using Basic Auth...")
        auth = auth_header.split(" ")[1]
        credentials = base64.b64decode(auth).decode("utf-8").split(":")
        username = credentials[0]
        password = credentials[1]

        correct_username = secrets.compare_digest(username, os.getenv("DEFAULT_USERNAME"))
        correct_password = secrets.compare_digest(password, os.getenv("DEFAULT_PASSWORD"))
        
        if correct_username and correct_password:
            logger.debug("Basic auth successful.")
            return {"auth": "basic", "user": username}
        else:
            logger.debug("Basic auth failed.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )

    # If Basic Auth is not provided, try to authenticate using API Key (JWT)
    elif authorization:
        print("Using API Key...")
        logger.debug(f"Authorization header received: {authorization[:5]}...")
        scheme, _, token = authorization.partition(" ")
        
        if scheme.lower() != "bearer":
            logger.debug("Invalid authentication scheme.")
            raise HTTPException(
                status_code=401, detail="Invalid authentication scheme")
        
        try:
            payload = jwt.decode(
                token, DEFAULT_SECRET_KEY, 
                algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            
            if username is None:
                logger.debug("JWT payload does not contain a username.")
                raise HTTPException(status_code=401, detail="Could not validate credentials")
            
            logger.debug("JWT validation successful.")
            return {"auth": "api_key", "user": username}
        
        except jwt.PyJWTError as e:
            logger.debug(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=401, detail="Could not validate credentials")

    else:
        logger.debug("No credentials provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Basic"},
        )