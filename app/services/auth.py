import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    assert os.getenv("TEST_USERNAME")
    assert os.getenv("TEST_PASSWORD")
    correct_username = secrets.compare_digest(credentials.username, os.getenv("TEST_USERNAME"))
    correct_password = secrets.compare_digest(credentials.password, os.getenv("TEST_PASSWORD"))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
