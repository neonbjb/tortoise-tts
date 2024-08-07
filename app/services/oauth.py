import os
from google_auth_oauthlib.flow import Flow
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2AuthorizationCodeBearer

# Configuration
CLIENT_SECRETS_FILE = 'client_secret.json'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
REDIRECT_URI = 'http://localhost:8000/callback'
OAUTH2_SCHEME = OAuth2AuthorizationCodeBearer(authorizationUrl='https://accounts.google.com/o/oauth2/auth',
                                              tokenUrl='https://oauth2.googleapis.com/token')

def get_authenticated_service():
    """
    Authenticate and return a service object for the YouTube API.
    """
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, SCOPES,
        redirect_uri=REDIRECT_URI
    )
    
    # Tell the user to go to the authorization URL.
    auth_url, _ = flow.authorization_url(prompt='consent')
    print('Please go to this URL: {}'.format(auth_url))
    
    # The user will get an authorization code. This code is used to get the access token.
    code = input('Enter the authorization code: ')
    flow.fetch_token(code=code)
    
    # You can use flow.credentials, or you can just get a requests session using flow.authorized_session.
    session = flow.authorized_session()
    print(session.get('https://www.googleapis.com/userinfo/v2/me').json())
    return session

async def get_current_user(token: str = Depends(OAUTH2_SCHEME)):
    # Assuming the token is valid and we can get user info
    # In practice, you would validate the token and fetch user info
    try:
        session = get_authenticated_service()
        user_info = session.get('https://www.googleapis.com/userinfo/v2/me').json()
        return user_info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
