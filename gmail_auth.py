"""
gmail_auth.py - Google OAuth 2.0 Authentication Manager for Smart Gmail Replier.

Handles loading existing tokens, refreshing expired tokens, and completing the
initial OAuth 2.0 browser authorization flow to generate token.json.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build

# Required Google OAuth 2.0 scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "credentials.json"
DEFAULT_TOKEN_PATH = BASE_DIR / "token.json"


class GmailAuthError(Exception):
    """Raised when Gmail authentication fails due to missing credentials or network issues."""
    pass


def get_gmail_credentials(
    credentials_path: Path = DEFAULT_CREDENTIALS_PATH,
    token_path: Path = DEFAULT_TOKEN_PATH,
    interactive: bool = True,
) -> Optional[Credentials]:
    """
    Retrieves or generates valid Google OAuth 2.0 credentials.
    
    Flow:
      1. If token_path exists, load and check validity.
      2. If expired and refresh_token is present, refresh the token.
      3. If no token, check for credentials_path. If missing, raise GmailAuthError.
      4. If credentials_path exists, initiate OAuth 2.0 desktop authorization flow.
      5. Save refreshed/new token to token_path.
    """
    creds: Optional[Credentials] = None
    credentials_path = Path(credentials_path)
    token_path = Path(token_path)

    # 1. Load saved token if available
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as e:
            print(f"[gmail_auth] Warning: Failed to parse existing token {token_path}: {e}", file=sys.stderr)
            creds = None

    # 2. Check token validity or refresh
    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            print("[gmail_auth] Refreshing expired OAuth token...", file=sys.stderr)
            creds.refresh(Request())
            # Save refreshed credentials
            with open(token_path, "w", encoding="utf-8") as token_file:
                token_file.write(creds.to_json())
            return creds
        except Exception as refresh_err:
            print(f"[gmail_auth] Failed to refresh token: {refresh_err}. Re-authenticating...", file=sys.stderr)
            creds = None

    # 3. If credentials.json is missing, provide explicit, friendly instructions
    if not credentials_path.exists():
        instructions = (
            "\n" + "=" * 70 + "\n"
            "❌ ERROR: Gmail OAuth Credentials File Not Found!\n"
            f"Expected file at: {credentials_path.resolve()}\n"
            "=" * 70 + "\n\n"
            "To enable live Gmail integration:\n"
            "1. Visit the Google Cloud Console: https://console.cloud.google.com/\n"
            "2. Create a project and enable the 'Gmail API' under APIs & Services > Library.\n"
            "3. Go to APIs & Services > OAuth consent screen:\n"
            "   - Set User Type to 'External', add your test Gmail address.\n"
            "4. Go to APIs & Services > Credentials:\n"
            "   - Click 'Create Credentials' > 'OAuth client ID'.\n"
            "   - Application type: 'Desktop app'.\n"
            "   - Download the JSON client secrets file.\n"
            f"5. Rename the downloaded file to 'credentials.json' and place it at:\n"
            f"   {credentials_path.resolve()}\n"
            "6. Re-run your command or start the MCP server to authenticate.\n"
            "=" * 70 + "\n"
        )
        raise GmailAuthError(instructions)

    if not interactive:
        raise GmailAuthError(
            f"Authentication required but interactive browser flow is disabled. "
            f"Run 'python gmail_auth.py' locally in your terminal to complete the authorization once."
        )

    # 4. Interactive OAuth 2.0 flow
    print("[gmail_auth] Initiating Google OAuth 2.0 authorization in your browser...", file=sys.stderr)
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=True)
        # Save token for future offline runs
        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        print(f"[gmail_auth] Successfully saved authorization token to {token_path.resolve()}", file=sys.stderr)
        return creds
    except Exception as flow_err:
        raise GmailAuthError(f"OAuth flow failed: {flow_err}") from flow_err


def get_gmail_service(
    credentials_path: Path = DEFAULT_CREDENTIALS_PATH,
    token_path: Path = DEFAULT_TOKEN_PATH,
    interactive: bool = True,
) -> Resource:
    """
    Builds and returns an authorized Google API Gmail v1 service client.
    """
    creds = get_gmail_credentials(credentials_path, token_path, interactive=interactive)
    return build("gmail", "v1", credentials=creds)


def is_authenticated(token_path: Path = DEFAULT_TOKEN_PATH) -> bool:
    """Checks whether a valid token.json exists without initiating an OAuth flow."""
    p = Path(token_path)
    if not p.exists():
        return False
    try:
        creds = Credentials.from_authorized_user_file(str(p), SCOPES)
        return bool(creds and (creds.valid or creds.refresh_token))
    except Exception:
        return False


if __name__ == "__main__":
    print("[gmail_auth] Testing Gmail Authentication Setup...")
    try:
        service = get_gmail_service()
        profile = service.users().getProfile(userId="me").execute()
        email_address = profile.get("emailAddress", "unknown")
        messages_total = profile.get("messagesTotal", 0)
        print(f"✅ Success! Authenticated as: {email_address} (Total Messages: {messages_total})")
    except GmailAuthError as auth_err:
        print(auth_err, file=sys.stderr)
        sys.exit(1)
    except Exception as ex:
        print(f"❌ Unexpected Error: {ex}", file=sys.stderr)
        sys.exit(1)
