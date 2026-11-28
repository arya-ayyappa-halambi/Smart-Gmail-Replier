"""
gmail_auth.py - Google OAuth 2.0 Authentication Manager for Smart Gmail Replier.

Handles loading existing tokens, refreshing expired tokens, and completing the
initial OAuth 2.0 browser authorization flow to generate token.json.
"""

import os
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import google_auth_oauthlib.flow as oauth_flow
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
import wsgiref.util

# Required Google OAuth 2.0 scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = BASE_DIR / "credentials.json"
DEFAULT_TOKEN_PATH = BASE_DIR / "token.json"

_LATEST_AUTH_URL: Optional[str] = None
_AUTH_URL_EVENT: threading.Event = threading.Event()

def get_latest_auth_url() -> Optional[str]:
    """Returns the most recently generated Google OAuth authorization URL."""
    return _LATEST_AUTH_URL

SUCCESS_REDIRECT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Signed In Successfully - MailMind AI</title>
  <meta http-equiv="refresh" content="0; url=http://localhost:8000/">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 20px;
      text-align: center;
    }
    .card {
      background: #1e293b;
      padding: 40px 48px;
      border-radius: 24px;
      border: 1px solid #334155;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
      max-width: 460px;
      width: 100%;
    }
    .icon-wrapper {
      width: 56px;
      height: 56px;
      background: rgba(37, 99, 235, 0.15);
      border: 2px solid #3b82f6;
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 20px;
      color: #60a5fa;
      font-size: 24px;
      font-weight: bold;
    }
    .spinner {
      width: 32px;
      height: 32px;
      border: 3px solid #3b82f6;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.75s linear infinite;
      margin: 16px auto 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    h2 { margin: 0 0 10px; font-size: 22px; font-weight: 700; color: #ffffff; letter-spacing: -0.02em; }
    p { margin: 0 0 20px; font-size: 14px; color: #94a3b8; line-height: 1.5; }
    .btn {
      display: inline-block;
      background: #2563eb;
      color: #ffffff;
      padding: 12px 28px;
      border-radius: 12px;
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .btn:hover { background: #1d4ed8; }
  </style>
  <script>
    // 1. Immediately redirect this active window to MailMind AI
    window.location.replace("http://localhost:8000/");

    // 2. If this window was opened as a popup from another tab, reload parent and close popup
    if (window.opener && !window.opener.closed) {
      try {
        window.opener.location.href = "http://localhost:8000/";
        setTimeout(function() { window.close(); }, 300);
      } catch(e) {}
    }
  </script>
</head>
<body>
  <div class="card">
    <div class="icon-wrapper">✓</div>
    <h2>Authentication Complete!</h2>
    <p>Signing you in and loading your inbox directly inside MailMind AI...</p>
    <div class="spinner"></div>
    <div style="margin-top: 24px;">
      <a href="http://localhost:8000/" class="btn">Enter MailMind AI</a>
    </div>
  </div>
</body>
</html>"""


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
            print(f"[gmail_auth] Failed to refresh token: {refresh_err}. Removing stale token file...", file=sys.stderr)
            token_path.unlink(missing_ok=True)
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
        global _LATEST_AUTH_URL, _AUTH_URL_EVENT
        _AUTH_URL_EVENT.clear()

        # Patch _RedirectWSGIApp so it responds with text/html and an immediate auto-redirect to http://localhost:8000/
        def _custom_html_wsgi_call(self, environ, start_response):
            start_response("200 OK", [("Content-type", "text/html; charset=utf-8")])
            self.last_request_uri = wsgiref.util.request_uri(environ)
            return [SUCCESS_REDIRECT_HTML.encode("utf-8")]

        oauth_flow._RedirectWSGIApp.__call__ = _custom_html_wsgi_call

        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        orig_auth_url = flow.authorization_url

        def _capturing_auth_url(**kwargs):
            global _LATEST_AUTH_URL
            url, state = orig_auth_url(**kwargs)
            _LATEST_AUTH_URL = url
            _AUTH_URL_EVENT.set()
            return url, state

        flow.authorization_url = _capturing_auth_url

        try:
            creds = flow.run_local_server(
                port=0,
                open_browser=True,
                success_message="Signed in successfully! Redirecting you to MailMind AI..."
            )
        finally:
            _LATEST_AUTH_URL = None
            _AUTH_URL_EVENT.clear()

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
    """Checks whether a valid and active token exists."""
    p = Path(token_path)
    if not p.exists():
        return False
    try:
        creds = Credentials.from_authorized_user_file(str(p), SCOPES)
        if not creds:
            return False
        if creds.valid:
            return True
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(p, "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
                return True
            except Exception as e:
                print(f"[gmail_auth] Expired token cannot be refreshed ({e}). Removing {p}", file=sys.stderr)
                p.unlink(missing_ok=True)
                return False
        return False
    except Exception:
        p.unlink(missing_ok=True)
        return False
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
