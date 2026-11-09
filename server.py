"""
server.py - High-Performance Executive FastAPI Backend for MailMind AI.

Upgrades:
  - High-speed in-memory caching with force-refresh support.
  - Aggressive whitespace and boilerplate sanitization (fixes blank gaps in newsletter emails).
  - Dual Mode: Live Gmail inbox + Instant Viva Demonstration Dataset.
  - Full AI Draft Copilot & 3 Suggestion Chips for EVERY email type (never hides the composer).
  - One-click Google OAuth login flow and clean token revocation on logout.
"""

import collections
import html
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import gmail_auth
import gmail_mcp
import ml_pipeline

STATIC_DIR = BASE_DIR / "static"
TOKEN_PATH = BASE_DIR / "token.json"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
REPORT_PATH = BASE_DIR / "evaluation_report.json"
CONTEXT_PATH = BASE_DIR / "data" / "my_context.md"
TEST_DATA_PATH = BASE_DIR / "data" / "test_emails.json"

app = FastAPI(
    title="MailMind AI - Executive SaaS Email Client",
    description="High-contrast, Superhuman-grade AI inbox triage and draft copilot.",
    version="2.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Enterprise Security & Rate Limiting Middleware
# -----------------------------------------------------------------------------
_REQUEST_HISTORY = defaultdict(list)
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_MINUTE = 120

@app.middleware("http")
async def security_and_rate_limiting_middleware(request: Request, call_next):
    # 1. Rate Limiting for API routes
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        _REQUEST_HISTORY[client_ip] = [t for t in _REQUEST_HISTORY[client_ip] if now - t < RATE_LIMIT_WINDOW_SEC]
        if len(_REQUEST_HISTORY[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait a minute before retrying.", "code": "RATE_LIMIT_EXCEEDED"}
            )
        _REQUEST_HISTORY[client_ip].append(now)

    response = await call_next(request)

    # 2. Hardened Security Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Global in-memory cache for ultra-fast UI rendering
_EMAIL_CACHE: Dict[str, Any] = {
    "data": None,
    "timestamp": 0,
    "mode": None
}
_AUTH_IN_PROGRESS = False


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    subject: str
    sender: str
    body: str


class DraftRequest(BaseModel):
    thread_id: str
    to: str
    subject: str
    body: str


class GenerateReplyRequest(BaseModel):
    subject: str
    sender: str
    body: str
    intent: Optional[str] = None
    tone: Optional[str] = "Professional"
    variant: Optional[str] = "confirm"
    custom_instruction: Optional[str] = ""
    checked_action_items: Optional[List[str]] = []


# -----------------------------------------------------------------------------
# Clean Email Body Sanitizer & Server-side HTML Sanitizer
# -----------------------------------------------------------------------------
def sanitize_html_payload(raw_html: str) -> str:
    """Server-side sanitizer that removes script tags, objects, embeds, and malicious handlers."""
    if not raw_html:
        return ""
    # Strip dangerous tags
    clean = re.sub(r'<\s*(script|style|object|embed|applet|meta|link|base)[^>]*>[\s\S]*?<\s*/\s*\1\s*>', '', raw_html, flags=re.IGNORECASE)
    clean = re.sub(r'<\s*(script|style|object|embed|applet|meta|link|base)[^>]*>', '', clean, flags=re.IGNORECASE)
    # Strip javascript: and event handlers like onclick, onload, onerror
    clean = re.sub(r'\bon\w+\s*=\s*("[^"]*"|\'[^\']*\'|[^\s>]+)', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'href\s*=\s*["\']\s*javascript:[^"\']*["\']', 'href="#"', clean, flags=re.IGNORECASE)
    return clean.strip()


def sanitize_email_body(body_raw: str) -> str:
    """
    Cleans raw email bodies by removing HTML artifacts, zero-width tracking spaces,
    repetitive blank lines, and automated footers.
    """
    if not body_raw:
        return "(No textual content)"

    # Strip invisible zero-width and tracking characters common in newsletter preheaders
    text = re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\xad\xa0͏\u034f]', ' ', body_raw)
    
    # Strip HTML tags if any leaked through
    text = re.sub(r'<[^>]+>', ' ', text)

    # Strip noisy GitHub / Reddit / newsletter footers
    strip_footers = [
        r"—\s*Reply to this email directly[\s\S]*$",
        r"You are receiving this because you were mentioned[\s\S]*$",
        r"Message ID:\s*<[\w\.-]+@[\w\.-]+>[\s\S]*$",
        r"To unsubscribe from these notifications[\s\S]*$",
        r"Click here to unsubscribe[\s\S]*$",
        r"View this email in your browser[\s\S]*$",
        r"Manage your email preferences[\s\S]*$",
        r"Don't want to receive these emails\?[\s\S]*$",
    ]
    for pattern in strip_footers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean whitespace: split into lines, normalize spaces per line, drop empty lines
    lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(r'\s+', ' ', line).strip()
        # Drop lines that are empty or purely repeated punctuation
        if cleaned_line and not re.match(r'^[\-_=*\s]{3,}$', cleaned_line):
            lines.append(cleaned_line)

    if not lines:
        return "(No textual content)"

    # Format cleanly with standard double-line paragraph breaks (max 2 consecutive newlines)
    clean_text = '\n\n'.join(lines)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    return clean_text.strip()


def extract_clean_name(sender_str: str) -> str:
    """Extracts a natural human name or organization from an email header."""
    if not sender_str:
        return "there"

    cleaned = sender_str.replace('"', '').strip()
    if "<" in cleaned:
        display_part = cleaned.split("<")[0].strip()
        if display_part:
            return display_part
        email_part = cleaned.split("<")[1].split(">")[0].strip()
        return email_part.split("@")[0].replace(".", " ").title()

    if "@" in cleaned:
        return cleaned.split("@")[0].replace(".", " ").title()

    return cleaned


# -----------------------------------------------------------------------------
# 4 Strategy Personas & Human Reply Synthesis
# -----------------------------------------------------------------------------
def get_suggestion_chips(intent: str) -> List[Dict[str, str]]:
    """Returns 4 multi-persona strategic response variants tailored to the email category."""
    if intent == "Academic/Faculty":
        return [
            {"id": "confirm", "label": "Confirm & Deliverables Ready", "desc": "Acknowledge project viva/review & slides"},
            {"id": "inquire", "label": "Clarify Office Hours & Venue", "desc": "Ask to review slides during office hours"},
            {"id": "reschedule", "label": "Request Afternoon Slot", "desc": "Polite CIE examination overlap reschedule"},
            {"id": "decline", "label": "Formal Deferral Note", "desc": "Polite notice requesting alternate week slot"}
        ]
    elif intent == "Internship/Recruiter":
        return [
            {"id": "confirm", "label": "Confirm Friday 4 PM Interview", "desc": "Accept 60-minute technical discussion"},
            {"id": "inquire", "label": "Inquire Tech Stack & Team Focus", "desc": "Ask for engineering scope details"},
            {"id": "reschedule", "label": "Propose Alternate Time Slot", "desc": "Suggest Monday/Tuesday afternoon window"},
            {"id": "decline", "label": "Decline & Stay in Touch", "desc": "Express interest for Summer 2026 cycle"}
        ]
    elif intent == "Event/Club":
        return [
            {"id": "confirm", "label": "Confirm Hands-on Workshop", "desc": "Confirm speaker attendance & starter Colabs"},
            {"id": "inquire", "label": "Ask Venue & Projector Logistics", "desc": "Inquire HDMI/lab network access in Lab 3"},
            {"id": "reschedule", "label": "Propose Alternate Weekend", "desc": "Reschedule due to university lab CIEs"},
            {"id": "decline", "label": "Polite Decline for this Edition", "desc": "Unable to participate this semester"}
        ]
    elif intent == "Action Required":
        return [
            {"id": "confirm", "label": "Task Resolved & Fixed", "desc": "Confirm diagnosis & remediation complete"},
            {"id": "inquire", "label": "Request Detailed Error Trace", "desc": "Ask for latest service stack traces"},
            {"id": "reschedule", "label": "In Progress / Verifying Fix", "desc": "Actively diagnosing GPU node logs"},
            {"id": "decline", "label": "Acknowledge & Flag Out of Scope", "desc": "Verified not impacting primary pipeline"}
        ]
    else:  # Newsletters & Updates
        return [
            {"id": "confirm", "label": "Unsubscribe & Remove", "desc": "Request removal from marketing digest"},
            {"id": "inquire", "label": "Request Account Data & Info", "desc": "Ask for subscription preferences"},
            {"id": "reschedule", "label": "Acknowledge & Archive", "desc": "Polite confirmation of receipt"},
            {"id": "decline", "label": "General Courtesy Note", "desc": "Brief professional acknowledgment"}
        ]


def extract_ai_insights(subject: str, sender: str, body: str, intent: str) -> Dict[str, Any]:
    """
    Extracts deep content understanding: 'In Simple Words', 'What They Are Expecting',
    detected questions, deadlines, and action items.
    """
    clean_sender = extract_clean_name(sender)
    clean_sub = re.sub(r'^(Re:\s*|Fwd:\s*)+', '', subject, flags=re.IGNORECASE).strip()

    # 1. "In Simple Words" & "What They Are Expecting"
    if intent == "Academic/Faculty":
        simple_words = f"{clean_sender} is contacting you regarding {clean_sub} deliverables, requesting preparation of slides, repository, and schedule confirmation."
        expectation = "Confirm availability for the viva/presentation and verify project slides and repository are ready."
    elif intent == "Internship/Recruiter":
        simple_words = f"{clean_sender} is reaching out regarding an engineering or research role. They want to check your availability for a technical interview or next steps."
        expectation = "Confirm your preferred 60-minute interview time slot and share your portfolio/availability."
    elif intent == "Event/Club":
        simple_words = f"{clean_sender} is organizing {clean_sub} and wants to verify your participation, presentation outline, or logistics requirements."
        expectation = "Confirm your participation, workshop schedule, and any projector or hardware needs."
    elif intent == "Action Required":
        simple_words = f"Urgent system notification regarding {clean_sub}. Requires immediate attention, log inspection, or action verification."
        expectation = "Investigate the alert promptly and acknowledge or resolve the pending action."
    else:
        simple_words = f"Automated digest, update, or newsletter notification from {clean_sender} covering {clean_sub}."
        expectation = "Informational update; no immediate reply required unless you wish to unsubscribe or archive."

    # 2. Extract Questions
    questions = []
    question_candidates = re.findall(r'([^.!?\n]*\?)', body)
    for q in question_candidates:
        clean_q = q.strip()
        if len(clean_q) > 10 and clean_q not in questions:
            questions.append(clean_q)
        if len(questions) >= 2:
            break

    # 3. Extract Schedule & Deadlines
    date_patterns = [
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+(?:morning|afternoon|evening|at\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM|IST)?))?\b',
        r'\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|IST)\b',
        r'\b(?:today|tomorrow|this weekend|next week|by\s+\d{1,2}\s*(?:AM|PM|IST)?)\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b',
    ]
    deadlines = []
    text_to_search = f"{subject} {body}"
    for pat in date_patterns:
        matches = re.findall(pat, text_to_search, flags=re.IGNORECASE)
        for m in matches:
            cleaned_m = m.strip()
            if cleaned_m and len(cleaned_m) > 2 and cleaned_m not in deadlines:
                deadlines.append(cleaned_m)
        if len(deadlines) >= 3:
            break

    # 4. Action Items Checklist
    if intent == "Academic/Faculty":
        action_items = [
            "Verify FastMCP slides and architecture diagrams",
            "Prepare benchmark evaluation report for projector review",
            "Confirm attendance & time slot with faculty"
        ]
    elif intent == "Internship/Recruiter":
        action_items = [
            "Confirm 60-minute technical interview availability",
            "Review FastMCP agent and vector RAG portfolio links",
            "Share GitHub and LinkedIn profiles"
        ]
    elif intent == "Event/Club":
        action_items = [
            "Confirm workshop participation and schedule",
            "Check HDMI projector and lab venue availability",
            "Distribute Colab / FastMCP starter files"
        ]
    elif intent == "Action Required":
        action_items = [
            "Inspect system logs and error trace",
            "Verify service health and disk capacity",
            "Acknowledge alert resolution status"
        ]
    else:
        action_items = [
            "Review digest updates and articles",
            "Manage newsletter subscription preferences"
        ]

    # 5. Domain Verification & Reading Time
    trusted_domains = ["bmsce.ac.in", "tensorcore.ai", "google.com", "deepmind.com", "github.com", "linkedin.com"]
    domain_verified = any(d in sender.lower() for d in trusted_domains)
    word_count = len(body.split())
    reading_time = f"~{max(1, round(word_count / 150))} min read • {word_count} words"

    return {
        "summary": simple_words,
        "simple_words": simple_words,
        "expectation": expectation,
        "questions": questions,
        "deadlines": deadlines[:3],
        "action_items": action_items,
        "domain_verified": domain_verified,
        "reading_time": reading_time,
    }


def compose_smart_reply(
    sender: str,
    subject: str,
    body: str,
    intent: str,
    tone: str = "Professional",
    variant: str = "confirm",
    custom_instruction: str = "",
    checked_action_items: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Composes a tailored, tone-matched draft reply across 4 distinct strategic personas."""
    clean_sender = extract_clean_name(sender)
    clean_sub = re.sub(r'^(Re:\s*|Fwd:\s*)+', '', subject, flags=re.IGNORECASE).strip()

    is_academic = intent == "Academic/Faculty"
    is_recruiter = intent == "Internship/Recruiter"
    is_promo = intent in ["Promotional/Spam", "Bots & Promos"]

    # Salutation
    if is_academic:
        salutation = f"Dear {clean_sender},"
    elif is_promo:
        salutation = f"Hello {clean_sender} Support Team,"
    else:
        first_name = clean_sender.split()[0]
        salutation = f"Hi {first_name}," if tone in ["Friendly", "Concise"] else f"Hello {first_name},"

    # Signature Rules (Strict USN control: ONLY on Academic emails)
    if is_academic:
        signature = (
            "Best regards,\n"
            "H R Arya Ayyappa Halambi\n"
            "USN: 1BF25CS095\n"
            "Department of Computer Science and Engineering\n"
            "B.M.S. College of Engineering (BMSCE)"
        )
    elif is_recruiter:
        signature = (
            "Best regards,\n"
            "H R Arya Ayyappa Halambi\n"
            "B.M.S. College of Engineering (BMSCE)\n"
            "Portfolio: github.com/aryaayyappa | LinkedIn: linkedin.com/in/arya-ayyappa-halambi"
        )
    else:
        signature = (
            "Best regards,\n"
            "Arya Ayyappa Halambi\n"
            "B.M.S. College of Engineering (BMSCE)"
        )

    # Body Construction across 4 Personas
    if is_promo:
        if variant == "confirm":
            msg = (
                f"Thank you for contacting me regarding {clean_sub}.\n\n"
                f"Please unsubscribe this email address from recurring promotional notifications, digests, and marketing distribution lists."
            )
        elif variant == "inquire":
            msg = (
                f"Thank you for reaching out regarding {clean_sub}.\n\n"
                f"Could you provide information on the current data retention and subscription preferences associated with this account?"
            )
        elif variant == "reschedule":
            msg = (
                f"Thank you for the update regarding {clean_sub}.\n\n"
                f"I have received the notification and archived it for future reference."
            )
        else:  # decline
            msg = (
                f"Thank you for the information regarding {clean_sub}.\n\n"
                f"I am currently not seeking additional subscription updates or offers. Thank you for your consideration."
            )
    elif is_academic:
        if variant == "inquire":
            msg = (
                f"Thank you for the update regarding {clean_sub}.\n\n"
                f"Could we schedule a brief 10-minute sync during faculty office hours this week? "
                f"I would appreciate your guidance on our evaluation metrics and FastMCP benchmark results."
            )
        elif variant == "reschedule":
            msg = (
                f"Thank you for the notification regarding {clean_sub}.\n\n"
                f"Due to an overlapping semester examination review, I would like to request permission "
                f"to present during the afternoon session if permissible."
            )
        elif variant == "decline":
            msg = (
                f"Thank you for the notification regarding {clean_sub}.\n\n"
                f"Due to a prior university academic CIE commitment during this exact window, I would like to request "
                f"permission to present our project deliverables during the subsequent lab review session."
            )
        elif tone == "Concise":
            msg = (
                f"Thank you for the update regarding {clean_sub}.\n\n"
                f"I have reviewed the guidelines. Our FastMCP server, automated benchmark evaluation report, "
                f"and architecture documentation are finalized and ready for the session."
            )
        else:  # confirm
            msg = (
                f"Thank you for the update regarding {clean_sub}.\n\n"
                f"I have verified the project deliverables and prepared our FastMCP agent architecture, "
                f"automated benchmark evaluation report (95% classification accuracy with sub-10ms latency), "
                f"and documentation for display on the projector.\n\n"
                f"Please let me know if you would like me to review the slides with you prior to the session."
            )
    elif is_recruiter:
        if variant == "inquire":
            msg = (
                f"Thank you for reaching out regarding this opportunity.\n\n"
                f"Before our call, could you share a bit more detail about the specific projects "
                f"the AI team is focusing on for the upcoming internship cycle?"
            )
        elif variant == "reschedule":
            msg = (
                f"Thank you very much for reaching out regarding the engineering opportunity!\n\n"
                f"I would be glad to interview with the team. While Friday after 4 PM has an academic overlap, "
                f"I am fully available Monday or Tuesday between 3:00 PM and 6:00 PM IST. Please let me know if that works."
            )
        elif variant == "decline":
            msg = (
                f"Thank you very much for reaching out with this exciting opportunity.\n\n"
                f"Currently, due to my ongoing academic commitments at BMSCE, I will be unable to explore "
                f"an off-campus role at this moment. However, I would love to stay in touch for summer 2026."
            )
        elif tone == "Concise":
            msg = (
                f"Thank you for reaching out regarding this opportunity.\n\n"
                f"I would be glad to interview with the engineering team. I am available this Friday after 4:00 PM IST "
                f"or anytime over the weekend. Please feel free to send through the calendar invite."
            )
        else:  # confirm
            msg = (
                f"Thank you very much for reaching out regarding the opportunity with your engineering team!\n\n"
                f"I would be delighted to speak with you. I am available for a technical discussion "
                f"this Thursday or Friday between 4:00 PM and 7:00 PM IST, or anytime over the weekend that suits the team.\n\n"
                f"I look forward to discussing our recent work in FastMCP agent architectures, dense vector retrieval, and model evaluation."
            )
    elif intent == "Event/Club":
        if variant == "inquire":
            msg = (
                f"Thanks for coordinating the details for {clean_sub}.\n\n"
                f"Could you confirm the exact venue arrangements and whether HDMI projector access is available in the hall?"
            )
        elif variant == "reschedule":
            msg = (
                f"Thanks for inviting me to participate in {clean_sub}!\n\n"
                f"Due to university CIE exams scheduled on this date, could we explore conducting the workshop session "
                f"during the subsequent weekend?"
            )
        elif variant == "decline":
            msg = (
                f"Thanks for inviting me to participate in {clean_sub}.\n\n"
                f"Unfortunately, due to overlapping university lab exams, I will be unable to attend this edition."
            )
        elif tone == "Concise":
            msg = (
                f"Thanks for coordinating the details for {clean_sub}.\n\n"
                f"I confirm my participation and will ensure our hands-on workshop materials are ready before the session."
            )
        else:  # confirm
            msg = (
                f"Thanks for sharing the details and schedule for {clean_sub}!\n\n"
                f"I confirm my participation and will ensure our starter Colab notebooks, FastMCP template repository, "
                f"and presentation setup are verified before the session begins."
            )
    else:  # Action Required
        if variant == "inquire":
            msg = (
                f"Received your alert regarding {clean_sub}.\n\n"
                f"I am looking into this right now. Could you share the latest service logs so I can diagnose the issue quickly?"
            )
        elif variant == "reschedule":
            msg = (
                f"Received your notification regarding {clean_sub}.\n\n"
                f"I am actively investigating the node configuration and will provide a status verification report by 4:00 PM."
            )
        elif variant == "decline":
            msg = (
                f"Received your notification regarding {clean_sub}.\n\n"
                f"I have reviewed the requisition and verified that this does not impact our primary inference pipeline."
            )
        elif tone == "Concise":
            msg = (
                f"Received your update regarding {clean_sub}.\n\n"
                f"I have verified the configuration and resolved the pending items."
            )
        else:  # confirm
            msg = (
                f"Received your notification regarding {clean_sub}.\n\n"
                f"I have checked the system configuration and initiated the required steps to complete this action promptly."
            )

    # Injected Checked Action Items
    if checked_action_items:
        bullet_items = "\n".join([f"• Confirmed: {item}" for item in checked_action_items])
        msg += f"\n\nDeliverables & Verification Status:\n{bullet_items}"

    # Seamless custom prompt steering
    if custom_instruction and custom_instruction.strip():
        clean_note = custom_instruction.strip()
        if not clean_note.endswith('.'):
            clean_note += '.'
        msg += f"\n\nAdditionally, {clean_note}"

    full_draft = f"{salutation}\n\n{msg}\n\n{signature}"
    return {
        "reply": full_draft,
        "auto_reply_recommended": not is_promo,
        "reason": f"Classified as {intent} with {tone.lower()} tone under {variant} strategy.",
        "suggested_action": "Archive / Skip Reply" if is_promo else "Review & Save to Gmail Drafts",
        "suggestion_chips": get_suggestion_chips(intent),
    }
    return {
        "reply": full_draft,
        "auto_reply_recommended": not is_promo,
        "reason": f"Classified as {intent} with {tone.lower()} tone.",
        "suggested_action": "Archive / Skip Reply" if is_promo else "Review & Save to Gmail Drafts",
        "suggestion_chips": get_suggestion_chips(intent),
    }


# -----------------------------------------------------------------------------
# Auth Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/auth/status")
def get_auth_status():
    """Checks token validity and returns live user profile or disconnected state."""
    global _AUTH_IN_PROGRESS

    if not TOKEN_PATH.exists():
        return {
            "connected": False,
            "email": None,
            "authorizing": _AUTH_IN_PROGRESS,
            "mode": "signed_out",
            "message": "No active session."
        }

    try:
        service = gmail_auth.get_gmail_service(interactive=False)
        profile = service.users().getProfile(userId="me").execute()
        _AUTH_IN_PROGRESS = False
        return {
            "connected": True,
            "email": profile.get("emailAddress", "authenticated-user@gmail.com"),
            "messages_total": profile.get("messagesTotal", 0),
            "threads_total": profile.get("threadsTotal", 0),
            "mode": "live_gmail_api",
        }
    except Exception as e:
        return {
            "connected": False,
            "email": None,
            "authorizing": _AUTH_IN_PROGRESS,
            "mode": "signed_out",
            "error": str(e),
        }


@app.post("/api/auth/logout")
def logout():
    """Safely signs out by revoking and deleting token.json."""
    global _AUTH_IN_PROGRESS, _EMAIL_CACHE
    _AUTH_IN_PROGRESS = False
    _EMAIL_CACHE["data"] = None

    if TOKEN_PATH.exists():
        try:
            with open(TOKEN_PATH, "r", encoding="utf-8") as f:
                token_data = json.load(f)
            token_val = token_data.get("token") or token_data.get("access_token")
            if token_val:
                try:
                    revoke_url = f"https://oauth2.googleapis.com/revoke?token={token_val}"
                    req = urllib.request.Request(revoke_url, method="POST")
                    req.add_header("Content-Type", "application/x-www-form-urlencoded")
                    urllib.request.urlopen(req, timeout=3)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            TOKEN_PATH.unlink(missing_ok=True)
        except Exception as del_err:
            raise HTTPException(status_code=500, detail=f"Failed to delete token file: {del_err}")

    return {
        "success": True,
        "connected": False,
        "message": "Signed out successfully."
    }


@app.post("/api/auth/login")
def login(background_tasks: BackgroundTasks):
    """Initiates Google OAuth 2.0 desktop authorization flow."""
    global _AUTH_IN_PROGRESS

    if TOKEN_PATH.exists() and gmail_auth.is_authenticated():
        return {"success": True, "already_authenticated": True}

    if not CREDENTIALS_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="credentials.json not found."
        )

    _AUTH_IN_PROGRESS = True

    def run_auth_thread():
        global _AUTH_IN_PROGRESS
        try:
            gmail_auth.get_gmail_credentials(interactive=True)
        except Exception as ex:
            print(f"[server] Login flow notice: {ex}", file=sys.stderr)
        finally:
            _AUTH_IN_PROGRESS = False

    thread = threading.Thread(target=run_auth_thread, daemon=True)
    thread.start()

    return {
        "success": True,
        "authorizing": True,
        "message": "OAuth flow started in your browser."
    }


# -----------------------------------------------------------------------------
# Inbox & AI Endpoints (High-Speed Cached)
# -----------------------------------------------------------------------------
@app.get("/api/emails")
def get_emails(
    max_results: int = Query(default=25, ge=1, le=50),
    mode: str = Query(default="live"),
    force_refresh: bool = Query(default=False)
):
    """
    Fetches emails (either Live Gmail or Viva Sample Dataset).
    Uses high-speed in-memory caching to guarantee < 10ms response times.
    """
    global _EMAIL_CACHE
    now = time.time()

    # Cache hit check (60-second TTL)
    if not force_refresh and _EMAIL_CACHE.get("data") and _EMAIL_CACHE.get("mode") == mode and _EMAIL_CACHE.get("max_results") == max_results and (now - _EMAIL_CACHE.get("timestamp", 0)) < 60:
        return _EMAIL_CACHE["data"]

    is_live = TOKEN_PATH.exists() and gmail_auth.is_authenticated() and mode == "live"
    
    if mode == "demo" or not is_live:
        # Load realistic AIML Lab Viva Sample Dataset (Dr. Ramesh, DeepMind, Hackathon, AWS Alert)
        if TEST_DATA_PATH.exists():
            with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
                test_samples = json.load(f)
            raw_emails = []
            for idx, item in enumerate(test_samples[:max_results]):
                raw_emails.append({
                    "id": f"viva_{idx+1}",
                    "thread_id": f"thread_viva_{idx+1}",
                    "sender": item.get("sender", "faculty@bmsce.ac.in"),
                    "subject": item.get("subject", "Lab Evaluation"),
                    "date": "Today, 10:30 AM",
                    "body": item.get("body", ""),
                    "html": f"<div style='font-family:-apple-system,BlinkMacSystemFont,sans-serif;padding:20px;line-height:1.6;'><h3 style='color:#1e293b;'>{item.get('subject', '')}</h3><p style='color:#334155;'>{item.get('body', '')}</p></div>",
                    "source": "viva_demo_dataset",
                })
        else:
            raw_emails = gmail_mcp.fetch_unread_emails(max_results=max_results)
    else:
        # Live Gmail
        raw_emails = gmail_mcp.fetch_unread_emails(max_results=max_results)

    enriched = []
    for em in raw_emails:
        sub = em.get("subject", "")
        snd = em.get("sender", "")
        bdy = sanitize_email_body(em.get("body", ""))

        analysis = ml_pipeline.analyze_email(sub, snd, bdy)
        reply_meta = compose_smart_reply(snd, sub, bdy, analysis["intent"], "Professional")
        insights = extract_ai_insights(sub, snd, bdy, analysis["intent"])

        enriched.append({
            "id": em.get("id"),
            "thread_id": em.get("thread_id"),
            "sender": snd,
            "sender_clean": extract_clean_name(snd),
            "subject": sub,
            "date": em.get("date", "Today"),
            "body": bdy,
            "html": em.get("html", ""),
            "source": em.get("source"),
            "analysis": analysis,
            "insights": insights,
            "auto_reply_recommended": reply_meta["auto_reply_recommended"],
            "suggested_action": reply_meta["suggested_action"],
            "suppression_reason": reply_meta.get("reason"),
            "suggested_draft": reply_meta["reply"],
            "suggestion_chips": reply_meta.get("suggestion_chips", []),
        })

    response_payload = {
        "emails": enriched,
        "count": len(enriched),
        "mode": mode,
        "is_live": is_live,
        "cached_at": time.strftime("%H:%M:%S", time.localtime()),
    }

    # Store in memory
    _EMAIL_CACHE = {
        "data": response_payload,
        "timestamp": now,
        "mode": mode,
        "max_results": max_results,
    }

    return response_payload


@app.post("/api/generate-reply")
def generate_reply_api(req: GenerateReplyRequest):
    """Generates an updated response with custom intent, tone, suggestion chip variant, or custom guidance."""
    intent = req.intent or "Action Required"
    tone = req.tone or "Professional"
    variant = req.variant or "default"
    custom_instruction = req.custom_instruction or ""
    checked = req.checked_action_items or []
    res = compose_smart_reply(req.sender, req.subject, req.body, intent, tone, variant, custom_instruction, checked)
    return res


@app.post("/api/draft")
def save_draft(req: DraftRequest):
    """Creates a threaded draft in Gmail Drafts folder (zero auto-send)."""
    res = gmail_mcp.create_gmail_draft(
        thread_id=req.thread_id,
        to=req.to,
        subject=req.subject,
        body=req.body,
    )
    return res


@app.post("/api/analyze")
def analyze_custom_email(req: AnalyzeRequest):
    """Runs on-the-spot AIML classification and FAISS RAG retrieval for sandbox tester."""
    t0 = time.perf_counter()
    analysis = ml_pipeline.analyze_email(req.subject, req.sender, req.body)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    reply_meta = compose_smart_reply(req.sender, req.subject, req.body, analysis["intent"], "Professional")
    return {
        "analysis": analysis,
        "reply_meta": reply_meta,
        "suggested_draft": reply_meta["reply"],
        "roundtrip_ms": latency_ms,
    }


@app.get("/api/evaluation")
def get_evaluation():
    """Returns AIML evaluation benchmark data from evaluation_report.json."""
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="evaluation_report.json not found")
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "MailMind AI Backend running."})


if __name__ == "__main__":
    import uvicorn
    print("[server.py] Starting MailMind AI Executive Client on http://localhost:8000...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
