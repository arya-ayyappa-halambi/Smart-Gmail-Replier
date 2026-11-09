"""
gmail_mcp.py - Custom FastMCP Server for Smart Gmail Replier.

Exposes three Model Context Protocol (MCP) tools:
  1. fetch_unread_emails(max_results: int = 5)
  2. analyze_email_ml(subject: str, sender: str, body: str)
  3. create_gmail_draft(thread_id: str, to: str, subject: str, body: str)
"""

import base64
import email
import json
import os
import sys
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

# Import local authentication and ML modules
import gmail_auth
import ml_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DRAFTS_LOG_PATH = DATA_DIR / "drafts_log.json"
TEST_EMAILS_PATH = DATA_DIR / "test_emails.json"

# Initialize FastMCP Server
mcp = FastMCP("Smart Gmail Replier", dependencies=["beautifulsoup4", "google-api-python-client"])


def _extract_body_and_html(payload: Dict[str, Any]) -> tuple[str, str]:
    """
    Recursively parses Gmail message payload to extract both:
      1. Clean plain text (stripped of HTML tags using BeautifulSoup for ML/prose)
      2. Rich HTML content (for sandboxed Apple Mail iframe preview)
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    plain_text_parts: List[str] = []
    html_parts: List[str] = []

    if body_data:
        try:
            decoded = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
            if "text/html" in mime_type:
                html_parts.append(decoded)
                soup = BeautifulSoup(decoded, "html.parser")
                plain_text_parts.append(soup.get_text(separator=" ").strip())
            else:
                plain_text_parts.append(decoded.strip())
        except Exception:
            pass

    parts = payload.get("parts", [])
    for part in parts:
        part_mime = part.get("mimeType", "")
        part_data = part.get("body", {}).get("data")
        if not part_data and "parts" in part:
            sub_plain, sub_html = _extract_body_and_html(part)
            if sub_plain:
                plain_text_parts.append(sub_plain)
            if sub_html:
                html_parts.append(sub_html)
            continue

        if part_data:
            try:
                decoded = base64.urlsafe_b64decode(part_data).decode("utf-8", errors="replace")
                if part_mime == "text/plain":
                    plain_text_parts.append(decoded.strip())
                elif part_mime == "text/html":
                    html_parts.append(decoded)
                    soup = BeautifulSoup(decoded, "html.parser")
                    plain_text_parts.append(soup.get_text(separator=" ").strip())
            except Exception:
                pass

    final_plain = "\n".join(plain_text_parts).strip() if plain_text_parts else payload.get("snippet", "")
    final_html = "\n".join(html_parts).strip() if html_parts else ""

    return final_plain, final_html


@mcp.tool()
def fetch_unread_emails(max_results: int = 25) -> List[Dict[str, Any]]:
    """
    Fetches unread emails from the user's primary Gmail inbox using high-speed batch requests.
    Returns clean sender, subject, date, threadId, clean prose body, and raw HTML payload.
    If no unread emails are found, gracefully retrieves latest inbox emails to ensure the inbox is never empty.
    If token.json is not present, returns realistic simulated lab emails for viva demonstration.
    """
    results: List[Dict[str, Any]] = []

    # Check if live Gmail service is available
    if gmail_auth.is_authenticated():
        try:
            service = gmail_auth.get_gmail_service(interactive=False)
            
            # First attempt: unread primary inbox messages
            response = (
                service.users()
                .messages()
                .list(userId="me", q="is:unread label:INBOX -label:SPAM", maxResults=max_results)
                .execute()
            )
            messages = response.get("messages", [])

            # Graceful fallback: if no unread messages, fetch recent primary inbox messages
            if not messages:
                response = (
                    service.users()
                    .messages()
                    .list(userId="me", q="label:INBOX -label:SPAM", maxResults=max_results)
                    .execute()
                )
                messages = response.get("messages", [])

            if messages:
                # High-speed batch retrieval: retrieves all messages in 1 HTTP roundtrip instead of N sequential calls
                batch = service.new_batch_http_request()
                batch_dict = {}

                def batch_cb(req_id, resp, exc):
                    if not exc and resp:
                        batch_dict[req_id] = resp

                for m in messages:
                    mid = m["id"]
                    batch.add(service.users().messages().get(userId="me", id=mid, format="full"), request_id=mid, callback=batch_cb)

                batch.execute()

                # Process in original chronological order
                for msg_meta in messages:
                    msg_id = msg_meta["id"]
                    full_msg = batch_dict.get(msg_id)
                    if not full_msg:
                        continue

                    headers = full_msg.get("payload", {}).get("headers", [])
                    header_dict = {h["name"].lower(): h["value"] for h in headers}

                    subject = header_dict.get("subject", "(No Subject)")
                    sender = header_dict.get("from", "Unknown Sender")
                    date_str = header_dict.get("date", "")
                    thread_id = full_msg.get("threadId", msg_id)
                    body, html_content = _extract_body_and_html(full_msg.get("payload", {}))

                    results.append({
                        "id": msg_id,
                        "thread_id": thread_id,
                        "sender": sender,
                        "subject": subject,
                        "date": date_str,
                        "body": body,
                        "html": html_content,
                        "source": "live_gmail_api",
                    })

                return results
        except Exception as e:
            print(f"[gmail_mcp] Notice: Could not read live Gmail ({e}). Falling back to simulated inbox.", file=sys.stderr)

    # Simulated inbox fallback for demonstration & viva testing when token.json is not present
    if TEST_EMAILS_PATH.exists():
        with open(TEST_EMAILS_PATH, "r", encoding="utf-8") as f:
            test_samples = json.load(f)
        for idx, item in enumerate(test_samples[:max_results]):
            results.append({
                "id": f"sim_msg_{idx+1}",
                "thread_id": f"sim_thread_{idx+1}",
                "sender": item.get("sender", "faculty@bmsce.ac.in"),
                "subject": item.get("subject", "AIML Skill Lab Project Schedule"),
                "date": "Today, 10:30 AM",
                "body": item.get("body", ""),
                "html": f"<div style='font-family:sans-serif;padding:16px;'><h3>{item.get('subject', '')}</h3><p>{item.get('body', '')}</p></div>",
                "source": "simulated_inbox (token.json not present)",
            })
    else:
        results.append({
            "id": "demo_01",
            "thread_id": "thread_demo_01",
            "sender": "dr.ramesh@bmsce.ac.in",
            "subject": "AIML Skill Lab Project Demonstration Schedule",
            "date": "Today, 09:15 AM",
            "body": "Dear Arya, Please be ready with your FastMCP Smart Replier demo on Monday at 10 AM.",
            "html": "<div style='font-family:sans-serif;padding:16px;'><h3>AIML Skill Lab Project Demonstration Schedule</h3><p>Dear Arya, Please be ready with your FastMCP Smart Replier demo on Monday at 10 AM.</p></div>",
            "source": "demo_fallback",
        })

    return results


@mcp.tool()
def analyze_email_ml(subject: str, sender: str, body: str) -> Dict[str, Any]:
    """
    Executes the AIML engine to classify email Intent and Urgency with confidence scores,
    evaluates tone/sentiment, checks action requirement, and retrieves the top-2 most relevant
    few-shot style and personal context matches using Dense Sentence Transformers & FAISS.
    """
    return ml_pipeline.analyze_email(subject=subject, sender=sender, body=body)


@mcp.tool()
def create_gmail_draft(thread_id: str, to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Creates a threaded draft reply in Gmail Drafts.
    Safety Guarantee: NEVER calls users().messages().send() - drafts are only saved for human review.
    Sets 'In-Reply-To' and 'References' headers to maintain proper email threading.
    """
    # Normalize subject with 'Re: ' if not present
    draft_subject = subject.strip()
    if not draft_subject.lower().startswith("re:"):
        draft_subject = f"Re: {draft_subject}"

    # Construct standard MIME Email message
    message = MIMEText(body, "plain", "utf-8")
    message["to"] = to
    message["subject"] = draft_subject
    if thread_id:
        message["In-Reply-To"] = thread_id
        message["References"] = thread_id

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    # If live Gmail is authenticated, save via Google API
    if gmail_auth.is_authenticated():
        try:
            service = gmail_auth.get_gmail_service(interactive=False)
            draft_body = {
                "message": {
                    "raw": raw_message,
                    "threadId": thread_id if thread_id and not thread_id.startswith("sim_") else None,
                }
            }
            # Remove threadId key if None
            if not draft_body["message"]["threadId"]:
                del draft_body["message"]["threadId"]

            draft_response = service.users().drafts().create(userId="me", body=draft_body).execute()
            return {
                "status": "success",
                "mode": "live_gmail_api",
                "draft_id": draft_response.get("id"),
                "thread_id": thread_id,
                "to": to,
                "subject": draft_subject,
                "preview": body[:120] + "..." if len(body) > 120 else body,
                "message": "Draft created in Gmail Drafts folder. Ready for your manual review and sending.",
            }
        except Exception as e:
            print(f"[gmail_mcp] Failed to save live Gmail draft ({e}). Logging locally.", file=sys.stderr)

    # Local fallback / offline logging
    draft_record = {
        "status": "success",
        "mode": "simulated_local_draft",
        "draft_id": f"draft_{int(os.times().elapsed * 1000)}",
        "thread_id": thread_id,
        "to": to,
        "subject": draft_subject,
        "body": body,
        "note": "Live Gmail token not present; saved to data/drafts_log.json for safety and verification.",
    }

    try:
        existing_drafts: List[Dict[str, Any]] = []
        if DRAFTS_LOG_PATH.exists():
            with open(DRAFTS_LOG_PATH, "r", encoding="utf-8") as lf:
                existing_drafts = json.load(lf)
        existing_drafts.append(draft_record)
        with open(DRAFTS_LOG_PATH, "w", encoding="utf-8") as lf:
            json.dump(existing_drafts, lf, indent=2)
    except Exception as log_err:
        print(f"[gmail_mcp] Notice: Could not append to drafts log: {log_err}", file=sys.stderr)

    return {
        "status": "success",
        "mode": "simulated_local_draft",
        "draft_id": draft_record["draft_id"],
        "thread_id": thread_id,
        "to": to,
        "subject": draft_subject,
        "preview": body[:120] + "..." if len(body) > 120 else body,
        "message": "Draft safely registered in local drafts log (data/drafts_log.json). Ready for review.",
    }


if __name__ == "__main__":
    print("[gmail_mcp] Starting FastMCP Smart Gmail Replier server via STDIO transport...", file=sys.stderr)
    mcp.run(transport="stdio")
