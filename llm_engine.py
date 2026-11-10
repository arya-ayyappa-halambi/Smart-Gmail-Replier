"""
llm_engine.py — High-Performance Foundation Model Engine for MailMind AI.

Supports local foundation model reasoning via Ollama (qwen2.5-coder:7b)
and optional Google Gemini API.

CRITICAL SAFETY PROTOCOL:
  The model 'vulnpilot-7b' belongs strictly to another project on this machine.
  It is STRICTLY PROTECTED and will NEVER be invoked, loaded, queried, or modified.
  All requests are hardcoded to target 'qwen2.5-coder:7b'.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mailmind.llm")

# -----------------------------------------------------------------------------
# Ironclad Safety Enforcers
# -----------------------------------------------------------------------------
BLOCKED_MODEL_KEYWORDS = ["vulnpilot", "vulnproject"]
DEFAULT_LOCAL_MODEL = "qwen2.5-coder:7b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# In-memory LRU cache to ensure sub-1ms UI responsiveness on cached reads
_INSIGHTS_CACHE: Dict[str, Dict[str, Any]] = {}
_REPLY_CACHE: Dict[str, str] = {}


def _assert_safe_model(model_name: str) -> None:
    """Raises an AssertionError if any blocked model is accidentally referenced."""
    for keyword in BLOCKED_MODEL_KEYWORDS:
        if keyword in model_name.lower():
            raise AssertionError(
                f"[SAFETY GUARD TRIGGERED] Model '{model_name}' matches blocked pattern '{keyword}'. "
                f"MailMind AI strictly forbids accessing other project models."
            )


def is_ollama_available() -> bool:
    """Checks if Ollama is running and has qwen2.5-coder:7b available."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return any("qwen2.5-coder" in m for m in models)
    except Exception:
        return False


def get_active_model_info() -> Dict[str, Any]:
    """Returns metadata for the UI status badge."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return {
            "connected": True,
            "provider": "gemini",
            "model_name": "Google Gemini 1.5 Flash",
            "badge": "⚡ Google Gemini Cloud",
            "is_local": False
        }

    ollama_ok = is_ollama_available()
    if ollama_ok:
        return {
            "connected": True,
            "provider": "ollama",
            "model_name": DEFAULT_LOCAL_MODEL,
            "badge": "⚡ Qwen 2.5 Coder (Local Ollama)",
            "is_local": True
        }

    return {
        "connected": True,
        "provider": "heuristic_fallback",
        "model_name": "Rule Heuristics",
        "badge": "⚡ Rule-Based Engine (Ollama Standby)",
        "is_local": True
    }


# -----------------------------------------------------------------------------
# FINE-GRAINED SUB-TOPIC & TONE DETECTOR
# -----------------------------------------------------------------------------
SUB_TOPIC_CONFIGS = {
    "academic_fee": {
        "label": "Fee Notice & Dues",
        "chips": [
            {"id": "confirm", "label": "Confirm Payment Completed", "desc": "Provide UTR / receipt for CIE clearance"},
            {"id": "inquire", "label": "Clarify Challan Submission", "desc": "Ask if physical submission is required at accounts"},
            {"id": "reschedule", "label": "Request 3-Day Extension", "desc": "Politely request grace period for bank transfer"},
            {"id": "decline", "label": "Prior Payment Proof", "desc": "Attach existing receipt showing dues cleared"}
        ]
    },
    "academic_review": {
        "label": "Skill Lab & Project Review",
        "chips": [
            {"id": "confirm", "label": "Confirm Deliverables Ready", "desc": "Acknowledge slot; architecture & Git logs ready"},
            {"id": "inquire", "label": "Clarify Rubrics & Weights", "desc": "Ask if prototype or doc has higher weight"},
            {"id": "reschedule", "label": "Request Afternoon Session", "desc": "Request alternate slot due to morning CIE clash"},
            {"id": "decline", "label": "Academic Deferral Note", "desc": "Polite request to schedule in makeup lab"}
        ]
    },
    "academic_exam": {
        "label": "CIE Exam & Timetable",
        "chips": [
            {"id": "confirm", "label": "Acknowledge Exam Schedule", "desc": "Confirm timetable noted and revision underway"},
            {"id": "inquire", "label": "Clarify Syllabus & Units", "desc": "Inquire whether Unit 5 or lab concepts are included"},
            {"id": "reschedule", "label": "Report Exam Clash", "desc": "Flag overlapping elective or lab evaluation clash"},
            {"id": "decline", "label": "Medical / Absence Notice", "desc": "Submit formal intimation regarding unavoidable absence"}
        ]
    },
    "academic_classroom": {
        "label": "Classroom Notes & Assignment",
        "chips": [
            {"id": "confirm", "label": "Acknowledge & Received", "desc": "Confirm notes and practice exercises received"},
            {"id": "inquire", "label": "Ask Concept / Syntax Query", "desc": "Ask clarifying question on problem statement"},
            {"id": "reschedule", "label": "Request Submission Grace", "desc": "Ask for 24-hr extension to refine test suites"},
            {"id": "decline", "label": "Doubts in Office Hours", "desc": "Request brief discussion during office hours"}
        ]
    },
    "academic_admin": {
        "label": "Academic Circular & Notice",
        "chips": [
            {"id": "confirm", "label": "Acknowledge Notification", "desc": "Confirm circular requirements noted"},
            {"id": "inquire", "label": "Request Policy Clarification", "desc": "Inquire regarding eligibility criteria and forms"},
            {"id": "reschedule", "label": "Request Appointment Slot", "desc": "Request meeting with department coordinator"},
            {"id": "decline", "label": "Formal Representation", "desc": "Submit formal request regarding attendance or exemptions"}
        ]
    },
    "dev_pull_request": {
        "label": "GitHub Pull Request & Review",
        "chips": [
            {"id": "confirm", "label": "Changes Committed & Pushed", "desc": "Confirm review feedback addressed on branch"},
            {"id": "inquire", "label": "Request Specific Examples", "desc": "Ask for clarification on requested architecture changes"},
            {"id": "reschedule", "label": "Track in Separate Issue", "desc": "Propose merging current PR and tracking remainder"},
            {"id": "decline", "label": "Clarify Design Rationale", "desc": "Explain why current architectural pattern was chosen"}
        ]
    },
    "dev_alert": {
        "label": "Cloud Alert & Incident",
        "chips": [
            {"id": "confirm", "label": "Issue Diagnosed & Mitigated", "desc": "Confirm alert acknowledged and root cause identified"},
            {"id": "inquire", "label": "Request Stack & Telemetry", "desc": "Ask for distributed traces and Prometheus metrics"},
            {"id": "reschedule", "label": "Active Triage / Monitoring", "desc": "Currently investigating logs; update in 30 mins"},
            {"id": "decline", "label": "Flag False Positive / Staging", "desc": "Verified non-impacting dev/staging telemetry"}
        ]
    },
    "dev_collaboration": {
        "label": "Technical RFC & Architecture",
        "chips": [
            {"id": "confirm", "label": "Proceed with Implementation", "desc": "Agree with proposed technical RFC and start coding"},
            {"id": "inquire", "label": "Ask Architectural Trade-offs", "desc": "Compare latency vs memory footprint considerations"},
            {"id": "reschedule", "label": "Suggest 15-Min Quick Sync", "desc": "Propose quick huddle on Google Meet to align"},
            {"id": "decline", "label": "Propose Alternative Design", "desc": "Recommend lightweight approach with fewer dependencies"}
        ]
    },
    "career_interview": {
        "label": "Interview Invitation",
        "chips": [
            {"id": "confirm", "label": "Accept Proposed Time Slot", "desc": "Confirm interview attendance and verify meeting link"},
            {"id": "inquire", "label": "Inquire Tech Format & Prep", "desc": "Ask about coding environment, DSA, or system design"},
            {"id": "reschedule", "label": "Propose Alternate Window", "desc": "Suggest alternate afternoon slot due to college CIEs"},
            {"id": "decline", "label": "Respectfully Decline Slot", "desc": "Decline current round; request considering future roles"}
        ]
    },
    "career_offer": {
        "label": "Offer Letter & Onboarding",
        "chips": [
            {"id": "confirm", "label": "Enthusiastically Accept Offer", "desc": "Confirm acceptance of terms and submit verification"},
            {"id": "inquire", "label": "Inquire Onboarding Details", "desc": "Ask about start date flexibility, laptop setup, paperwork"},
            {"id": "reschedule", "label": "Request Consideration Window", "desc": "Request 3 days to review offer terms with mentors"},
            {"id": "decline", "label": "Respectfully Decline Offer", "desc": "Polite decline citing existing academic commitments"}
        ]
    },
    "career_inquiry": {
        "label": "Career & Internship Opportunity",
        "chips": [
            {"id": "confirm", "label": "Express Strong Interest", "desc": "Share resume highlights and portfolio links for role"},
            {"id": "inquire", "label": "Ask Role Scope & Projects", "desc": "Inquire about team missions, core stack, responsibilities"},
            {"id": "reschedule", "label": "Connect Later this Semester", "desc": "Express interest for upcoming summer internship cycle"},
            {"id": "decline", "label": "Not Looking Currently", "desc": "Polite acknowledgment, connect on LinkedIn for future"}
        ]
    },
    "admin_invoice": {
        "label": "Invoice & Transaction Receipt",
        "chips": [
            {"id": "confirm", "label": "Confirm Payment & Record", "desc": "Acknowledge receipt and archive transaction record"},
            {"id": "inquire", "label": "Request Itemized Breakdown", "desc": "Ask for tax invoice or itemized charge verification"},
            {"id": "reschedule", "label": "Dispute Duplicate Charge", "desc": "Flag unexpected charge or duplicate debit for refund"},
            {"id": "decline", "label": "Cancel Subscription / Plan", "desc": "Request immediate cancellation and billing halt"}
        ]
    },
    "admin_security": {
        "label": "Security Alert & 2FA",
        "chips": [
            {"id": "confirm", "label": "Acknowledge Authorized Login", "desc": "Confirm sign-in was recognized and legitimate"},
            {"id": "inquire", "label": "Request IP & Device Log", "desc": "Ask for session geolocation and user agent details"},
            {"id": "reschedule", "label": "Revoke Session & Change Pass", "desc": "Report unrecognized activity and initiate password reset"},
            {"id": "decline", "label": "Ignore / Archive Notification", "desc": "Standard automated verification notice"}
        ]
    },
    "newsletter": {
        "label": "Newsletter & Digest",
        "chips": [
            {"id": "confirm", "label": "Acknowledge & Archive", "desc": "Polite confirmation of receipt"},
            {"id": "inquire", "label": "Update Topic Preferences", "desc": "Request preference center to tune frequency"},
            {"id": "reschedule", "label": "Save to Reading List", "desc": "Archive to technical reference notebook"},
            {"id": "decline", "label": "Unsubscribe & Remove", "desc": "Request removal from marketing distribution list"}
        ]
    },
    "general": {
        "label": "General Correspondence",
        "chips": [
            {"id": "confirm", "label": "Confirm & Acknowledge", "desc": "Acknowledge message and outline next steps"},
            {"id": "inquire", "label": "Request Clarification", "desc": "Ask clarifying questions on specific points"},
            {"id": "reschedule", "label": "Propose Follow-up Time", "desc": "Suggest connecting later in the week"},
            {"id": "decline", "label": "Polite Deferral Note", "desc": "Respectful decline citing current project workload"}
        ]
    }
}


def _has_keywords(text: str, patterns: List[str]) -> bool:
    """Checks whether any keyword pattern appears as a distinct word in text."""
    for pat in patterns:
        if re.search(r'\b' + re.escape(pat) + r'\b', text, re.IGNORECASE):
            return True
    return False


def detect_sub_topic_and_tone(subject: str, body: str, sender: str, broad_intent: str) -> Dict[str, str]:
    """
    Performs multi-signal topic and tone classification across 15+ sub-topics
    with strict word-boundary matching and sender domain authority.
    """
    text = f"{subject} {body}".lower()
    sub_lower = subject.lower()
    snd = sender.lower()

    is_academic_sender = any(k in snd for k in ["bmsce.ac.in", "edu", "college", "university", "principal", "accounts", "faculty"])

    # 1. High-Priority Direct Topic Identification
    if _has_keywords(sub_lower, ["fee", "fees", "dues", "challan", "tuition", "payment due"]) or (is_academic_sender and _has_keywords(text, ["fee", "fees", "dues", "challan", "tuition", "payment due"])):
        sub_topic = "academic_fee"
    elif _has_keywords(sub_lower, ["skill lab", "skill labs", "review 1", "review 2", "review 3", "viva", "rubric"]) or (is_academic_sender and _has_keywords(text, ["skill lab", "review 1", "viva", "lab 3"])):
        sub_topic = "academic_review"
    elif _has_keywords(sub_lower, ["cie", "exam timetable", "hall ticket", "seating", "semester end", "internal test"]) or (is_academic_sender and _has_keywords(text, ["cie", "exam timetable", "hall ticket"])):
        sub_topic = "academic_exam"
    elif _has_keywords(sub_lower, ["classroom", "typedef", "lecture notes", "assignment", "homework"]) or "classroom.google.com" in snd:
        sub_topic = "academic_classroom"
    elif is_academic_sender and _has_keywords(text, ["circular", "attendance", "shortage", "registration", "guidelines"]):
        sub_topic = "academic_admin"
    elif _has_keywords(sub_lower, ["pull request", "pr #"]) or any(k in snd for k in ["github.com", "gitlab", "sourcery", "dependabot"]):
        sub_topic = "dev_pull_request"
    elif _has_keywords(sub_lower, ["alert", "incident", "alarm", "crash", "outage"]) or any(k in snd for k in ["sentry", "pagerduty", "aws", "datadog", "grafana"]):
        sub_topic = "dev_alert"
    elif _has_keywords(sub_lower, ["api", "architecture guide", "sqlite wal", "cdc stream", "docker", "caddy", "nginx", "rfc"]):
        sub_topic = "dev_collaboration"
    elif _has_keywords(sub_lower, ["interview", "screening", "technical round"]):
        sub_topic = "career_interview"
    elif _has_keywords(sub_lower, ["offer letter", "congratulations on your offer", "internship offer", "compensation"]):
        sub_topic = "career_offer"
    elif _has_keywords(sub_lower, ["unstop", "hackathon", "internship", "hiring", "opportunity", "voyagehack"]) or "unstop" in snd:
        sub_topic = "career_inquiry"
    elif _has_keywords(sub_lower, ["invoice", "receipt", "order #", "billing"]) or any(k in snd for k in ["uber", "swiggy", "zomato", "amazon"]):
        sub_topic = "admin_invoice"
    elif _has_keywords(sub_lower, ["verification code", "security alert", "2fa", "otp", "password reset", "unauthorized"]):
        sub_topic = "admin_security"
    elif any(k in snd for k in ["newsletter", "linkedin", "substack", "medium", "reddit", "insideapple.apple.com", "codepen", "leetcode", "store"]):
        sub_topic = "newsletter"
    elif broad_intent == "Academic/Faculty":
        sub_topic = "academic_admin"
    elif broad_intent == "Internship/Recruiter":
        sub_topic = "career_inquiry"
    elif broad_intent == "Action Required":
        sub_topic = "dev_collaboration"
    else:
        sub_topic = "general"

    # 2. Tone Detection
    if _has_keywords(text, ["final reminder", "urgent", "defaulters will not", "immediate action", "strictly prohibited", "mandatory"]):
        tone = "Urgent / Strict"
    elif is_academic_sender or _has_keywords(text, ["principal", "hereby directed", "as per directions", "official notification"]):
        tone = "Formal / Administrative"
    elif any(k in snd for k in ["github", "gitlab"]) or _has_keywords(text, ["pull request", "wal storage", "stack trace", "traceback", "commit"]):
        tone = "Technical / Instructional"
    elif "?" in body or _has_keywords(text, ["availability", "preferred time", "could you", "please confirm", "let me know"]):
        tone = "Inquiring / Scheduling"
    elif _has_keywords(text, ["discount", "limited time", "save $", "exclusive offer", "join now", "impressions last week"]):
        tone = "Promotional / Marketing"
    else:
        tone = "Professional / Neutral"

    config = SUB_TOPIC_CONFIGS.get(sub_topic, SUB_TOPIC_CONFIGS["general"])

    return {
        "sub_topic": sub_topic,
        "sub_topic_label": config["label"],
        "incoming_tone": tone
    }


def generate_dynamic_suggestion_chips(
    sub_topic: str,
    subject: str = "",
    intent: str = ""
) -> List[Dict[str, str]]:
    """Returns 4 multi-persona strategic response variants tailored specifically to this sub-topic."""
    config = SUB_TOPIC_CONFIGS.get(sub_topic)
    if not config:
        # Fallback to category default
        if intent == "Academic/Faculty":
            config = SUB_TOPIC_CONFIGS["academic_admin"]
        elif intent == "Internship/Recruiter":
            config = SUB_TOPIC_CONFIGS["career_inquiry"]
        elif intent == "Action Required":
            config = SUB_TOPIC_CONFIGS["dev_collaboration"]
        else:
            config = SUB_TOPIC_CONFIGS["general"]

    return config["chips"]


# -----------------------------------------------------------------------------
# UNIFIED SINGLE-PASS FOUNDATION MODEL INFERENCE
# -----------------------------------------------------------------------------
def unified_email_reasoning(
    sender: str,
    subject: str,
    body: str,
    intent: str,
    tone: str = "Professional",
    variant: str = "confirm",
    custom_instruction: str = "",
    checked_action_items: Optional[List[str]] = None,
    chip_label: Optional[str] = "",
    chip_desc: Optional[str] = ""
) -> Dict[str, Any]:
    """
    Executes a single unified call to qwen2.5-coder:7b to extract deep comprehension
    AND draft an authentic, factual, context-aware reply in one pass (~12-18s total).
    """
    checked = checked_action_items or []
    cache_key = f"unified_v2|{sender}|{subject}|{body[:250]}|{intent}|{tone}|{variant}|{chip_label}|{custom_instruction}|{','.join(checked)}"
    if cache_key in _INSIGHTS_CACHE:
        return _INSIGHTS_CACHE[cache_key]

    # Detect fine-grained sub-topic & tone
    meta = detect_sub_topic_and_tone(subject, body, sender, intent)
    sub_topic = meta["sub_topic"]
    sub_topic_label = meta["sub_topic_label"]
    detected_tone = meta["incoming_tone"]

    # 4 Dynamic suggestion chips tailored specifically to this sub-topic
    chips = generate_dynamic_suggestion_chips(sub_topic, subject, intent)

    # Resolve active strategy from chosen chip
    active_chip = next((c for c in chips if c["id"] == variant), chips[0])
    strategy_label = chip_label or active_chip["label"]
    strategy_desc = chip_desc or active_chip["desc"]

    fallback_insights = _generate_heuristic_insights(subject, sender, body, intent, sub_topic)
    fallback_reply = _generate_heuristic_reply(
        sender=sender,
        subject=subject,
        body=body,
        intent=intent,
        sub_topic=sub_topic,
        tone=tone,
        variant=variant,
        strategy_label=strategy_label,
        strategy_desc=strategy_desc,
        custom_instruction=custom_instruction,
        checked_action_items=checked
    )

    if not is_ollama_available():
        res = {
            "insights": fallback_insights,
            "reply": fallback_reply,
            "suggestion_chips": chips,
            "sub_topic": sub_topic_label,
            "incoming_tone": detected_tone
        }
        _INSIGHTS_CACHE[cache_key] = res
        return res

    _assert_safe_model(DEFAULT_LOCAL_MODEL)

    is_academic = intent == "Academic/Faculty"

    # Strict Attribution Rule
    if is_academic:
        signature_inst = (
            "This is an academic email to university faculty/administration. Sign off formally with:\n"
            "Best regards,\n"
            "Arya Ayyappa Halambi\n"
            "USN: 1BF25CS095\n"
            "Department of Computer Science and Engineering\n"
            "B.M.S. College of Engineering (BMSCE)"
        )
    else:
        signature_inst = (
            "This is an external / developer / recruiter email (NOT academic faculty).\n"
            "DO NOT include any USN or college degree details.\n"
            "Sign off cleanly as:\n"
            "Best regards,\n"
            "Arya Ayyappa Halambi"
        )

    action_note = ""
    if checked:
        items_str = "\n".join([f"• Confirmed: {item}" for item in checked])
        action_note = f"\nExplicitly mention these confirmed checklist points in the reply:\n{items_str}\n"

    custom_note = ""
    if custom_instruction.strip():
        custom_note = f"\nUser steering instruction: Incorporate this specific guidance: '{custom_instruction.strip()}'\n"

    prompt = (
        "You are an AI email assistant drafting an authentic, highly context-aware response for Arya Ayyappa Halambi.\n"
        "Carefully read the incoming email below. Notice specific dates, room/lab numbers, assignment names, code details, or questions.\n\n"
        f"Incoming Email:\n"
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"Category: {intent} (Sub-Topic: {sub_topic_label}, Sender Tone: {detected_tone})\n"
        f"Body:\n{body[:2200]}\n\n"
        f"Chosen Response Strategy: {strategy_label} — {strategy_desc}\n"
        f"Tone: {tone} (Natural, direct, professional, free of generic filler phrases).\n"
        f"{action_note}"
        f"{custom_note}"
        f"\nSignature Requirement:\n{signature_inst}\n\n"
        "Draft an email-specific reply that follows the Chosen Response Strategy and directly references the concrete facts, dates, deliverables, or questions in this email.\n"
        "Return a valid JSON object with exactly these 4 keys:\n"
        "{\n"
        '  "simple_words": "1-2 sentence plain English explanation summarizing what the sender is communicating.",\n'
        '  "expectation": "Concrete deliverable, action, or deadline expected by the sender.",\n'
        '  "action_items": ["Specific practical action 1", "Specific practical action 2"],\n'
        '  "reply": "Full drafted email response (salutation, body paragraphs addressing email specifics, and formal signature)."\n'
        "}\n\n"
        "Return ONLY the valid JSON object. No other text."
    )

    try:
        payload = {
            "model": DEFAULT_LOCAL_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.25,
                "num_predict": 280
            }
        }
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=35) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = data.get("response", "{}")

            # Robust JSON extraction without requiring slow grammar CFG constraints
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except Exception:
                    parsed = {}
            else:
                parsed = {}

            simple_words = parsed.get("simple_words") or fallback_insights["simple_words"]
            expectation = parsed.get("expectation") or fallback_insights["expectation"]
            action_items = parsed.get("action_items") or fallback_insights["action_items"]
            reply_text = parsed.get("reply", "").strip()

            # Clean any leading Subject: header line if model returned it
            reply_text = re.sub(r'^Subject:\s*.*?\n+', '', reply_text, flags=re.IGNORECASE).strip()

            if not reply_text or len(reply_text) < 30:
                reply_text = fallback_reply

            insights_res = {
                "simple_words": str(simple_words).strip(),
                "expectation": str(expectation).strip(),
                "questions": fallback_insights["questions"],
                "action_items": [str(a).strip() for a in action_items if a],
                "deadlines": fallback_insights["deadlines"],
                "domain_verified": fallback_insights["domain_verified"],
                "reading_time": fallback_insights["reading_time"]
            }

            result = {
                "insights": insights_res,
                "reply": reply_text,
                "suggestion_chips": chips,
                "sub_topic": sub_topic_label,
                "incoming_tone": detected_tone
            }
            _INSIGHTS_CACHE[cache_key] = result
            return result

    except Exception as ex:
        logger.warning(f"Ollama unified reasoning fallback triggered: {ex}")
        res = {
            "insights": fallback_insights,
            "reply": fallback_reply,
            "suggestion_chips": chips,
            "sub_topic": sub_topic_label,
            "incoming_tone": detected_tone
        }
        _INSIGHTS_CACHE[cache_key] = res
        return res


def analyze_email_with_llm(
    subject: str,
    sender: str,
    body: str,
    intent: str
) -> Dict[str, Any]:
    """Compatibility wrapper that calls unified_email_reasoning()."""
    res = unified_email_reasoning(
        sender=sender,
        subject=subject,
        body=body,
        intent=intent,
        tone="Professional",
        variant="confirm"
    )
    return res["insights"]


def generate_persona_reply_with_llm(
    sender: str,
    subject: str,
    body: str,
    intent: str,
    tone: str = "Professional",
    variant: str = "confirm",
    custom_instruction: str = "",
    checked_action_items: Optional[List[str]] = None,
    chip_label: Optional[str] = "",
    chip_desc: Optional[str] = ""
) -> str:
    """Invokes unified reasoning for the specified persona / tone."""
    res = unified_email_reasoning(
        sender=sender,
        subject=subject,
        body=body,
        intent=intent,
        tone=tone,
        variant=variant,
        custom_instruction=custom_instruction,
        checked_action_items=checked_action_items,
        chip_label=chip_label,
        chip_desc=chip_desc
    )
    return res["reply"]


# -----------------------------------------------------------------------------
# CONTEXT-ANCHORED DYNAMIC FALLBACK ENGINE (Never Cookie-Cutter)
# -----------------------------------------------------------------------------
def _clean_sender_name(sender_str: str) -> str:
    if not sender_str:
        return "there"
    cleaned = sender_str.replace('"', '').strip()
    if "<" in cleaned:
        display = cleaned.split("<")[0].strip()
        if display:
            return display
        email_part = cleaned.split("<")[1].split(">")[0].strip()
        return email_part.split("@")[0].replace(".", " ").title()
    return cleaned


def _extract_body_entities(body: str) -> Dict[str, Any]:
    """Extracts dates, venues, deadlines, and questions from the body."""
    date_patterns = [
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(?:\d{2,4})?\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b'
    ]
    dates = []
    for pat in date_patterns:
        for m in re.findall(pat, body, flags=re.IGNORECASE):
            if m not in dates:
                dates.append(m)
            if len(dates) >= 4:
                break

    venues = []
    for v in re.findall(r'\b(?:Lab\s*\d+|Room\s*\d+|Auditorium|Hall\s*\d+|CSE\s*Dept)\b', body, re.IGNORECASE):
        if v not in venues:
            venues.append(v)

    questions = []
    for line in body.splitlines():
        line_clean = line.strip()
        if "?" in line_clean and len(line_clean) > 8:
            questions.append(line_clean)
        if len(questions) >= 3:
            break

    imperatives = []
    for line in body.splitlines():
        l = line.strip()
        if re.search(r'\b(bring|submit|prepare|verify|note|attend|required|due|review|conducted|eligibility|schedule)\b', l, re.IGNORECASE) and 12 < len(l) < 140:
            cleaned_l = re.sub(r'^[0-9\.\-\*\•\s]+', '', l)
            if cleaned_l not in imperatives:
                imperatives.append(cleaned_l)
        if len(imperatives) >= 3:
            break

    return {
        "dates": dates,
        "venues": venues,
        "questions": questions,
        "imperatives": imperatives
    }


def _generate_heuristic_insights(subject: str, sender: str, body: str, intent: str, sub_topic: str = "general") -> Dict[str, Any]:
    clean_sender = _clean_sender_name(sender)
    clean_sub = re.sub(r'^(Re:\s*|Fwd:\s*)+', '', subject, flags=re.IGNORECASE).strip()
    entities = _extract_body_entities(body)

    dates_str = f" on {', '.join(entities['dates'][:2])}" if entities['dates'] else ""
    venues_str = f" at {entities['venues'][0]}" if entities['venues'] else ""

    if sub_topic == "academic_fee":
        simple_words = f"{clean_sender} is notifying about tuition/fee dues for {clean_sub}{dates_str}, requiring payment verification for exam eligibility."
        expectation = f"Complete fee payment before deadline{dates_str} and retain transaction receipt for CIE clearance."
    elif sub_topic == "academic_review":
        simple_words = f"{clean_sender} has scheduled project review assessments for {clean_sub}{dates_str}{venues_str}."
        expectation = f"Prepare updated architecture diagrams, functional prototype, and Git commit logs for evaluation{dates_str}."
    elif sub_topic == "academic_exam":
        simple_words = f"{clean_sender} has published the examination circular / schedule regarding {clean_sub}{dates_str}."
        expectation = f"Review syllabus, timetable{dates_str}, and confirm exam hall ticket eligibility."
    elif sub_topic == "academic_classroom":
        simple_words = f"{clean_sender} posted course announcements and study materials regarding {clean_sub}."
        expectation = "Review posted reference notes and complete tutorial practice exercises."
    elif sub_topic == "dev_pull_request":
        simple_words = f"{clean_sender} submitted code review feedback on pull request {clean_sub}."
        expectation = "Address requested changes, verify local tests, and push updated commits to the branch."
    elif sub_topic == "dev_alert":
        simple_words = f"Automated monitoring alert from {clean_sender} flagging service status for {clean_sub}."
        expectation = "Investigate error telemetry, verify CPU/disk metrics, and confirm service recovery."
    elif sub_topic == "career_interview":
        simple_words = f"{clean_sender} reached out regarding technical interview scheduling for {clean_sub}{dates_str}."
        expectation = "Confirm interview slot availability and verify video meeting credentials."
    elif sub_topic == "career_inquiry":
        simple_words = f"{clean_sender} shared engineering career / hackathon opportunity details for {clean_sub}."
        expectation = "Review opportunity guidelines, tech stack focus, and submit application if interested."
    else:
        simple_words = f"{clean_sender} communicated details regarding {clean_sub}{dates_str}."
        expectation = f"Review the notification and follow up with the requested next steps{dates_str}."

    sender_lower = sender.lower()
    domain_verified = any(d in sender_lower for d in ["@bmsce.ac.in", "@google.com", "@microsoft.com", "@apple.com", "@github.com", "@reddit.com", "@leetcode.com"])
    word_count = len(body.split())
    reading_time = f"{max(1, round(word_count / 180))} min read"

    action_items = []
    if entities["imperatives"]:
        for imp in entities["imperatives"][:2]:
            action_items.append(imp[:75])
    else:
        action_items = [f"Review requirements for '{clean_sub}'", "Confirm next step with sender"]

    return {
        "simple_words": simple_words,
        "expectation": expectation,
        "questions": entities["questions"],
        "action_items": action_items,
        "deadlines": entities["dates"],
        "domain_verified": domain_verified,
        "reading_time": reading_time
    }


def _generate_heuristic_reply(
    sender: str,
    subject: str,
    body: str,
    intent: str,
    sub_topic: str,
    tone: str,
    variant: str,
    strategy_label: str,
    strategy_desc: str,
    custom_instruction: str,
    checked_action_items: List[str]
) -> str:
    clean_sender = _clean_sender_name(sender)
    clean_sub = re.sub(r'^(Re:\s*|Fwd:\s*)+', '', subject, flags=re.IGNORECASE).strip()
    is_academic = intent == "Academic/Faculty"
    entities = _extract_body_entities(body)

    dates_str = f" scheduled for {', '.join(entities['dates'][:2])}" if entities['dates'] else ""
    venues_str = f" at {entities['venues'][0]}" if entities['venues'] else ""

    salutation = f"Dear {clean_sender}," if is_academic else f"Hi {clean_sender},"

    # Sub-topic specific context generation
    if sub_topic == "academic_fee":
        if variant == "inquire":
            body_msg = (
                f"Thank you for the notification regarding {clean_sub}.\n\n"
                f"Could you please clarify whether the payment challan must be submitted physically at the department accounts office, "
                f"or if verifying the online transaction UTR receipt is sufficient for CIE exam clearance?"
            )
        elif variant == "reschedule":
            body_msg = (
                f"Thank you for the notice regarding {clean_sub}{dates_str}.\n\n"
                f"Due to an active bank transaction processing delay, could I please request a short 3-day extension to complete "
                f"the fee payment without impacting my eligibility for the upcoming CIE examinations?"
            )
        elif variant == "decline":
            body_msg = (
                f"Thank you for the notice regarding {clean_sub}.\n\n"
                f"I would like to clarify that my fee dues were already settled previously. I have attached the transaction receipt "
                f"and challan reference for your verification and clearance in the college portal."
            )
        else:
            body_msg = (
                f"Thank you for the notice regarding {clean_sub}.\n\n"
                f"I have noted the deadline{dates_str}. I am completing the transaction verification and will retain the "
                f"fee payment receipt to ensure complete clearance for the upcoming CIE examinations."
            )

    elif sub_topic == "academic_review":
        if variant == "inquire":
            body_msg = (
                f"Thank you for informing us about {clean_sub}{dates_str}.\n\n"
                f"Could you please confirm if the evaluation rubrics emphasize the working prototype demo "
                f"or if equal weightage is assigned to the architecture documentation and Git commit history?"
            )
        elif variant == "reschedule":
            body_msg = (
                f"Thank you for the schedule update regarding {clean_sub}{dates_str}.\n\n"
                f"Our team has an overlapping CIE lab evaluation during that morning slot. Would it be possible "
                f"for our batch to be reviewed in the afternoon session on the same day?"
            )
        elif variant == "decline":
            body_msg = (
                f"Thank you for the update regarding {clean_sub}{dates_str}.\n\n"
                f"Due to an unavoidable scheduling conflict, our team would like to request permission to be evaluated "
                f"during the makeup lab session next week. All our deliverables will be completed and ready."
            )
        else:
            body_msg = (
                f"Thank you for the detailed guidelines regarding {clean_sub}{dates_str}{venues_str}.\n\n"
                f"Our team is preparing the required deliverables, including the updated architecture diagrams, "
                f"functional prototype, and Git commit logs. We will be present during the scheduled review slot."
            )

    elif sub_topic == "academic_classroom":
        if variant == "inquire":
            body_msg = (
                f"Thank you for sharing the materials regarding {clean_sub}.\n\n"
                f"I have a quick question regarding the problem statement and edge-case handling mentioned in the tutorial. "
                f"Could you clarify the expected time-complexity bounds for this exercise?"
            )
        else:
            body_msg = (
                f"Thank you for sharing the lecture notes and practice exercises regarding {clean_sub}.\n\n"
                f"I have reviewed the syllabus topics and will work through the assigned practice problems in preparation for class."
            )

    elif sub_topic == "dev_pull_request":
        if variant == "inquire":
            body_msg = (
                f"Thanks for the thorough review on {clean_sub}.\n\n"
                f"Could you share a quick reference or preferred pattern for the documentation layout you mentioned so I can align the PR accordingly?"
            )
        elif variant == "reschedule":
            body_msg = (
                f"Thanks for reviewing {clean_sub}.\n\n"
                f"To keep this PR focused and maintainable, would it be acceptable to merge the current modular docs and track the "
                f"extended CDC streams documentation in a dedicated follow-up issue?"
            )
        else:
            body_msg = (
                f"Thanks for reviewing {clean_sub}.\n\n"
                f"I have inspected the requested changes and pushed the updated documentation commits to the branch. "
                f"All CI checks are passing and ready for re-review."
            )

    elif sub_topic == "career_interview":
        if variant == "reschedule":
            body_msg = (
                f"Thank you very much for the invitation to interview for {clean_sub}.\n\n"
                f"I have an academic CIE examination commitment during the proposed slot{dates_str}. Would it be possible "
                f"to connect on Monday or Tuesday afternoon between 2:00 PM and 5:00 PM IST instead?"
            )
        elif variant == "inquire":
            body_msg = (
                f"Thank you for reaching out regarding the interview for {clean_sub}.\n\n"
                f"Could you please share more details regarding the interview format (coding environment, technical topics, or duration) "
                f"so I can adequately prepare?"
            )
        elif variant == "decline":
            body_msg = (
                f"Thank you for considering me for {clean_sub}.\n\n"
                f"Due to my ongoing academic semester workload at BMSCE, I will have to respectfully pass on this round, "
                f"but I would welcome staying in touch for future opportunities."
            )
        else:
            body_msg = (
                f"Thank you very much for the interview invitation for {clean_sub}.\n\n"
                f"I am pleased to confirm my availability for the proposed discussion{dates_str}. I look forward to speaking with the team."
            )

    else:
        # General specific fallback
        body_msg = (
            f"Thank you for contacting me regarding {clean_sub}.\n\n"
            f"I have reviewed your message{dates_str} and will ensure the required actions are followed up promptly."
        )

    # Attach checked items
    if checked_action_items:
        items_str = "\n".join([f"• Confirmed: {item}" for item in checked_action_items])
        body_msg += f"\n\nDeliverable Checklist:\n{items_str}"

    if custom_instruction.strip():
        body_msg += f"\n\nNote: {custom_instruction.strip()}"

    if is_academic:
        sig = (
            "Best regards,\n"
            "Arya Ayyappa Halambi\n"
            "USN: 1BF25CS095\n"
            "Department of Computer Science and Engineering\n"
            "B.M.S. College of Engineering (BMSCE)"
        )
    else:
        sig = "Best regards,\nArya Ayyappa Halambi"

    return f"{salutation}\n\n{body_msg}\n\n{sig}"
