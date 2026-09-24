"""ntfy tools — let the model ping Aaron's phone in real time via ntfy.sh.

One tool, one job: send_ntfy posts a push notification to Aaron's personal
topic. The topic URL lives only in .env (TOPIC_URL) — never in source — so it
cannot leak through the public repo. It is read at call time (after agent.py's
load_dotenv) and there is deliberately no parameter that can redirect a send:
prompt injection can choose the words, never the destination.

Real-time only by construction: there is no delay/scheduled-publish parameter
and no timer support. Anything that needs to happen later is the heartbeat's
or a cron's job, not this module's.
"""

import functools
import json
import os
import urllib.error
import urllib.request

# Aaron's phone-ping channel (re-armed 2026-09-24 at his explicit request).
# The URL is a credential — anyone who knows it can ping the phone — so it
# lives in .env as TOPIC_URL, read at call time. Never hardcode it back here.
# Channel record: workspace/shared/config/ntfy_channel.md
def _topic_url() -> str:
    """Read the topic URL from the environment, erroring clearly if absent."""
    url = os.environ.get("TOPIC_URL", "").strip()
    if not url:
        raise NtfyError("TOPIC_URL is not set — add it to .env.")
    return url

# Priorities ntfy.sh accepts, as sent on the wire.
VALID_PRIORITIES = ("min", "low", "default", "high", "urgent")

# Seconds to wait for ntfy.sh before giving up, so a network hiccup never
# stalls the agent's tool loop for long.
_TIMEOUT = 10


class NtfyError(Exception):
    """Raised when a ping cannot be sent. Reported to the model as an ordinary
    "Error: ..." string, like every other refusal."""


def _tool(fn):
    """Turn network failures into error strings instead of unwinding into the
    tool-call loop — same contract as workspace_tool._tool."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NtfyError as e:
            return f"Error: {e}"
        except urllib.error.HTTPError as e:
            return f"Error: {fn.__name__} failed: ntfy.sh returned HTTP {e.code}."
        except (urllib.error.URLError, OSError) as e:
            return f"Error: {fn.__name__} failed: {e}"

    return wrapper


@_tool
def send_ntfy(
    message: str,
    title: str = "📡 AMADEUS → Aaron",
    priority: str = "default",
    tags: str = "bell",
) -> str:
    """Send a real-time push notification to Aaron's phone via ntfy.sh."""
    if not message or not message.strip():
        raise NtfyError("message is empty — nothing to send.")
    if priority not in VALID_PRIORITIES:
        raise NtfyError(
            f"priority must be one of {', '.join(VALID_PRIORITIES)} (got '{priority}')."
        )
    url = _topic_url()

    body = message.encode("utf-8")
    headers = {
        "Title": (title or "📡 AMADEUS → Aaron").encode("utf-8"),
        "Priority": priority,
        "Tags": tags or "bell",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    ntfy_id = payload.get("id", "unknown")
    return (
        f"Ping accepted by ntfy.sh (HTTP {payload.get('code', 200)}). "
        f"Message id: {ntfy_id}."
    )


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_ntfy",
            "description": (
                "Sends a real-time push notification to Aaron's phone via ntfy.sh. "
                "For time-sensitive pings only — reminders, schedule changes, error "
                "alerts, AFK pages. Non-urgent items belong in the next session brief "
                "instead; do not spam the channel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Body text of the notification.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short headline shown above the body. Defaults to '📡 AMADEUS → Aaron'.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": list(VALID_PRIORITIES),
                        "description": "Urgency: 'high' only when Aaron needs to look now; 'default' for FYIs.",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated emoji/tag names shown with the notification (e.g. 'bell', 'rotating_light').",
                    },
                },
                "required": ["message"],
            },
        },
    }
]

FUNCTIONS = {"send_ntfy": send_ntfy}
