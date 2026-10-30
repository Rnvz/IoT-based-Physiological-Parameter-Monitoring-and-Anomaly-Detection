#!/usr/bin/env python3
"""
CLI Helper to Tag Benign Recording Scenarios
Communicates with FastAPI backend (/session/scenario).

Usage:
    python scripts/tag_scenario.py REST --note "kamar, duduk"
    python scripts/tag_scenario.py --show
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple

DEFAULT_BASE_URL = "http://localhost:8000"
ENV_PATH = os.path.join(os.path.dirname(__file__), "../backend/.env")

VALID_SCENARIOS = [
    "REST",
    "TYPING",
    "LIGHT_ACTIVITY",
    "MOTION",
    "FINGER_SHIFT",
    "SENSOR_OFF",
    "OTHER"
]

def load_env_config() -> Tuple[str, Optional[str]]:
    """Loads API base URL and API Key from backend/.env or environment variables."""
    base_url = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")

    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k == "API_KEY" and not api_key:
                        api_key = v
                    elif k == "API_BASE_URL" and not base_url:
                        base_url = v
        except Exception:
            pass

    if not base_url:
        base_url = DEFAULT_BASE_URL
    return base_url.rstrip("/"), api_key

def get_scenario(base_url: str, api_key: Optional[str]) -> Dict[str, Any]:
    url = f"{base_url}/session/scenario"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        print(f"[ERROR] HTTP {e.code}: {err_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Connection failed to {url}: {e}", file=sys.stderr)
        sys.exit(1)

def set_scenario(base_url: str, api_key: Optional[str], scenario: str, note: Optional[str]) -> Dict[str, Any]:
    url = f"{base_url}/session/scenario"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {"scenario": scenario}
    if note is not None:
        payload["note"] = note

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        print(f"[ERROR] HTTP {e.code}: {err_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Connection failed to {url}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    default_url, default_key = load_env_config()

    parser = argparse.ArgumentParser(
        description="Tag benign recording scenarios for model evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=VALID_SCENARIOS,
        help=f"Benign scenario identifier ({', '.join(VALID_SCENARIOS)})"
    )
    parser.add_argument(
        "--note", "-n",
        default=None,
        help="Optional note describing recording condition (max 200 chars)"
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Print the current active scenario from GET /session/scenario"
    )
    parser.add_argument(
        "--base-url",
        default=default_url,
        help=f"Backend base URL (default: {default_url})"
    )
    parser.add_argument(
        "--api-key",
        default=default_key,
        help="X-API-Key for authentication"
    )

    args = parser.parse_args()

    if not args.scenario and not args.show:
        parser.print_help()
        sys.exit(1)

    if args.scenario:
        res = set_scenario(
            base_url=args.base_url,
            api_key=args.api_key,
            scenario=args.scenario,
            note=args.note
        )
        print("Scenario updated successfully:")
        print(json.dumps(res, indent=2))

    if args.show and not args.scenario:
        res = get_scenario(
            base_url=args.base_url,
            api_key=args.api_key
        )
        print("Current active scenario:")
        print(json.dumps(res, indent=2))
    elif args.show and args.scenario:
        res = get_scenario(
            base_url=args.base_url,
            api_key=args.api_key
        )
        print("\nVerified from GET /session/scenario:")
        print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
