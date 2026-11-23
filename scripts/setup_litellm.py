"""
Set up budgets, customers and the app's virtual key on the LiteLLM proxy.

Run once after starting the proxy (docker compose -f litellm/docker-compose.yml up -d):

    uv run python scripts/setup_litellm.py

What it does (safe to rerun; existing items are left alone):
  1. Creates the "HyperBudget" budget tier for end users.
  2. Registers each demo user (James, George, Mike, Sherlock) as a customer
     on that budget, so the "user" field the app sends is billed to it.
  3. Generates a virtual key for the app, limited to the chat and embeddings
     models, with its own budget and rate limits, and writes it to .env as
     LITELLM_API_KEY. The key is never printed.

The master key is read from litellm/.env and is only used by this script,
never by the app.
"""

import sys
from pathlib import Path

import requests
from dotenv import dotenv_values, set_key

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from llm_clients import (  # noqa: E402
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_PROXY_URL,
    DEMO_USERS,
)

APP_ENV_PATH = PROJECT_ROOT / ".env"
PROXY_ENV_PATH = PROJECT_ROOT / "litellm" / ".env"

# End-user pricing tier: each demo user gets this budget and these limits
END_USER_BUDGET = {
    "budget_id": "HyperBudget",
    "max_budget": 0.01,        # [USD per budget_duration]
    "budget_duration": "1d",
    "rpm_limit": 20,           # [requests per minute]
    "tpm_limit": 20000,        # [tokens per minute]
}

# The app's virtual key: what the whole application may spend and how fast
APP_KEY_SETTINGS = {
    "key_alias": "smartphone-assistant",
    "models": [DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDINGS_MODEL],
    "max_budget": 0.50,        # [USD per budget_duration]
    "budget_duration": "1d",
    "rpm_limit": 30,           # [requests per minute]
    "tpm_limit": 30000,        # [tokens per minute]
}


def admin_session(master_key: str) -> requests.Session:
    """
    Build an HTTP session authenticated with the proxy's master key.

    :param
        master_key (str): LITELLM_MASTER_KEY from litellm/.env.

    :returns
        requests.Session: Session that sends the master key on every request.
    """
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"})
    return session


def ensure_budget(session: requests.Session, proxy_url: str, budget: dict) -> str:
    """
    Create the end-user budget tier unless it already exists.

    :param
        session (requests.Session): Authenticated admin session.
        proxy_url (str): Proxy base URL.
        budget (dict): Budget fields for /budget/new, including budget_id.

    :returns
        str: "created" or "exists".
    """
    info = session.post(f"{proxy_url}/budget/info", json={"budgets": [budget["budget_id"]]}, timeout=10)
    if info.ok and info.json():
        return "exists"

    session.post(f"{proxy_url}/budget/new", json=budget, timeout=10).raise_for_status()
    return "created"


def ensure_customer(session: requests.Session, proxy_url: str, user_id: str, budget_id: str) -> str:
    """
    Register an end user (LiteLLM "customer") on a budget unless already registered.

    :param
        session (requests.Session): Authenticated admin session.
        proxy_url (str): Proxy base URL.
        user_id (str): End user name, matching the "user" field the app sends.
        budget_id (str): Budget tier to attach the user to.

    :returns
        str: "created" or "exists".
    """
    info = session.get(f"{proxy_url}/customer/info", params={"end_user_id": user_id}, timeout=10)
    if info.ok:
        return "exists"

    session.post(f"{proxy_url}/customer/new", json={"user_id": user_id, "budget_id": budget_id}, timeout=10).raise_for_status()
    return "created"


def ensure_app_key(session: requests.Session, proxy_url: str, settings: dict, env_path: Path) -> str:
    """
    Generate the app's virtual key and store it in .env, unless .env already has a working one.

    :param
        session (requests.Session): Authenticated admin session.
        proxy_url (str): Proxy base URL.
        settings (dict): Fields for /key/generate (alias, models, budget, limits).
        env_path (Path): The app's .env file; LITELLM_API_KEY is written here.

    :returns
        str: "created" or "exists".
    """
    existing_key = dotenv_values(env_path).get("LITELLM_API_KEY")
    if existing_key:
        check = requests.get(
            f"{proxy_url}/key/info",
            headers={"Authorization": f"Bearer {existing_key}"},
            timeout=10,
        )
        if check.ok:
            return "exists"

    response = session.post(f"{proxy_url}/key/generate", json=settings, timeout=10)
    response.raise_for_status()

    # Write the key straight to .env without printing it
    set_key(str(env_path), "LITELLM_API_KEY", response.json()["key"], quote_mode="never")
    return "created"


def ensure_app_settings(env_path: Path, proxy_url: str) -> None:
    """
    Add the proxy URL and model names to the app's .env if they are missing.

    :param
        env_path (Path): The app's .env file.
        proxy_url (str): Proxy base URL to store as LITELLM_BASE_URL.

    :returns
        None.
    """
    current = dotenv_values(env_path)
    defaults = {
        "LITELLM_BASE_URL": proxy_url,
        "LITELLM_MODEL": DEFAULT_CHAT_MODEL,
        "LITELLM_EMBEDDINGS_MODEL": DEFAULT_EMBEDDINGS_MODEL,
    }
    for name, value in defaults.items():
        if not current.get(name):
            set_key(str(env_path), name, value, quote_mode="never")


def main() -> None:
    """Run the setup steps in order and report what was created."""
    proxy_url = dotenv_values(APP_ENV_PATH).get("LITELLM_BASE_URL") or DEFAULT_PROXY_URL
    master_key = dotenv_values(PROXY_ENV_PATH).get("LITELLM_MASTER_KEY")
    if not master_key:
        sys.exit(f"LITELLM_MASTER_KEY not found in {PROXY_ENV_PATH}")

    session = admin_session(master_key)

    print(f"Budget {END_USER_BUDGET['budget_id']}: {ensure_budget(session, proxy_url, END_USER_BUDGET)}")
    for user_id in DEMO_USERS:
        print(f"Customer {user_id}: {ensure_customer(session, proxy_url, user_id, END_USER_BUDGET['budget_id'])}")

    ensure_app_settings(APP_ENV_PATH, proxy_url)
    print(f"App virtual key: {ensure_app_key(session, proxy_url, APP_KEY_SETTINGS, APP_ENV_PATH)} (stored in .env)")


if __name__ == "__main__":
    main()
