"""Smoke test: confirms PINECONE_API_KEY and ANTHROPIC_API_KEY are valid
before any ingestion or eval work begins. Distinguishes auth failures from
network/timeout failures so the error message is actionable.
"""
import os
import sys

from dotenv import load_dotenv

# Load PINECONE_API_KEY, PINECONE_INDEX_NAME, ANTHROPIC_API_KEY from .env
load_dotenv()


def check_pinecone() -> bool:
    """Confirm PINECONE_API_KEY is set and can authenticate with Pinecone."""
    # Validate the key is present before ever calling the API
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("[Pinecone] FAIL - PINECONE_API_KEY not set in .env")
        return False

    from pinecone import Pinecone
    from pinecone.exceptions import UnauthorizedException

    try:
        # list_indexes is a lightweight authenticated call - just enough
        # to confirm the key works, without creating or touching an index
        pc = Pinecone(api_key=api_key)
        pc.list_indexes()
        print("[Pinecone] OK - authenticated and reachable")
        return True
    except UnauthorizedException as e:
        print(f"[Pinecone] FAIL - invalid API key: {e}")
        return False
    except Exception as e:
        # Anything else (DNS failure, timeout, service outage, ...) is a
        # connection problem rather than a bad key
        print(f"[Pinecone] FAIL - network/connection error: {e}")
        return False


def check_anthropic() -> bool:
    """Confirm ANTHROPIC_API_KEY is set and can authenticate with Claude."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[Anthropic] FAIL - ANTHROPIC_API_KEY not set in .env")
        return False

    import anthropic

    try:
        # models.list is a lightweight authenticated call - confirms the
        # key works without spending tokens on a real completion
        client = anthropic.Anthropic(api_key=api_key)
        client.models.list(limit=1)
        print("[Anthropic] OK - authenticated and reachable")
        return True
    except anthropic.AuthenticationError as e:
        print(f"[Anthropic] FAIL - invalid API key: {e}")
        return False
    except anthropic.APIConnectionError as e:
        print(f"[Anthropic] FAIL - network/connection error: {e}")
        return False
    except Exception as e:
        print(f"[Anthropic] FAIL - unexpected error: {e}")
        return False


def main() -> int:
    """Run both checks and report a combined pass/fail result."""
    pinecone_ok = check_pinecone()
    anthropic_ok = check_anthropic()

    if pinecone_ok and anthropic_ok:
        print("\nAll connections OK.")
        return 0

    print("\nOne or more connections failed. See messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
