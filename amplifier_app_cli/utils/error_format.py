"""Safe error message formatting utilities.

Ensures exceptions always have useful display messages, even when
their str() representation is empty (e.g., TimeoutError, CancelledError).

This module addresses a common issue where certain Python exceptions
have empty string representations, leading to unhelpful error messages
like "Error: " with nothing after the colon.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


# Friendly messages for specific exception types known to have empty str()
# These provide user-actionable context when the exception itself has none
FRIENDLY_MESSAGES: dict[type, str] = {
    TimeoutError: "Request timed out. This may indicate network issues or a slow API response.",
    asyncio.CancelledError: "Operation was cancelled.",
    ConnectionResetError: "Connection was reset by the server.",
    BrokenPipeError: "Connection was closed unexpectedly.",
    KeyboardInterrupt: "Operation interrupted by user.",
}


def format_error_message(e: BaseException, *, include_type: bool = True) -> str:
    """Format an exception into a useful display message.

    Handles exceptions with empty str() representations by falling back
    to type name and/or friendly messages.

    Args:
        e: The exception to format
        include_type: Whether to include the exception type name

    Returns:
        A non-empty, user-friendly error message

    Examples:
        >>> format_error_message(TimeoutError())
        'TimeoutError: Request timed out. This may indicate network issues or a slow API response.'

        >>> format_error_message(ValueError("invalid input"))
        'ValueError: invalid input'

        >>> format_error_message(ValueError("invalid input"), include_type=False)
        'invalid input'
    """
    error_str = str(e)
    error_type = type(e).__name__

    # If we have a message, use it
    if error_str:
        if include_type and error_type not in error_str:
            return f"{error_type}: {error_str}"
        return error_str

    # No message - check for friendly fallback
    for exc_type, friendly_msg in FRIENDLY_MESSAGES.items():
        if isinstance(e, exc_type):
            return f"{error_type}: {friendly_msg}"

    # Last resort: just the type name with indicator
    return f"{error_type}: (no additional details)"


def format_cross_provider_resume_error(e: BaseException) -> str | None:
    """Translate thinking-block/signature 400s into a friendly explanation.

    Providers reject transcripts containing extended-thinking blocks that were
    produced by a *different* provider/model (their signatures fail
    validation), surfacing as a cryptic InvalidRequestError 400 mentioning
    "thinking" and/or "signature". This detects that shape and points the user
    at cross-provider resume as the likely cause.

    Args:
        e: The exception to inspect

    Returns:
        Human-readable explanation string, or None if the error does not
        look like a cross-provider resume failure.
    """
    error_text = str(e)
    haystack = f"{type(e).__name__}: {error_text}".lower()

    compact = haystack.replace("_", "").replace(" ", "")
    looks_like_bad_request = (
        "invalidrequest" in compact or "400" in haystack or "bad request" in haystack
    )
    mentions_thinking_signature = "thinking" in haystack or "signature" in haystack
    if not (looks_like_bad_request and mentions_thinking_signature):
        return None

    return (
        f"{error_text}\n\n"
        "This usually means the session was resumed under a different provider "
        "than the one that created it: thinking blocks in the transcript carry "
        "provider-specific signatures that other providers reject with a 400 error.\n"
        "To fix, resume the session with its original provider:\n"
        "  amplifier run --resume <session-id> --provider <original-provider> --model <original-model>\n"
        "(The session's original model is recorded in its metadata.json under "
        "'model' / 'model_history'.)"
    )


def print_error(console: "Console", e: BaseException, *, verbose: bool = False) -> None:
    """Print an error to the console with proper formatting.

    Args:
        console: Rich console for output
        e: The exception to display
        verbose: If True, also print the traceback
    """
    error_msg = format_error_message(e)
    console.print(f"[red]Error:[/red] {error_msg}")

    if verbose:
        console.print_exception()


def get_error_context(e: BaseException) -> dict:
    """Extract context from an exception for logging/events.

    Useful for emitting error events with structured data.

    Args:
        e: The exception to extract context from

    Returns:
        Dict with error_type, error_message, and traceback info
    """
    return {
        "error_type": type(e).__name__,
        "error_message": format_error_message(e, include_type=False),
        "error_repr": repr(e),
        "traceback": traceback.format_exc(),
    }
