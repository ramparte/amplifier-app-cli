"""Tests for the /compact slash command and token-pressure /status.

Wires the real SimpleContextManager into a mocked session so the command
path (process_input -> handle_command -> _compact_context) is exercised
end-to-end, including the before/after summary and the persistent shrink.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# The context manager lives in a sibling module checkout; make it importable
# without requiring an installed package.
_CTX = Path(__file__).resolve().parents[2] / "amplifier-module-context-simple"
if _CTX.exists():
    sys.path.insert(0, str(_CTX))

from amplifier_module_context_simple import SimpleContextManager  # noqa: E402


def _make_processor(context):
    """CommandProcessor with a mocked session exposing a real context manager."""
    from amplifier_app_cli.main import CommandProcessor

    mounts = {"context": context, "providers": {}, "tools": {}}
    session = MagicMock()
    session.coordinator = MagicMock()
    session.coordinator.session_state = {"active_mode": None, "mode_discovery": None}
    session.coordinator.get.side_effect = lambda key: mounts.get(key)
    session.coordinator.session_id = "test-session"

    cp = CommandProcessor(session, "test-bundle")
    return cp


async def _fill(context, pairs):
    for i in range(pairs):
        await context.add_message(
            {"role": "user", "content": f"message {i} with some padding content here"}
        )
        await context.add_message(
            {"role": "assistant", "content": f"response {i} with some padding content here"}
        )


def test_compact_is_registered():
    from amplifier_app_cli.main import CommandProcessor

    assert "/compact" in CommandProcessor.COMMANDS
    assert CommandProcessor.COMMANDS["/compact"]["action"] == "compact_context"


def test_compact_routes_through_process_input():
    ctx = SimpleContextManager(max_tokens=1000)
    cp = _make_processor(ctx)
    action, data = cp.process_input("/compact")
    assert action == "compact_context"


@pytest.mark.asyncio
async def test_compact_command_shrinks_and_reports():
    ctx = SimpleContextManager(
        max_tokens=1000, compact_threshold=0.9, target_usage=0.5, protected_recent=0.1
    )
    await _fill(ctx, 50)
    before = len(ctx.messages)

    cp = _make_processor(ctx)
    result = await cp.handle_command("compact_context", {})

    assert result.startswith("✓ Compacted context")
    assert len(ctx.messages) < before  # persistent shrink actually happened


@pytest.mark.asyncio
async def test_compact_command_nothing_to_do():
    ctx = SimpleContextManager(max_tokens=100_000)
    await _fill(ctx, 3)
    cp = _make_processor(ctx)

    result = await cp.handle_command("compact_context", {})
    assert result.startswith("Nothing to compact")


@pytest.mark.asyncio
async def test_status_shows_token_pressure_and_nudge():
    ctx = SimpleContextManager(max_tokens=1000, compact_threshold=0.9)
    await _fill(ctx, 60)  # push over threshold
    cp = _make_processor(ctx)

    status = await cp._get_status()
    assert "Context:" in status
    assert "/compact" in status  # the nudge is present when context is full
