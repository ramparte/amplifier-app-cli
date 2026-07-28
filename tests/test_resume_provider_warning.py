"""Tests for cross-provider resume handling.

Covers:
- model_history provenance: previous model appended when it changes across saves
- resume mismatch warning: emitted when the session's saved provider differs
  from the resolved provider, silent when they match
- friendly translation of thinking-block/signature InvalidRequestError 400s
"""

import asyncio

from amplifier_app_cli.commands.run import build_resume_mismatch_warning
from amplifier_app_cli.incremental_save import IncrementalSaveHook
from amplifier_app_cli.incremental_save import updated_model_history
from amplifier_app_cli.session_store import SessionStore
from amplifier_app_cli.utils.error_format import format_cross_provider_resume_error


def _config(module: str, model: str) -> dict:
    return {"providers": [{"module": module, "config": {"default_model": model}}]}


# ---------------------------------------------------------------------------
# model_history provenance
# ---------------------------------------------------------------------------


def test_model_history_appended_when_model_changes():
    existing = {"model": "claude-opus-4-6"}
    assert updated_model_history(existing, "gpt-5") == ["claude-opus-4-6"]


def test_model_history_accumulates_previous_models():
    existing = {"model": "gpt-5", "model_history": ["claude-opus-4-6"]}
    assert updated_model_history(existing, "gemini-2.5-pro") == [
        "claude-opus-4-6",
        "gpt-5",
    ]


def test_model_history_unchanged_when_model_same():
    existing = {"model": "claude-opus-4-6", "model_history": ["gpt-5"]}
    assert updated_model_history(existing, "claude-opus-4-6") == ["gpt-5"]


def test_model_history_ignores_unknown_or_missing_previous_model():
    assert updated_model_history({"model": "unknown"}, "gpt-5") == []
    assert updated_model_history({}, "gpt-5") == []


class _FakeContext:
    def __init__(self, messages):
        self._messages = messages

    async def get_messages(self):
        return self._messages


class _FakeCoordinator:
    def __init__(self, context):
        self._context = context

    def get(self, name):
        return self._context if name == "context" else None


class _FakeSession:
    def __init__(self, context):
        self.coordinator = _FakeCoordinator(context)


def test_model_history_appended_across_incremental_saves(tmp_path):
    """End-to-end: two incremental saves under different providers preserve
    the original model in metadata.json's model_history."""
    store = SessionStore(tmp_path)
    messages = [{"role": "user", "content": "hi"}]
    session = _FakeSession(_FakeContext(messages))

    # Session exists before incremental saves fire (as in production)
    store.save("sess-1", [], {"session_id": "sess-1"})

    hook1 = IncrementalSaveHook(
        session,
        store,
        "sess-1",
        "bundle:foundation",
        _config("provider-anthropic", "claude-opus-4-6"),
    )
    asyncio.run(hook1.on_tool_post("tool:post", {"tool_name": "t"}))

    messages.append({"role": "assistant", "content": "hello"})
    hook2 = IncrementalSaveHook(
        session,
        store,
        "sess-1",
        "bundle:foundation",
        _config("provider-openai", "gpt-5"),
    )
    asyncio.run(hook2.on_tool_post("tool:post", {"tool_name": "t"}))

    metadata = store.get_metadata("sess-1")
    assert metadata is not None
    assert metadata["model"] == "gpt-5"  # current model preserved for display
    assert metadata["model_history"] == ["claude-opus-4-6"]


# ---------------------------------------------------------------------------
# resume provider mismatch warning
# ---------------------------------------------------------------------------


def test_provider_mismatch_warning_on_resume():
    warning = build_resume_mismatch_warning(
        "provider-anthropic/claude-opus-4-6",
        _config("provider-openai", "gpt-5"),
        "abc123",
    )
    assert warning is not None
    # Names both models
    assert "claude-opus-4-6" in warning
    assert "gpt-5" in warning
    # Shows exact flags to resume with the original provider
    assert "--resume abc123" in warning
    assert "--provider anthropic" in warning
    assert "--model claude-opus-4-6" in warning


def test_no_provider_mismatch_warning_when_providers_match():
    warning = build_resume_mismatch_warning(
        "provider-anthropic/claude-opus-4-6",
        _config("provider-anthropic", "claude-opus-4-6"),
        "abc123",
    )
    assert warning is None


def test_no_provider_mismatch_warning_for_mount_name_format():
    """Real persisted metadata uses provider mount names ("anthropic/<model>",
    written by main.py's final save), not module ids ("provider-anthropic").
    A same-provider resume of such a session must not warn."""
    warning = build_resume_mismatch_warning(
        "anthropic/claude-opus-4-6",
        _config("provider-anthropic", "claude-opus-4-6"),
        "abc123",
    )
    assert warning is None


def test_provider_mismatch_warning_for_mount_name_format():
    """Cross-provider resume with the actually-persisted mount-name format
    still warns, and the advice names usable flags."""
    warning = build_resume_mismatch_warning(
        "anthropic/claude-opus-4-6",
        _config("provider-openai", "gpt-5"),
        "abc123",
    )
    assert warning is not None
    assert "anthropic/claude-opus-4-6" in warning
    assert "gpt-5" in warning
    assert "--provider anthropic" in warning
    assert "--model claude-opus-4-6" in warning


def test_following_warning_advice_suppresses_warning():
    """Following the warning's own advice (--provider anthropic --model
    claude-opus-4-6) must suppress the warning on that resume."""
    saved = "anthropic/claude-opus-4-6"
    base_config = {
        "providers": [
            {"module": "provider-openai", "config": {"default_model": "gpt-5"}},
            {
                "module": "provider-anthropic",
                "config": {"default_model": "claude-sonnet-4-5"},
            },
        ]
    }
    assert build_resume_mismatch_warning(saved, base_config, "abc123") is not None

    # Simulate the CLI override applied by `--provider anthropic --model
    # claude-opus-4-6` (run.py promotes the target entry to priority 0 and
    # sets default_model) — the warning must go away.
    override_config = {
        "providers": [
            {"module": "provider-openai", "config": {"default_model": "gpt-5"}},
            {
                "module": "provider-anthropic",
                "config": {"default_model": "claude-opus-4-6", "priority": 0},
            },
        ]
    }
    assert build_resume_mismatch_warning(saved, override_config, "abc123") is None


def test_provider_mismatch_warning_for_bare_model_name():
    # Older/incremental metadata records just the model name
    warning = build_resume_mismatch_warning(
        "claude-opus-4-6", _config("provider-openai", "gpt-5"), "abc123"
    )
    assert warning is not None
    assert "claude-opus-4-6" in warning
    assert "gpt-5" in warning

    # Same bare model -> no warning
    assert (
        build_resume_mismatch_warning(
            "gpt-5", _config("provider-openai", "gpt-5"), "abc123"
        )
        is None
    )


def test_no_provider_mismatch_warning_without_saved_model():
    config = _config("provider-openai", "gpt-5")
    assert build_resume_mismatch_warning(None, config, "abc123") is None
    assert build_resume_mismatch_warning("unknown", config, "abc123") is None
    # No providers resolved -> nothing to compare against
    assert build_resume_mismatch_warning("claude-opus-4-6", {}, "abc123") is None


def test_provider_mismatch_uses_highest_precedence_provider():
    config = {
        "providers": [
            {
                "module": "provider-openai",
                "config": {"default_model": "gpt-5", "priority": 100},
            },
            {
                "module": "provider-anthropic",
                "config": {"default_model": "claude-opus-4-6", "priority": 0},
            },
        ]
    }
    # Saved provider matches the priority-0 (active) provider -> no warning
    assert (
        build_resume_mismatch_warning(
            "provider-anthropic/claude-opus-4-6", config, "abc123"
        )
        is None
    )


# ---------------------------------------------------------------------------
# friendly translation of thinking/signature 400s
# ---------------------------------------------------------------------------


class InvalidRequestError(Exception):
    """Stand-in for provider SDK InvalidRequestError (matched by name/content)."""


def test_friendly_translation_for_thinking_signature_400():
    err = InvalidRequestError(
        'Error code: 400 - {"type": "error", "error": {"type": "invalid_request_error",'
        ' "message": "messages.1.content.0.thinking.signature: Invalid signature"}}'
    )
    message = format_cross_provider_resume_error(err)
    assert message is not None
    assert "different provider" in message
    assert "--provider" in message
    # Original error text is preserved for debugging
    assert "Invalid signature" in message


def test_friendly_translation_ignores_unrelated_errors():
    # Not a 400 / invalid request at all
    assert format_cross_provider_resume_error(ValueError("plain failure")) is None
    # A 400 that has nothing to do with thinking blocks or signatures
    assert (
        format_cross_provider_resume_error(
            InvalidRequestError("400 bad request: max_tokens too large")
        )
        is None
    )
