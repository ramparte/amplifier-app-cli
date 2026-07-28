"""Tests for `amplifier session change-provider` (Phase 2).

All tests use tmp-dir session stores (SessionStore(base_dir=tmp_path)) and
never touch the user's real ~/.amplifier.
"""

import json

from amplifier_app_cli.commands.session import change_session_provider
from amplifier_app_cli.commands.session import parse_provider_model
from amplifier_app_cli.commands.session import strip_thinking_blocks
from amplifier_app_cli.session_store import SessionStore

import pytest


def _make_session(tmp_path, session_id="sess-1", model="anthropic/claude-opus-4-6"):
    """Create an isolated session with a transcript containing thinking blocks."""
    store = SessionStore(base_dir=tmp_path)
    transcript = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "pondering...", "signature": "sig-1"},
                {"type": "text", "text": "hi there"},
            ],
        },
        {"role": "user", "content": "more"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "again...", "signature": "sig-2"},
                {"type": "text", "text": "sure"},
            ],
        },
    ]
    metadata = {"session_id": session_id, "model": model}
    store.save(session_id, transcript, metadata)
    return store


def test_change_provider_strips_thinking_blocks_on_provider_change(tmp_path):
    store = _make_session(tmp_path)

    result = change_session_provider(store, "sess-1", "openai/gpt-5")

    assert result.changed is True
    assert result.provider_changed is True
    assert result.thinking_blocks_stripped == 2

    transcript, _ = store.load("sess-1")
    for message in transcript:
        content = message.get("content")
        if isinstance(content, list):
            assert all(block.get("type") != "thinking" for block in content)
    # Non-thinking content survives
    assistant_texts = [
        block["text"]
        for message in transcript
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "text"
    ]
    assert assistant_texts == ["hi there", "sure"]


def test_change_provider_same_provider_leaves_transcript_untouched(tmp_path):
    store = _make_session(tmp_path)
    transcript_file = tmp_path / "sess-1" / "transcript.jsonl"
    before_bytes = transcript_file.read_bytes()

    result = change_session_provider(store, "sess-1", "anthropic/claude-opus-4-8")

    assert result.changed is True
    assert result.provider_changed is False
    assert result.thinking_blocks_stripped == 0
    # Transcript file bytes are identical: never rewritten
    assert transcript_file.read_bytes() == before_bytes
    # Thinking blocks still present
    transcript, _ = store.load("sess-1")
    thinking = [
        block
        for message in transcript
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "thinking"
    ]
    assert len(thinking) == 2


def test_change_provider_updates_model_and_model_history(tmp_path):
    store = _make_session(tmp_path, model="anthropic/claude-opus-4-6")

    change_session_provider(store, "sess-1", "openai/gpt-5")

    metadata = store.get_metadata("sess-1")
    assert metadata["model"] == "openai/gpt-5"
    assert metadata["model_history"] == ["anthropic/claude-opus-4-6"]


def test_change_provider_creates_backups_before_modification(tmp_path):
    store = _make_session(tmp_path)
    session_dir = tmp_path / "sess-1"
    original_transcript = (session_dir / "transcript.jsonl").read_bytes()
    original_metadata = (session_dir / "metadata.json").read_bytes()

    result = change_session_provider(store, "sess-1", "openai/gpt-5")

    transcript_backups = list(session_dir.glob("transcript.jsonl.bak-*"))
    metadata_backups = list(session_dir.glob("metadata.json.bak-*"))
    assert len(transcript_backups) == 1
    assert len(metadata_backups) == 1
    assert set(result.backup_paths) == {transcript_backups[0], metadata_backups[0]}
    # Backups hold the PRE-modification contents
    assert transcript_backups[0].read_bytes() == original_transcript
    assert metadata_backups[0].read_bytes() == original_metadata
    # And the live files were actually modified afterwards
    assert (session_dir / "metadata.json").read_bytes() != original_metadata
    backup_metadata = json.loads(metadata_backups[0].read_text())
    assert backup_metadata["model"] == "anthropic/claude-opus-4-6"


def test_change_provider_is_idempotent_noop_on_same_target(tmp_path):
    store = _make_session(tmp_path)

    first = change_session_provider(store, "sess-1", "openai/gpt-5")
    assert first.changed is True

    session_dir = tmp_path / "sess-1"
    backups_after_first = sorted(p.name for p in session_dir.glob("*.bak-*"))
    metadata_after_first = store.get_metadata("sess-1")

    second = change_session_provider(store, "sess-1", "openai/gpt-5")
    assert second.changed is False
    assert second.backup_paths == []
    assert second.thinking_blocks_stripped == 0
    # No new backups, metadata unchanged
    assert sorted(p.name for p in session_dir.glob("*.bak-*")) == backups_after_first
    assert store.get_metadata("sess-1") == metadata_after_first


def test_change_provider_rejects_bare_model_target(tmp_path):
    store = _make_session(tmp_path)

    with pytest.raises(ValueError, match="provider/model"):
        change_session_provider(store, "sess-1", "claude-opus-4-8")

    with pytest.raises(ValueError, match="provider/model"):
        parse_provider_model("gpt-5")


def test_change_provider_strip_helper_handles_string_content():
    transcript = [
        {"role": "assistant", "content": "plain string, nothing to strip"},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "x"}],
        },
    ]
    stripped, count = strip_thinking_blocks(transcript)
    assert count == 1
    assert stripped[0]["content"] == "plain string, nothing to strip"
    assert stripped[1]["content"] == []


def test_change_provider_metadata_passes_phase1_mismatch_check(tmp_path):
    """After migration, resuming under the new provider yields no mismatch warning."""
    from amplifier_app_cli.commands.run import build_resume_mismatch_warning

    store = _make_session(tmp_path)
    change_session_provider(store, "sess-1", "openai/gpt-5")

    metadata = store.get_metadata("sess-1")
    config_data = {
        "providers": [
            {"module": "provider-openai", "config": {"model": "gpt-5", "priority": 1}}
        ]
    }
    assert (
        build_resume_mismatch_warning(metadata["model"], config_data, "sess-1") is None
    )
