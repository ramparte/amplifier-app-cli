"""Primary run command for the Amplifier CLI."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any

import click

if TYPE_CHECKING:
    pass

from rich.panel import Panel

from amplifier_foundation.exceptions import BundleError, BundleValidationError

from ..console import console
from ..session_store import extract_session_mode
from ..effective_config import get_effective_config_summary
from ..lib.settings import AppSettings
from ..paths import create_config_manager
from ..runtime.config import resolve_config
from ..types import (
    ExecuteSingleProtocol,
    InteractiveChatProtocol,
    SearchPathProviderProtocol,
)

logger = logging.getLogger(__name__)


def _split_saved_model(saved_model: str) -> tuple[str | None, str]:
    """Split a persisted metadata model value into (provider, model_name).

    Saved values come in three shapes:
        "anthropic/claude-opus-4-6" -> ("anthropic", "claude-opus-4-6")
            (mount-name form, written by main.py's final save)
        "provider-anthropic/claude-opus-4-6" -> ("provider-anthropic", "claude-opus-4-6")
        "claude-opus-4-6" -> (None, "claude-opus-4-6")
            (bare model, written by incremental saves)
    """
    if "/" in saved_model:
        provider_part, model_part = saved_model.split("/", 1)
        return provider_part, model_part
    return None, saved_model


def _normalize_provider(provider: str) -> str:
    """Normalize a provider identifier for comparison.

    Persisted metadata records the provider as a mount name ("anthropic")
    while resolved config uses module ids ("provider-anthropic"). Both
    normalize to "anthropic" so same-provider resumes compare equal.
    """
    return provider.removeprefix("provider-")


def _resolve_active_provider(config_data: dict) -> tuple[str | None, str | None]:
    """Return (provider_module, model_name) for the highest-precedence provider.

    The active provider is the entry with the lowest ``priority`` value
    (default 100) in config_data["providers"] — mirrors the CLI override logic.
    """
    providers_list = config_data.get("providers", [])
    best_entry: dict | None = None
    best_priority = float("inf")
    for entry in providers_list:
        if not isinstance(entry, dict):
            continue
        entry_config = entry.get("config")
        priority = (
            entry_config.get("priority", 100) if isinstance(entry_config, dict) else 100
        )
        if priority < best_priority:
            best_priority = priority
            best_entry = entry
    if best_entry is None:
        return None, None
    entry_config = best_entry.get("config")
    model_name = None
    if isinstance(entry_config, dict):
        model_name = entry_config.get("model") or entry_config.get("default_model")
    return best_entry.get("module"), model_name


def build_resume_mismatch_warning(
    saved_model: str | None, config_data: dict, session_id: str
) -> str | None:
    """Detect a cross-provider resume and build an actionable warning.

    Compares the session's persisted model (metadata.json "model") against the
    provider/model about to be used. Returns a warning message naming both
    models and the exact flags to resume with the original provider, or None
    when they match (or when there is not enough information to compare).

    This is warn-and-continue: callers must not block the resume.
    """
    if not saved_model or saved_model == "unknown":
        return None

    current_module, current_model = _resolve_active_provider(config_data)
    if current_module is None:
        return None

    saved_provider, saved_model_name = _split_saved_model(saved_model)

    if saved_provider is not None:
        # Provider recorded: warn only when resuming under a different provider.
        # Compare normalized names: metadata stores mount names ("anthropic")
        # while config uses module ids ("provider-anthropic").
        mismatch = _normalize_provider(saved_provider) != _normalize_provider(
            current_module
        )
    else:
        # Only the bare model name recorded: a different model is the best
        # available signal that a different provider may be in play
        mismatch = current_model is not None and saved_model_name != current_model
    if not mismatch:
        return None

    provider_flag = (saved_provider or "<original-provider>").removeprefix("provider-")
    current_display = (
        f"{current_module}/{current_model}" if current_model else current_module
    )
    return (
        "This session was created with a different provider/model.\n"
        f"  Session model:  {saved_model}\n"
        f"  About to use:   {current_display}\n"
        "\n"
        "Resuming under a different provider can fail with a cryptic\n"
        "provider 400 error (e.g. invalid thinking-block signatures).\n"
        "To resume with the original provider, run:\n"
        "\n"
        f"  amplifier run --resume {session_id} "
        f"--provider {provider_flag} --model {saved_model_name}"
    )


def register_run_command(
    cli: click.Group,
    *,
    interactive_chat: InteractiveChatProtocol,
    execute_single: ExecuteSingleProtocol,
    get_module_search_paths: SearchPathProviderProtocol,
    check_first_run: Callable[[], bool],
    prompt_first_run_init: Callable[[Any], bool],
):
    """Register the run command on the root CLI group."""

    @cli.command()
    @click.argument("prompt", required=False)
    @click.option("--bundle", "-B", help="Bundle to use for this session")
    @click.option("--provider", "-p", default=None, help="LLM provider to use")
    @click.option("--model", "-m", help="Model to use (provider-specific)")
    @click.option("--max-tokens", type=int, help="Maximum output tokens")
    @click.option(
        "--mode",
        type=click.Choice(["chat", "single"]),
        default="single",
        help="Execution mode",
    )
    @click.option("--resume", help="Resume specific session with new prompt")
    @click.option("--verbose", "-v", is_flag=True, help="Verbose output")
    @click.option(
        "--output-format",
        type=click.Choice(["text", "json", "json-trace"]),
        default="text",
        help="Output format: text (markdown), json (response only), json-trace (full execution detail)",
    )
    def run(
        prompt: str | None,
        bundle: str | None,
        provider: str,
        model: str | None,
        max_tokens: int | None,
        mode: str,
        resume: str | None,
        verbose: bool,
        output_format: str,
    ):
        """Execute a prompt or start an interactive session."""
        from ..session_store import SessionStore

        # Handle --resume flag
        if resume:
            store = SessionStore()
            try:
                resume = store.find_session(resume)
            except FileNotFoundError:
                console.print(f"[red]Error:[/red] No session found matching '{resume}'")
                sys.exit(1)
            except ValueError as e:
                from ..utils.error_format import format_error_message

                console.print(f"[red]Error:[/red] {format_error_message(e)}")
                sys.exit(1)

            try:
                transcript, metadata = store.load(resume)
                console.print(f"[green]✓[/green] Resuming session: {resume}")
                console.print(f"  Messages: {len(transcript)}")

                # Detect bundle from saved session
                if not bundle:
                    saved_bundle, _legacy = extract_session_mode(metadata)
                    if saved_bundle:
                        bundle = saved_bundle
                        console.print(f"  Using saved bundle: {bundle}")

            except Exception as exc:
                console.print(f"[red]Error loading session:[/red] {exc}")
                sys.exit(1)

            # Determine mode based on prompt presence
            if prompt is None and sys.stdin.isatty():
                # No prompt, no pipe → interactive mode
                mode = "chat"
            else:
                # Has prompt or piped input → single-shot mode
                if prompt is None:
                    prompt = sys.stdin.read()
                    if not prompt or not prompt.strip():
                        console.print(
                            "[red]Error:[/red] Prompt required when resuming in single mode"
                        )
                        sys.exit(1)
                mode = "single"
        else:
            transcript = None
            metadata = None

        config_manager = create_config_manager()

        # Check for active bundle from settings (via 'amplifier bundle use')
        # CLI --bundle flag takes precedence over settings
        if not bundle:
            bundle_settings = config_manager.get_merged_settings().get("bundle", {})
            if isinstance(bundle_settings, dict):
                bundle = bundle_settings.get("active")

        # Default to foundation bundle when no explicit bundle is configured
        if not bundle:
            bundle = "foundation"

        # Check if first run init is needed
        # This runs unconditionally - --provider just selects from configured providers,
        # it doesn't bypass the need for configuration
        if check_first_run():
            if sys.stdin.isatty():
                prompt_first_run_init(console)
            else:
                # Non-interactive context (CI, Docker, shadow env)
                # Auto-init from environment variables
                from .init import auto_init_from_env

                auto_init_from_env(console)

        # Agent loading is now handled via foundation's bundle.load_agent_metadata()
        app_settings = AppSettings()

        # Track configuration source for display (always bundle mode now)
        config_source_name = f"bundle:{bundle}"

        # Resolve configuration using unified function (single source of truth)
        try:
            config_data, prepared_bundle = resolve_config(
                bundle_name=bundle,
                app_settings=app_settings,
                console=console,
            )
        except FileNotFoundError as exc:
            # Bundle not found - display error gracefully without traceback
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)
        except BundleValidationError as exc:
            # Bundle validation failed (e.g., malformed YAML, missing required fields)
            console.print()
            console.print(
                Panel(
                    str(exc),
                    title="[bold white on red] Bundle Validation Error [/bold white on red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            sys.exit(1)
        except BundleError as exc:
            # General bundle error (loading, resolution, etc.)
            console.print()
            console.print(
                Panel(
                    str(exc),
                    title="[bold white on red] Bundle Error [/bold white on red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            sys.exit(1)

        search_paths = get_module_search_paths()

        # Handle provider/model CLI overrides
        if model and not provider:
            # Require --provider when using --model for clarity
            console.print(
                "[red]Error:[/red] --model requires --provider\n"
                "Specify which provider to use: --provider anthropic --model claude-opus-4-6\n"
                "Run 'amplifier provider use --help' for configuration options"
            )
            sys.exit(1)

        if provider:
            provider_module = (
                provider if provider.startswith("provider-") else f"provider-{provider}"
            )
            providers_list = config_data.get("providers", [])

            # Find the target provider
            target_idx = None
            for i, entry in enumerate(providers_list):
                if isinstance(entry, dict) and entry.get("module") == provider_module:
                    target_idx = i
                    break

            if target_idx is None:
                console.print(
                    f"[red]Error:[/red] Provider '{provider}' not configured\n"
                    f"Available providers: {', '.join(p.get('module', '?').replace('provider-', '') for p in providers_list if isinstance(p, dict))}\n"
                    f"Run 'amplifier provider use --help' for configuration options"
                )
                sys.exit(1)

            # Clone ALL providers (keep multi-provider setup intact)
            updated_providers: list[dict[str, Any]] = []
            for i, entry in enumerate(providers_list):
                entry_copy = {**entry}
                entry_copy["config"] = dict(entry.get("config") or {})

                if i == target_idx:
                    # Promote this provider to priority 0 (highest)
                    entry_copy["config"]["priority"] = 0

                    if model:
                        entry_copy["config"]["default_model"] = model
                    if max_tokens:
                        entry_copy["config"]["max_tokens"] = max_tokens

                updated_providers.append(entry_copy)

            config_data["providers"] = updated_providers

            # CRITICAL: Update the prepared bundle's mount plan with modified providers
            # The bundle was already prepared with original config, we need to update it
            if prepared_bundle and hasattr(prepared_bundle, "mount_plan"):
                prepared_bundle.mount_plan["providers"] = updated_providers

            # Hint orchestrator if it supports default provider configuration
            session_cfg = config_data.setdefault("session", {})
            orchestrator_cfg = session_cfg.get("orchestrator")
            if isinstance(orchestrator_cfg, dict):
                orchestrator_config = dict(orchestrator_cfg.get("config") or {})
                orchestrator_config["default_provider"] = provider_module
                orchestrator_cfg["config"] = orchestrator_config
            elif isinstance(orchestrator_cfg, str):
                # Convert shorthand into dict form with default provider hint
                # Preserve orchestrator_source when converting to dict format
                orchestrator_dict: dict[str, Any] = {
                    "module": orchestrator_cfg,
                    "config": {"default_provider": provider_module},
                }
                if "orchestrator_source" in session_cfg:
                    orchestrator_dict["source"] = session_cfg["orchestrator_source"]
                session_cfg["orchestrator"] = orchestrator_dict

            orchestrator_meta = config_data.setdefault("orchestrator", {})
            if isinstance(orchestrator_meta, dict):
                meta_config = dict(orchestrator_meta.get("config") or {})
                meta_config["default_provider"] = provider_module
                orchestrator_meta["config"] = meta_config
        elif max_tokens:
            # Allow --max-tokens without --provider (applies to priority provider)
            providers_list = config_data.get("providers", [])
            if not providers_list:
                console.print(
                    "[yellow]Warning:[/yellow] No providers configured; ignoring --max-tokens"
                )
            else:
                # Find provider with lowest priority number (highest precedence)
                min_priority = float("inf")
                target_idx = 0
                for i, entry in enumerate(providers_list):
                    if isinstance(entry, dict):
                        entry_config = entry.get("config", {})
                        priority = (
                            entry_config.get("priority", 100)
                            if isinstance(entry_config, dict)
                            else 100
                        )
                        if priority < min_priority:
                            min_priority = priority
                            target_idx = i

                updated_providers: list[dict[str, Any]] = []
                for i, entry in enumerate(providers_list):
                    entry_copy = {**entry}
                    if i == target_idx:
                        entry_copy["config"] = dict(entry.get("config") or {})
                        entry_copy["config"]["max_tokens"] = max_tokens
                    updated_providers.append(entry_copy)

                config_data["providers"] = updated_providers

                # CRITICAL: Update the prepared bundle's mount plan with modified providers
                if prepared_bundle and hasattr(prepared_bundle, "mount_plan"):
                    prepared_bundle.mount_plan["providers"] = updated_providers

        # Cross-provider resume check: warn (but never block) when the session
        # was produced by a different provider than the one about to be used.
        # Placed after CLI overrides so --provider/--model suppress the warning.
        if resume and metadata:
            mismatch_warning = build_resume_mismatch_warning(
                metadata.get("model"), config_data, resume
            )
            if mismatch_warning:
                console.print()
                console.print(
                    Panel(
                        mismatch_warning,
                        title="[bold black on yellow] Provider mismatch on resume [/bold black on yellow]",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )

        # Run update check (uses unified startup_checker with settings.yaml)
        from ..utils.startup_checker import check_and_notify

        asyncio.run(check_and_notify())

        if mode == "chat":
            # Interactive mode - supports optional initial_prompt for auto-execution
            # Check for piped input if no prompt provided
            initial_prompt = prompt
            if initial_prompt is None and not sys.stdin.isatty():
                initial_prompt = sys.stdin.read()
                if initial_prompt is not None and not initial_prompt.strip():
                    initial_prompt = None

            if resume:
                # Resume existing session (transcript loaded earlier)
                if transcript is None:
                    console.print("[red]Error:[/red] Failed to load session transcript")
                    sys.exit(1)
                # Display conversation history before resuming (reuse session.py's display)
                from .session import _display_session_history

                _display_session_history(transcript, metadata or {})
                asyncio.run(
                    interactive_chat(
                        config_data,
                        search_paths,
                        verbose,
                        session_id=resume,
                        bundle_name=config_source_name,
                        prepared_bundle=prepared_bundle,
                        initial_prompt=initial_prompt,
                        initial_transcript=transcript,
                    )
                )
            else:
                # New session - banner displayed by interactive_chat
                session_id = str(uuid.uuid4())
                asyncio.run(
                    interactive_chat(
                        config_data,
                        search_paths,
                        verbose,
                        session_id=session_id,
                        bundle_name=config_source_name,
                        prepared_bundle=prepared_bundle,
                        initial_prompt=initial_prompt,
                    )
                )
        else:
            # Single-shot mode
            if prompt is None:
                # Allow piping prompt content via stdin
                if not sys.stdin.isatty():
                    prompt = sys.stdin.read()
                    if prompt is not None and not prompt.strip():
                        prompt = None
                if prompt is None:
                    console.print("[red]Error:[/red] Prompt required in single mode")
                    sys.exit(1)

            # Always persist single-shot sessions
            if resume:
                # Resume existing session with context
                if transcript is None:
                    console.print("[red]Error:[/red] Failed to load session transcript")
                    sys.exit(1)
                asyncio.run(
                    execute_single(
                        prompt,
                        config_data,
                        search_paths,
                        verbose,
                        session_id=resume,
                        bundle_name=config_source_name,
                        output_format=output_format,
                        prepared_bundle=prepared_bundle,
                        initial_transcript=transcript,
                    )
                )
            else:
                # Create new session
                session_id = str(uuid.uuid4())
                if output_format == "text":
                    config_summary = get_effective_config_summary(
                        config_data, config_source_name
                    )
                    console.print(f"\n[dim]Session ID: {session_id}[/dim]")
                    console.print(f"[dim]{config_summary.format_banner_line()}[/dim]")
                asyncio.run(
                    execute_single(
                        prompt,
                        config_data,
                        search_paths,
                        verbose,
                        session_id=session_id,
                        bundle_name=config_source_name,
                        output_format=output_format,
                        prepared_bundle=prepared_bundle,
                    )
                )

    return run


__all__ = ["register_run_command"]
