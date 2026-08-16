"""Tests for role-specific generator/judge provider selection (Checkpoint
Phase 3: direct Qwen generator integration). No live network calls --
credential presence is faked via environment variables only, and
`build_retrieval_context`/checkpoint I/O are monkeypatched to explode if
ever reached, proving the fail-closed ordering."""

from __future__ import annotations

import pytest

from rag import llm_providers, run_generation_eval
from rag.llm_providers import GroqProvider, QwenProvider, RequiredProviderUnavailable


def _clear_all_provider_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "QWEN_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_BASE_URL",
        "GEN_EVAL_GENERATOR_PROVIDER",
        "GEN_EVAL_JUDGE_PROVIDER",
        "GEN_EVAL_REQUIRE_PROVIDER",
        "GEN_EVAL_REQUIRE_GENERATOR_PROVIDER",
        "GEN_EVAL_REQUIRE_JUDGE_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_providers, "_ollama_reachable", lambda: None)


def test_qwen_generator_and_groq_judge_are_selected_independently(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example-workspace.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    monkeypatch.setenv("GEN_EVAL_JUDGE_PROVIDER", "groq")

    detection = llm_providers.detect_available_providers()
    generator, judge = run_generation_eval._select_evaluation_providers(detection)

    assert isinstance(generator, QwenProvider)
    assert isinstance(judge, GroqProvider)
    assert generator.name == "qwen"
    assert judge.name == "groq"
    assert generator.model != judge.model


def test_qwen_key_present_does_not_make_the_judge_become_qwen(monkeypatch):
    """A Qwen key must never, by itself, cause the judge role to resolve
    to Qwen -- only an explicit GEN_EVAL_JUDGE_PROVIDER=qwen could."""
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example-workspace.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    # No GEN_EVAL_JUDGE_PROVIDER override, no other hosted key, no Ollama.

    detection = llm_providers.detect_available_providers()
    generator, judge = run_generation_eval._select_evaluation_providers(detection)

    assert isinstance(generator, QwenProvider)
    assert judge is None  # legacy fallback: no hosted key detected (qwen is never auto-detected), no Ollama


def test_groq_key_present_does_not_make_the_generator_become_groq_when_qwen_is_required(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example-workspace.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    monkeypatch.setenv("GEN_EVAL_JUDGE_PROVIDER", "groq")

    detection = llm_providers.detect_available_providers()
    generator, judge = run_generation_eval._select_evaluation_providers(detection)

    assert isinstance(generator, QwenProvider)  # never Groq, despite GROQ_API_KEY being present
    assert isinstance(judge, GroqProvider)


def test_missing_qwen_generator_credential_raises_before_construction(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    # QWEN_API_KEY / DASHSCOPE_API_KEY intentionally absent.

    detection = llm_providers.detect_available_providers()
    with pytest.raises(RequiredProviderUnavailable, match="GEN_EVAL_GENERATOR_PROVIDER"):
        run_generation_eval._select_evaluation_providers(detection)


def test_neither_role_override_set_preserves_legacy_groq_both_roles_behavior(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")

    detection = llm_providers.detect_available_providers()
    generator, judge = run_generation_eval._select_evaluation_providers(detection)

    assert isinstance(generator, GroqProvider)
    assert isinstance(judge, GroqProvider)
    assert generator.model == llm_providers.GROQ_GENERATOR_MODEL
    assert judge.model == llm_providers.GROQ_JUDGE_MODEL


def test_enforce_required_generator_provider_passes_when_resolved_correctly(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example-workspace.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    monkeypatch.setenv("GEN_EVAL_REQUIRE_GENERATOR_PROVIDER", "qwen")

    detection = llm_providers.detect_available_providers()
    run_generation_eval.enforce_required_role_provider("generator", detection)  # must not raise


def test_enforce_required_judge_provider_fails_when_only_generator_is_qwen(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example-workspace.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    monkeypatch.setenv("GEN_EVAL_REQUIRE_JUDGE_PROVIDER", "groq")
    # No GROQ_API_KEY at all -- judge cannot legitimately resolve to groq.

    detection = llm_providers.detect_available_providers()
    with pytest.raises(RequiredProviderUnavailable, match="GEN_EVAL_REQUIRE_JUDGE_PROVIDER"):
        run_generation_eval.enforce_required_role_provider("judge", detection)


def test_qwen_can_never_accidentally_satisfy_a_require_role_provider_check(monkeypatch):
    """Qwen is never auto-detected (see detect_available_providers), so
    GEN_EVAL_REQUIRE_GENERATOR_PROVIDER=qwen can only be satisfied by an
    explicit GEN_EVAL_GENERATOR_PROVIDER=qwen -- never by accident."""
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example-workspace.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("GEN_EVAL_REQUIRE_GENERATOR_PROVIDER", "qwen")
    # No GEN_EVAL_GENERATOR_PROVIDER override.

    detection = llm_providers.detect_available_providers()
    assert detection.provider_name is None  # Qwen key alone never registers here
    with pytest.raises(RequiredProviderUnavailable):
        run_generation_eval.enforce_required_role_provider("generator", detection)


def test_no_secret_value_is_logged_in_role_selection_errors(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    # No credential present at all.

    detection = llm_providers.detect_available_providers()
    with pytest.raises(RequiredProviderUnavailable) as exc_info:
        run_generation_eval._select_evaluation_providers(detection)

    message = str(exc_info.value)
    assert "QWEN_API_KEY" not in message or "=" not in message.split("QWEN_API_KEY")[-1][:3]
    # No dataclass/field dump, no bearer/authorization text.
    assert "Bearer" not in message
    assert "Authorization" not in message


def test_swapping_role_environment_variables_changes_the_evaluation_fingerprint():
    common_kwargs = dict(
        winner_config="B",
        winner_mode="dense",
        collection_fingerprint="fp-fixed",
    )
    fingerprint_qwen_generator = run_generation_eval._evaluation_fingerprint(
        generator_name="qwen",
        generator_model="qwen3.7-flash",
        judge_name="groq",
        judge_model="openai/gpt-oss-20b",
        **common_kwargs,
    )
    fingerprint_swapped = run_generation_eval._evaluation_fingerprint(
        generator_name="groq",
        generator_model="openai/gpt-oss-20b",
        judge_name="qwen",
        judge_model="qwen3.7-flash",
        **common_kwargs,
    )
    fingerprint_same_as_first = run_generation_eval._evaluation_fingerprint(
        generator_name="qwen",
        generator_model="qwen3.7-flash",
        judge_name="groq",
        judge_model="openai/gpt-oss-20b",
        **common_kwargs,
    )

    assert fingerprint_qwen_generator != fingerprint_swapped
    assert fingerprint_qwen_generator == fingerprint_same_as_first


def test_missing_required_qwen_generator_fails_before_ingestion_or_checkpoint_write(monkeypatch):
    _clear_all_provider_env(monkeypatch)
    monkeypatch.setenv("GEN_EVAL_GENERATOR_PROVIDER", "qwen")
    # QWEN_API_KEY intentionally absent -> must fail before any setup.

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("build_retrieval_context must never be called when a required role credential is missing")

    monkeypatch.setattr(run_generation_eval, "build_retrieval_context", _must_not_be_called)

    def _checkpoint_must_not_be_written(*args, **kwargs):
        raise AssertionError("checkpoint must never be written when a required role credential is missing")

    monkeypatch.setattr(run_generation_eval, "_write_checkpoint_atomic", _checkpoint_must_not_be_written)

    with pytest.raises(RequiredProviderUnavailable):
        run_generation_eval.run()


def test_report_construction_includes_provider_fields_and_never_a_secret_field():
    """Requirement 9/10: the persisted report separates generator/judge
    provider name from model, and never carries a base_url/workspace/key
    field. Inspects run_generation_eval.run's report dict literal
    directly -- a real end-to-end run needs live ingestion/embeddings,
    which this checkpoint must not perform."""
    import inspect

    source = inspect.getsource(run_generation_eval.run)
    report_start = source.index("report = {")
    report_block = source[report_start : source.index("}", source.index('"rows"', report_start)) + 1]

    for required_field in ('"generator_provider"', '"generator_model"', '"judge_provider"', '"judge_model"'):
        assert required_field in report_block
    for forbidden_field in ("base_url", "api_key", "workspace", "authorization", "Authorization"):
        assert forbidden_field not in report_block
