from unittest.mock import MagicMock, patch

from config.settings import settings
from observability import langfuse_tracing


def test_setup_is_noop_when_keys_absent(monkeypatch):
    langfuse_tracing._enabled = False
    langfuse_tracing._client = None
    monkeypatch.setattr(settings, "langfuse_public_key", None)
    monkeypatch.setattr(settings, "langfuse_secret_key", None)

    langfuse_tracing.setup()

    assert langfuse_tracing.is_enabled() is False


def test_setup_enables_when_keys_present(monkeypatch):
    langfuse_tracing._enabled = False
    langfuse_tracing._client = None
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    monkeypatch.setattr(settings, "langfuse_host", "https://cloud.langfuse.com")

    with patch("langfuse.Langfuse") as mock_langfuse_cls, \
         patch("openinference.instrumentation.dspy.DSPyInstrumentor") as mock_instrumentor_cls:
        mock_langfuse_cls.return_value = MagicMock()
        mock_instrumentor_cls.return_value = MagicMock()

        langfuse_tracing.setup()

        assert langfuse_tracing.is_enabled() is True
        mock_instrumentor_cls.return_value.instrument.assert_called_once()

    langfuse_tracing._enabled = False
    langfuse_tracing._client = None


def test_setup_never_raises_on_sdk_failure(monkeypatch):
    langfuse_tracing._enabled = False
    langfuse_tracing._client = None
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")

    with patch("langfuse.Langfuse", side_effect=RuntimeError("boom")):
        langfuse_tracing.setup()  # must not raise

    assert langfuse_tracing.is_enabled() is False
