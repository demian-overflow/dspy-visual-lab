import logging

from config.settings import settings

logger = logging.getLogger(__name__)

_enabled = False
_client = None


def setup() -> None:
    global _enabled, _client
    if _enabled:
        return

    if not (settings.langfuse_public_key and settings.langfuse_secret_key and settings.langfuse_host):
        logger.info("Langfuse disabled (LANGFUSE_* not set) — LLM tracing is a no-op")
        return

    try:
        from langfuse import Langfuse
        from openinference.instrumentation.dspy import DSPyInstrumentor

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        DSPyInstrumentor().instrument()
        _enabled = True
        logger.info("Langfuse observability enabled (host=%s)", settings.langfuse_host)
    except Exception:
        logger.exception("Langfuse init failed — continuing without LLM tracing")
        _client = None
        _enabled = False


def is_enabled() -> bool:
    return _enabled


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)
