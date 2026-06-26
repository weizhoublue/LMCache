# SPDX-License-Identifier: Apache-2.0

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
import logging

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

_ADAPTER_LOGGER_NAME = "lmcache.integration.vllm.vllm_v1_adapter"


@contextmanager
def _capture_adapter_warnings() -> Iterator[list[logging.LogRecord]]:
    """Capture WARNING records emitted by the adapter logger."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    logger = logging.getLogger(_ADAPTER_LOGGER_NAME)
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def _make_connector(
    *,
    engine: object = None,
    lookup_client: object = None,
    kv_role: str = "kv_both",
) -> LMCacheConnectorV1Impl:
    """Create a mock LMCacheConnectorV1Impl instance with mocked dependencies.

    Args:
        engine: The mock engine.
        lookup_client: The mock lookup client.
        kv_role: The role of the KV cache.

    Returns:
        The mock connector instance.
    """
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine,
        lookup_client=lookup_client,
    )
    connector.kv_role = kv_role
    connector.config = SimpleNamespace(
        get_extra_config_value=lambda key, default: default
    )
    connector.load_specs = {}
    connector._unfinished_requests = {}
    return connector


def test_get_num_new_matched_tokens_lookup_client_none() -> None:
    """Test get_num_new_matched_tokens returns 0 and logs warning when lookup_client is None."""
    connector = _make_connector(lookup_client=None, kv_role="kv_both")
    request = SimpleNamespace(request_id="req-test-matched-tokens")

    with _capture_adapter_warnings() as records:
        res = connector.get_num_new_matched_tokens(
            request=request,  # type: ignore[arg-type]
            num_computed_tokens=0,
        )

    assert res == 0
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert any(
        "req-test-matched-tokens" in r.getMessage()
        and "lookup_client is None" in r.getMessage()
        for r in warnings
    ), f"Expected warning about lookup_client is None, got: {warnings}"


def test_update_state_after_alloc_lookup_client_none() -> None:
    """Test update_state_after_alloc logs warning and returns when lookup_client is None."""
    connector = _make_connector(lookup_client=None, kv_role="kv_both")
    request = SimpleNamespace(request_id="req-test-state-after-alloc")

    with _capture_adapter_warnings() as records:
        connector.update_state_after_alloc(
            request=request,  # type: ignore[arg-type]
            num_external_tokens=0,
        )

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert any(
        "req-test-state-after-alloc" in r.getMessage()
        and "lookup_client is None" in r.getMessage()
        for r in warnings
    ), f"Expected warning about lookup_client is None, got: {warnings}"
