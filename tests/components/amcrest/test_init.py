"""Test the Amcrest integration init."""

import asyncio
import logging
import threading
import time
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from amcrest import AmcrestError
import pytest

from homeassistant.components import amcrest
from homeassistant.components.amcrest.const import DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import (
    CONF_HOST,
    CONF_NAME,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_USERNAME,
)
from homeassistant.core import HomeAssistant

from .conftest import SERIAL_FROM_FLOW

from tests.common import MockConfigEntry


@pytest.mark.usefixtures("mock_patch_platforms")
async def test_setup_entry_uses_unique_id_for_identifiers_when_serial_fetch_fails(
    hass: HomeAssistant,
    mock_amcrest_init_entry: MockConfigEntry,
) -> None:
    """Test config-entry setup uses entry.unique_id even if device serial fetch fails."""
    entry = mock_amcrest_init_entry

    api = MagicMock()

    async def _raise_serial():
        raise AmcrestError

    api.async_serial_number = _raise_serial()
    api.get_base_url.return_value = "http://1.2.3.4"

    async_forward = AsyncMock()

    with (
        patch("homeassistant.components.amcrest.AmcrestChecker", return_value=api),
        patch("homeassistant.components.amcrest.dr.async_get") as mock_async_get,
        patch("homeassistant.components.amcrest.DeviceInfo") as mock_device_info,
        patch.object(hass.config_entries, "async_forward_entry_setups", async_forward),
    ):
        device_registry = MagicMock()
        mock_async_get.return_value = device_registry

        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED

    # Device registry should be keyed by the stable unique_id, not entry_id.
    device_registry.async_get_or_create.assert_called_once()
    identifiers = device_registry.async_get_or_create.call_args.kwargs["identifiers"]
    assert identifiers == {(DOMAIN, SERIAL_FROM_FLOW)}

    # DeviceInfo should also use the stable unique_id.
    assert mock_device_info.call_args.kwargs["identifiers"] == {
        (DOMAIN, SERIAL_FROM_FLOW)
    }


@pytest.mark.usefixtures("mock_patch_platforms")
async def test_setup_entry_requires_unique_id(hass: HomeAssistant) -> None:
    """Test config-entry setup fails when entry.unique_id is missing."""
    entry = MockConfigEntry(
        title="Amcrest Camera",
        domain=DOMAIN,
        unique_id=None,
        data={
            CONF_HOST: "1.2.3.4",
            CONF_PORT: 80,
            CONF_USERNAME: "user",
            CONF_PASSWORD: "pass",
            CONF_NAME: "Amcrest Camera",
        },
    )
    entry.add_to_hass(hass)

    api = MagicMock()
    api.get_base_url.return_value = "http://1.2.3.4"

    with patch("homeassistant.components.amcrest.AmcrestChecker", return_value=api):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.SETUP_ERROR


@pytest.mark.usefixtures("mock_patch_platforms")
async def test_setup_entry_starts_event_monitor_after_forward_entry_setups(
    hass: HomeAssistant,
    mock_amcrest_init_entry: MockConfigEntry,
    mock_amcrest_checker_serial_ok_api: MagicMock,
) -> None:
    """Event monitor starts only after platforms are forwarded."""
    entry = mock_amcrest_init_entry
    api = mock_amcrest_checker_serial_ok_api

    call_order: list[str] = []

    async def track_forward(_entry: object, _platforms: object) -> None:
        call_order.append("forward")

    mock_thread = MagicMock(spec=threading.Thread)

    def track_start(*_args: object, **_kwargs: object) -> MagicMock:
        call_order.append("monitor")
        return mock_thread

    with (
        patch("homeassistant.components.amcrest.AmcrestChecker", return_value=api),
        patch("homeassistant.components.amcrest.dr.async_get") as mock_async_get,
        patch("homeassistant.components.amcrest.DeviceInfo"),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(side_effect=track_forward),
        ),
        patch.object(amcrest, "_start_event_monitor", side_effect=track_start),
    ):
        mock_async_get.return_value = MagicMock()
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert call_order == ["forward", "monitor"]


@pytest.mark.usefixtures("mock_patch_platforms")
async def test_unload_entry_stops_event_monitor_and_joins_thread(
    hass: HomeAssistant,
    mock_amcrest_init_entry: MockConfigEntry,
    mock_amcrest_checker_serial_ok_api: MagicMock,
) -> None:
    """Unload signals the monitor thread and waits for it via executor join."""

    def wait_for_stop(
        _hass_inner: HomeAssistant,
        _name: str,
        _api: MagicMock,
        _event_codes: set[str],
        stop_event: threading.Event | None,
    ) -> None:
        assert stop_event is not None
        stop_event.wait(timeout=120)

    entry = mock_amcrest_init_entry
    api = mock_amcrest_checker_serial_ok_api

    async_forward = AsyncMock()

    with (
        patch("homeassistant.components.amcrest.AmcrestChecker", return_value=api),
        patch("homeassistant.components.amcrest.dr.async_get") as mock_async_get,
        patch("homeassistant.components.amcrest.DeviceInfo"),
        patch.object(hass.config_entries, "async_forward_entry_setups", async_forward),
        patch.object(amcrest, "_monitor_events", side_effect=wait_for_stop),
    ):
        mock_async_get.return_value = MagicMock()

        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        runtime_data = cast(amcrest.AmcrestConfigEntryData, entry.runtime_data)
        monitor_thread = runtime_data["event_monitor_thread"]
        assert monitor_thread.is_alive()

        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.NOT_LOADED
    assert not monitor_thread.is_alive()


@pytest.mark.usefixtures("mock_patch_platforms")
async def test_unload_entry_warns_when_event_monitor_join_times_out(
    hass: HomeAssistant,
    mock_amcrest_init_entry: MockConfigEntry,
    mock_amcrest_checker_serial_ok_api: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Join uses a timeout and warns if the monitor thread keeps running."""
    monkeypatch.setattr(amcrest, "MONITOR_THREAD_JOIN_TIMEOUT", 0.05)

    def delay_then_observe_stop(
        _hass_inner: HomeAssistant,
        _name: str,
        _api: MagicMock,
        _event_codes: set[str],
        stop_event: threading.Event | None,
    ) -> None:
        assert stop_event is not None
        time.sleep(0.3)
        stop_event.wait(timeout=30)

    entry = mock_amcrest_init_entry
    api = mock_amcrest_checker_serial_ok_api
    async_forward = AsyncMock()

    caplog.clear()
    with (
        caplog.at_level(logging.WARNING, logger="homeassistant.components.amcrest"),
        patch("homeassistant.components.amcrest.AmcrestChecker", return_value=api),
        patch("homeassistant.components.amcrest.dr.async_get") as mock_async_get,
        patch("homeassistant.components.amcrest.DeviceInfo"),
        patch.object(hass.config_entries, "async_forward_entry_setups", async_forward),
        patch.object(amcrest, "_monitor_events", side_effect=delay_then_observe_stop),
    ):
        mock_async_get.return_value = MagicMock()

        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.NOT_LOADED
    assert any(
        "event monitor thread did not exit within" in record.message
        for record in caplog.records
    )

    await asyncio.sleep(0.5)
