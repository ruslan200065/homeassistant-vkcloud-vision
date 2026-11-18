import asyncio
import shutil
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant import loader
from homeassistant.bootstrap import async_load_base_functionality
from homeassistant.config_entries import ConfigEntries, ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from PIL import Image

from custom_components.vkcloud_vision import async_setup
from custom_components.vkcloud_vision.api.vkcloud.auth import VKCloudOAuthAuth
from custom_components.vkcloud_vision.api.vkcloud.exceptions import (
    VKCloudVisionAPIError,
    VKCloudVisionDetectionError,
)
from custom_components.vkcloud_vision.api.vkcloud.vision.base_client import (
    VKCloudVisionBaseClient,
)
from custom_components.vkcloud_vision.api.vkcloud.vision.response import (
    VKCloudVisionObjectDetectionResponse,
)
from custom_components.vkcloud_vision.bounding_boxes import BoundingBoxes
from custom_components.vkcloud_vision.const import DOMAIN, BoundingBoxesType
from custom_components.vkcloud_vision.diagnostics import (
    async_get_config_entry_diagnostics,
)


def reply(payload, status=200):
    @asynccontextmanager
    async def response(*args, **kwargs):
        yield SimpleNamespace(status=status, json=AsyncMock(return_value=payload))

    return response


@pytest.fixture
def session():
    return Mock()


@pytest.fixture
def client(monkeypatch, session):
    monkeypatch.setattr(
        "custom_components.vkcloud_vision.api.vkcloud.vision.base_client.async_get_clientsession",
        lambda hass: session,
    )
    return VKCloudVisionBaseClient(
        None, Mock(get_access_token=AsyncMock(return_value="token"))
    )


@pytest.mark.asyncio
async def test_partial_detection_and_exact_snapshot(client, session):
    expected = {"prob": 0.9, "eng": "dog"}
    body = {
        "object": [{"name": "door_1", "status": 1}],
        "multiobject": [
            {"name": "door_2", "status": 0, "labels": [expected, {"prob": 0.1}]},
            {"name": "door_1", "status": 0, "labels": []},
        ],
    }
    session.post.side_effect = reply({"status": 200, "body": body})
    result = VKCloudVisionObjectDetectionResponse(
        await client._make_request("/v1/objects/detect", {}), prob_threshold=0.5
    )
    assert result.has_errors
    assert "door_1" in result.error_message
    assert result.labels_for_image("door_1") == []
    assert result.labels_for_image("door_2") == [expected]
    session.post.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"objects": []}])
async def test_empty_results_are_not_indexed(client, session, body):
    session.post.side_effect = reply({"status": 200, "body": body})
    assert await client._make_request("/endpoint", {}) == body


@pytest.mark.asyncio
async def test_temporary_failures_do_not_sleep_after_final_attempt(
    client, session, monkeypatch
):
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    session.post.side_effect = reply(None, status=503)
    with pytest.raises(VKCloudVisionAPIError, match="2 attempt"):
        await client._make_request("/endpoint", {}, max_retries=2)
    assert session.post.call_count == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried(client, session):
    session.post.side_effect = reply(
        {"status": 200, "body": {"object": [{"name": "door", "status": 1}]}}
    )
    with pytest.raises(VKCloudVisionDetectionError, match="permanent error"):
        await client._make_request("/endpoint", {})
    session.post.assert_called_once()


@pytest.mark.asyncio
async def test_oauth_exchange_is_serialized_and_persists_rotation(monkeypatch, session):
    monkeypatch.setattr(
        "custom_components.vkcloud_vision.api.vkcloud.auth.async_get_clientsession",
        lambda hass: session,
    )
    session.post.side_effect = reply(
        {"access_token": "new-access", "refresh_token": "rotated", "expires_in": 600}
    )
    store = Mock()
    auth = VKCloudOAuthAuth(
        None, "client", refresh_token="previous", on_refresh_token=store
    )
    assert await asyncio.gather(auth.get_access_token(), auth.get_access_token()) == [
        "new-access",
        "new-access",
    ]
    session.post.assert_called_once()
    store.assert_called_once_with("rotated")
    assert auth.get_refresh_token() == "rotated"


@pytest.fixture
def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_atomic_image_output_and_path_validation(tmp_path, image_bytes, monkeypatch):
    boxes = BoundingBoxes(
        image_bytes, [{"coord": [20, 20, 5, 5]}], BoundingBoxesType.NO_LABELS
    )
    output = tmp_path / "frame.png"
    with pytest.raises(HomeAssistantError, match="allowed"):
        boxes.save_image(str(output), lambda path: False)
    assert not output.exists()

    assert boxes.save_image(str(output), lambda path: True) == str(output)
    assert output.read_bytes() == image_bytes

    def broken_save(self, stream, **kwargs):
        stream.write(b"incomplete")
        raise OSError("disk failure")

    monkeypatch.setattr(Image.Image, "save", broken_save)
    with pytest.raises(OSError, match="disk failure"):
        boxes.save_image(str(output), lambda path: True)
    assert output.read_bytes() == image_bytes
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.asyncio
async def test_home_assistant_action_schema_and_camera_paths(monkeypatch, tmp_path):
    hass = HomeAssistant(str(tmp_path))
    hass.config_entries = Mock(
        async_loaded_entries=Mock(return_value=[SimpleNamespace(options={})])
    )
    detect_objects = AsyncMock(return_value={"response_type": "action_done"})
    monkeypatch.setattr(
        "custom_components.vkcloud_vision.get_vision_entity",
        lambda hass: SimpleNamespace(
            async_detect_objects=detect_objects, record_action=Mock()
        ),
    )
    await async_setup(hass, {})
    data = {
        "entity_id": ["camera.yard", "camera.door"],
        "file_out": "/config/www/{camera}.jpg",
        "num_snapshots": 2,
        "snapshot_index": 2,
        "object_classes": [" Person "],
        "region": {"left": 1, "top": 2, "right": 3, "bottom": 4},
    }
    result = await hass.services.async_call(
        DOMAIN, "detect_objects", data, blocking=True, return_response=True
    )
    assert set(result) == {"camera.yard", "camera.door"}
    calls = detect_objects.await_args_list
    assert [call.kwargs["file_out"] for call in calls] == [
        "/config/www/door.jpg",
        "/config/www/yard.jpg",
    ]
    assert calls[0].kwargs["snapshot_index"] == 2
    assert calls[0].kwargs["object_classes"] == ["Person"]
    assert calls[0].kwargs["region"] == (1, 2, 3, 4)
    assert calls[0].kwargs["modes"] == ["multiobject"]

    with pytest.raises(HomeAssistantError, match="must not exceed"):
        await hass.services.async_call(
            DOMAIN,
            "detect_objects",
            {**data, "snapshot_index": 3},
            blocking=True,
            return_response=True,
        )

    with pytest.raises(HomeAssistantError, match="multiple cameras"):
        await hass.services.async_call(
            DOMAIN,
            "detect_objects",
            {**data, "file_out": "/config/www/frame.jpg"},
            blocking=True,
            return_response=True,
        )

    hass.config_entries.async_loaded_entries.return_value = []
    with pytest.raises(HomeAssistantError, match="not configured"):
        await hass.services.async_call(
            DOMAIN, "detect_objects", data, blocking=True, return_response=True
        )


@pytest.mark.asyncio
async def test_entry_reload_restores_image_processing_entity(
    tmp_path, monkeypatch, image_bytes
):
    hass = HomeAssistant(str(tmp_path))
    hass.config_entries = ConfigEntries(hass, {})
    component = Path(__file__).parents[1] / "custom_components" / DOMAIN
    shutil.copytree(component, tmp_path / "custom_components" / DOMAIN)
    # The test uses the real platform and entry manager, with camera and upload
    # dependencies marked as loaded because no camera or upload is needed.
    monkeypatch.setattr(
        "custom_components.vkcloud_vision.api.vkcloud.vision.base_client.async_get_clientsession",
        lambda hass: Mock(post=reply({"status": 200, "body": {"multiobject": []}})),
    )
    monkeypatch.setattr(
        "custom_components.vkcloud_vision.image_processing.async_get_image",
        AsyncMock(return_value=SimpleNamespace(content=image_bytes)),
    )
    loader.async_setup(hass)
    hass.config.skip_pip = True
    await async_load_base_functionality(hass)
    hass.config.components.update({"camera", "file_upload"})
    entry = ConfigEntry(
        version=4,
        minor_version=1,
        domain=DOMAIN,
        title="VK Cloud Vision",
        data={"api_key": "test"},
        source="user",
        options={},
        unique_id=None,
        discovery_keys={},
        subentries_data=(),
    )
    entity_id = "image_processing.vkcloud_vision"
    try:
        await hass.config_entries.async_add(entry)
        await hass.async_block_till_done()
        assert entry.state is ConfigEntryState.LOADED
        assert hass.states.get(entity_id) is not None

        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        assert hass.states.get(entity_id).state == "unavailable"
        with pytest.raises(HomeAssistantError, match="not configured"):
            await hass.services.async_call(
                DOMAIN,
                "detect_objects",
                {"entity_id": "camera.door"},
                blocking=True,
                return_response=True,
            )

        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        clock = Mock(return_value=100.0)
        monkeypatch.setattr(
            "custom_components.vkcloud_vision.image_processing.monotonic", clock
        )

        async def detect(camera):
            return (
                await hass.services.async_call(
                    DOMAIN,
                    "detect_objects",
                    {"entity_id": camera, "cooldown_sec": 30},
                    blocking=True,
                    return_response=True,
                )
            )[camera]

        first, second = await asyncio.gather(
            detect("camera.door"), detect("camera.door")
        )
        assert {first["response_type"], second["response_type"]} == {
            "action_done",
            "skipped",
        }
        assert (await detect("camera.yard"))["response_type"] == "action_done"
        clock.return_value = 130.0
        assert (await detect("camera.door"))["response_type"] == "action_done"
        assert "T" in hass.states.get(entity_id).state

        diagnostics = await async_get_config_entry_diagnostics(hass, entry)
        assert diagnostics["auth_method"] == "api_key"
        stats = next(
            item
            for item in diagnostics["actions_since_load"]
            if item["camera"] == "camera.door"
        )
        assert stats["action"] == "detect_objects"
        assert stats["counts"] == {"action_done": 2, "skipped": 1}
        assert stats["last_success_at"] == stats["last_completed_at"]
        assert "test" not in str(diagnostics)
    finally:
        await hass.async_stop(force=True)


def test_object_class_filter_applies_to_response_and_annotations():
    label = {"eng": "cat", "rus": "кошка", "prob": 0.9}
    body = {
        "multiobject": [
            {
                "name": "door_1",
                "labels": [
                    label,
                    {"eng": "person", "prob": 0.9},
                    {"eng": "cat", "prob": 0.1},
                ],
            }
        ]
    }
    result = VKCloudVisionObjectDetectionResponse(body, 0.5, [" КОШКА "])
    assert result.data["multiobject"][0]["labels"] == [label]
    assert result.labels_for_image("door_1") == [label]
    assert len(body["multiobject"][0]["labels"]) == 3


def test_object_region_filters_by_box_center():
    inside = {"eng": "person", "prob": 0.9, "coord": [0, 0, 20, 20]}
    outside = {"eng": "person", "prob": 0.9, "coord": [5, 6, 11, 12]}
    no_box = {"eng": "person", "prob": 0.9}
    inverted = {"eng": "person", "prob": 0.9, "coord": [15, 5, 5, 15]}
    response = VKCloudVisionObjectDetectionResponse(
        {
            "multiobject": [
                {"name": "door_1", "labels": [inside, outside, no_box, inverted]}
            ]
        },
        region=(10, 10, 30, 30),
    )
    assert response.labels_for_image("door_1") == [inside]
