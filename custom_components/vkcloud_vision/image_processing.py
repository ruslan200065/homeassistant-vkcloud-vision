# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""VK Cloud Vision image processing platform."""

from __future__ import annotations

import asyncio

from homeassistant.components.camera import async_get_image
from homeassistant.components.image_processing import DOMAIN as IMAGE_PROCESSING_DOMAIN
from homeassistant.components.image_processing import ImageProcessingEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType

from .api.vkcloud.exceptions import VKCloudVisionAPIError, VKCloudVisionAuthError
from .api.vkcloud.vision import VKCloudVision
from .bounding_boxes import BoundingBoxes
from .const import DOMAIN, LOGGER, BoundingBoxesType, ResponseType

DEFAULT_IMAGE_TIMEOUT = 10
MAX_IMAGE_RETRIES = 3
RETRY_IMAGE_DELAY = 1


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the image processing entity for the active config entry."""
    async_add_entities([VKCloudVisionEntity(entry)])


class VKCloudVisionEntity(ImageProcessingEntity):
    """VK Cloud Vision image processing entity."""

    entity_id = f"{IMAGE_PROCESSING_DOMAIN}.vkcloud_vision"
    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self._entry = entry
        self._client: VKCloudVision = entry.runtime_data
        self._attr_name = "VK Cloud Vision"
        self._attr_unique_id = "vkcloud_vision"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, self._attr_unique_id)},
            name="VK Cloud Vision",
            manufacturer="VK Cloud",
            model="Vision",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._last_detection = None

    @property
    def state(self) -> str | None:
        """Return the state of the entity."""
        return self._last_detection

    def process_image(self, _image: bytes) -> None:
        raise HomeAssistantError("Use `vkcloud_vision.detect_objects` instead")

    async def async_process_image(self, image: bytes) -> None:
        self.process_image(image)

    async def async_detect_objects(
        self,
        camera_id: str,
        modes: list[str],
        prob_threshold: float,
        file_out: str | None,
        bounding_boxes: str,
        num_snapshots: int,
        snapshot_interval_sec: float,
        max_retries: int,
        snapshot_index: int = 1,
    ) -> JsonObjectType:
        """Detect objects with optional bounding box drawing."""
        images_data = await self._async_get_images(
            camera_id, num_snapshots, snapshot_interval_sec
        )
        images_meta = [
            {"name": f"{split_entity_id(camera_id)[1]}_{i + 1}"}
            for i in range(num_snapshots)
        ]

        image = images_data[snapshot_index - 1]
        name = images_meta[snapshot_index - 1]["name"]
        output_path = None
        try:
            response = await self._client.objects.detect(
                files=images_data,
                modes=modes,
                images=images_meta,
                prob_threshold=prob_threshold,
                max_retries=max_retries,
            )
        except VKCloudVisionAPIError as err:
            return await self._failed_with_snapshot(err, image, file_out)

        if file_out:
            output_path = await self._save_image(
                image,
                response.labels_for_image(name),
                file_out,
                BoundingBoxesType(bounding_boxes),
            )

        self._last_detection = dt_util.utcnow().isoformat()
        self.async_write_ha_state()

        return {
            "response": response.data,
            "file_out": output_path,
            "response_type": ResponseType.PARTIAL_ACTION_DONE
            if response.has_errors
            else ResponseType.ACTION_DONE,
            "error": response.error_message,
        }

    async def recognize_text(
        self, camera_id: str, lang: str | None, max_retries: int
    ) -> JsonObjectType:
        """Recognize text in an image."""
        image_data = await self._async_get_image(camera_id)
        image_meta = {"name": split_entity_id(camera_id)[1]}

        try:
            response = await self._client.text.scene_text_recognize(
                files=[image_data],
                images=[image_meta],
                lang=lang,
                max_retries=max_retries,
            )
        except VKCloudVisionAPIError as err:
            raise self._action_error(err) from err

        self._last_detection = dt_util.utcnow().isoformat()
        self.async_write_ha_state()

        return {
            "response": response.data,
            "response_type": ResponseType.PARTIAL_ACTION_DONE
            if response.has_errors
            else ResponseType.ACTION_DONE,
            "error": response.error_message,
        }

    async def recognize_faces(
        self,
        camera_id: str,
        space: int,
        create_new: bool,
        update_embedding: bool,
        confidence_threshold: float,
        file_out: str | None,
        bounding_boxes: str,
        max_retries: int,
        tag_to_alias_map: dict | None = None,
    ) -> JsonObjectType:
        """Recognize faces in an image."""
        image_data = await self._async_get_image(camera_id)
        image_meta = {"name": split_entity_id(camera_id)[1]}

        output_path = None
        try:
            response = await self._client.persons.recognize(
                files=[image_data],
                space=space,
                images=[image_meta],
                create_new=create_new,
                update_embedding=update_embedding,
                confidence_threshold=confidence_threshold,
                tag_to_alias_map=tag_to_alias_map,
                max_retries=max_retries,
            )
        except VKCloudVisionAPIError as err:
            return await self._failed_with_snapshot(err, image_data, file_out)

        if file_out:
            output_path = await self._save_image(
                image_data,
                response.persons,
                file_out,
                BoundingBoxesType(bounding_boxes),
            )

        self._last_detection = dt_util.utcnow().isoformat()
        self.async_write_ha_state()

        return {
            "response": response.data,
            "file_out": output_path,
            "response_type": ResponseType.PARTIAL_ACTION_DONE
            if response.has_errors
            else ResponseType.ACTION_DONE,
            "error": response.error_message,
        }

    async def _failed_with_snapshot(
        self, err: VKCloudVisionAPIError, image: bytes, file_out: str | None
    ) -> JsonObjectType:
        error = self._action_error(err)
        path = (
            await self._save_image(image, [], file_out, BoundingBoxesType.NONE)
            if file_out
            else None
        )
        return {
            "response": None,
            "file_out": path,
            "response_type": ResponseType.ERROR,
            "error": str(error),
        }

    def _action_error(self, err: VKCloudVisionAPIError) -> HomeAssistantError:
        if isinstance(err, VKCloudVisionAuthError):
            self._entry.async_start_reauth(self.hass)
        return HomeAssistantError(str(err))

    async def _save_image(
        self, image: bytes, labels: list[dict], file_out: str, mode: BoundingBoxesType
    ) -> str:
        boxes = BoundingBoxes(image, labels, mode)
        try:
            return await self.hass.async_add_executor_job(
                boxes.save_image, file_out, self.hass.config.is_allowed_path
            )
        except (OSError, RuntimeError, ValueError) as err:
            raise HomeAssistantError(f"Unable to save image: {err}") from err

    async def _async_get_image(self, camera_id: str) -> bytes:
        """Get a single image from camera with retry logic."""
        last_error = None

        for attempt in range(MAX_IMAGE_RETRIES):
            try:
                camera_image = await async_get_image(
                    self.hass, camera_id, timeout=DEFAULT_IMAGE_TIMEOUT
                )
                return camera_image.content
            except (HomeAssistantError, TimeoutError) as err:
                last_error = str(err) or "Camera timed out"
                LOGGER.warning(
                    "Failed to get image from %s (attempt %d/%d): %s",
                    camera_id,
                    attempt + 1,
                    MAX_IMAGE_RETRIES,
                    last_error,
                )

                if attempt < MAX_IMAGE_RETRIES - 1:
                    await asyncio.sleep(RETRY_IMAGE_DELAY * (2**attempt))

        raise HomeAssistantError(
            f"Failed to get image from {camera_id} after {MAX_IMAGE_RETRIES} attempts. Last error: {last_error}"
        )

    async def _async_get_images(
        self, camera_id: str, num_snapshots: int, snapshot_interval_sec: float
    ) -> list[bytes]:
        """Get multiple snapshots from camera with a small interval."""
        images_data = []

        for i in range(num_snapshots):
            image_data = await self._async_get_image(camera_id)
            images_data.append(image_data)
            if i < num_snapshots - 1:
                await asyncio.sleep(snapshot_interval_sec)

        return images_data
