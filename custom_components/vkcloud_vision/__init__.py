"""VK Cloud Vision integration"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import voluptuous as vol
from homeassistant.auth.permissions.const import POLICY_CONTROL
from homeassistant.components.image_processing import DOMAIN as IMAGE_PROCESSING_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import (
    EntityServiceResponse,
    HomeAssistant,
    ServiceCall,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError, Unauthorized
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_component import DATA_INSTANCES
from homeassistant.helpers.entity_platform import async_get_platforms
from homeassistant.helpers.service import async_extract_entity_ids
from homeassistant.helpers.typing import ConfigType

from .api.vkcloud.auth import VKCloudAuth
from .api.vkcloud.vision import VKCloudVision
from .const import (
    ATTR_BOUNDING_BOXES,
    ATTR_CONFIDENCE_THRESHOLD,
    ATTR_CREATE_NEW,
    ATTR_FILE_OUT,
    ATTR_LANG,
    ATTR_MAX_RETRIES,
    ATTR_MODES,
    ATTR_NUM_SNAPSHOTS,
    ATTR_PROB_THRESHOLD,
    ATTR_SNAPSHOT_INDEX,
    ATTR_SNAPSHOT_INTERVAL_SEC,
    ATTR_SPACE,
    ATTR_UPDATE_EMBEDDING,
    CONF_API_KEY,
    CONF_CLIENT_ID,
    CONF_CLIENT_SECRET,
    CONF_CREATE_NEW,
    CONF_PERSON_ALIASES,
    CONF_REFRESH_TOKEN,
    CONF_UPDATE_EMBEDDING,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CREATE_NEW,
    DEFAULT_FACE_BOUNDING_BOXES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODES,
    DEFAULT_NUM_SNAPSHOTS,
    DEFAULT_OBJECT_BOUNDING_BOXES,
    DEFAULT_PROB_THRESHOLD,
    DEFAULT_SNAPSHOT_INTERVAL_SEC,
    DEFAULT_SPACE,
    DEFAULT_UPDATE_EMBEDDING,
    DOMAIN,
    LOGGER,
    SERVICE_DETECT_OBJECTS,
    SERVICE_RECOGNIZE_FACES,
    SERVICE_RECOGNIZE_TEXT,
    VALID_MODES,
    BoundingBoxesType,
    ResponseType,
)
from .image_processing import VKCloudVisionEntity

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


def get_vision_entity(hass: HomeAssistant) -> VKCloudVisionEntity:
    """Get the VK Cloud Vision entity."""
    for platform in async_get_platforms(hass, DOMAIN):
        for entity in platform.entities.values():
            if isinstance(entity, VKCloudVisionEntity):
                return entity
    raise HomeAssistantError("VK Cloud Vision entity is not setup yet")


async def async_setup(hass: HomeAssistant, _config: ConfigType) -> bool:
    """Register the VK Cloud Vision actions."""

    async def handle_action(call: ServiceCall) -> EntityServiceResponse:
        entries = hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            raise HomeAssistantError("VK Cloud Vision is not configured or loaded")
        entry = entries[0]
        entity = get_vision_entity(hass)
        data = call.data
        cameras = sorted(
            entity_id
            for entity_id in await async_extract_entity_ids(call)
            if entity_id.startswith("camera.")
        )
        if not cameras:
            raise HomeAssistantError("Select at least one camera")

        if call.context.user_id is not None:
            user = await hass.auth.async_get_user(call.context.user_id)
            if user is None or any(
                not user.permissions.check_entity(camera, POLICY_CONTROL)
                for camera in cameras
            ):
                raise Unauthorized(context=call.context)

        kwargs = {ATTR_MAX_RETRIES: data[ATTR_MAX_RETRIES]}
        file_out = data.get(ATTR_FILE_OUT)
        if call.service == SERVICE_DETECT_OBJECTS:
            method = entity.async_detect_objects
            kwargs.update(
                modes=data[ATTR_MODES],
                prob_threshold=data[ATTR_PROB_THRESHOLD],
                bounding_boxes=data[ATTR_BOUNDING_BOXES],
                num_snapshots=data[ATTR_NUM_SNAPSHOTS],
                snapshot_interval_sec=data[ATTR_SNAPSHOT_INTERVAL_SEC],
                snapshot_index=data[ATTR_SNAPSHOT_INDEX],
            )
            if kwargs[ATTR_SNAPSHOT_INDEX] > kwargs[ATTR_NUM_SNAPSHOTS]:
                raise HomeAssistantError("snapshot_index must not exceed num_snapshots")
        elif call.service == SERVICE_RECOGNIZE_FACES:
            method = entity.recognize_faces
            space = data[ATTR_SPACE]
            kwargs.update(
                space=space,
                create_new=data.get(
                    ATTR_CREATE_NEW,
                    entry.options.get(CONF_CREATE_NEW, DEFAULT_CREATE_NEW),
                ),
                update_embedding=data.get(
                    ATTR_UPDATE_EMBEDDING,
                    entry.options.get(CONF_UPDATE_EMBEDDING, DEFAULT_UPDATE_EMBEDDING),
                ),
                confidence_threshold=data[ATTR_CONFIDENCE_THRESHOLD],
                bounding_boxes=data[ATTR_BOUNDING_BOXES],
                tag_to_alias_map={
                    f"person{alias['person_id']}": alias["alias"]
                    for alias in entry.options.get(CONF_PERSON_ALIASES, [])
                    if alias["space"] == space
                },
            )
        else:
            method = entity.recognize_text
            kwargs[ATTR_LANG] = data.get(ATTR_LANG)

        if file_out and len(cameras) > 1 and "{camera}" not in file_out:
            raise HomeAssistantError(
                "Use {camera} in file_out when selecting multiple cameras"
            )

        result = {}
        for camera in cameras:
            if call.service != SERVICE_RECOGNIZE_TEXT:
                kwargs[ATTR_FILE_OUT] = (
                    file_out.replace("{camera}", camera.split(".", 1)[1])
                    if file_out
                    else None
                )
            try:
                result[camera] = await method(camera, **kwargs)
            except HomeAssistantError as err:
                result[camera] = {
                    "response": None,
                    "file_out": None,
                    "response_type": ResponseType.ERROR,
                    "error": str(err),
                }
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_DETECT_OBJECTS,
        handle_action,
        schema=cv.make_entity_service_schema(
            {
                vol.Optional(ATTR_MODES, default=DEFAULT_MODES): vol.All(
                    cv.ensure_list, [vol.In(VALID_MODES)], vol.Length(min=1)
                ),
                vol.Optional(
                    ATTR_PROB_THRESHOLD, default=DEFAULT_PROB_THRESHOLD
                ): vol.All(vol.Coerce(float), vol.Range(min=0.01, max=1.0)),
                vol.Optional(ATTR_FILE_OUT): cv.string,
                vol.Optional(
                    ATTR_BOUNDING_BOXES, default=DEFAULT_OBJECT_BOUNDING_BOXES
                ): vol.In(["none", "no_labels", "rus", "eng"]),
                vol.Optional(
                    ATTR_NUM_SNAPSHOTS, default=DEFAULT_NUM_SNAPSHOTS
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
                vol.Optional(ATTR_SNAPSHOT_INDEX, default=1): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=100)
                ),
                vol.Optional(
                    ATTR_SNAPSHOT_INTERVAL_SEC, default=DEFAULT_SNAPSHOT_INTERVAL_SEC
                ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=10)),
                vol.Optional(ATTR_MAX_RETRIES, default=DEFAULT_MAX_RETRIES): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=10)
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RECOGNIZE_TEXT,
        handle_action,
        schema=cv.make_entity_service_schema(
            {
                vol.Optional(ATTR_LANG): vol.In(["rus", "eng"]),
                vol.Optional(ATTR_MAX_RETRIES, default=DEFAULT_MAX_RETRIES): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=10)
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RECOGNIZE_FACES,
        handle_action,
        schema=cv.make_entity_service_schema(
            {
                vol.Required(ATTR_SPACE, default=DEFAULT_SPACE): vol.All(
                    vol.Coerce(int), vol.Range(min=0, max=9)
                ),
                vol.Optional(ATTR_CREATE_NEW): cv.boolean,
                vol.Optional(ATTR_UPDATE_EMBEDDING): cv.boolean,
                vol.Optional(
                    ATTR_CONFIDENCE_THRESHOLD, default=DEFAULT_CONFIDENCE_THRESHOLD
                ): vol.All(vol.Coerce(float), vol.Range(min=0.01, max=1.0)),
                vol.Optional(ATTR_FILE_OUT): cv.string,
                vol.Optional(
                    ATTR_BOUNDING_BOXES, default=DEFAULT_FACE_BOUNDING_BOXES
                ): vol.In(
                    [
                        BoundingBoxesType.NONE.value,
                        BoundingBoxesType.NO_LABELS.value,
                        BoundingBoxesType.TAG.value,
                        BoundingBoxesType.ALIAS.value,
                    ]
                ),
                vol.Optional(ATTR_MAX_RETRIES, default=DEFAULT_MAX_RETRIES): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=10)
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up a config entry."""

    credentials = entry.data

    def store_refresh_token(token: str) -> None:
        nonlocal credentials
        if entry.data is not credentials:
            return
        hass.config_entries.async_update_entry(
            entry, data={**entry.data, CONF_REFRESH_TOKEN: token}
        )
        credentials = entry.data

    auth_client = VKCloudAuth(
        hass,
        api_key=entry.data.get(CONF_API_KEY),
        client_id=entry.data.get(CONF_CLIENT_ID),
        client_secret=entry.data.get(CONF_CLIENT_SECRET),
        refresh_token=entry.data.get(CONF_REFRESH_TOKEN),
        on_refresh_token=store_refresh_token,
    )
    client = VKCloudVision(hass, auth_client)
    entry.runtime_data = client
    # The image_processing component does not expose module-level entry hooks.
    component = hass.data[DATA_INSTANCES][IMAGE_PROCESSING_DOMAIN]
    return await component.async_setup_entry(entry)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload config entry."""
    component = hass.data[DATA_INSTANCES][IMAGE_PROCESSING_DOMAIN]
    return await component.async_unload_entry(entry)


# Migration (keeps old OAuth entries working until user reconfigures)
async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate config entry to latest version."""
    if config_entry.version > 4:
        return False

    if config_entry.version in (1, 2):
        # v1 → v3: Bump version, keep OAuth data
        # v2 → v3: Bump version, no data changes (supports both auth types)
        new_data = dict(config_entry.data)
        hass.config_entries.async_update_entry(config_entry, version=3, data=new_data)
        LOGGER.info("Migrated VK Cloud Vision config entry to v3")

    if config_entry.version == 3:
        # v3 → v4: Replace single training_mode with separate create_new/update_embedding options
        new_options = dict(config_entry.options)
        new_options[CONF_CREATE_NEW] = new_options.pop("training_mode", False)
        new_options[CONF_UPDATE_EMBEDDING] = True
        hass.config_entries.async_update_entry(
            config_entry, version=4, options=new_options
        )
        LOGGER.info("Migrated VK Cloud Vision config entry from v3 to v4")

    return True
