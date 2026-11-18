"""Home Assistant diagnostics for VK Cloud Vision."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from . import get_vision_entity
from .const import CONF_API_KEY


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict:
    """Return action statistics without including credentials or recognition data."""
    try:
        actions = get_vision_entity(hass).action_diagnostics()
    except HomeAssistantError:
        actions = []
    return {
        "auth_method": "api_key" if entry.data.get(CONF_API_KEY) else "oauth",
        "entry_version": entry.version,
        "actions_since_load": actions,
    }
