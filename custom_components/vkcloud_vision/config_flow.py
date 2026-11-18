# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant import data_entry_flow
from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.helpers.selector import (
    FileSelector,
    FileSelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    ObjectSelector,
    ObjectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .api.vkcloud.auth import VKCloudAuth
from .api.vkcloud.exceptions import VKCloudVisionAPIError, VKCloudVisionAuthError
from .api.vkcloud.vision import VKCloudVision
from .const import (
    CONF_ALIAS,
    CONF_CLIENT_ID,
    CONF_CLIENT_SECRET,
    CONF_CONFIRM_DELETE,
    CONF_CONFIRM_TRUNCATE,
    CONF_CREATE_NEW,
    CONF_DELETE_PERSON_SPACE,
    CONF_PERSON_ALIASES,
    CONF_PERSON_IDS,
    CONF_PHOTO,
    CONF_REFRESH_TOKEN,
    CONF_SPACE,
    CONF_TRUNCATE_SPACE,
    CONF_UPDATE_EMBEDDING,
    DEFAULT_CREATE_NEW,
    DEFAULT_SPACE,
    DEFAULT_UPDATE_EMBEDDING,
    DOMAIN,
    LOGGER,
    SECTION_PERSON_ALIASES,
    SECTION_TRAINING_MODE,
)


class VKCloudVisionConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for VK Cloud Vision."""

    VERSION = 4

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Connect a new VK Cloud account."""
        return await self._async_credentials_form("user", user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Replace the credentials for an existing account."""
        return await self._async_credentials_form(
            "reconfigure", user_input, self._get_reconfigure_entry()
        )

    async def async_step_reauth(self, _entry_data: Mapping[str, Any]) -> ConfigFlowResult:
        """Start a credential update after an authentication failure."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Accept new credentials for the existing account."""
        return await self._async_credentials_form(
            "reauth_confirm", user_input, self._get_reauth_entry()
        )

    async def _async_credentials_form(
        self,
        step_id: str,
        user_input: dict[str, Any] | None,
        entry: ConfigEntry | None = None,
    ) -> ConfigFlowResult:
        """Validate OAuth credentials and discard the client secret."""
        errors = {}
        if user_input is not None:
            client_id = user_input[CONF_CLIENT_ID].strip()
            client_secret = user_input[CONF_CLIENT_SECRET].strip()
            if not client_id or not client_secret:
                errors["base"] = "invalid_auth"
            else:
                auth = VKCloudAuth(self.hass, client_id=client_id, client_secret=client_secret)
                try:
                    await auth.get_access_token()
                except VKCloudVisionAuthError:
                    errors["base"] = "invalid_auth"
                except VKCloudVisionAPIError:
                    errors["base"] = "cannot_connect"
                else:
                    if refresh_token := auth.get_refresh_token():
                        data = {CONF_CLIENT_ID: client_id, CONF_REFRESH_TOKEN: refresh_token}
                        if entry is None:
                            return self.async_create_entry(title="VK Cloud Vision", data=data)
                        return self.async_update_reload_and_abort(entry, data=data)
                    errors["base"] = "invalid_auth"

        schema = vol.Schema({
            vol.Required(CONF_CLIENT_ID): str,
            vol.Required(CONF_CLIENT_SECRET): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        })
        if entry is not None and CONF_CLIENT_ID in entry.data:
            schema = self.add_suggested_values_to_schema(schema, {CONF_CLIENT_ID: entry.data[CONF_CLIENT_ID]})
        return self.async_show_form(
            step_id=step_id,
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "obtain_token_url": "https://msk.cloud.vk.com/app/services/machinelearning/vision/access/"
            },
        )

    @staticmethod
    def async_get_options_flow(_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return VKCloudVisionOptionsFlow()


class VKCloudVisionOptionsFlow(OptionsFlow):
    """VK Cloud Vision options flow."""

    async def async_step_init(self, _user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Show the main menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["face_recognition", "manual_training", "truncate_space", "delete_persons"],
        )

    async def async_step_face_recognition(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage face recognition options (training mode and aliases)."""
        if user_input is not None:
            new_opts = dict(self.config_entry.options)
            new_opts.update(user_input.get(SECTION_TRAINING_MODE, {}))
            new_opts.update(user_input.get(SECTION_PERSON_ALIASES, {}))
            return self.async_create_entry(data=new_opts)

        create_new_default = self.config_entry.options.get(CONF_CREATE_NEW, DEFAULT_CREATE_NEW)
        update_embedding_default = self.config_entry.options.get(CONF_UPDATE_EMBEDDING, DEFAULT_UPDATE_EMBEDDING)
        existing_aliases = self.config_entry.options.get(CONF_PERSON_ALIASES, [])

        training_schema = vol.Schema({
            vol.Required(CONF_CREATE_NEW, default=create_new_default): bool,
            vol.Required(CONF_UPDATE_EMBEDDING, default=update_embedding_default): bool,
        })

        alias_schema = vol.Schema({
            vol.Optional(CONF_PERSON_ALIASES, default=existing_aliases): ObjectSelector(
                ObjectSelectorConfig(
                    fields={
                        "alias": {
                            "required": True,
                            "selector": {"text": None},
                        },
                        "person_id": {
                            "required": True,
                            "selector": {"number": {"min": 1, "mode": "box"}},
                        },
                        "space": {
                            "required": True,
                            "selector": {"number": {"min": 0, "max": 9, "mode": "box"}},
                        },
                    },
                    multiple=True,
                    translation_key="person_aliases",
                )
            )
        })

        schema = self.add_suggested_values_to_schema(
            vol.Schema({
                vol.Required(SECTION_TRAINING_MODE): data_entry_flow.section(
                    training_schema,
                    {"collapsed": False},
                ),
                vol.Required(SECTION_PERSON_ALIASES): data_entry_flow.section(
                    alias_schema,
                    {"collapsed": False},
                ),
            }),
            self.config_entry.options,
        )

        return self.async_show_form(
            step_id="face_recognition",
            data_schema=schema,
        )

    async def async_step_truncate_space(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Truncate a person space."""
        errors: dict[str, str] = {}

        if user_input is not None:
            space = user_input.get(CONF_TRUNCATE_SPACE, DEFAULT_SPACE)
            confirm = user_input.get(CONF_CONFIRM_TRUNCATE, False)

            if not confirm:
                errors["base"] = "confirm_truncate"
            else:
                client: VKCloudVision = self.config_entry.runtime_data
                try:
                    await client.persons.truncate(space)
                except Exception as err:
                    errors["base"] = "truncate_failed"
                    LOGGER.error("Truncate space %s failed: %s", space, err)
                else:
                    self._remove_aliases(space)
                    return self.async_abort(reason="truncate_success")

        space_default = user_input.get(CONF_TRUNCATE_SPACE, DEFAULT_SPACE) if user_input else DEFAULT_SPACE

        data_schema = vol.Schema({
            vol.Required(CONF_TRUNCATE_SPACE, default=space_default): vol.All(
                NumberSelector(
                    NumberSelectorConfig(min=0, max=9, mode=NumberSelectorMode.BOX),
                ),
                vol.Coerce(int),
            ),
            vol.Required(CONF_CONFIRM_TRUNCATE, default=False): bool,
        })

        return self.async_show_form(
            step_id="truncate_space",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_delete_persons(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Delete specific persons from a space."""
        errors: dict[str, str] = {}

        if user_input is not None:
            space = user_input.get(CONF_DELETE_PERSON_SPACE, DEFAULT_SPACE)
            person_ids_str = user_input.get(CONF_PERSON_IDS, "")
            confirm = user_input.get(CONF_CONFIRM_DELETE, False)

            if not confirm:
                errors["base"] = "confirm_delete"
            else:
                person_ids = []
                if person_ids_str:
                    try:
                        person_ids = [int(value.strip()) for value in person_ids_str.split(",") if value.strip()]
                    except ValueError:
                        errors["base"] = "invalid_person_ids"
                if not person_ids or any(person_id < 1 for person_id in person_ids):
                    errors["base"] = "invalid_person_ids"
                if not errors:
                    client: VKCloudVision = self.config_entry.runtime_data
                    try:
                        await client.persons.delete(space, person_ids)
                    except Exception as err:
                        errors["base"] = "delete_failed"
                        LOGGER.error("Delete persons from space %s failed: %s", space, err)
                    else:
                        self._remove_aliases(space, set(person_ids))
                        return self.async_abort(reason="delete_success")

        space_default = user_input.get(CONF_DELETE_PERSON_SPACE, DEFAULT_SPACE) if user_input else DEFAULT_SPACE
        person_ids_default = user_input.get(CONF_PERSON_IDS, "") if user_input else ""

        data_schema = vol.Schema({
            vol.Required(CONF_DELETE_PERSON_SPACE, default=space_default): vol.All(
                NumberSelector(
                    NumberSelectorConfig(min=0, max=9, mode=NumberSelectorMode.BOX),
                ),
                vol.Coerce(int),
            ),
            vol.Required(CONF_PERSON_IDS, default=person_ids_default): TextSelector(),
            vol.Required(CONF_CONFIRM_DELETE, default=False): bool,
        })

        return self.async_show_form(
            step_id="delete_persons",
            data_schema=data_schema,
            errors=errors,
        )

    def _remove_aliases(self, space: int, person_ids: set[int] | None = None) -> None:
        options = dict(self.config_entry.options)
        aliases = options.get(CONF_PERSON_ALIASES, [])
        remaining = [
            alias
            for alias in aliases
            if alias["space"] != space
            or (person_ids is not None and alias["person_id"] not in person_ids)
        ]
        if len(remaining) != len(aliases):
            options[CONF_PERSON_ALIASES] = remaining
            self.hass.config_entries.async_update_entry(
                self.config_entry, options=options
            )

    async def async_step_manual_training(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle manual training with photo upload."""
        errors: dict[str, str] = {}
        if user_input is not None:
            return await self._process_manual_training(user_input, errors)

        return self.async_show_form(
            step_id="manual_training",
            data_schema=self._build_manual_training_schema(),
            errors=errors,
        )

    def _build_manual_training_schema(self) -> vol.Schema:
        """Build the manual training form schema."""
        return vol.Schema({
            vol.Required(CONF_SPACE, default=DEFAULT_SPACE): vol.All(
                NumberSelector(NumberSelectorConfig(min=0, max=9, mode=NumberSelectorMode.BOX)),
                vol.Coerce(int),
            ),
            vol.Required(CONF_PHOTO): FileSelector(FileSelectorConfig(accept=".jpg,.jpeg,.png,.tiff")),
            vol.Optional(CONF_ALIAS): TextSelector(),
        })

    def _read_uploaded_photo(self, file_id: str) -> bytes:
        """Read uploaded photo bytes from file ID."""
        with process_uploaded_file(self.hass, file_id) as path:
            return path.read_bytes()

    async def _process_manual_training(self, user_input: dict[str, Any], errors: dict[str, str]) -> ConfigFlowResult:
        """Process manual training form submission."""

        try:
            file_id = user_input[CONF_PHOTO]
            space = user_input[CONF_SPACE]
            alias = user_input.get(CONF_ALIAS, "").strip()
            photo_bytes = await self.hass.async_add_executor_job(self._read_uploaded_photo, file_id)

            client: VKCloudVision = self.config_entry.runtime_data
            response = await client.persons.recognize(
                files=[photo_bytes],
                space=space,
                images=[{"name": "training_photo"}],
                create_new=True,
                update_embedding=True,
                max_retries=1,
            )

            if response.has_errors:
                raise Exception(response.error_message)

            person_tags = [p["tag"] for p in response.persons if p.get("tag", "undefined") != "undefined"]
            if not person_tags:
                errors["base"] = "no_faces"
                return self.async_show_form(
                    step_id="manual_training",
                    data_schema=self._build_manual_training_schema(),
                    errors=errors,
                )

            person_ids = [int(t.replace("person", "")) for t in person_tags]
            aliases = self._update_aliases(person_ids, space, alias)

            new_opts = dict(self.config_entry.options)
            new_opts[CONF_PERSON_ALIASES] = aliases
            self.hass.config_entries.async_update_entry(self.config_entry, options=new_opts)

            tags_str = ", ".join(person_tags)
            return self.async_abort(
                reason="manual_training_success",
                description_placeholders={"tags": tags_str},
            )

        except Exception as err:
            LOGGER.exception("Manual training failed", exc_info=err)
            errors["base"] = "training_failed"
            return self.async_show_form(
                step_id="manual_training",
                data_schema=self._build_manual_training_schema(),
                errors=errors,
            )

    def _update_aliases(self, person_ids: list[int], space: int, alias: str) -> list[dict[str, Any]]:
        aliases = [dict(item) for item in self.config_entry.options.get(CONF_PERSON_ALIASES, [])]
        first_pid = person_ids[0]

        if alias:
            found = False
            for entry in aliases:
                if entry.get("person_id") == first_pid and entry.get("space") == space:
                    entry["alias"] = alias
                    found = True
                    break

            if not found:
                aliases.append({"person_id": first_pid, "space": space, "alias": alias})

        for pid in person_ids:
            if not any(a["person_id"] == pid and a["space"] == space for a in aliases):
                aliases.append({"person_id": pid, "space": space, "alias": f"person{pid}"})

        return aliases
