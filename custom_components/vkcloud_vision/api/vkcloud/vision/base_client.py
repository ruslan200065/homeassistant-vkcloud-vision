"""Base client for VK Cloud Vision API requests."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
import json
import logging
from typing import Any

from aiohttp import (
    ClientConnectionError,
    ClientError,
    ClientSession,
    ClientTimeout,
    FormData,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util.json import JsonObjectType

from ..auth import VKCloudAuth
from ..exceptions import (
    VKCloudVisionAPIError,
    VKCloudVisionAuthError,
    VKCloudVisionDetectionError,
    VKCloudVisionDetectionStatus,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
RETRY_DELAY = 1


class VKCloudVisionBaseClient:
    def __init__(
        self,
        hass: HomeAssistant,
        auth: VKCloudAuth,
        base_url: str = "https://smarty.mail.ru/api",
    ) -> None:
        """Initialize the base client."""
        self._hass = hass
        self._auth = auth
        self._base_url = base_url
        self._session: ClientSession = async_get_clientsession(hass)

    async def _make_request(
        self,
        endpoint: str,
        meta: dict[str, Any],
        files: list[bytes] | None = None,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
        *,
        allow_partial: bool = True,
    ) -> JsonObjectType:
        """Make an API request with multipart/form-data."""
        access_token = await self._auth.get_access_token()
        if not access_token:
            raise VKCloudVisionAuthError("Failed to obtain access token")

        url = f"{self._base_url}{endpoint}"

        query_params = {
            "oauth_token": access_token,
            "oauth_provider": "mcs",
        }
        if params:
            query_params.update(params)

        return await self._execute_request_with_retries(
            url, query_params, meta, files, max_retries, allow_partial
        )

    def _prepare_form_data(
        self, meta: dict[str, Any], files: list[bytes] | None
    ) -> FormData:
        """Prepare multipart form data for the request using file names from meta."""

        data = FormData(default_to_multipart=True)

        data.add_field("meta", json.dumps(meta))

        if files is None or len(files) == 0:
            return data

        if "images" not in meta or not isinstance(meta["images"], list):
            raise ValueError("meta must contain 'images' key with a list")

        images = meta["images"]
        if len(files) != len(images):
            raise ValueError("Number of files must match number of images in meta")

        names = [img.get("name") for img in images]
        if any(name is None for name in names):
            raise ValueError("All images in meta must have a 'name' key")

        if len(set(names)) != len(names):
            raise ValueError("All image names in meta must be unique")

        for file_data, name in zip(files, names):
            data.add_field(name, file_data, filename=name)

        return data

    async def _execute_request(
        self,
        url: str,
        query_params: dict[str, Any],
        data: FormData,
        allow_partial: bool,
    ) -> JsonObjectType:
        """Execute single request attempt."""
        async with self._session.post(
            url,
            params=query_params,
            data=data,
            allow_redirects=False,
            timeout=ClientTimeout(total=DEFAULT_TIMEOUT),
        ) as response:
            if response.status in (408, 429, 500, 502, 503, 504):
                raise TimeoutError(f"VK Cloud Vision returned HTTP {response.status}")
            if response.status in (401, 403):
                raise VKCloudVisionAuthError(
                    "VK Cloud Vision rejected the credentials", response.status
                )
            if response.status >= 300:
                raise VKCloudVisionAPIError(
                    "VK Cloud Vision request failed", response.status
                )

            try:
                result = await response.json()
            except (ValueError, ClientError) as err:
                raise VKCloudVisionAPIError("Invalid JSON response") from err

            if not isinstance(result, dict):
                raise VKCloudVisionAPIError("Invalid response envelope")
            if result.get("status") != 200:
                raise VKCloudVisionAPIError(
                    "VK Cloud Vision returned an API error",
                    api_status=result.get("status"),
                )

            body = result.get("body")
            if not isinstance(body, dict):
                raise VKCloudVisionAPIError("Missing or invalid response body")

            results = []
            for mode, images in body.items():
                # Management endpoints can return metadata alongside image results.
                if not isinstance(images, list):
                    continue
                for image in images:
                    if not isinstance(image, dict):
                        raise VKCloudVisionAPIError("Invalid image result")
                    results.append((mode, image))

            # Keep successful results even when another mode or image failed.
            failed = [
                (mode, image) for mode, image in results if image.get("status", 0) != 0
            ]
            if failed and (not allow_partial or len(failed) == len(results)):
                mode, image = next(
                    (
                        (mode, image)
                        for mode, image in failed
                        if image.get("status") != 2
                    ),
                    failed[0],
                )
                status = image.get("status")
                if status not in (1, 2):
                    raise VKCloudVisionAPIError("Unknown image processing status")
                raise VKCloudVisionDetectionError(
                    mode, str(image.get("name", "unknown")), status
                )
            return body

    async def _execute_request_with_retries(
        self,
        url: str,
        query_params: dict[str, Any],
        meta: dict[str, Any],
        files: list[bytes] | None,
        max_retries: int,
        allow_partial: bool,
    ) -> JsonObjectType:
        """Retry temporary failures with bounded backoff."""
        for attempt in range(1, max_retries + 1):
            data = self._prepare_form_data(meta, files)
            try:
                return await self._execute_request(
                    url, query_params, data, allow_partial
                )
            except VKCloudVisionDetectionError as err:
                if (
                    err.detection_status
                    is not VKCloudVisionDetectionStatus.TEMPORARY_ERROR
                    or attempt == max_retries
                ):
                    raise
            except (TimeoutError, ClientConnectionError):
                if attempt == max_retries:
                    raise VKCloudVisionAPIError(
                        f"VK Cloud Vision unavailable after {attempt} attempt(s)"
                    ) from None
            except ClientError:
                raise VKCloudVisionAPIError(
                    "Unable to contact VK Cloud Vision"
                ) from None
            _LOGGER.debug(
                "Retrying VK Cloud Vision request after attempt %d/%d",
                attempt,
                max_retries,
            )
            await asyncio.sleep(min(RETRY_DELAY * 2 ** (attempt - 1), 8))
        raise ValueError("max_retries must be positive")
