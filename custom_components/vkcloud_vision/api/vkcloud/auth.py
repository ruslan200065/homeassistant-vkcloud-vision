"""VK Cloud Vision authorization helpers."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from time import monotonic

from aiohttp import ClientError, ClientSession, ClientTimeout
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .exceptions import VKCloudVisionAPIError, VKCloudVisionAuthError


class VKCloudAuthBase(ABC):
    """Base class for all VK Cloud auth implementations."""

    @abstractmethod
    async def get_access_token(self) -> str:
        """Return a valid access token (raises on failure)."""


class VKCloudOAuthAuth(VKCloudAuthBase):
    def __init__(
        self,
        hass: HomeAssistant,
        client_id: str,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        on_refresh_token: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize VK Cloud Vision authorization helper."""
        self._client_id: str = client_id
        self._client_secret: str | None = client_secret
        self._refresh_token: str | None = refresh_token

        self._session: ClientSession = async_get_clientsession(hass)
        self._token_url: str = "https://mcs.mail.ru/auth/oauth/v1/token"

        self._access_token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()
        self._on_refresh_token = on_refresh_token

    async def get_access_token(self) -> str:
        """Return a cached token or exchange credentials under a lock."""
        async with self._lock:
            if self._access_token and monotonic() < self._expires_at:
                return self._access_token

            payload = {"client_id": self._client_id}
            if self._refresh_token:
                payload.update(
                    refresh_token=self._refresh_token, grant_type="refresh_token"
                )
            elif self._client_secret:
                payload.update(
                    client_secret=self._client_secret, grant_type="client_credentials"
                )
            else:
                raise VKCloudVisionAuthError("VK Cloud Vision credentials are missing")

            try:
                async with self._session.post(
                    self._token_url,
                    json=payload,
                    timeout=ClientTimeout(total=10),
                    allow_redirects=False,
                ) as response:
                    if response.status in (400, 401, 403):
                        raise VKCloudVisionAuthError(
                            "VK Cloud Vision rejected the credentials", response.status
                        )
                    if response.status != 200:
                        raise VKCloudVisionAPIError(
                            "VK Cloud OAuth request failed", response.status
                        )
                    data = await response.json()
            except (ClientError, TimeoutError, ValueError):
                raise VKCloudVisionAPIError(
                    "Unable to obtain a VK Cloud OAuth token"
                ) from None

            token = data.get("access_token") if isinstance(data, dict) else None
            if not isinstance(token, str) or not token:
                raise VKCloudVisionAPIError("VK Cloud OAuth returned no access token")

            refresh_token = data.get("refresh_token")
            if (
                isinstance(refresh_token, str)
                and refresh_token
                and refresh_token != self._refresh_token
            ):
                if self._on_refresh_token:
                    self._on_refresh_token(refresh_token)
                self._refresh_token = refresh_token

            expires_in = data.get("expires_in", 3600)
            if not isinstance(expires_in, (int, float)) or not 0 < expires_in < float(
                "inf"
            ):
                expires_in = 3600
            self._access_token = token
            self._expires_at = monotonic() + expires_in - min(60, expires_in / 10)
            return token

    def get_refresh_token(self) -> str | None:
        """Return the current refresh token."""
        return self._refresh_token


class VKCloudApiKeyAuth(VKCloudAuthBase):
    """Static service token."""

    def __init__(self, _hass: HomeAssistant, api_key: str) -> None:
        self._api_key = api_key.strip()

    async def get_access_token(self) -> str:
        if not self._api_key:
            raise VKCloudVisionAuthError(
                "Static API key (service token) is missing or empty"
            )

        return self._api_key


class VKCloudAuth:
    """Public auth wrapper – chooses the right implementation automatically."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        api_key: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        on_refresh_token: Callable[[str], None] | None = None,
    ) -> None:
        if api_key:
            self._impl: VKCloudAuthBase = VKCloudApiKeyAuth(hass, api_key)
        else:
            # Old OAuth path (only used for existing config entries)
            if not client_id:
                raise VKCloudVisionAuthError(
                    "Either api_key or client_id must be provided"
                )
            self._impl = VKCloudOAuthAuth(
                hass,
                client_id=client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
                on_refresh_token=on_refresh_token,
            )

    async def get_access_token(self) -> str:
        return await self._impl.get_access_token()

    def get_refresh_token(self) -> str | None:
        """Return the refresh token if using OAuth, otherwise None."""
        if isinstance(self._impl, VKCloudOAuthAuth):
            return self._impl.get_refresh_token()
        return None
