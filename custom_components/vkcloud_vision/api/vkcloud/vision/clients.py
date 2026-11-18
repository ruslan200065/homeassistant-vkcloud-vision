# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Any

from .base_client import VKCloudVisionBaseClient
from .response import (
    VKCloudVisionFaceRecognitionResponse,
    VKCloudVisionObjectDetectionResponse,
    VKCloudVisionTextRecognitionResponse,
)


class VKCloudVisionObjectsClient(VKCloudVisionBaseClient):
    """Client for objects-related VK Cloud Vision API endpoints."""

    async def detect(
        self,
        files: list[bytes],
        modes: list[str],
        images: list[dict[str, str]],
        prob_threshold: float,
        max_retries: int = 3,
        object_classes: list[str] | None = None,
        region: tuple[int, int, int, int] | None = None,
    ) -> VKCloudVisionObjectDetectionResponse:
        """Detect objects in a photo."""
        meta = {
            "mode": modes,  # e.g., ["object", "object2", "scene"]
            "images": images,  # Expected format: [{"name": str}]
        }
        raw_response = await self._make_request(
            "/v1/objects/detect", meta, files, max_retries=max_retries
        )
        return VKCloudVisionObjectDetectionResponse(
            raw_response=raw_response,
            prob_threshold=prob_threshold,
            object_classes=object_classes,
            region=region,
        )


class VKCloudVisionTextClient(VKCloudVisionBaseClient):
    """Client for text-related VK Cloud Vision API endpoints."""

    async def scene_text_recognize(
        self,
        files: list[bytes],
        images: list[dict[str, str]],
        lang: str | None = None,
        max_retries: int = 3,
    ) -> VKCloudVisionTextRecognitionResponse:
        """Recognize text in scene photos."""
        images_meta = [
            {"name": img["name"], **({"lang": lang} if lang else {})} for img in images
        ]
        meta: dict[str, Any] = {"images": images_meta}
        raw_response = await self._make_request(
            "/v1/scene_text/recognize", meta, files, max_retries=max_retries
        )
        return VKCloudVisionTextRecognitionResponse(raw_response)


class VKCloudVisionPersonsClient(VKCloudVisionBaseClient):
    """Client for persons-related VK Cloud Vision API endpoints."""

    async def set(
        self,
        files: list[bytes],
        space: int,
        images: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Set a relationship between a photo and person_id."""
        meta = {
            "space": str(space),
            "images": images,  # Expected format: [{"name": str, "person_id": int}]
        }
        return await self._make_request(
            "/v1/persons/set", meta, files, max_retries=1, allow_partial=False
        )

    async def delete(
        self,
        space: int,
        person_id: int | list[int],
    ) -> dict[str, Any]:
        """Delete a relationship between a photo and person_id."""
        person_ids = [person_id] if isinstance(person_id, int) else person_id
        meta = {
            "space": str(space),
            "images": [{"name": str(pid), "person_id": pid} for pid in person_ids],
        }
        return await self._make_request(
            "/v1/persons/delete", meta, max_retries=1, allow_partial=False
        )

    async def truncate(self, space: int) -> dict[str, Any]:
        """Clear the entire space."""
        meta = {"space": str(space)}
        return await self._make_request(
            "/v1/persons/truncate", meta, max_retries=1, allow_partial=False
        )

    async def recognize(
        self,
        files: list[bytes],
        space: int,
        images: list[dict[str, str]],
        create_new: bool = False,
        update_embedding: bool = True,
        confidence_threshold: float = 0.1,
        tag_to_alias_map: dict | None = None,
        max_retries: int = 3,
    ) -> VKCloudVisionFaceRecognitionResponse:
        """Recognize a person in a photo."""
        meta = {
            "space": str(space),
            "create_new": create_new,
            "update_embedding": update_embedding,
            "images": images,  # Expected format: [{"name": str}]
        }
        raw_response = await self._make_request(
            "/v1/persons/recognize",
            meta,
            files,
            max_retries=1 if create_new or update_embedding else max_retries,
        )
        return VKCloudVisionFaceRecognitionResponse(
            raw_response,
            confidence_threshold=confidence_threshold,
            tag_to_alias_map=tag_to_alias_map,
        )
