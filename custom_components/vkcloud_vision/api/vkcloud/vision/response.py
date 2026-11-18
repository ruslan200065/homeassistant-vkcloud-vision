"""Vision response class for VK Cloud Vision API."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import cast

from homeassistant.util.json import JsonObjectType, JsonValueType

from ..exceptions import VKCloudVisionAPIError


def _result_items(value: JsonValueType, field: str) -> list[JsonObjectType]:
    """Validate a list of objects returned by Vision."""
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise VKCloudVisionAPIError(f"Invalid VK Cloud Vision field: {field}")
    return cast(list[JsonObjectType], value)


def _passes_threshold(value: JsonValueType, threshold: float) -> bool:
    return isinstance(value, (int, float)) and value >= threshold


class VKCloudVisionObjectDetectionResponse:
    """Class to handle and parse VK Cloud Vision object detection API responses."""

    def __init__(
        self,
        raw_response: JsonObjectType,
        prob_threshold: float = 0.1,
        object_classes: list[str] | None = None,
    ):
        """Initialize with API response."""
        self._errors: list[str] = []
        self._prob_threshold = prob_threshold
        self._object_classes = {
            name.strip().casefold() for name in object_classes or []
        }
        self._data = self._process_response(raw_response)

    @property
    def data(self) -> JsonObjectType:
        """Return the processed API response data."""
        return cast(JsonObjectType, self._data)

    @property
    def has_errors(self) -> bool:
        """Return True if any errors were found in the response."""
        return bool(self._errors)

    @property
    def error_message(self) -> str | None:
        """Return concatenated error message or None if no errors."""
        return "; ".join(self._errors) if self._errors else None

    def labels_for_image(self, name: str) -> list[JsonObjectType]:
        """Return labels for the requested image across successful modes."""
        return [
            label
            for images in self._data.values()
            for image in images
            if image.get("name") == name and image.get("status", 0) == 0
            for label in cast(list[JsonObjectType], image.get("labels", []))
        ]

    def _matches_class(self, label: JsonObjectType) -> bool:
        if not self._object_classes:
            return True
        return any(
            isinstance(value := label.get(key), str)
            and value.strip().casefold() in self._object_classes
            for key in ("eng", "rus")
        )

    def _process_response(
        self, response: JsonObjectType
    ) -> dict[str, list[JsonObjectType]]:
        """Process and filter API response."""
        processed = {}
        for mode, result in response.items():
            processed[mode] = []
            for image in _result_items(result, mode):
                image_name = image.get("name", "unknown")
                status = image.get("status", 0)

                if status != 0:
                    error = image.get("error", "unknown error")
                    self._errors.append(f"{image_name} ({mode}) {error}")

                processed_image = image.copy()
                if "labels" in image:
                    processed_image["labels"] = [
                        label
                        for label in _result_items(image["labels"], "labels")
                        if _passes_threshold(label.get("prob", 0), self._prob_threshold)
                        and self._matches_class(label)
                    ]

                processed[mode].append(processed_image)

        return processed


class VKCloudVisionFaceRecognitionResponse:
    """Parse response from /v1/persons/recognize."""

    def __init__(
        self,
        raw_response: dict,
        confidence_threshold: float = 0.1,
        tag_to_alias_map: dict | None = None,
    ):
        self._persons: list[dict] = []
        self._aliases_changed: bool = False
        self._errors: list[str] = []
        self._confidence_threshold = confidence_threshold
        self._tag_to_alias_map = tag_to_alias_map or {}
        self._parse_persons(raw_response)

    @property
    def data(self) -> JsonObjectType:
        """Return the processed API response data."""
        return {
            "persons": cast(JsonValueType, self._persons),
            "aliases_changed": self._aliases_changed,
        }

    @property
    def persons(self) -> list[dict]:
        return self._persons

    @property
    def aliases_changed(self) -> bool:
        return self._aliases_changed

    @property
    def has_errors(self) -> bool:
        return bool(self._errors)

    @property
    def error_message(self) -> str | None:
        return "; ".join(self._errors) if self._errors else None

    def _parse_persons(self, response: JsonObjectType) -> None:
        objects = _result_items(response.get("objects", []), "objects")
        self._aliases_changed = cast(bool, response.get("aliases_changed", False))

        if not objects:
            return

        # We only handle a single image per request
        obj = objects[0]
        status = obj.get("status", 0)
        if status != 0:
            self._errors.append(
                f"{obj.get('name', 'unknown')}: {obj.get('error', 'unknown')}"
            )

        self._persons = []
        for person in _result_items(obj.get("persons", []), "persons"):
            if not _passes_threshold(
                person.get("confidence", 0), self._confidence_threshold
            ):
                continue

            tag = person.get("tag")
            if not isinstance(tag, str):
                tag = "undefined"
            self._persons.append(
                {**person, "alias": self._tag_to_alias_map.get(tag, tag)}
            )


class VKCloudVisionTextRecognitionResponse:
    """Parse response from /v1/scene_text/recognize."""

    def __init__(self, raw_response: dict):
        self._objects: list[dict] = []
        self._errors: list[str] = []
        self._parse_response(raw_response)

    @property
    def data(self) -> JsonObjectType:
        """Return the processed API response data."""
        return {"objects": cast(JsonValueType, self._objects)}

    @property
    def words(self) -> list[dict]:
        """Return raw words array for the first image."""
        if self._objects:
            return self._objects[0].get("words", [])
        return []

    @property
    def text(self) -> str | None:
        """Return joined text string for the first image (backward compatible)."""
        if self._objects:
            return self._objects[0].get("text", "")
        return None

    @property
    def has_errors(self) -> bool:
        return bool(self._errors)

    @property
    def error_message(self) -> str | None:
        return "; ".join(self._errors) if self._errors else None

    def _parse_response(self, response: JsonObjectType) -> None:
        """Parse text recognition response objects."""
        objects = _result_items(response.get("objects", []), "objects")
        if not objects:
            return

        for obj in objects:
            status = obj.get("status", 0)
            name = obj.get("name", "unknown")
            if status != 0:
                self._errors.append(f"{name}: {obj.get('error', 'unknown error')}")

            words = _result_items(obj.get("words", []), "words")

            # Generate backward-compatible text by joining word["text"] with newlines
            text = "\n".join(
                word_text
                for word in words
                if isinstance(word_text := word.get("text"), str)
            )

            processed_obj = obj.copy()
            processed_obj["words"] = words
            processed_obj["text"] = text
            self._objects.append(processed_obj)
