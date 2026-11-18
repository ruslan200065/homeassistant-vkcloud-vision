"""Utility functions for image processing in VK Cloud Vision integration."""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import io
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from homeassistant.exceptions import HomeAssistantError
from PIL import Image, ImageFont, UnidentifiedImageError
from PIL.ImageDraw import Draw, ImageDraw
from propcache.api import cached_property

from .const import LOGGER, BoundingBoxesType


class BoundingBoxes:
    """Helper class for image processing tasks."""

    def __init__(
        self, image_data: bytes, labels: list[dict[str, Any]], mode: BoundingBoxesType
    ) -> None:
        """Initialize the BoundingBoxes class with image data and labels."""
        self.image_data = image_data
        self.labels = labels
        self.mode = mode

    @cached_property
    def _font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            font_path = os.path.join(
                os.path.dirname(__file__), "fonts", "Tuffy_Bold.ttf"
            )
            return ImageFont.truetype(font_path, 20)
        except Exception as err:
            LOGGER.warning("Failed to load custom font: %s. Using default font.", err)
            return ImageFont.load_default()

    def save_image(
        self, output_path: str, is_allowed_path: Callable[[str], bool]
    ) -> str:
        """Draw bounding boxes with labels and save image."""
        path = Path(output_path)
        if not path.is_absolute():
            raise HomeAssistantError("file_out must be an absolute path")
        path = path.resolve()
        if not is_allowed_path(str(path)):
            raise HomeAssistantError(
                "file_out must be an absolute path allowed by Home Assistant"
            )
        image_format = Image.registered_extensions().get(path.suffix.lower())
        if image_format not in {"JPEG", "PNG", "WEBP", "TIFF"}:
            raise HomeAssistantError("file_out must use JPEG, PNG, WebP or TIFF")

        try:
            with Image.open(io.BytesIO(self.image_data)) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError) as err:
            raise HomeAssistantError("Unable to process image: bad data") from err

        draw = Draw(image)

        if self.mode != BoundingBoxesType.NONE:
            for label in self.labels:
                coord = label.get("coord")
                if (
                    not isinstance(coord, (list, tuple))
                    or len(coord) != 4
                    or not all(
                        isinstance(v, int | float) and abs(v) < float("inf")
                        for v in coord
                    )
                    or coord[0] > coord[2]
                    or coord[1] > coord[3]
                ):
                    continue

                text = None
                score = 0.0
                if self.mode == BoundingBoxesType.RUS:
                    text = label.get("rus")
                    score = label.get("prob", 0.0)
                elif self.mode == BoundingBoxesType.ENG:
                    text = label.get("eng")
                    score = label.get("prob", 0.0)
                elif self.mode == BoundingBoxesType.TAG:
                    text = label.get("tag")
                    score = label.get("similarity", 0.0)
                elif self.mode == BoundingBoxesType.ALIAS:
                    text = label.get("alias", label.get("tag"))
                    score = label.get("similarity", 0.0)

                self._draw_box(
                    draw,
                    tuple(coord),
                    text if isinstance(text, str) else None,
                    score if isinstance(score, (int, float)) else 0.0,
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = NamedTemporaryFile(dir=path.parent, delete=False)
        temporary_path = Path(temporary.name)
        try:
            with temporary:
                image.save(temporary, format=image_format)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        LOGGER.debug("Image saved: %s", output_path)

        return output_path

    def _draw_box(
        self,
        draw: ImageDraw,
        coord: tuple[int, int, int, int],
        text_label: str | None = None,
        probability: float = 0,
        color: tuple[int, int, int] = (255, 255, 0),
    ) -> None:
        """Draw a bounding box on an image using direct coordinates.

        Args:
            draw: ImageDraw object
            coord: Tuple of (x1, y1, x2, y2) coordinates
            text: Label text to display
            color: RGB color tuple for the box
        """
        line_width = 3
        font_height = 20
        x1, y1, x2, y2 = coord

        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw the label if text is provided
        if text_label:
            text = f"{text_label} {probability:.0%}" if probability else text_label

            draw.text(
                (x1, max(0, y1 - line_width - font_height)),
                text,
                fill=color,
                font=self._font,
                font_size=font_height,
            )
