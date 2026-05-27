from __future__ import annotations

from enum import Enum


class Visibility(str, Enum):
    PLAYER = "player"
    GM = "gm"
    CHARACTER_PRIVATE = "character_private"
