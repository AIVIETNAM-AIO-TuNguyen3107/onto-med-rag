"""Pipeline stage interfaces — swap implementations without touching orchestration."""

from __future__ import annotations

from typing import Protocol

from src.schemas.entity import MedicalEntity


class Extractor(Protocol):
    def extract(self, text: str) -> list[MedicalEntity]: ...


class Linker(Protocol):
    def link(self, text: str, entity_type: str) -> list[str]: ...
