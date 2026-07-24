from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EntityType(StrEnum):
    SYMPTOM = "TRIỆU_CHỨNG"
    TEST_NAME = "TÊN_XÉT_NGHIỆM"
    TEST_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
    DIAGNOSIS = "CHẨN_ĐOÁN"
    MEDICATION = "THUỐC"


class Assertion(StrEnum):
    NEGATED = "isNegated"
    FAMILY = "isFamily"
    HISTORICAL = "isHistorical"


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    source_path: str | None = None


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    index: int
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def valid_interval(self) -> "Chunk":
        if self.end < self.start:
            raise ValueError("chunk end precedes start")
        if self.end - self.start != len(self.text):
            raise ValueError("chunk interval length does not equal text length")
        return self


class SpanProposal(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    type: EntityType
    source: str
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_interval(self) -> "SpanProposal":
        if self.end <= self.start:
            raise ValueError("proposal must have positive length")
        if self.end - self.start != len(self.text):
            raise ValueError("proposal interval length does not equal text length")
        return self


class LinkCandidate(BaseModel):
    identifier: str
    name: str
    terminology_type: str | None = None
    score: float = 0.0
    retrieval_sources: list[str] = Field(default_factory=list)
    component_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    text: str
    type: EntityType
    candidates: list[str] | None = None
    assertions: list[Assertion] = Field(default_factory=list)
    position: tuple[int, int]

    @field_validator("assertions")
    @classmethod
    def unique_assertions(cls, values: list[Assertion]) -> list[Assertion]:
        order = [Assertion.NEGATED, Assertion.FAMILY, Assertion.HISTORICAL]
        return [item for item in order if item in set(values)]

    @model_validator(mode="after")
    def validate_fields(self) -> "Entity":
        start, end = self.position
        if start < 0 or end <= start:
            raise ValueError("entity position must be a positive interval")
        if end - start != len(self.text):
            raise ValueError("entity position length does not equal text length")
        linkable = self.type in {EntityType.DIAGNOSIS, EntityType.MEDICATION}
        if linkable and self.candidates is None:
            self.candidates = []
        if not linkable and self.candidates is not None:
            raise ValueError("candidates are only allowed for diagnoses and medications")
        return self

    def output_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "type": self.type.value,
        }
        if self.candidates is not None:
            data["candidates"] = self.candidates
        data["assertions"] = [value.value for value in self.assertions]
        data["position"] = list(self.position)
        return data


class StageResult(BaseModel):
    stage: str
    status: str
    document_id: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
