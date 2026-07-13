"""Output schema types for competition submission."""

from typing import Literal, TypedDict

EntityType = Literal[
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
]

AssertionType = Literal["isNegated", "isFamily", "isHistorical"]

ASSERTION_TYPES = frozenset({"isNegated", "isFamily", "isHistorical"})
LINKED_TYPES = frozenset({"CHẨN_ĐOÁN", "THUỐC"})
ASSERTION_ELIGIBLE = frozenset({"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"})


class MedicalEntity(TypedDict):
    text: str
    position: list[int]  # [start, end) character offsets
    type: EntityType
    assertions: list[AssertionType]
    candidates: list[str]
