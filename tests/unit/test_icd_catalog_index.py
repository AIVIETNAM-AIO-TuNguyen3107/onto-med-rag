import json
from pathlib import Path

from clinical_nlp.icd_linking import ICDIndex


def test_catalog_index_preserves_hierarchy_and_contextual_z_codes(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "icd.jsonl"
    rows = [
        {
            "model": "chapter",
            "code": "XXI",
            "name_vi": "Yếu tố ảnh hưởng đến tình trạng sức khỏe",
            "path": "chapter:XXI",
            "is_leaf": False,
        },
        {
            "model": "type",
            "code": "Z52",
            "name_vi": "Người hiến cơ quan và mô",
            "path": "chapter:XXI/type:Z52",
            "is_leaf": False,
        },
        {
            "model": "disease",
            "code": "Z52.0",
            "name_vi": "Hiến máu",
            "path": "chapter:XXI/type:Z52/disease:Z520",
            "is_leaf": True,
        },
        {
            "model": "disease",
            "code": "U13/9",
            "name_vi": "Mã khẩn cấp không hợp lệ",
            "path": "chapter:XXI/disease:U139",
            "is_leaf": True,
        },
    ]
    catalog.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    index = ICDIndex.from_catalog(catalog)
    assert "U13/9" not in index.concepts
    concept = index.concepts["Z52.0"]
    assert concept.models == ("disease",)
    assert concept.is_leaf is True
    assert "Hiến máu" in concept.hierarchies[0]

    candidates = index.retrieve("hiến máu")
    assert candidates[0].identifier == "Z52.0"
    assert candidates[0].score < 1.0
    assert candidates[0].metadata["hierarchies"]

    saved = tmp_path / "index.json"
    index.save(saved)
    loaded = ICDIndex.load(saved)
    assert loaded.concepts["Z52.0"] == concept
