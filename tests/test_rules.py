DRUG_LIST = (
    "Danh sách thuốc trước nhập viện chính xác và đầy đủ. "
    "1. amlodipine 10 mg po daily "
    "2. aspirin 81 mg po daily "
    "3. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau"
)


def test_numbered_drugs_extracted():
    from src.extract.rules import extract_entities

    entities = extract_entities(DRUG_LIST)
    drugs = [e for e in entities if e["type"] == "THUỐC"]
    assert len(drugs) >= 3
    assert all("isHistorical" in e["assertions"] for e in drugs)


def test_dieu_tri_symptom():
    from src.extract.rules import extract_entities

    entities = extract_entities(DRUG_LIST)
    symptoms = [e for e in entities if e["type"] == "TRIỆU_CHỨNG"]
    assert any("sốt" in e["text"] for e in symptoms)
