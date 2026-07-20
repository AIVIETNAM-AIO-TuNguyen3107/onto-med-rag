from pathlib import Path

from src.pipeline.pipeline import ClinicalNLPipeline, build_default_pipeline
from src.extract.rules import RuleExtractor
from src.kb.load import load_kb_root
from src.linking.fuzzy import FuzzyLinker


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INPUT = (ROOT / "data/examples/sample_input.txt").read_text(encoding="utf-8")


def test_pipeline_processes_sample():
    pipeline = build_default_pipeline(ROOT / "data" / "kb")
    entities = pipeline.process(SAMPLE_INPUT)
    assert len(entities) > 0
    for ent in entities:
        s, e = ent["position"]
        assert ent["text"] == SAMPLE_INPUT[s:e]


def test_custom_extractor_linker_injection():
    kb = load_kb_root(ROOT / "data" / "kb")
    pipeline = ClinicalNLPipeline(RuleExtractor(), FuzzyLinker(kb))
    entities = pipeline.process(SAMPLE_INPUT)
    assert any(e["type"] == "CHẨN_ĐOÁN" for e in entities)
