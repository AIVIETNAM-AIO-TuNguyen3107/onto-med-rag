from src.kb.load import load_concepts
from src.linking.fuzzy import link_concept


def test_load_sample_rxnorm():
    concepts = load_concepts(
        __import__("pathlib").Path("data/kb/rxnorm/concepts.sample.tsv")
    )
    assert ("308135", "amlodipine 10 MG Oral Tablet") in concepts


def test_link_amLODipine():
    concepts = load_concepts(
        __import__("pathlib").Path("data/kb/rxnorm/concepts.sample.tsv")
    )
    codes = link_concept("amlodipine 10 mg po daily", concepts, top_k=1)
    assert codes == ["308135"]
