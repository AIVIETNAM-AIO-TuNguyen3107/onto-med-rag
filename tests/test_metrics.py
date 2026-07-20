from src.eval.metrics import _jaccard, _wer


def test_wer_identical():
    assert _wer(["a", "b"], ["a", "b"]) == 0.0


def test_wer_empty_ref():
    assert _wer([], ["a"]) == 1.0


def test_jaccard_both_empty():
    assert _jaccard(set(), set()) == 1.0


def test_jaccard_disjoint():
    assert _jaccard({"a"}, {"b"}) == 0.0
