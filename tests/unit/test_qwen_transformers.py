from __future__ import annotations

from clinical_nlp.config import ModelConfig
from clinical_nlp.llm.qwen_transformers import _model_load_kwargs, _normalize_messages


def test_normalize_messages_wraps_string_content() -> None:
    messages = _normalize_messages(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "already blocks"}]},
        ]
    )
    assert messages[0]["content"] == [{"type": "text", "text": "be brief"}]
    assert messages[1]["content"] == [{"type": "text", "text": "already blocks"}]


def test_model_load_kwargs_default_uses_auto_dtype() -> None:
    kwargs = _model_load_kwargs(ModelConfig(model_id="models/qwen_transformers"))
    assert kwargs["torch_dtype"] == "auto"
    assert "quantization_config" not in kwargs


def test_model_load_kwargs_bnb_4bit() -> None:
    kwargs = _model_load_kwargs(
        ModelConfig(model_id="models/qwen_transformers", quantization="bnb_4bit")
    )
    assert "torch_dtype" not in kwargs
    quant = kwargs["quantization_config"]
    assert quant.load_in_4bit is True
    assert quant.bnb_4bit_quant_type == "nf4"
    assert quant.bnb_4bit_use_double_quant is True
