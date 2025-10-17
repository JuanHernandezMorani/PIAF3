import pytest

torch = pytest.importorskip("torch")

from src.model import FiLM, MultimodalYoloStub


def test_film_modulation_shapes():
    film = FiLM(in_channels=16, dim_context=8, hidden_dims=(32, 32))
    x = torch.randn(2, 16, 8, 8)
    context = torch.randn(2, 8)
    out = film(x, context)
    assert out.shape == x.shape


def test_multimodal_model_forward_accepts_context():
    model = MultimodalYoloStub(in_channels=5, num_classes=3, dim_context=10)
    x = torch.randn(2, 5, 128, 128)
    ctx = torch.randn(2, 10)
    outputs = model(x, ctx)

    logits = outputs["logits"]
    assert logits.shape[0] == x.shape[0]
    assert logits.shape[1] == 3
    assert logits.shape[-1] == logits.shape[-2]

    backbone_feats = outputs["backbone_features"]
    neck_feats = outputs["neck_features"]

    assert len(backbone_feats) == len(neck_feats) == 4
    assert backbone_feats[-1].shape[1] == model.backbone.out_channels[-1]
    assert neck_feats[0].shape[1] == model.neck.out_channels
