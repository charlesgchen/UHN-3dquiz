import torch

from scripts.train_frozen_embedding_mlp import EmbeddingMLP


def test_accepted_encoder_mlp_shape_and_parameter_count():
    model = EmbeddingMLP(input_dim=640, hidden_dim=16, dropout=0.5)

    assert sum(parameter.numel() for parameter in model.parameters()) == 10_307
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(4, 640))
    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()


def test_hidden_dim_zero_selects_the_linear_ablation():
    model = EmbeddingMLP(input_dim=640, hidden_dim=0, dropout=0.5)
    assert isinstance(model.network, torch.nn.Linear)
