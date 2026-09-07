import torch

from secure_fl.model.mlp import AdultMLP


def test_model_output_shape():
    model = AdultMLP(input_dim=10)
    x = torch.randn(8, 10)
    y = model(x)
    assert y.shape == (8,)
