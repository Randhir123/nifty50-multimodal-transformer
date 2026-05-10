"""Unit tests for the ImageCNN encoder."""

import torch
from src.models.image_cnn import ImageCNN, ImageCNNConfig


def test_cnn_output_shape():
    config = ImageCNNConfig(image_size=32, in_channels=2, output_dim=16)
    model = ImageCNN(config)
    
    x = torch.randn(4, 2, 32, 32)
    out = model.encode_images(x)
    assert out.shape == (4, 16)


def test_cnn_train_eval_mode():
    config = ImageCNNConfig()
    model = ImageCNN(config)
    x = torch.randn(2, 2, 32, 32)
    
    model.train()
    train_out = model.encode_images(x)
    
    model.eval()
    eval_out = model.encode_images(x)
    
    assert not torch.allclose(train_out, eval_out)
    assert not torch.isnan(train_out).any()


def test_cnn_gradient_flows():
    config = ImageCNNConfig()
    model = ImageCNN(config)
    x = torch.randn(2, 2, 32, 32)
    
    loss = model(x).sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert torch.any(param.grad != 0), f"No gradient for {name}"