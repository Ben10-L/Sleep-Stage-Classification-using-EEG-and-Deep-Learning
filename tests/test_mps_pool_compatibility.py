"""Cross-device safety checks for MPS-compatible temporal pooling."""
import pytest
import torch
from eeg_sleep.models import get_model, MPSCompatibleAvgPool1d


@pytest.mark.parametrize('length, bins', [(47, 16), (120, 40), (375, 48)])
def test_pooling_output_and_gradient(length, bins):
    values = torch.randn(2, 3, length, requires_grad=True)
    out = MPSCompatibleAvgPool1d(bins)(values)
    assert out.shape == (2, 3, bins)
    out.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize('architecture', ['cnn', 'lstm', 'transformer'])
def test_model_forward_backward(architecture):
    model = get_model(architecture)
    inputs = torch.randn(2, 2, 3000)
    logits = model(inputs)
    assert logits.shape == (2, 5)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_mps_when_available():
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        pytest.skip('MPS is not available in this environment')
    for arch in ('cnn', 'lstm', 'transformer'):
        model = get_model(arch).to('mps')
        output = model(torch.randn(2, 2, 3000, device='mps'))
        output.sum().backward()
        torch.mps.synchronize()
