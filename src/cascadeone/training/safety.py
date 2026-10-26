import hashlib

import torch


def freeze(model):
    model.requires_grad_(False)
    model.eval()
    for p in model.parameters():
        p.grad = None
    return model


def assert_frozen(*models):
    for model in models:
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("upstream parameter unexpectedly trainable")
        if any(m.training for m in model.modules()):
            raise RuntimeError("upstream module unexpectedly in training mode")
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("upstream parameter unexpectedly has gradient")


def assert_optimizer(adapter, optimizer, *upstreams):
    assert_frozen(*upstreams)
    params = [p for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in adapter.parameters() if p.requires_grad}
    if not expected or len(params) != len({id(p) for p in params}) or {id(p) for p in params} != expected:
        raise RuntimeError("optimizer must contain exactly the trainable FusionAdapter parameters")
    if any(id(p) in expected for model in upstreams for p in model.parameters()):
        raise RuntimeError("adapter shares parameters with upstream")


def state_hash(model):
    """Full streaming hash including buffers; no second complete model in RAM."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        flat = value.detach().reshape(-1)
        for chunk in flat.split(1_000_000):
            digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_report(fast, strong, adapter=None):
    groups = {'fast': fast.model, 'strong': strong.model}
    if adapter is not None:
        groups['fusion'] = adapter
    report = {name: {'total': sum(p.numel() for p in model.parameters()),
                     'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad]}
              for name, model in groups.items()}
    total = sum(r['total'] for r in report.values())
    report['trainable_percent'] = 100 * sum(r['trainable'] for r in report.values()) / max(total, 1)
    return report
