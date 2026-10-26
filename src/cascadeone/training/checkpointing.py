import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file, save_file


def compatibility(adapter, fast, strong):
    return {'fast': asdict(fast.identity), 'strong': asdict(strong.identity),
            'fusion_type': adapter.fusion_type, 'fusion_interface_version': adapter.interface_version,
            'fusion_insertion_point': strong.insertion_point, 'fusion_config': adapter.config}


def save_adapter(path, adapter, fast, strong, metadata=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights = path / 'adapter.safetensors'
    save_file({key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()}, str(weights))
    manifest = {'compatibility': compatibility(adapter, fast, strong),
                'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'metadata': metadata or {}}
    payload = json.dumps(manifest, sort_keys=True).encode()
    manifest['artifact_hash'] = hashlib.sha256(payload).hexdigest()
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def load_adapter(path, adapter, fast, strong):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    digest = manifest.pop('artifact_hash')
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != digest:
        raise ValueError('artifact manifest checksum mismatch')
    if manifest['compatibility'] != compatibility(adapter, fast, strong):
        raise ValueError('adapter/backend compatibility mismatch')
    weights = path / 'adapter.safetensors'
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['weights_sha256']:
        raise ValueError('adapter weights checksum mismatch')
    adapter.load_state_dict(load_file(str(weights)), strict=True)
    adapter.eval()
    return manifest
