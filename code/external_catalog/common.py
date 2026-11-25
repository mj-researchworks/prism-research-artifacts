from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_VERSION = 'external_schema_v1'
DENSE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
SUPPORTED_FOLDS = ('fold3', 'fold4')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def jsonish(value):
    if is_dataclass(value):
        return jsonish(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonish(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported trace value: {type(value).__name__}')


def verify_snapshot(root=ROOT):
    snapshot = load(root / 'source_snapshot.json')
    for rel, expected in snapshot['files'].items():
        path = root / rel
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Frozen snapshot mismatch: {rel}')
    return snapshot


def fold_paths(fold):
    if fold not in SUPPORTED_FOLDS:
        raise ValueError(f'Unsupported fold {fold!r}; choose one of {SUPPORTED_FOLDS}')
    return {
        'model_dir': ROOT / f'e5_{fold}_best_model',
        'cache': ROOT / 'artifacts' / f'external_e5_{fold}.pt',
        'build_manifest': ROOT / 'artifacts' / f'build_manifest_{fold}.json',
        'split': ROOT / 'outputs' / fold / 'test_conversations.json',
    }


def model_file_hashes(model_dir):
    if not model_dir.is_dir():
        raise FileNotFoundError(f'Missing fold model directory: {model_dir}')
    return {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted(model_dir.rglob('*')) if path.is_file()
    }


def configure(planner_mode='auto', device='cpu', fold='fold4'):
    # Explicit values reproduce the user's final agent configuration.
    paths = fold_paths(fold)
    settings = {
        'AUTONOMOUS_REASONING_BUDGET': 'adaptive',
        'AUTONOMOUS_RETRIEVAL_BACKEND': 'conserve_stack',
        'AUTONOMOUS_PLAN_POLICY_MODE': 'validator',
        'AUTONOMOUS_PLANNER_MODE': planner_mode,
        'AUTONOMOUS_EXECUTION_SCOPE': 'full',
        'AUTONOMOUS_INTENT_MODE': 'lightweight',
        'AUTONOMOUS_REFLECTION_MODE': 'agent',
        'AUTONOMOUS_REFLECTION_WRITES_INTENT': '1',
        'AUTONOMOUS_CRITIC_MODE': 'hybrid',
        'AUTONOMOUS_RESPONSE_MODE': 'template',
        'MACPRS_USE_TRIPLET_RERANKER': '1',
        'MACPRS_TRIPLET_MODEL_PATH': str(paths['model_dir']),
        'MACPRS_TRIPLET_PRODUCT_TEXT_PATH': str(ROOT / 'data/product_base_retrieval/product_base_retrieval_text.json'),
        'MACPRS_TRIPLET_PRODUCT_CACHE_PATH': str(paths['cache']),
        'MACPRS_TRIPLET_DEVICE': device,
        'SEARCH_DENSE_MODEL_NAME': DENSE_MODEL,
        'SEARCH_DENSE_INDEX_BASE': 'indexes',
        'TOKENIZERS_PARALLELISM': 'false',
        'USE_TF': '0',
        'TRANSFORMERS_NO_TF': '1',
    }
    for role in ('PLANNER', 'REFLECTION', 'CRITIC'):
        settings[f'AUTONOMOUS_{role}_PROVIDER'] = 'deepseek'
        settings[f'AUTONOMOUS_{role}_MODEL'] = 'deepseek-v4-flash'
        settings[f'AUTONOMOUS_{role}_TIMEOUT'] = '60'
    # No paid calls during explicitly selected offline rule diagnostics.
    if planner_mode == 'rule':
        settings['AUTONOMOUS_REFLECTION_MODE'] = 'rule'
        settings['AUTONOMOUS_CRITIC_MODE'] = 'rule'
    os.environ.update(settings)
    os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/hf_modules'))
    return settings


def effective_environment():
    prefixes = ('AUTONOMOUS_', 'MACPRS_', 'SEARCH_', 'ROUTE_', 'DEEPSEEK_')
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(prefixes) and not any(s in k for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
