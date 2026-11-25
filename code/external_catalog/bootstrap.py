"""Copy a narrow, uploadable frozen snapshot; never copy old indexes/caches."""
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, digest, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if source == ROOT or source in ROOT.parents:
        raise ValueError('Source must be a different directory from the external runtime')
    if (ROOT / 'source_snapshot.json').exists() or (ROOT / 'src').exists():
        raise FileExistsError('Snapshot already exists. It will not be overwritten.')
    needed = ['src', 'e5_fold4_best_model', 'outputs/fold4/test_conversations.json',
              'externaldata/data_50k/product_base_table.json',
              'externaldata/data_50k/external_catalogue_manifest.json',
              'scripts/build_dense_retrieval_index.py', 'scripts/build_triplet_product_cache.py']
    for rel in needed:
        if not (source / rel).exists():
            raise FileNotFoundError(source / rel)
    files = {}
    for rel in ['src', 'e5_fold4_best_model', 'outputs/fold4/test_conversations.json',
                'scripts/build_dense_retrieval_index.py', 'scripts/build_triplet_product_cache.py']:
        src, dst = source / rel, ROOT / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
            paths = [p for p in dst.rglob('*') if p.is_file()]
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            paths = [dst]
        for path in paths:
            key = str(path.relative_to(ROOT))
            original_digest = digest(source / key)
            if digest(path) != original_digest:
                raise ValueError(f'Copy verification failed: {key}')
            files[key] = original_digest
    input_dir = ROOT / 'input'
    input_dir.mkdir()
    shutil.copy2(source / needed[3], input_dir / 'catalogue_source.json')
    shutil.copy2(source / needed[4], input_dir / 'source_manifest.json')
    save(ROOT / 'source_snapshot.json', {
        'source_root': str(source), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': files, 'external_source_sha256': digest(input_dir / 'catalogue_source.json'),
        'note': 'Runtime source and Fold4 model copied unchanged. Old product caches/indexes/results excluded. Frozen routing/taxonomy knowledge retained.',
    })
    print(f'Snapshot copied and hash-verified: {ROOT}')


if __name__ == '__main__':
    main()
