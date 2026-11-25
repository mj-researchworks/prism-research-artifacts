"""Build genuine external embeddings with frozen models, without LLM calls."""
import argparse
import os
import subprocess
import sys

from .common import DENSE_MODEL, ROOT, SUPPORTED_FOLDS, digest, fold_paths, load, model_file_hashes, save, verify_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--fold', choices=SUPPORTED_FOLDS, default='fold4')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('TRANSFORMERS_NO_TF', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/hf_modules'))
    from .validate import validate
    validate(require_embeddings=False, fold=args.fold)
    paths = fold_paths(args.fold)
    manifest_path = paths['build_manifest']
    if manifest_path.exists():
        print(validate(fold=args.fold))
        print('Existing full build verified; nothing overwritten.')
        return
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from scripts.build_triplet_product_cache import resolve_device
    device = resolve_device(args.device)
    rows = load(ROOT / 'data/product_base_retrieval/product_base_retrieval_text.json')
    ids = [r['product_id'] for r in rows]
    dense_dir = ROOT / 'indexes/_global'
    stamp = {'retrieval_text_sha256': digest(ROOT / 'data/product_base_retrieval/product_base_retrieval_text.json'),
             'dense_model': DENSE_MODEL}
    dense_stamp = ROOT / 'artifacts/dense_build.json'
    if dense_stamp.exists():
        old = load(dense_stamp)
        if any(old.get(k) != v for k, v in stamp.items()):
            raise ValueError('Existing dense build has incompatible source/model')
        for rel, expected in old['files'].items():
            if digest(ROOT / rel) != expected:
                raise ValueError(f'Partial dense build was modified: {rel}')
        print('Reusing verified completed dense stage.', flush=True)
    else:
        print(f'Building {len(rows):,} dense vectors on {device}.', flush=True)
        model = SentenceTransformer(DENSE_MODEL, device=device)
        matrix = model.encode([r['text'] for r in rows], batch_size=args.batch_size,
                              normalize_embeddings=True, show_progress_bar=True).astype('float32')
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(np.ascontiguousarray(matrix))
        dense_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(dense_dir / 'faiss.bin'))
        save(dense_dir / 'ids.json', ids)
        save(dense_stamp, {**stamp, 'device': device, 'dimension': int(matrix.shape[1]),
                          'files': {rel: digest(ROOT / rel) for rel in ['indexes/_global/faiss.bin', 'indexes/_global/ids.json']}})
        del model, matrix, index
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Use the exact cache encoder shipped with the frozen runtime.
    cache_path = paths['cache']
    if cache_path.exists():
        raise FileExistsError('An unverified E5 cache exists without a completed build manifest. Move it aside before rebuilding.')
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/build_triplet_product_cache.py'),
                    '--model-dir', str(paths['model_dir']),
                    '--product-text-json', str(ROOT / 'data/product_base_retrieval/product_base_retrieval_text.json'),
                    '--out', str(cache_path), '--device', device,
                    '--batch-size', str(args.batch_size)], check=True, cwd=ROOT)
    snapshot = verify_snapshot()
    rels = ['indexes/_global/faiss.bin', 'indexes/_global/ids.json', str(cache_path.relative_to(ROOT))]
    save(manifest_path, {**stamp, 'device': device,
                        'fold': args.fold,
                        'model_files': model_file_hashes(paths['model_dir']),
                        'files': {rel:digest(ROOT/rel) for rel in rels}})
    print(validate(fold=args.fold))


if __name__ == '__main__':
    main()
