"""Reject stale/mixed artifacts before any paid inference."""
import argparse

from .common import ROOT, SUPPORTED_FOLDS, digest, fold_paths, load, model_file_hashes, verify_snapshot


def validate(require_embeddings=True, fold='fold4'):
    snapshot = verify_snapshot()
    paths = fold_paths(fold)
    manifest = load(ROOT / 'data/external_manifest.json')
    if digest(ROOT / 'input/catalogue_source.json') != manifest['source_sha256']:
        raise ValueError('External source no longer matches prepared data')
    for rel, expected in manifest['files'].items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f'Prepared catalogue hash mismatch: {rel}')
    products = load(ROOT / 'data/product_base_table.json')
    rows = load(ROOT / 'data/product_base_retrieval/product_base_retrieval_text.json')
    ids = load(ROOT / 'data/product_base_retrieval/ids.json')
    row_ids = [r['product_id'] for r in rows]
    expected = set(products)
    if len(expected) != manifest['products'] or row_ids != ids or set(ids) != expected or len(ids) != len(expected):
        raise ValueError('Catalogue/retrieval-text/ID mismatch or duplicates')
    for filename in ('product_category_map.json', 'product_base_category_map.json'):
        if set(load(ROOT / 'data' / filename)) != expected:
            raise ValueError(f'Category map mismatch: {filename}')
    bm25 = load(ROOT / 'data/product_base_retrieval/bm25_index.json')
    if bm25['n_docs'] != len(ids) or len(bm25['doc_len']) != len(ids):
        raise ValueError('BM25 document/ID mismatch')
    report = {'products': len(expected), 'data_alignment': 'pass', 'frozen_snapshot': 'pass',
              'source_sha256': manifest['source_sha256'], 'embedding_alignment': 'not_checked'}
    if not require_embeddings:
        return report
    builds = load(paths['build_manifest'])
    import faiss
    import torch
    import numpy as np
    text_hash = manifest['files']['data/product_base_retrieval/product_base_retrieval_text.json']
    if builds['retrieval_text_sha256'] != text_hash:
        raise ValueError('Embeddings were built from a different retrieval-text file')
    if builds.get('fold') != fold:
        raise ValueError(f'Build manifest belongs to {builds.get("fold")}, not {fold}')
    model_hashes = model_file_hashes(paths['model_dir'])
    if builds['model_files'] != model_hashes:
        raise ValueError(f'E5 cache belongs to a different {fold} model')
    for rel, expected_hash in builds['files'].items():
        if digest(ROOT / rel) != expected_hash:
            raise ValueError(f'Built artifact hash mismatch: {rel}')
    dense_ids = load(ROOT / 'indexes/_global/ids.json')
    index = faiss.read_index(str(ROOT / 'indexes/_global/faiss.bin'))
    if dense_ids != ids or index.ntotal != len(ids) or index.d != 384:
        raise ValueError('Dense index/ID count, order or embedding dimension mismatch')
    sample = index.reconstruct_n(0, min(16, index.ntotal))
    if not np.isfinite(sample).all() or not np.allclose(np.linalg.norm(sample, axis=1), 1, atol=.01):
        raise ValueError('Dense vectors are not finite normalized embeddings')
    cache = torch.load(paths['cache'], map_location='cpu', weights_only=True)
    embeddings = cache.get('embeddings', {})
    if set(embeddings) != expected or cache.get('count') != len(expected):
        raise ValueError('E5 cache misses external products or includes stale products')
    dim = load(paths['model_dir'] / 'config.json')['hidden_size']
    for pid, vector in embeddings.items():
        if vector.ndim != 1 or len(vector) != dim or not torch.isfinite(vector).all():
            raise ValueError(f'Invalid E5 embedding: {pid}')
    report.update({'embedding_alignment': 'pass', 'dense_vectors': index.ntotal,
                   'e5_embeddings': len(embeddings), 'dense_model': builds['dense_model']})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--fold', choices=SUPPORTED_FOLDS, default='fold4')
    args = parser.parse_args()
    import json
    print(json.dumps(validate(not args.data_only, fold=args.fold), indent=2))


if __name__ == '__main__':
    main()
