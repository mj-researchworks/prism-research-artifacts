"""Offline schema normalization, without retrieval-policy or agent changes."""
from __future__ import annotations

import math
import re
from collections import Counter

from .common import NORMALIZATION_VERSION, ROOT, digest, load, save, verify_snapshot

CATEGORIES = {
    'All_Beauty': 'Beauty & Health', 'Amazon_Fashion': 'Clothing & Fashion',
    'Books': 'Books & Media', 'Electronics': 'Electronics',
    'Grocery_and_Gourmet_Food': 'Grocery & Food', 'Home_and_Kitchen': 'Home & Living',
    'Office_Products': 'Office & Stationery', 'Pet_Supplies': 'Pet Supplies',
    'Sports_and_Outdoors': 'Sports & Outdoors', 'Toys_and_Games': 'Toys & Kids',
    'Video_Games': 'Electronics',
}
LABELS = ['Compatible Mountings', 'Camera Flash', 'Included Components', 'Item Dimensions LxWxH',
          'Connector Type', 'Cable Type', 'Compatible Devices', 'Special Feature',
          'Compatible Phone Models', 'Number of Processors', 'Computer Memory Type',
          'Hard Drive Interface', 'Batteries', 'Best Sellers Rank', 'Date First Available',
          'Personal computer design type', 'Memory Storage Capacity', 'Screen Size',
          'Ram Memory Installed Size', 'CPU Model', 'Color', 'Material', 'Manufacturer',
          'Product Dimensions', 'Item Weight', 'Size', 'Style', 'Model Name', 'Model Number',
          'Connectivity Technology', 'Form Factor', 'Operating System', 'Hardware Interface',
          'Wireless Communication Technology', 'Mounting Type', 'Power Source', 'Voltage',
          'Item model number', 'Is Discontinued By Manufacturer', 'Capacity', 'Unit Count',
          'Item Form', 'Flavor', 'Package Dimensions', 'Brand']
NEXT_LABEL = re.compile(r'\b(?:' + '|'.join(re.escape(v) for v in sorted(LABELS, key=len, reverse=True)) + r')\s*:', re.I)


def clean_brand(value):
    value = ' '.join(str(value or '').split()).strip()
    value = re.sub(r'^brand\s*:\s*', '', value, flags=re.I)
    match = NEXT_LABEL.search(value)
    if match:
        value = value[:match.start()].strip(' |,;')
    # Do not guess unknown field boundaries or treat long attribute blobs as brands.
    return '' if ':' in value or len(value) > 100 else value


def price_value(value):
    value = str(value or '').strip().replace(',', '')
    if not re.fullmatch(r'\$?\s*\d+(?:\.\d+)?', value):
        return None
    result = float(value.replace('$', '').strip())
    return result if math.isfinite(result) and result > 0 else None


def normalize_product(pid, product):
    title = ' '.join(str(product.get('title') or '').split())
    if not title:
        raise ValueError(f'Missing title: {pid}')
    attrs = dict(product.get('attributes') or {})
    category = str(product.get('category') or '').strip()
    if category not in CATEGORIES and category not in CATEGORIES.values():
        raise ValueError(f'Unmapped source category {category!r}; add a documented category mapping first')
    brand = clean_brand(product.get('brand'))
    price = price_value(attrs.get('Price') or product.get('price'))
    canonical = {k: v for k, v in attrs.items() if k != 'External Retrieval Text'}
    canonical['Brand'] = brand
    canonical.pop('Price', None)
    if price is not None:
        canonical['price'] = str(price)
    normalized = {
        **product, 'product_id': str(pid), 'title': title, 'brand': brand,
        'color': str(product.get('color') or attrs.get('Color') or '').strip(),
        'category': CATEGORIES.get(category, category), 'l1': CATEGORIES.get(category, category), 'l2': None,
        'external_category': category, 'attributes': attrs,
        'ranking_metadata': canonical, 'all_original_attributes': canonical,
    }
    source_text = str(attrs.get('External Retrieval Text') or '').strip()
    parts = [f'Product: {title}', f'Category: {normalized["category"]}']
    if brand:
        parts.append(f'Brand: {brand}')
    if price is not None:
        parts.append(f'Price: {price}')
    if normalized['color']:
        parts.append(f'Color: {normalized["color"]}')
    if source_text:
        parts.append(f'Source details: {source_text}')
    return normalized, ' | '.join(parts)


def bm25(rows):
    postings, idf, lengths = {}, {}, []
    for doc, row in enumerate(rows):
        counts = Counter(re.findall(r'[a-z0-9]+', row['text'].lower()))
        lengths.append(sum(counts.values()))
        for token, tf in counts.items():
            postings.setdefault(token, []).append([doc, tf])
    count = len(rows)
    for token, values in postings.items():
        idf[token] = math.log(1 + (count - len(values) + .5) / (len(values) + .5))
    return {'schema': 'prism_bm25_v1', 'n_docs': count, 'avg_doc_len': sum(lengths)/count,
            'k1': 1.5, 'b': .75, 'doc_len': lengths, 'idf': idf, 'postings': postings}


def main():
    snapshot = verify_snapshot()
    source = ROOT / 'input/catalogue_source.json'
    source_hash = digest(source)
    if source_hash != snapshot['external_source_sha256']:
        raise ValueError('Source catalogue changed since snapshot; create a new experiment snapshot')
    manifest_path = ROOT / 'data/external_manifest.json'
    if manifest_path.exists():
        old = load(manifest_path)
        if old['source_sha256'] != source_hash or old['normalization_version'] != NORMALIZATION_VERSION:
            raise ValueError('Prepared catalogue belongs to a different source/version')
        for rel, expected in old['files'].items():
            if digest(ROOT / rel) != expected:
                raise ValueError(f'Prepared file changed: {rel}')
        print('Prepared catalogue already matches its manifest; nothing overwritten.')
        return
    source_products = load(source)
    if not isinstance(source_products, dict) or not source_products:
        raise ValueError('Expected a nonempty product-id-keyed source catalogue')
    products, rows, categories = {}, [], {}
    for pid, product in source_products.items():
        if str(product.get('product_id', pid)) != str(pid):
            raise ValueError(f'Source key/product_id mismatch: {pid}')
        item, text = normalize_product(pid, product)
        products[str(pid)] = item
        rows.append({'product_id': str(pid), 'text': text})
        categories[str(pid)] = {'l1': item['l1'], 'l2': None}
    print(f'Normalized {len(products):,} products; building BM25.', flush=True)
    output = {
        'data/product_base_table.json': products,
        'data/product_base_category_map.json': categories,
        'data/product_category_map.json': categories,
        'data/product_base_retrieval/product_base_retrieval_text.json': rows,
        'data/product_base_retrieval/ids.json': list(products),
        'data/product_base_retrieval/bm25_index.json': bm25(rows),
    }
    for rel, content in output.items():
        path = ROOT / rel
        if path.exists():
            raise FileExistsError(f'Partial preparation exists: {path}; inspect before retrying')
        save(path, content)
    save(manifest_path, {
        'normalization_version': NORMALIZATION_VERSION, 'source_sha256': source_hash,
        'products': len(products), 'categories': dict(Counter(p['category'] for p in products.values())),
        'prices_readable': sum('price' in p['ranking_metadata'] for p in products.values()),
        'brands_changed': sum(products[k]['brand'] != str(p.get('brand') or '') for k, p in source_products.items()),
        'brands_unknown': sum(not p['brand'] for p in products.values()),
        'category_mapping': CATEGORIES, 'files': {rel: digest(ROOT / rel) for rel in output},
        'policy': 'Schema conversion only; no inferred product types, invented prices, model tuning, or catalogue-specific ranking rules.',
    })
    print('Catalogue prepared. Next build fresh dense and E5 product artifacts.')


if __name__ == '__main__':
    main()
