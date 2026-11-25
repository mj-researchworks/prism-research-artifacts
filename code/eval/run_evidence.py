"""Incremental research evidence, independent of inference and metric logic."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import uuid


class EvidenceWriteError(RuntimeError):
    """Stop evaluation rather than spend more API credits without evidence."""


def plain(value):
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [plain(v) for v in value]
    if isinstance(value, Path): return str(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def environment():
    prefixes = ('AUTONOMOUS_', 'MACPRS_', 'SEARCH_', 'ROUTE_', 'DEEPSEEK_')
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(prefixes) and not any(s in k.upper() for s in
                ('API_KEY', 'SECRET', 'PASSWORD', 'TOKEN', 'CREDENTIAL', 'AUTHORIZATION'))}


class RunEvidence:
    def __init__(self, *, root, out_dir, args, conversations, split_dir):
        self.root = Path(root)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.run_id = f'{stamp}_{uuid.uuid4().hex[:8]}'
        self.directory = Path(out_dir) / f'evidence_{args.split}_{self.run_id}'
        self.completed_turns = 0
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.manifest = {
                'schema_version': 1, 'run_id': self.run_id, 'status': 'running',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'fold': args.fold, 'split': args.split,
                'arguments': plain(vars(args)), 'environment': environment(),
                'python_version': sys.version, 'platform': platform.platform(),
                'selected_conversation_ids': [str(c.get('conv_id')) for c in conversations],
                'source_files': {}, 'assets': {}, 'packages': {},
                'note': ('Full state is inference evidence. Historical reference labels are scored '
                         'separately and are not supplied to the runner. Running status means incomplete; '
                         'JSONL records are written and closed after each event. No automatic resume.'),
            }
            self._save_manifest()
            self._snapshot_source()
            self._record_assets(Path(split_dir), args.split)
            for package in ('torch','transformers','sentence-transformers','faiss-cpu','numpy'):
                try: version = metadata.version(package)
                except metadata.PackageNotFoundError: version = None
                self.manifest['packages'][package] = version
            self._save_manifest()
        except Exception as exc:
            raise EvidenceWriteError(f'Cannot initialize research evidence: {exc}') from exc

    def _snapshot_source(self):
        paths = list(self.root.joinpath('src').rglob('*.py'))
        paths += [self.root/'eval'/'autonomous_full_recommendation_eval.py',
                  self.root/'eval'/'run_evidence.py']
        paths += list(self.root.joinpath('src','taxonomy').glob('*.json'))
        for path in sorted(set(paths)):
            if not path.is_file(): continue
            rel = path.relative_to(self.root)
            target = self.directory/'source_snapshot'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            self.manifest['source_files'][str(rel)] = digest(target)

    def _record_assets(self, split_dir, split):
        paths = [split_dir/f'{split}_conversations.json',
                 self.root/'data'/'product_base_table.json',
                 self.root/'data'/'product_category_map.json',
                 self.root/'src'/'taxonomy'/'route_embeddings.npy']
        retrieval = self.root/'data'/'product_base_retrieval'
        paths += list(retrieval.glob('*.json'))
        model = os.environ.get('MACPRS_TRIPLET_MODEL_PATH')
        if model:
            p = Path(model)
            paths += list(p.rglob('*')) if p.is_dir() else [p]
        for key in ('MACPRS_TRIPLET_PRODUCT_CACHE_PATH','MACPRS_TRIPLET_PRODUCT_TEXT_PATH'):
            if os.environ.get(key): paths.append(Path(os.environ[key]))
        index = Path(os.environ.get('SEARCH_DENSE_INDEX_BASE', 'indexes'))
        if not index.is_absolute(): index = self.root/index
        if index.is_dir(): paths += list(index.rglob('*'))
        for path in sorted(set(paths)):
            if path.is_file() and '__pycache__' not in path.parts:
                self.manifest['assets'][str(path)] = {
                    'sha256': digest(path), 'bytes': path.stat().st_size}
            elif not path.exists():
                self.manifest['assets'][str(path)] = {'missing': True}

    def _save_manifest(self):
        # Atomic replacement keeps the previous valid manifest on interruption.
        temp = self.directory/'manifest.pending.json'
        temp.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False),encoding='utf-8')
        temp.replace(self.directory/'manifest.json')

    def append(self, name, record):
        try:
            payload = json.dumps(plain(record), ensure_ascii=False)
            with (self.directory/name).open('a',encoding='utf-8') as stream:
                stream.write(payload+'\n')
                stream.flush()
        except Exception as exc:
            raise EvidenceWriteError(f'Cannot save {name}; stopping evaluation: {exc}') from exc

    def turn(self, *, cid, turn_index, user_message, response, state, latency_ms):
        state = plain(state)
        response_type = 'unknown'
        for observation in reversed(state.get('tool_observations') or []):
            if observation.get('component') == 'response_tool':
                match = re.search(r'\btype=([a-z_]+)',str(observation.get('summary','')))
                if match: response_type = match.group(1)
                break
        recommendations = state.get('last_recommendations') or []
        ids = [str(p['product_id']) for p in recommendations if p.get('product_id')]
        self.append('turn_states.jsonl', {
            'run_id':self.run_id, 'conversation_id':str(cid), 'turn_index':turn_index,
            'runtime_turn_id':state.get('turn_id'), 'user_message':user_message,
            'response':response, 'latency_ms':latency_ms, 'status':'runner_returned',
            'response_type':response_type,
            'newly_displayed_ids':ids if response_type=='recommendation' else [],
            'displayed_products':recommendations if response_type=='recommendation' else [],
            'retained_ids':ids if response_type=='acceptance' else [],
            'display_classification_known':response_type!='unknown',
            'reflection_notes':state.get('reflection_notes') or [],
            'critic_notes':state.get('critic_notes') or [],
            'state':state,
        })
        self.completed_turns += 1

    def finish(self, *, summary, output_files):
        self.manifest.update(status='completed',
                             finished_utc=datetime.now(timezone.utc).isoformat(),
                             completed_turn_records=self.completed_turns,
                             final_summary=plain(summary),
                             output_files={str(p):digest(p) for p in output_files})
        try: self._save_manifest()
        except Exception as exc: raise EvidenceWriteError(f'Cannot finalize manifest: {exc}') from exc
