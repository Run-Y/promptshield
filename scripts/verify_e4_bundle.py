"""Verify a distributed E4 material bundle without third-party dependencies."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError('Unsafe manifest path')
        if not path.is_file(): errors.append('Missing: ' + row['path'])
        elif path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            errors.append('Hash mismatch: ' + row['path'])
    print(json.dumps({'variant': manifest['variant'], 'checked_files': len(manifest['files']),
        'omitted_large_weights': manifest.get('omitted_large_weights', []), 'errors': errors}, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__': main()
