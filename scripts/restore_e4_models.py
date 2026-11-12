"""Restore exact FP32 model bytes from an E4 compact bundle; Python 3 stdlib only."""
import hashlib
import json
import lzma
import struct
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    result = (root / name).resolve()
    if root.resolve() not in result.parents: raise ValueError('Path outside bundle')
    return result


def restore_one(root, entry):
    target = safe_path(root, entry['path'])
    if target.exists():
        if digest(target) != entry['sha256']: raise RuntimeError('Existing model differs; refusing overwrite: ' + str(target))
        print('Already verified:', entry['path'], flush=True)
        return
    payload = safe_path(root, entry['payload'])
    if digest(payload) != entry['payload_sha256']: raise RuntimeError('Damaged payload: ' + str(payload))
    reference = None
    if entry.get('xor_reference'):
        base = safe_path(root, entry['xor_reference'])
        if digest(base) != entry['reference_sha256']: raise RuntimeError('Reference model hash mismatch')
        reference = base.open('rb')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.restoring')
    # Exclusive creation preserves any earlier interrupted attempt for inspection.
    try:
        with payload.open('rb') as source, temporary.open('xb') as output:
            if source.read(4) != b'PSW1': raise RuntimeError('Unknown compact payload format')
            while True:
                header = source.read(8)
                if not header: break
                if len(header) != 8: raise RuntimeError('Truncated payload header')
                size, packed_size = struct.unpack('<II', header)
                if not (0 < size <= 8 * 1024 * 1024 and 0 < packed_size <= 16 * 1024 * 1024):
                    raise RuntimeError('Invalid payload block length')
                packed = source.read(packed_size)
                if len(packed) != packed_size: raise RuntimeError('Truncated payload block')
                shuffled = lzma.decompress(packed, memlimit=256 * 1024 * 1024)
                padded_size = size + (-size % 4)
                if len(shuffled) != padded_size: raise RuntimeError('Invalid decoded block length')
                raw = bytearray(padded_size); quarter = padded_size // 4
                for lane in range(4): raw[lane::4] = shuffled[lane * quarter:(lane + 1) * quarter]
                raw = bytes(raw[:size])
                if reference:
                    baseline = reference.read(size)
                    if len(baseline) != size: raise RuntimeError('Reference length mismatch')
                    raw = (int.from_bytes(raw, 'little') ^ int.from_bytes(baseline, 'little')).to_bytes(size, 'little')
                output.write(raw)
        if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
            raise RuntimeError('Restored weight hash mismatch; original files are untouched')
        temporary.rename(target)
        print('Restored and verified:', entry['path'], flush=True)
    finally:
        if reference: reference.close()


def main():
    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / 'COMPACT_MODELS.json').read_text(encoding='utf-8'))
    if index['format'] != 'PSW1-byte-lanes-lzma-xor': raise RuntimeError('Unknown compact bundle version')
    for entry in index['models']: restore_one(root, entry)
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = safe_path(root, row['path'])
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise RuntimeError('Original bundle file verification failed: ' + row['path'])
    print('SUCCESS: all three original models and all', len(manifest['files']), 'original bundle files match SHA-256.', flush=True)


if __name__ == '__main__': main()
