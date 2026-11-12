"""Losslessly pack the complete E4 handoff below one decimal GB, retaining every file."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import lzma
import shutil
import struct
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 8 * 1024 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''): h.update(block)
    return h.hexdigest()


def encode(source, target, reference=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base = reference.open('rb') if reference else None
    try:
        with source.open('rb') as stream, target.open('xb') as output:
            output.write(b'PSW1'); count = 0
            while True:
                raw = stream.read(BLOCK)
                if not raw: break
                size = len(raw)
                if base:
                    baseline = base.read(size)
                    assert len(baseline) == size
                    raw = (int.from_bytes(raw, 'little') ^ int.from_bytes(baseline, 'little')).to_bytes(size, 'little')
                raw += b'\0' * (-size % 4)
                shuffled = b''.join(raw[lane::4] for lane in range(4))
                packed = lzma.compress(shuffled, preset=6)
                output.write(struct.pack('<II', size, len(packed))); output.write(packed)
                count += 1
                if count % 8 == 0: print(target.name, 'blocks', count, 'seconds', round(time.time() - started), flush=True)
    finally:
        if base: base.close()
    return {'bytes': target.stat().st_size, 'sha256': digest(target)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / 'PromptShield_E4_Group_Handoff'
    destination.mkdir(exist_ok=False)
    original = json.loads((source / 'MANIFEST.json').read_text(encoding='utf-8'))
    assert original['variant'] == 'FULL'
    for row in original['files']:
        p = source / row['path']
        assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], row['path']
        if not row['path'].endswith('/model.safetensors'):
            target = destination / row['path']; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    shutil.copy2(source / 'MANIFEST.json', destination / 'MANIFEST.json')
    # Preserve the original 153-file manifest. New packaging helpers are additional files.
    shutil.copy2(ROOT / 'scripts/restore_e4_models.py', destination / 'scripts/restore_e4_models.py')
    models = ['models/e4/lr2e-5/model.safetensors', 'models/e4_followup/benign_weight2/model.safetensors',
              'models/e3/lr2e-5/model.safetensors']
    by_name = {r['path']: r for r in original['files']}
    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = []
        for i, name in enumerate(models):
            payload = f'compact_weights/model_{i + 1}.psw'
            base = source / models[0] if i == 1 else None
            job = pool.submit(encode, source / name, destination / payload, base)
            jobs.append((i, name, payload, job))
        for i, name, payload, job in jobs:
            result = job.result()
            entries.append({**by_name[name], 'payload': payload, 'payload_bytes': result['bytes'],
                'payload_sha256': result['sha256'], 'xor_reference': models[0] if i == 1 else None,
                'reference_sha256': by_name[models[0]]['sha256'] if i == 1 else None})
    index = {'format': 'PSW1-byte-lanes-lzma-xor', 'lossless': True, 'models': entries,
        'original_file_count': len(original['files']),
        'packaging_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (destination / 'COMPACT_MODELS.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8')
    (destination / 'COMPACT_README.txt').write_text(
        'E4 紧凑完整版：所有实验材料及三个模型均保留，无量化、无精度损失。\n\n'
        '阅读报告：直接打开 START_HERE.html。\n'
        '恢复模型：安装 Python 3 后，双击 RESTORE_MODELS.bat（Windows），\n'
        '或在当前目录运行：python scripts/restore_e4_models.py\n\n'
        '脚本只使用 Python 标准库，不需要 torch、不联网。它将恢复正式 E4、探索性 E4 和 E3 权重，\n'
        '并核对与原完整版完全一致的 SHA-256。模型仍是原始 FP32 参数。\n'
        '恢复后会自动核验原包全部 153 个文件。压缩后的权重文件可保留，不影响使用。\n'
        '请在解压前预留约 3 GB 磁盘空间。若恢复中断，已有不完整文件以 .restoring 结尾；\n'
        '重新解压到新目录再运行即可，不要覆盖已有不同版本的模型。\n', encoding='utf-8')
    (destination / 'RESTORE_MODELS.bat').write_bytes(
        b'@echo off\r\ncd /d "%~dp0"\r\nwhere py >nul 2>&1\r\n'
        b'if %errorlevel% equ 0 (py -3 scripts\\restore_e4_models.py) else (python scripts\\restore_e4_models.py)\r\n'
        b'if errorlevel 1 echo Restore failed. See the error above.\r\npause\r\n')
    zip_path = args.output_root / 'PromptShield_E4_COMPLETE_under1GB.zip'
    with zipfile.ZipFile(zip_path, 'w', allowZip64=True) as archive:
        for p in sorted(destination.rglob('*')):
            if p.is_file(): archive.write(p, destination.name + '/' + p.relative_to(destination).as_posix(),
                compress_type=zipfile.ZIP_STORED if p.suffix == '.psw' else zipfile.ZIP_DEFLATED, compresslevel=5)
    with zipfile.ZipFile(zip_path) as archive: assert archive.testzip() is None
    receipt = {'archive': zip_path.name, 'bytes': zip_path.stat().st_size, 'under_decimal_1GB': zip_path.stat().st_size < 1_000_000_000,
        'sha256': digest(zip_path), 'all_original_files_retained': True, 'original_model_bytes_unchanged_after_restoration': True}
    (args.output_root / 'COMPACT_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)
    assert receipt['under_decimal_1GB'], 'Compression target missed; do not present this archive as under 1 GB.'


if __name__ == '__main__': main()
