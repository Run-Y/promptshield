"""Build full and review E4 ZIPs with offline reports and per-file SHA-256 manifests."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'PromptShield_E4_Group_Handoff'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def md_html(path):
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
        value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
        return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)', r'<a href="\2">\1</a>', value)
    lines = path.read_text(encoding='utf-8').splitlines()
    output, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            block = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            output.append('<pre><code>' + html.escape('\n'.join(block)) + '</code></pre>')
        elif line.startswith('| '):
            rows = []
            while i < len(lines) and lines[i].startswith('| '):
                rows.append([c.strip() for c in lines[i].strip('|').split('|')]); i += 1
            output.append('<div class="table"><table><tr>' + ''.join('<th>' + inline(c) + '</th>' for c in rows[0]) + '</tr>')
            for row in rows[2:]: output.append('<tr>' + ''.join('<td>' + inline(c) + '</td>' for c in row) + '</tr>')
            output.append('</table></div>'); continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)', line)
            if match:
                image = (path.parent / match.group(2)).resolve()
                assert image.is_relative_to(ROOT) and image.suffix == '.png'
                output.append('<img alt="' + html.escape(match.group(1)) + '" src="data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode() + '">')
        elif line.startswith('#'):
            level = min(6, len(line) - len(line.lstrip('#')) + 1)
            output.append(f'<h{level}>' + inline(line.lstrip('#').strip()) + f'</h{level}>')
        elif line.startswith('> '): output.append('<blockquote>' + inline(line[2:]) + '</blockquote>')
        elif line.startswith('- '): output.append('<p>• ' + inline(line[2:]) + '</p>')
        elif line.strip(): output.append('<p>' + inline(line) + '</p>')
        i += 1
    return '\n'.join(output)


def source_files():
    singles = ['README.md', 'requirements.txt', 'requirements-e4.txt', 'requirements-e4-lock.txt', 'run_e4.ps1',
        'experiments/__init__.py', 'configs/data_sources.json', 'configs/e3.json', 'configs/e4.json', 'configs/e4_followup.json',
        'results/e1/handoff/combined_scores.parquet']
    directories = ['docs/e4', 'scripts', 'experiments/e4', 'experiments/e4_followup', 'experiments/e5',
        'results/e4', 'results/e4_followup', 'results/e3/handoff', 'data/processed', 'data/audit',
        'models/e4/lr2e-5', 'models/e4_followup/benign_weight2', 'models/e3/lr2e-5']
    files = {ROOT / p for p in singles}
    for d in directories:
        files.update(p for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ['.pyc', '.tmp'])
    for p in files: assert p.is_file(), p
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pr-url', default='')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / PREFIX
    destination.mkdir(exist_ok=True)
    if (destination / 'MANIFEST.json').exists(): raise RuntimeError('Use a new output directory; preserve finalized bundles.')
    for experiment in ['e4', 'e4_followup']:
        assert json.loads((ROOT / f'results/{experiment}/verification.json').read_text())['status'] == 'passed'
    checkpoint_rows = json.loads((ROOT / 'docs/e4/CHECKPOINTS.json').read_text())['checkpoints']
    for row in checkpoint_rows: assert digest(ROOT / row['path']) == row['sha256']
    files = source_files()
    required = sum(p.stat().st_size for p in files) * 3 + 1024**3
    assert shutil.disk_usage(args.output_root).free > required, 'Insufficient space; ask the user to clear space.'
    for p in files:
        target = destination / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    git = {'upstream': 'https://github.com/Run-Y/promptshield',
        'fork': 'https://github.com/dreamerYWQ/promptshield', 'pull_request_url': args.pr_url,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'merge_status_at_packaging': 'Pull request opened; upstream merge is pending maintainer action'}
    (destination / 'GIT_DELIVERY_STATUS.json').write_text(json.dumps(git, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    intro = '''# PromptShield E4 组员材料包

先打开 START_HERE.html：它包含实验全过程、主报告和 D3 补充报告，图表已嵌入，可离线阅读。

- 完整复现版含正式 E4、探索性 E4 和 E3 成本比较模型的权重。
- 阅读版含全部文字、代码、过程日志、分数、图表和冻结数据，不含三个大权重文件。
- 在解压后的目录运行 `python scripts/verify_e4_bundle.py` 核验文件。
- 正式 E4 始终使用 2e-5 / epoch 1；D3 补充实验未替换主模型，且不能当作新的盲测。
- E6 从原始分数统一选择正式阈值。E3/E5 支持材料注明来源，不计作 E4 原创贡献。
- Git 更新地址与提交哈希见 GIT_DELIVERY_STATUS.json。上游合并请求待仓库维护者处理。

详细复现命令、组员分工、学术披露说明见 docs/e4/TEAM_HANDOFF_ZH.md。
'''
    (destination / 'START_HERE.md').write_text(intro, encoding='utf-8')
    head = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PromptShield E4 完整实验与交接</title><style>
body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1160px;margin:32px auto;padding:0 24px;color:#223542}h1{font-size:32px}h2{border-bottom:2px solid #dce7ec;padding-bottom:8px;margin-top:42px}h3{margin-top:28px}a{color:#087e7d}nav,.note{background:#edf5f5;padding:16px 24px;border-radius:10px}nav a{margin-right:22px}img{display:block;max-width:100%;height:auto;margin:20px auto}.table{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{border:1px solid #d9e3e8;padding:8px;text-align:left;vertical-align:top}th{background:#edf4f8}code{background:#eef2f5;padding:2px 4px;overflow-wrap:anywhere}pre{overflow-x:auto;padding:14px;background:#eef2f5}blockquote{border-left:3px solid #75a1b5;margin-left:0;padding-left:16px;color:#425866}section{scroll-margin-top:12px}</style><body>
<h1>PromptShield E4 完整实验与交接</h1><p class="note">主实验已核验完成。正式模型为 RoBERTa，学习率 2e-5、epoch 1。D3 补充实验属于探索性分析，不替代正式模型。</p><nav><a href="#process">全过程与分工</a><a href="#main">主实验报告</a><a href="#followup">D3 误报与改进</a><a href="#git">Git 状态</a></nav>'''
    sections = [head]
    for anchor, source in [('process', 'docs/e4/TEAM_HANDOFF_ZH.md'), ('main', 'results/e4/report_section.md'),
                           ('followup', 'results/e4_followup/D3误报诊断与改进实验.md')]:
        sections.append('<section id="' + anchor + '">' + md_html(ROOT / source) + '</section>')
    sections.append('<section id="git"><h2>Git 更新</h2><pre>' + html.escape(json.dumps(git, indent=2, ensure_ascii=False)) + '</pre></section></body></html>')
    (destination / 'START_HERE.html').write_text('\n'.join(sections), encoding='utf-8')
    records = [{'path': p.relative_to(destination).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
               for p in sorted(destination.rglob('*')) if p.is_file()]
    large = [r for r in records if r['path'].endswith('/model.safetensors')]
    assert len(large) == 3
    small = [r for r in records if r not in large]
    archives = []
    for variant, entries in [('FULL', records), ('REVIEW', small)]:
        manifest = {'variant': variant, 'git': git, 'files': entries,
            'omitted_large_weights': [] if variant == 'FULL' else [r['path'] for r in large]}
        payload = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
        if variant == 'FULL': (destination / 'MANIFEST.json').write_text(payload, encoding='utf-8')
        archive_path = args.output_root / f'PromptShield_E4_{variant}.zip'
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as archive:
            archive.writestr(PREFIX + '/MANIFEST.json', payload)
            for row in entries: archive.write(destination / row['path'], PREFIX + '/' + row['path'])
        with zipfile.ZipFile(archive_path) as archive: assert archive.testzip() is None
        archives.append({'file': archive_path.name, 'bytes': archive_path.stat().st_size,
                         'sha256': digest(archive_path), 'files': len(entries) + 1})
        print(json.dumps(archives[-1]), flush=True)
    receipt = {'archives': archives, 'all_source_files_copied': len(files), 'git': git}
    (args.output_root / 'DELIVERY_RECEIPT.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.output_root / 'SHA256SUMS.txt').write_text('\n'.join(r['sha256'] + '  ' + r['file'] for r in archives) + '\n', encoding='utf-8')
    print('Bundle complete and ZIP CRC verified.', flush=True)


if __name__ == '__main__': main()
