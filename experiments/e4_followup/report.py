"""Reproducible figures and bilingual experiment handoff for the exploratory follow-up."""
from __future__ import annotations
import base64
import html
import re
from pathlib import Path
from experiments.e4.common import ROOT, RESULTS as BASE, read_json, sha256, write_json
from experiments.e4_followup.analyze import OUT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(frame):
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return '—' if pd.isna(x) else f'{x:.4f}'
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                       '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                      ['| ' + ' | '.join(cell(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def render_markdown(text):
    """Render the restricted, generated report format without extra dependencies."""
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)', r'<a href="\2">\1</a>', value)
        value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
        return re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
    parts, lines = [], text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip(): i += 1; continue
        if line.startswith('| '):
            rows = []
            while i < len(lines) and lines[i].startswith('| '):
                rows.append([x.strip() for x in lines[i].strip('|').split('|')]); i += 1
            parts.append('<table><tr>' + ''.join('<th>' + inline(x) + '</th>' for x in rows[0]) + '</tr>')
            for row in rows[2:]: parts.append('<tr>' + ''.join('<td>' + inline(x) + '</td>' for x in row) + '</tr>')
            parts.append('</table>'); continue
        if line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)', line)
            if match:
                image_path = OUT / match.group(2)
                parts.append('<img alt="' + html.escape(match.group(1)) + '" src="data:image/png;base64,' + base64.b64encode(image_path.read_bytes()).decode() + '">')
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            parts.append(f'<h{level}>' + inline(line[level:].strip()) + f'</h{level}>')
        elif line.startswith('> '): parts.append('<blockquote>' + inline(line[2:]) + '</blockquote>')
        elif line.startswith('- '): parts.append('<p>• ' + inline(line[2:]) + '</p>')
        else: parts.append('<p>' + inline(line) + '</p>')
        i += 1
    return '\n'.join(parts)


def figures():
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
    baseline = pd.read_parquet(BASE / 'handoff/roberta_scores.parquet')
    thresholds = pd.read_csv(OUT / 'thresholds_source_only.csv', float_precision='round_trip')
    cutoff = thresholds.query("model == 'original' and policy == 'R95_exploratory'").threshold.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for ds, split, gold, label, color, style in [
        ('d1', 'validation', 0, 'D1 validation benign', '#137a72', '-'),
        ('d1', 'validation', 1, 'D1 validation attack', '#137a72', '--'),
        ('d3', 'test', 0, 'D3 benign', '#cf5538', '-'),
        ('d3', 'test', 1, 'D3 attack', '#cf5538', '--')]:
        s = np.sort(baseline.query('dataset == @ds and split == @split and gold == @gold').score_attack)
        logit = np.log(s / (1 - s))
        ax.step(logit, np.arange(1, len(s) + 1) / len(s), where='post', label=label, color=color, linestyle=style, lw=2)
    ax.axvline(0, color='#555555', alpha=.6, lw=1)
    ax.axvline(np.log(cutoff / (1 - cutoff)), color='#8862ac', alpha=.8, lw=1.4, linestyle=':')
    ax.set(xlabel='Attack-score logit: log(p / (1 - p))', ylabel='Cumulative fraction', ylim=(0, 1.02),
           title='Benign scores shift sharply between D1 and D3')
    ax.text(.13, .45, 'Vertical lines: p = 0.5 and D1-only R95 threshold', transform=ax.transAxes, fontsize=9)
    ax.legend(loc='center left', bbox_to_anchor=(.13, .74)); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(folder / '01_score_shift.png'); fig.savefig(folder / '01_score_shift.svg'); plt.close(fig)
    rates = pd.read_csv(OUT / 'd3_technique_rates.csv').query('gold == 0')
    labels = {'hard_negative': 'Hard negative', 'innocent_use_of_ignore': 'Innocent use of "ignore"',
              'legitimate_roleplay_request': 'Legitimate role-play', 'quoting_attack_for_analysis': 'Quoted attack for analysis'}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bars = ax.barh([labels[t] for t in rates.technique], rates.positive_rate * 100, color='#cf5538', height=.62)
    for bar, row in zip(bars, rates.itertuples()):
        ax.text(2, bar.get_y() + bar.get_height() / 2, f'{row.predicted_attack}/{row.n}  ({row.positive_rate:.1%})', va='center', color='white', fontweight='bold')
    ax.set(xlim=(0, 105), xlabel='False-positive rate (%)', title='D3 benign categories: original model, threshold 0.5')
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(folder / '02_benign_categories.png'); fig.savefig(folder / '02_benign_categories.svg'); plt.close(fig)
    results = pd.read_csv(OUT / 'all_policy_metrics.csv')
    views = [('original', 'default_0.5', 'Original / 0.5', '#374c62'),
             ('original', 'R95_exploratory', 'Original / D1-R95', '#137a72'),
             ('benign_weight2', 'default_0.5', 'Weight 2 / 0.5', '#cf5538'),
             ('benign_weight2', 'R95_exploratory', 'Weight 2 / D1-R95', '#8862ac')]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for model, policy, label, color in views:
        rows = results.query('model == @model and policy == @policy').set_index('dataset_split')
        axes[0].plot(range(4), rows.loc[['d1_test', 'd2_test', 'd3_test', 'd4_test'], 'fpr'] * 100, marker='o', label=label, color=color)
        axes[1].plot(range(3), rows.loc[['d1_test', 'd2_test', 'd3_test'], 'attack_recall'] * 100, marker='o', label=label, color=color)
    for ax in axes: ax.set(ylim=(-3, 105), ylabel='Rate (%)'); ax.grid(alpha=.15)
    axes[0].set(xticks=range(4), xticklabels=['D1', 'D2', 'D3', 'D4'], title='False-positive rate (lower is better)')
    axes[1].set(xticks=range(3), xticklabels=['D1', 'D2', 'D3'], title='Attack recall (higher is better)')
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle('Source-only interventions have cross-source tradeoffs; D4 has no attacks', fontsize=12)
    fig.tight_layout(rect=[0, .12, 1, .95]); fig.savefig(folder / '03_tradeoffs.png'); fig.savefig(folder / '03_tradeoffs.svg'); plt.close(fig)


def main():
    figures()
    selection = read_json(OUT / 'selection_frozen.json')
    metrics = pd.read_csv(OUT / 'all_policy_metrics.csv')
    groups = pd.read_csv(OUT / 'd3_technique_rates.csv').query('gold == 0')
    cases = pd.read_csv(OUT / 'd3_diagnostic_cases.csv')
    masks = pd.read_csv(OUT / 'd3_case_word_masking.csv')
    policies = pd.read_csv(OUT / 'thresholds_source_only.csv', float_precision='round_trip')
    tuning = pd.read_csv(OUT / 'tuning.csv')
    percent = lambda value: f'{value:.2%}'
    summary = []
    for model, policy in [('original', 'default_0.5'), ('original', 'R95_exploratory'),
                          ('benign_weight2', 'default_0.5'), ('benign_weight2', 'R95_exploratory')]:
        rows = metrics.query('model == @model and policy == @policy').set_index('dataset_split')
        summary.append({'模型 / 阈值规则': f'{model} / {policy}', 'D3误报率': percent(rows.loc['d3_test', 'fpr']),
            'D3攻击召回': percent(rows.loc['d3_test', 'attack_recall']), 'D2攻击召回': percent(rows.loc['d2_test', 'attack_recall']),
            'D1测试攻击召回': percent(rows.loc['d1_test', 'attack_recall']), 'D4误报率': percent(rows.loc['d4_test', 'fpr'])})
    category_table = groups[['technique', 'n', 'predicted_attack', 'positive_rate']].rename(columns={'technique': '正常样本类别', 'n': '正常样本数', 'predicted_attack': '误报数', 'positive_rate': '误报率'})
    category_table['误报率'] = category_table['误报率'].map(percent)
    text = f'''# D3 高误报诊断与仅使用 D1 开发数据的补充实验

## 结论

D3 的高误报主要表现为来源变化后正常文本分数整体升高，以及模型未充分区分“讨论、引用攻击”与“真正执行攻击”。原模型在 D3 将 265 条正常文本中的 219 条判成攻击，FPR = 219/265 = **82.64%**；280 条攻击全部检出。不是把全体 545 条作分母，也不是类别编码反转。

固定的 D1 验证集 R95 阈值能把 D3 FPR 降至 **40.00%（106/265）**，仍检出 280/280 条攻击，但 D2 攻击召回从 **81.67% 降到 50.00%**。这是明显的取舍，不能称为全面修复。正常类损失加权使 D3 FPR 降至 66.42%，但验证集模型选择没有选中它。**保留原 E4 模型和正式结果；将本次结果作为补充探索，交给 E6 讨论运行阈值。**

## 实验边界

这是看过既有 D3/D4 结果后提出的探索性分析，不是新的盲测。训练只使用 D1 train；epoch、模型和阈值选择只使用 D1 validation。D2/D3/D4 文本、标签、模板均未进入训练或数值参数选择；测试数据用于错误诊断和冻结后的评估。两项干预（正常类权重 2、验证集召回下限 95%）在新的外部评估前固定，没有按 D3 结果继续搜索。

候选训练于 `{read_json(OUT / 'protocol_frozen.json')['time_utc']}` 冻结协议；模型与阈值最终于 `{selection['time_utc']}` 冻结；新的外部推理完成于 `{read_json(OUT / 'external_evaluation_receipt.json')['time_utc']}`。所有时间为 UTC。原模型权重和原始分数哈希未改变。公开全部 2 个模型 × 4 个阈值策略 × 5 个分割的 40 行结果，包括恶化项。即便没有测试集调参，方法设计已受到测试诊断启发，仍需新的独立来源作确认。

## 原因与证据

1. **正常文本分布不同。** D1 train 正常文本长度中位数 47 字符；D1 validation 为 50，D3 为 167。D1 正常样本多为普通知识/编程问句，D3 正常样本包含安全分析、引用恶意文本、合法角色扮演和工作流安全要求。D3 中 212/265 = 80% 的正常样本标为 hard_negative。D3 的 265 条正常、280 条攻击接近平衡；类别比例本身不能解释条件误报率。D3 数据来源与语义类别可参见 [原始数据卡](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-5k)。
2. **来源变化后，攻击分数失去可迁移的操作意义。** D1 validation 正常文本分数中位数 0.000377，D3 正常文本却是 0.990409；D3 攻击中位数 0.999523。D3 AUROC 仍有 0.9624，说明排名信息仍较好，但默认 0.5 的工作点很差。这与阈值/分数迁移失败一致，不等于已证明概率校准良好，也不表示存在跨来源通用的安全阈值。
3. **上下文与表面词的依赖是合理机制，但证据不能夸大。** 从 D1 train 单独提取的 30 个攻击关联词（文档频次至少 10，平滑文档对数优势比排序）在 D1 validation 正常样本中仅覆盖 8/1043 = 0.77%，在 D3 正常样本中覆盖 83/265 = 31.32%。D3 有这些词的正常样本误报 81/83 = 97.59%，没有这些词也误报 138/182 = 75.82%。所以不能把问题归结为一个关键词或直接删除关键词。该关联词表不是模型的注意力或因果解释。
4. **具体遮挡实验支持局部敏感性。** 对“配置邮件摘要工具忽略姓名大小写”的正常请求，原分数 0.845645；遮挡 `ignore` 后为 0.108383，遮挡普通词 `Show` 后为 0.017690。其余三个预先按行号选出的案例，即使遮挡最有影响的单词仍高于 0.5，提示上下文模式和饱和分数也在作用。遮挡引入分布变化，只是局部敏感性证据，不能据此声称某个词是误报的唯一原因。相关研究 [PIGuard（ACL 2025）](https://aclanthology.org/2025.acl-long.1468/) 也研究了触发词导致的过度防御；这里没有复现该论文算法。
5. **已排除的实现问题。** 原 E4 全量分数已复算；本次再次验证数据清单哈希、ID 与标签对齐、阈值方向及 checkpoint 哈希。D3 最大 77 个 RoBERTa token，低于长度上限 256，没有截断。现有证据不支持把高误报解释为截断、标签反转或评估分母错误；尚未进行逐条人工标签复核，不能声称所有 D3 标签都绝对正确。

![分数分布](figures/01_score_shift.png)

{table(category_table)}

![正常类别误报率](figures/02_benign_categories.png)

## 两项固定干预的结果

{table(pd.DataFrame(summary))}

**A. 原模型 + D1 验证集 R95 阈值。** 用 D1 validation 在所有唯一分数边界中选择“攻击召回至少 95%”的最高阈值（预测规则 score >= threshold），得到 `0.9952170848846436`。验证集召回 895/942 = 95.01%，误报 2/1043 = 0.19%。该约束只保证这份验证集的经验结果，不保证 D1 test、D2 或未来数据的召回。相对默认 0.5，D3 少 113 条误报（下降 42.64 个百分点），但 D2 多漏 19/60 条攻击，D1 test 多漏 71/1051 条攻击；D3 Macro-F1 从 0.5073 升至 0.7954，D2 却从 0.8963 降至 0.7277。

**B. 正常类损失权重 2。** 从同一 RoBERTa 预训练 revision 重新训练；seed 6405、学习率 2e-5、3 epochs、batch 16、FP32、AdamW、数据顺序与原 2e-5 分支相同。每个 batch 使用 `sum(class_weight[y] * CE) / sum(class_weight[y])`，权重为正常 2、攻击 1；不增加 D3 样本，不进行重采样。D1 validation 在默认 0.5 下选择 epoch {selection['weighted_best_epoch']}，Macro-F1 {selection['weighted_dev_macro_f1']:.6f}，低于原模型 {selection['original_dev_macro_f1']:.6f}，因此预定规则选择 `original`。三轮训练纯更新耗时约 {tuning.train_seconds.sum():.1f} 秒，峰值显存约 {tuning.peak_allocated_mib.max():.0f} MiB。

加权模型默认 0.5 在 D3 少误报 43 条，但 D3 AUROC 从 0.9624 降为 0.9425；D2 FPR 从 1.79% 升到 7.14%，召回从 81.67% 降至 78.33%。这更像改变错误权衡，未解决跨来源语义区分问题。虽然加权模型按 tF1 调阈值后的验证 F1 更高，本次预注册的模型选择指标是默认 0.5 下的验证 F1，不能事后换标准来选它。

![来源间取舍](figures/03_tradeoffs.png)

## tF1 / t5 为什么没有解决 D3

两者这里只提供参考实现，不替代 E6 的正式阈值交付。tF1 在 D1 validation 最大化 Macro-F1（同分取最高阈值）；t5 在验证 FPR <= 5% 内最大化攻击召回（同分取最高阈值）。原模型分别得到 0.113247 与 0.000815，反而比 0.5 更低。它们在 D3 的 FPR 为 87.17%、99.25%。因此“在 D1 验证集保证 5% FPR”不等于“D3 上也会低于 5%”。仅做正温度缩放且继续使用 0.5，也不会改变二分类 logit 差值的正负，因此不能单靠这一操作改变误报判断。

## 如何使用这些结果

- 保留原 E4 模型与正式分数，报告明确陈述其跨来源过度防御问题。不要因为看到了 D3 的改善，改用加权模型或继续搜索更好的 D3 阈值。
- 如果业务事先明确愿意用更多漏报换更少误报，可将 R95 作为候选操作策略；本次结果不足以支持它作为默认安全策略。它在 D3 仍误报 40%，且 D2 攻击召回只有 50%。
- 更可能解决语义区分问题的后续方向是独立收集正常安全分析、引用攻击、合法角色扮演等困难负例，且同时保留对应真实攻击，按来源/模板划分训练和验证。不要复制 D3/D4 测试文本或套其模板来造训练数据。由于本项目协议仅允许 D1 训练，这次没有偷偷增加外部训练数据。
- 未来实验应在重新训练前固定数据、方法、候选范围和目标，然后使用新的未查看过的测试来源。不同误报与漏报代价应预先声明。看到当前测试结果后再拟定的新实验仍须标为探索。

## 不确定性

`paired_differences.csv` 给出 2000 次、seed 6405 的逐样本配对 bootstrap 差值区间。原模型 R95 相对 0.5 的 D3 FPR 下降 42.64 个百分点，描述性 95% 区间约 [36.97, 48.30]；D2 召回下降 31.67 个百分点，区间约 [21.63, 43.33]。D3 为合成模板数据，样本之间可能相关，这些区间没有校正模板聚类、方法选择或单随机种子的不确定性，不能当作跨来源泛化保证。D3 280/280 检出也不能证明真实世界 100% 召回。

## 典型误报与遮挡证据

案例固定为每个正常类别按 source_row 排序的第一个基线误报，不按遮挡结果挑选。全文来自本地冻结的 D3 测试数据，仅用于解释。
'''
    for row in cases.itertuples():
        top = masks[masks.sample_id == row.sample_id].sort_values('drop', ascending=False).head(3)
        text += f'\n### {row.sample_id} · {row.technique}\n\n> {row.text}\n\n原攻击分数：{row.score_attack:.6f}。\n\n'
        text += table(top[['word', 'original_score', 'masked_score', 'drop']]) + '\n'
    text += '\n## 完整验证阈值\n\n' + table(policies[['model', 'policy', 'threshold', 'fp', 'fn', 'macro_f1', 'attack_recall', 'fpr']])
    text += '\n\n完整精度以 `thresholds_source_only.csv` 为准；表中四位小数不能用于实际推理。\n'
    for split in ['d1_validation', 'd1_test', 'd2_test', 'd3_test', 'd4_test']:
        text += f'\n## 全部结果：{split}\n\n' + table(metrics[metrics.dataset_split == split][['model', 'policy', 'tn', 'fp', 'fn', 'tp', 'fpr', 'attack_recall', 'macro_f1', 'auroc']]) + '\n'
    text += '''
## 复现、文件和审计

代码位于 `experiments/e4_followup/`，参数在 `configs/e4_followup.json`，输出在 `results/e4_followup/`，探索模型在 `models/e4_followup/benign_weight2/`。原 E4 文件保持不变。训练仅在干净的独立目录运行 `python -m experiments.e4_followup.train_source`；训练脚本会拒绝覆盖已冻结结果。随后运行 `normalize_thresholds`、`analyze`、`report`（相同模块前缀）。重复 analyze 使用已缓存的新模型分数并重新核验哈希，不会重新选择模型。

审计时发现 pandas 默认 CSV 解析在边界阈值带来约 1.11e-16 的精度差；在任何新的测试推理前改为 round-trip 解析，仅用 D1 validation 重算阈值，模型选择和验证混淆矩阵不变，保留修正前文件及 `precision_correction.json`。另外，推理第一次因 attention 后端与训练不一致而未通过 D1 validation 分数复现检查，在对新模型进行后续测试推理前改回原 SDPA / deterministic 设置；最终检查通过。上述工程修正没有改变实验策略。

`verification.json` 记录验证项；`protocol_frozen.json`、`selection_frozen.json`、`external_evaluation_receipt.json` 记录先后顺序与哈希；`weighted_scores.parquet` 保存全 5086 行新模型分数。诊断的词表、分数分位数、长度分组、逐词遮挡、全部 40 行指标以及配对差值均保留。本补充实验由 AI 协助编写与运行，提交者应自行复核理解，并在课程 AI disclosure 中如实披露，不把它写成无辅助的个人原创实验。

## English report paragraph

D3 exhibits pronounced over-defense under the original RoBERTa detector: 219 of 265 benign examples are flagged at threshold 0.5, despite an AUROC of 0.9624. Benign-score medians shift from 0.000377 on D1 validation to 0.990409 on D3. Category-level errors and local masking diagnostics suggest sensitivity to attack-related surface patterns and insufficient discrimination between quoted or discussed instructions and active attacks; these observations do not establish a unique causal mechanism. Two exploratory interventions were fixed before new external evaluation. A threshold selected solely on D1 validation to retain at least 95% empirical attack recall reduces D3 FPR to 40.00% with 100% observed D3 recall, but reduces D2 recall from 81.67% to 50.00%. Doubling the benign training-loss weight reduces D3 FPR to 66.42%, yet underperforms the original model on the prespecified validation selection metric and is not selected. The original E4 model is retained. All fitting and numerical selection use D1 development data, but the study was motivated by previously inspected test behavior and therefore remains exploratory; confirmation requires a fresh held-out source.
'''
    (OUT / 'D3误报诊断与改进实验.md').write_text(text, encoding='utf-8')
    # Small standalone HTML overview with embedded scientific figures, plus full readable text.
    sections = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>D3 误报诊断与改进实验</title><style>body{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:36px auto;padding:0 24px;color:#203342}h1{font-size:30px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}td,th{padding:8px;border:1px solid #d9e1e7;text-align:left}th{background:#edf3f6}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:14px;background:#f5f8fa;padding:20px}a{color:#137a72}.lead{background:#edf6f3;padding:20px;border-left:5px solid #137a72}</style>',
        '<h1>D3 误报诊断与改进实验</h1><p class="lead"><b>D3 误报可以降低，但目前没有无代价的跨来源修复。</b><br>只用 D1 验证集确定的 R95 阈值：D3 FPR 82.64% → 40.00%；D2 攻击召回 81.67% → 50.00%。原 E4 模型保留，本次结果属于探索性补充。</p>',
        '<h2>固定干预的取舍</h2>', pd.DataFrame(summary).to_html(index=False, border=0)]
    for path in sorted((OUT / 'figures').glob('*.png')):
        sections.append('<img alt="' + html.escape(path.stem) + '" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">')
    sections.extend(['<h2>完整分析、结果与审计</h2><p>实际推理请使用 CSV 中完整精度的阈值。以下包含全部策略和反例。</p>', render_markdown(text), '</html>'])
    (OUT / 'D3误报分析.html').write_text(''.join(sections), encoding='utf-8')
    write_json(OUT / 'report_artifacts.json', {p.relative_to(OUT).as_posix(): sha256(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix in ['.md', '.html', '.png', '.svg']})
    print('Created report and three figure pairs.')


if __name__ == '__main__': main()
