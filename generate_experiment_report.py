#!/usr/bin/env python3
"""Create a resize/LR experiment Excel report from experiment_results/ (or a ZIP archive).

No third-party dependencies. Reads only results/fold_results.csv, not NPZ files.

Usage:
  python3 generate_experiment_report.py --results experiment_results
  python3 generate_experiment_report.py --results experiment_results.zip
  python3 generate_experiment_report.py --results /scratch/.../resize_lr_experiment/experiment_results --output report.xlsx

The workbook mirrors the supplied report's two worksheets: Fold Results and
Model Selection. Only complete five-fold runs are eligible for model selection.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from xml.etree import ElementTree as ET

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
RNS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PN = 'http://schemas.openxmlformats.org/package/2006/relationships'
CN = 'http://schemas.openxmlformats.org/package/2006/content-types'
ET.register_namespace('', NS)
ET.register_namespace('r', RNS)

STRATEGIES = [
    ('resize', 'Full image resize'),
    ('big_center_crop_resize', 'Big center crop + resize'),
    ('small_center_crop_resize', 'Small center crop + resize'),
]
# Match the ordering in the supplied workbook (NOT sorted numerical order).
LR_ORDER = [0.001, 0.0001, 0.0003]
MODELS = [*(f'efficientnet_b{i}' for i in range(8)),
          'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']

COLORS = {
    'dark': '1F4E78', 'pale': 'D9EAF7', 'white': 'FFFFFF',
    'gray': 'F3F4F6', 'blue': 'C9DAF8', 'green': 'C6EFCE',
    'yellow': 'FFEB9C', 'text': '233247', 'grid': 'D9E2EA',
}


def tag(name: str) -> str:
    return f'{{{NS}}}{name}'


def column(n: int) -> str:
    out = ''
    while n:
        n, d = divmod(n - 1, 26)
        out = chr(65 + d) + out
    return out


def fmt_model(model: str) -> str:
    x = re.fullmatch(r'efficientnet_b(\d+)', model)
    if x:
        return f'EfficientNet-B{x.group(1)}'
    x = re.fullmatch(r'efficientnet_v2_(\w+)', model)
    if x:
        return f'EfficientNet-V2-{x.group(1).upper()}'
    return model.replace('_', ' ').title()


def model_key(model: str):
    return (0, MODELS.index(model)) if model in MODELS else (1, model)


def config_key(key):
    lr, strategy = key
    try:
        lr_order = (0, LR_ORDER.index(lr))
    except ValueError:
        lr_order = (1, lr)
    order = [x[0] for x in STRATEGIES]
    return (lr_order, order.index(strategy) if strategy in order else 99, strategy)


def read_runs(root: Path, expected_folds: int):
    """Return model -> (learning-rate, resize-strategy) -> run details."""
    if not root.exists():
        raise FileNotFoundError(f'Results path does not exist: {root}')

    records = {}
    sources = []
    if root.is_dir():
        for path in root.rglob('fold_results.csv'):
            if path.parent.name == 'results':
                sources.append((str(path), lambda p=path: p.read_text(encoding='utf-8-sig')))
    elif zipfile.is_zipfile(root):
        # Do not extract a multi-GB archive or load all files into memory.
        with zipfile.ZipFile(root) as z:
            members = [name for name in z.namelist()
                       if name.endswith('/results/fold_results.csv')]
            for member in sorted(members):
                sources.append((member, z.read(member).decode('utf-8-sig')))
    else:
        raise ValueError('Provide a directory or ZIP archive of experiment results.')

    for source, content in sources:
        try:
            data = content() if callable(content) else content
            rows = list(csv.DictReader(io.StringIO(data)))
            if not rows:
                print(f'WARNING: empty {source}', file=sys.stderr)
                continue
            needed = {'fold', 'model', 'experiment_id', 'learning_rate',
                      'resize_strategy', 'test_accuracy', 'test_macro_f1'}
            if not needed.issubset(rows[0]):
                raise ValueError(f'Missing columns: {sorted(needed - set(rows[0]))}')
            first = rows[0]
            model = first['model'].strip().lower()
            run_id = first['experiment_id'].strip()
            lr = float(first['learning_rate'])
            strategy = first['resize_strategy'].strip()
            folds = {}
            for row in rows:
                if row['model'].strip().lower() != model or row['experiment_id'].strip() != run_id:
                    raise ValueError('Mixed model or experiment IDs in one fold_results.csv')
                if float(row['learning_rate']) != lr or row['resize_strategy'].strip() != strategy:
                    raise ValueError('Mixed learning rates or resize strategies in one fold_results.csv')
                fold = int(row['fold'])
                if not 1 <= fold <= expected_folds:
                    raise ValueError(f'Unexpected fold number {fold}')
                if fold in folds:
                    raise ValueError(f'Duplicate fold number {fold}')
                acc = float(row['test_accuracy'])
                f1 = float(row['test_macro_f1'])
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in (acc, f1)):
                    raise ValueError(f'Non-finite/out-of-range metric in fold {fold}')
                folds[fold] = {'accuracy': acc, 'f1': f1}
            key = (model, (lr, strategy))
            if key in records:
                raise ValueError(f'Duplicate model/configuration {key}; see {source}')
            records[key] = {'model': model, 'id': run_id, 'lr': lr,
                            'strategy': strategy, 'folds': folds,
                            'complete': len(folds) == expected_folds}
        except (OSError, ValueError, KeyError, UnicodeError) as exc:
            print(f'WARNING: skipped {source}: {exc}', file=sys.stderr)

    if not records:
        raise ValueError('No valid results/fold_results.csv files found.')
    return records


def select_runs(records, expected_folds):
    per_model = defaultdict(list)
    for (model, _), run in records.items():
        if run['complete']:
            metrics = [run['folds'][i] for i in range(1, expected_folds + 1)]
            run['mean_acc'] = mean(x['accuracy'] for x in metrics)
            run['mean_f1'] = mean(x['f1'] for x in metrics)
            run['best_fold'] = max(run['folds'],
                                   key=lambda i: (run['folds'][i]['f1'], -i))
            per_model[model].append(run)
    selected = {}
    for model, runs in per_model.items():
        # Primary criterion: mean TEST macro F1 across all five CV folds.
        # Tie-break: mean test accuracy, then deterministic experiment ID.
        selected[model] = sorted(runs, key=lambda r: (-r['mean_f1'],
                                                       -r['mean_acc'], r['id']))[0]
    # Top four highlighted by mean F1 (not by the winning single fold).
    best_order = sorted(selected, key=lambda m: (-selected[m]['mean_f1'],
                                                -selected[m]['mean_acc'], m))
    top_four = set(best_order[:4])
    # Match the attachment's 'Selected-Fold Rank', based on highest F1 fold.
    fold_order = sorted(selected, key=lambda m: (-selected[m]['folds'][selected[m]['best_fold']]['f1'],
                                                  -selected[m]['mean_f1'], m))
    fold_ranks = {m: idx + 1 for idx, m in enumerate(fold_order)}
    return selected, top_four, fold_ranks


class Styles:
    """Tiny dependency-free OOXML styles registry."""
    def __init__(self):
        self.entries = []
        self.indices = {}

    def get(self, background='white', font='normal', number='general', align='center', wrap=False):
        key = (background, font, number, align, wrap)
        if key not in self.indices:
            self.indices[key] = len(self.entries)
            self.entries.append(key)
        return self.indices[key]

    def xml(self) -> bytes:
        root = ET.Element(tag('styleSheet'))
        numfmts = ET.SubElement(root, tag('numFmts'), count='2')
        ET.SubElement(numfmts, tag('numFmt'), numFmtId='164', formatCode='0.0000')
        ET.SubElement(numfmts, tag('numFmt'), numFmtId='165', formatCode='0.000000')
        fonts = ET.SubElement(root, tag('fonts'), count='6')
        # Normal, bold, title white, header white, subtitle, emphasized.
        for size, bold, color, italic in [
            (10, False, COLORS['text'], False),
            (10, True, COLORS['text'], False),
            (16, True, 'FFFFFF', False),
            (10, True, 'FFFFFF', False),
            (10, False, COLORS['text'], True),
            (10, True, '245B38', False),
        ]:
            f = ET.SubElement(fonts, tag('font'))
            if bold: ET.SubElement(f, tag('b'))
            if italic: ET.SubElement(f, tag('i'))
            ET.SubElement(f, tag('sz'), val=str(size))
            ET.SubElement(f, tag('color'), rgb='FF' + color)
            ET.SubElement(f, tag('name'), val='Aptos')
        fills = ET.SubElement(root, tag('fills'), count='9')
        for fill_name in ['none', 'gray125', 'dark', 'pale', 'white', 'gray',
                          'blue', 'green', 'yellow']:
            fill = ET.SubElement(fills, tag('fill'))
            if fill_name == 'none':
                ET.SubElement(fill, tag('patternFill'), patternType='none')
            elif fill_name == 'gray125':
                ET.SubElement(fill, tag('patternFill'), patternType='gray125')
            else:
                pattern = ET.SubElement(fill, tag('patternFill'), patternType='solid')
                ET.SubElement(pattern, tag('fgColor'), rgb='FF' + COLORS[fill_name])
                ET.SubElement(pattern, tag('bgColor'), indexed='64')
        borders = ET.SubElement(root, tag('borders'), count='2')
        border = ET.SubElement(borders, tag('border'))
        for side in ('left', 'right', 'top', 'bottom', 'diagonal'):
            ET.SubElement(border, tag(side))
        border = ET.SubElement(borders, tag('border'))
        for side in ('left', 'right', 'top', 'bottom'):
            edge = ET.SubElement(border, tag(side), style='hair')
            ET.SubElement(edge, tag('color'), rgb='FF' + COLORS['grid'])
        ET.SubElement(border, tag('diagonal'))
        cs = ET.SubElement(root, tag('cellStyleXfs'), count='1')
        ET.SubElement(cs, tag('xf'), numFmtId='0', fontId='0', fillId='0', borderId='0')
        ids = {'normal': 0, 'bold': 1, 'title': 2, 'header': 3, 'subtitle': 4,
               'winner': 5}
        fillids = {'none': 0, 'dark': 2, 'pale': 3, 'white': 4, 'gray': 5,
                   'blue': 6, 'green': 7, 'yellow': 8}
        nums = {'general': 0, 'percent': 10, 'f1': 164, 'lr': 165}
        xs = ET.SubElement(root, tag('cellXfs'), count=str(len(self.entries)))
        for bg, font, number, align, wrap in self.entries:
            x = ET.SubElement(xs, tag('xf'), numFmtId=str(nums[number]),
                              fontId=str(ids[font]), fillId=str(fillids[bg]),
                              borderId='0' if font in ('title', 'subtitle') else '1',
                              xfId='0', applyAlignment='1', applyFont='1',
                              applyFill='1', applyBorder='1', applyNumberFormat='1')
            ET.SubElement(x, tag('alignment'), horizontal=align,
                          vertical='center', wrapText='1' if wrap else '0')
        cs = ET.SubElement(root, tag('cellStyles'), count='1')
        ET.SubElement(cs, tag('cellStyle'), name='Normal', xfId='0', builtinId='0')
        return xml_bytes(root)


def xml_bytes(elem) -> bytes:
    content = ET.tostring(elem, encoding='utf-8')
    # OOXML package-level XML uses a default namespace (rather than ns0).
    if elem.tag in (f'{{{PN}}}Relationships', f'{{{CN}}}Types'):
        content = content.replace(b'ns0:', b'').replace(b'xmlns:ns0=', b'xmlns=')
    return b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + content


class Sheet:
    def __init__(self, styles: Styles, widths, freeze=(0, 0)):
        self.styles = styles
        self.widths = widths
        self.freeze = freeze
        self.rows = defaultdict(list)
        self.heights = {}
        self.merged = []
        self.autofilter = None

    def add(self, row, col, value=None, bg='white', font='normal',
            number='general', align='center', wrap=False):
        sty = self.styles.get(bg, font, number, align, wrap)
        attrs = {'r': f'{column(col)}{row}', 's': str(sty)}
        if value is None:
            return
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            cell = ET.Element(tag('c'), attrs)
            ET.SubElement(cell, tag('v')).text = str(value)
        else:
            cell = ET.Element(tag('c'), {**attrs, 't': 'inlineStr'})
            si = ET.SubElement(cell, tag('is'))
            ET.SubElement(si, tag('t')).text = str(value)
        self.rows[row].append((col, cell))

    def merge(self, r1, c1, r2, c2):
        self.merged.append(f'{column(c1)}{r1}:{column(c2)}{r2}')

    def xml(self) -> bytes:
        root = ET.Element(tag('worksheet'))
        views = ET.SubElement(root, tag('sheetViews'))
        view = ET.SubElement(views, tag('sheetView'), workbookViewId='0', showGridLines='0')
        x, y = self.freeze
        if x or y:
            ET.SubElement(view, tag('pane'), xSplit=str(x), ySplit=str(y),
                          topLeftCell=f'{column(x + 1)}{y + 1}', activePane='bottomRight',
                          state='frozen')
        ET.SubElement(root, tag('sheetFormatPr'), defaultRowHeight='17')
        cols = ET.SubElement(root, tag('cols'))
        for a, b, width in self.widths:
            ET.SubElement(cols, tag('col'), min=str(a), max=str(b),
                          width=str(width), customWidth='1')
        data = ET.SubElement(root, tag('sheetData'))
        for rn in sorted(self.rows):
            kw = {'r': str(rn)}
            if rn in self.heights:
                kw.update(ht=str(self.heights[rn]), customHeight='1')
            row = ET.SubElement(data, tag('row'), kw)
            for col, cell in sorted(self.rows[rn], key=lambda item: item[0]):
                row.append(cell)
        if self.merged:
            merges = ET.SubElement(root, tag('mergeCells'), count=str(len(self.merged)))
            for ref in self.merged:
                ET.SubElement(merges, tag('mergeCell'), ref=ref)
        if self.autofilter:
            ET.SubElement(root, tag('autoFilter'), ref=self.autofilter)
        ET.SubElement(root, tag('pageMargins'), left='0.25', right='0.25',
                      top='0.5', bottom='0.5', header='0.2', footer='0.2')
        return xml_bytes(root)


def make_workbook(records, selected, top_four, fold_ranks, expected_folds):
    styles = Styles()
    models = sorted({model for model, _ in records}, key=model_key)
    configs = sorted({config for _, config in records}, key=config_key)
    col_max = 2 + 3 * len(models)
    fold_sheet = Sheet(styles, [(1, 1, 33), (2, 2, 8)] +
                       [(3 + 3 * i, 5 + 3 * i, 15) for i in range(len(models))],
                       freeze=(2, 5))
    # Note: width groups apply uniformly to each three-column model block.
    fold_sheet.add(1, 1, 'Resize / Learning-Rate Fold-Level Test Results', bg='dark', font='title', align='left')
    fold_sheet.merge(1, 1, 1, col_max)
    fold_sheet.add(2, 1,
        'Best configuration per model = highest mean test Macro F1 across complete CV folds. '
        'Winning fold highlighted (green for top 4, yellow otherwise); top 4 mean-F1 models blue; B0 control light blue.',
        bg='pale', font='subtitle', align='left', wrap=True)
    fold_sheet.merge(2, 1, 2, col_max)
    fold_sheet.heights.update({1: 28, 2: 38, 4: 30, 5: 24})
    fold_sheet.add(4, 1, 'Configuration', bg='dark', font='header')
    fold_sheet.add(4, 2, 'Fold', bg='dark', font='header')
    fold_sheet.merge(4, 1, 5, 1)
    fold_sheet.merge(4, 2, 5, 2)
    for mi, model in enumerate(models):
        c = 3 + 3 * mi
        fold_sheet.add(4, c, fmt_model(model), bg='dark', font='header')
        fold_sheet.merge(4, c, 4, c + 2)
        for j, header in enumerate(['Config ID', 'Accuracy', 'Macro F1']):
            fold_sheet.add(5, c + j, header, bg='dark', font='header')

    for idx, (lr, strategy) in enumerate(configs):
        row0 = 6 + idx * expected_folds
        descr = dict(STRATEGIES).get(strategy, strategy.replace('_', ' '))
        label = f'Configuration {idx + 1}\nLR {lr:g}\n{descr}'
        fold_sheet.add(row0, 1, label, bg='gray' if idx % 2 else 'white',
                       font='bold', wrap=True)
        fold_sheet.merge(row0, 1, row0 + expected_folds - 1, 1)
        for fn in range(1, expected_folds + 1):
            row = row0 + fn - 1
            fold_sheet.heights[row] = 24
            fold_sheet.add(row, 2, fn, bg='gray' if idx % 2 else 'white')
            for mi, model in enumerate(models):
                run = records.get((model, (lr, strategy)))
                if run is None:
                    continue
                cv = run['folds'].get(fn)
                is_selected = selected.get(model) is run
                is_best = is_selected and run['best_fold'] == fn
                if is_best:
                    bg, font = ('green', 'winner') if (model in top_four or model == 'efficientnet_b0') else ('yellow', 'bold')
                elif is_selected and model in top_four:
                    bg, font = 'blue', 'normal'
                elif is_selected and model == 'efficientnet_b0':
                    bg, font = 'pale', 'normal'
                else:
                    bg, font = ('gray' if idx % 2 else 'white'), 'normal'
                c = 3 + 3 * mi
                fold_sheet.add(row, c, run['id'], bg=bg, font=font)
                if cv:
                    fold_sheet.add(row, c + 1, cv['accuracy'], bg=bg, font=font, number='percent')
                    fold_sheet.add(row, c + 2, cv['f1'], bg=bg, font=font, number='f1')
    footer_row = 7 + len(configs) * expected_folds
    fold_sheet.add(footer_row, 1, 'Outer CV test folds are for configuration comparison; not the separately quarantined final holdout.',
                   bg='pale', font='subtitle', align='left')
    fold_sheet.merge(footer_row, 1, footer_row, col_max)
    fold_sheet.heights[footer_row] = 24

    summary = Sheet(styles,
                    [(1, 1, 19), (2, 2, 23), (3, 3, 14), (4, 4, 19),
                     (5, 5, 17), (6, 6, 35), (7, 7, 22), (8, 8, 22),
                     (9, 9, 17), (10, 10, 29)], freeze=(0, 4))
    summary.add(1, 1, 'Best Configuration and Selected Fold by Model',
                bg='dark', font='title', align='left')
    summary.merge(1, 1, 1, 10)
    summary.add(2, 1,
                f'Best config = highest {expected_folds}-fold mean test Macro F1. '
                'Selected-fold rank = best single-fold test Macro F1 (not mean ranking).',
                bg='pale', font='subtitle', align='left')
    summary.merge(2, 1, 2, 10)
    summary.heights.update({1: 29, 2: 25, 4: 38})
    headers = ['Selected-Fold Rank', 'Model', 'Config ID', 'Configuration',
               'Learning Rate', 'Resize Strategy', 'Mean Test Accuracy',
               'Mean Test Macro F1', 'Selected Fold', 'Selected Fold Test Macro F1']
    for j, h in enumerate(headers, 1):
        summary.add(4, j, h, bg='dark', font='header', wrap=True)
    summary.autofilter = f'A4:J{4 + len(models)}'
    config_indices = {v: i + 1 for i, v in enumerate(configs)}
    for index, model in enumerate(models, start=5):
        summary.heights[index] = 25
        run = selected.get(model)
        bg = ('green' if model in top_four else 'gray' if (index - 5) % 2 else 'yellow')
        font = 'bold' if model in top_four else 'normal'
        if run is None:
            vals = ['', fmt_model(model), 'No complete run', '', '', '', '', '', '', '']
        else:
            k = (run['lr'], run['strategy'])
            vals = [fold_ranks[model], fmt_model(model), run['id'],
                    f'Configuration {config_indices[k]}', run['lr'],
                    dict(STRATEGIES).get(run['strategy'], run['strategy'].replace('_', ' ')),
                    run['mean_acc'], run['mean_f1'], run['best_fold'],
                    run['folds'][run['best_fold']]['f1']]
        for j, value in enumerate(vals, 1):
            fmt = 'lr' if j == 5 else 'percent' if j == 7 else 'f1' if j in (8, 10) else 'general'
            summary.add(index, j, value, bg=bg, font=font, number=fmt,
                        align='left' if j in (2, 4, 6) else 'center')
    return fold_sheet, summary, styles


def build_xlsx(destination: Path, sheets, styles):
    content_types = ET.Element(f'{{{CN}}}Types')
    for ext, typ in [('rels', 'application/vnd.openxmlformats-package.relationships+xml'),
                     ('xml', 'application/xml')]:
        ET.SubElement(content_types, f'{{{CN}}}Default', Extension=ext, ContentType=typ)
    for part, typ in [
        ('/xl/workbook.xml', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'),
        ('/xl/styles.xml', 'application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml'),
        *[(f'/xl/worksheets/sheet{i}.xml', 'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
          for i in range(1, len(sheets) + 1)],
    ]:
        ET.SubElement(content_types, f'{{{CN}}}Override', PartName=part, ContentType=typ)

    root_rels = ET.Element(f'{{{PN}}}Relationships')
    ET.SubElement(root_rels, f'{{{PN}}}Relationship', Id='rId1',
                  Type=RNS + '/officeDocument', Target='xl/workbook.xml')
    work_rels = ET.Element(f'{{{PN}}}Relationships')
    for i in range(1, len(sheets) + 1):
        ET.SubElement(work_rels, f'{{{PN}}}Relationship', Id=f'rId{i}',
                      Type=RNS + '/worksheet', Target=f'worksheets/sheet{i}.xml')
    ET.SubElement(work_rels, f'{{{PN}}}Relationship', Id=f'rId{len(sheets) + 1}',
                  Type=RNS + '/styles', Target='styles.xml')
    wb = ET.Element(tag('workbook'))
    wb_sheets = ET.SubElement(wb, tag('sheets'))
    for i, name in enumerate(('Fold Results', 'Model Selection'), 1):
        ET.SubElement(wb_sheets, tag('sheet'),
                      {'name': name, 'sheetId': str(i), f'{{{RNS}}}id': f'rId{i}'})

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', xml_bytes(content_types))
        z.writestr('_rels/.rels', xml_bytes(root_rels))
        z.writestr('xl/workbook.xml', xml_bytes(wb))
        z.writestr('xl/_rels/workbook.xml.rels', xml_bytes(work_rels))
        z.writestr('xl/styles.xml', styles.xml())
        for i, sheet in enumerate(sheets, 1):
            z.writestr(f'xl/worksheets/sheet{i}.xml', sheet.xml())


def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    default_results = script_dir / 'experiment_results'
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=default_results,
                        help='experiment_results folder or ZIP archive (default: script folder / experiment_results)')
    parser.add_argument('--output', type=Path, default=None,
                        help='output .xlsx path (default: resize_lr_fold_level_test_results.xlsx next to input)')
    parser.add_argument('--expected-folds', type=int, default=5,
                        help='number of CV folds per completed run (default 5)')
    args = parser.parse_args(argv)
    if args.expected_folds < 1:
        parser.error('--expected-folds must be positive')
    output = args.output or args.results.resolve().parent / 'resize_lr_fold_level_test_results.xlsx'
    if output.suffix.lower() != '.xlsx':
        parser.error('--output must have a .xlsx extension')
    records = read_runs(args.results, args.expected_folds)
    selected, top_four, fold_ranks = select_runs(records, args.expected_folds)
    sheets = make_workbook(records, selected, top_four, fold_ranks, args.expected_folds)
    build_xlsx(output, sheets[:2], sheets[2])
    completed = sum(r['complete'] for r in records.values())
    print(f'Wrote {output} ({completed}/{len(records)} complete runs; {len(selected)} models).')
    if completed < len(records):
        print(f'NOTE: {len(records) - completed} incomplete runs shown in Fold Results but excluded from model selection.')
    print('Top models by mean test macro F1:')
    for model in sorted(selected, key=lambda m: -selected[m]['mean_f1'])[:4]:
        r = selected[model]
        print(f'  {fmt_model(model):20s} {r["id"]:9s} mean F1={r["mean_f1"]:.4f} '
              f'best fold={r["best_fold"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
