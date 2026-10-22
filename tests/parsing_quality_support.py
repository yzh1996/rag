"""解析质量测试的公共工具，不参与正式上传流程。

分工：读取标准答案 → 两条解析路线 → 逐项核对 → 保存可阅读的报告。
“程序没有报错”和“内容正确”是两个独立结论，报告分别记录。
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

from tests.parsing_quality_preview import build_original_preview, render_original_panel

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / '测试资源' / '文档解析质量'
ARTIFACTS = ROOT / 'tests' / '测试产物' / '文档解析质量'
LABELS = {'pass': '通过', 'fail': '不通过', 'not_executed': '未执行', 'not_applicable': '不适用'}


def load_cases():
    manifest = json.loads((FIXTURES / 'manifest.json').read_text(encoding='utf-8'))
    cases = []
    for relative in manifest['cases']:
        golden = FIXTURES / relative
        record = json.loads(golden.read_text(encoding='utf-8'))
        record['path'] = str(golden.parent / record['file'])
        record['golden_path'] = str(golden)
        cases.append(record)
    return cases


def compact(text):
    # 仅忽略空白，不替换 O/0、负号、小数点、大小写或否定词。
    return re.sub(r'\s+', '', str(text))


def character_error_rate(expected, actual):
    """编辑距离 / 标准字数；漏字、多字、错字都会计入，可大于 100%。"""
    expected, actual = compact(expected), compact(actual)
    previous = list(range(len(actual) + 1))
    for row, left in enumerate(expected, 1):
        current = [row]
        for column, right in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left != right)))
        previous = current
    return previous[-1] / max(1, len(expected))


class _HtmlTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == 'tr' and self.row:
            self.rows.append(self.row)


def structured_rows(documents):
    """只承认已输出的表格结构；不回读源 Excel/PDF 给解析结果补答案。"""
    rows = []
    for document in documents:
        for line in document['text'].splitlines():
            if line.strip().startswith('|') and line.strip().endswith('|'):
                cells = [cell.strip() for cell in re.split(r'(?<!\\)\|', line.strip())[1:-1]]
                if cells and not all(re.fullmatch(r':?-+:?', cell or ' ') for cell in cells):
                    rows.append(cells)
        parser = _HtmlTables()
        parser.feed(document['metadata'].get('text_as_html', ''))
        rows.extend(parser.rows)
    return rows


def serialize(documents):
    return [{'text': doc.page_content, 'metadata': dict(doc.metadata)} for doc in documents]


def extract_case(case, route, *, run_ocr=False):
    """原生路线直接调用 LangChain；增强路线才经过项目的校验/OCR/清洗。"""
    from config.settings import settings
    from langchain_community.document_loaders import (
        Docx2txtLoader, PDFPlumberLoader, PyMuPDFLoader, UnstructuredExcelLoader,
    )

    path = case['path']
    result = dict(route=route, execution='success', error=None, error_type=None, documents=[], raw_documents=[],
                  ocr_opt_in=run_ocr, ocr_required_pages=[], ocr_used_pages=[])
    started = time.perf_counter()
    try:
        if route == 'native':
            # 不调用 DocumentLoader：它的“raw”入口已经含项目专属增强。
            suffix = Path(path).suffix.lower()
            loader_class = {'.docx': Docx2txtLoader, '.xlsx': UnstructuredExcelLoader}.get(suffix)
            if suffix == '.pdf':
                loader_class = PDFPlumberLoader if case['loader'] == 'pdfplumber' else PyMuPDFLoader
            result['loader'] = loader_class.__name__
            documents = loader_class(path).load()
            result['raw_documents'] = serialize(documents)
        else:
            from core.content_processing import clean_documents, recognize_documents
            from core.document_loader import DocumentLoader

            # 设置只在本次调用生效，不改 .env、不改全局默认值；退出后自动恢复。
            with ExitStack() as stack:
                for key, value in case.get('settings', {}).items():
                    stack.enter_context(patch.object(settings, key, value))
                stack.enter_context(patch.object(settings, 'OCR_ENABLED', run_ocr))
                result['effective_settings'] = {key: getattr(settings, key) for key in [
                    'DOCUMENT_CLEANING_ENABLED', 'REMOVE_REPEATED_PAGE_EDGES', 'OCR_ENABLED',
                    'OCR_PROVIDER', 'OCR_MIN_TEXT_CHARS', 'MAX_PDF_PAGES', 'MAX_ARCHIVE_ENTRIES',
                ]}
                documents = DocumentLoader().load_raw_file(path, metadata=case.get('input_metadata'), loader_name=case['loader'])
                result['raw_documents'] = serialize(documents)
                result['ocr_required_pages'] = [doc.metadata['page'] for doc in documents if doc.metadata.get('ocr_required')]
                # 未明确开启真实 OCR 时，保留已提取内容供比较，绝不调用外部识别服务。
                if run_ocr:
                    documents = recognize_documents(path, documents)
                result['ocr_used_pages'] = [doc.metadata['page'] for doc in documents if doc.metadata.get('ocr_used')]
                documents = clean_documents(documents)
        result['documents'] = serialize(documents)
    except Exception as exc:
        # 保存错误到报告后，再由测试断言失败，避免只剩一条异常而没有对照产物。
        result['execution'] = 'error'
        result['error_type'] = type(exc).__name__
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    result['text'] = '\n\n'.join(doc['text'] for doc in result['documents'])
    if case.get('ocr_pages'):
        result['ocr_note'] = ('原生加载器不执行 OCR；扫描文字缺失属于基线能力限制。' if route == 'native'
                              else '已明确开启真实 OCR。' if run_ocr else '未执行真实 OCR；图片中文字的质量尚未验证。')
    return result


def evaluate(case, result):
    route = result['route']
    if case.get('expected_rejection'):
        if route == 'native' and case.get('settings'):
            return [dict(name='项目资源保护策略', status='not_applicable',
                         expected='项目自定义限制', actual='原生加载器不负责项目配置的页数/解压限制')]
        # 环境缺包等故障不能冒充“成功拒绝损坏文件”。
        rejected = result['execution'] == 'error' and result.get('error_type') in case['rejection_types']
        message_ok = route == 'native' or case.get('error_contains', '') in (result['error'] or '')
        return [dict(name='拒绝异常文件', status='pass' if rejected and message_ok else 'fail',
                     expected=case.get('error_contains') or '拒绝文件', actual=result['error'] or '未拒绝')]
    checks = []
    documents = result['documents']
    text = result['text']
    for rule in case['checks']:
        item = dict(name=rule['name'], kind=rule['kind'], status='fail')
        if rule.get('provided_by_user') and route == 'native':
            item.update(status='not_applicable', expected=rule.get('values'), actual='由上传人提供，不属于原生解析器职责')
            checks.append(item)
            continue
        if rule.get('requires_ocr') and route == 'project' and not result['ocr_opt_in']:
            item.update(status='not_executed', expected=rule.get('value'), actual='未开启真实 OCR')
            checks.append(item)
            continue
        kind = rule['kind']
        if kind == 'ocr_detection':
            item.update(expected=rule['pages'], actual=result['ocr_required_pages'])
            if route == 'native':
                item.update(status='not_applicable', actual='原生加载器无扫描页检测步骤')
            else:
                item['status'] = 'pass' if item['actual'] == item['expected'] else 'fail'
            checks.append(item)
            continue
        if result['execution'] == 'error':
            item.update(expected=rule.get('value', rule), actual=result['error'])
            checks.append(item)
            continue
        passed = False
        if kind == 'contains':
            value = compact(rule['value'])
            # 在原文上匹配并容忍字段内空白，不能把相邻单元格的数字粘起来。
            pattern = r'\s*'.join(re.escape(char) for char in value)
            # 防止 0001089 被误判成精确匹配 000108。
            if value and value[-1].isascii() and value[-1].isalnum():
                pattern += r'(?![A-Za-z0-9.])'
            if value and value[0].isascii() and value[0].isalnum():
                pattern = r'(?<![A-Za-z0-9.])' + pattern
            passed = re.search(pattern, text) is not None
            item.update(expected=rule['value'], actual='已找到' if passed else text)
        elif kind == 'absent':
            passed = compact(rule['value']) not in compact(text)
            item.update(expected=f"不应包含：{rule['value']}", actual=text if not passed else '已去除')
        elif kind == 'order':
            positions = [text.find(value) for value in rule['values']]
            passed = all(pos >= 0 for pos in positions) and all(a < b for a, b in zip(positions, positions[1:]))
            item.update(expected=rule['values'], actual=dict(zip(rule['values'], positions)))
        elif kind == 'table':
            actual_rows = structured_rows(documents)
            # 同一行必须完整对应；数值类型转换、合并归属不做猜测补全。
            row_counts = Counter(tuple(compact(cell) for cell in row) for row in actual_rows)
            expected_counts = Counter(tuple(compact(cell) for cell in row) for row in rule['rows'])
            passed = all(row_counts[row] >= count for row, count in expected_counts.items())
            item.update(expected=rule['rows'], actual=actual_rows)
        elif kind == 'pages':
            values = [doc['metadata'].get('page') for doc in documents]
            # 只在比较时把原生 PDF 的 0 起始页码转为 1 起始，原始 JSON 保留原值。
            if route == 'native' and Path(case['path']).suffix == '.pdf':
                values = [value + 1 if isinstance(value, int) else value for value in values]
            passed = values == rule['values']
            item.update(expected=rule['values'], actual=values)
        elif kind == 'metadata_values':
            values = [doc['metadata'].get(rule['key']) for doc in documents]
            passed = all(value in values for value in rule['values'])
            item.update(expected={rule['key']: rule['values']}, actual=values)
        elif kind == 'source':
            values = [doc['metadata'].get('source') for doc in documents]
            passed = bool(values) and all(value and Path(value).name == rule['filename'] for value in values)
            item.update(expected=rule['filename'], actual=values)
        elif kind == 'heading':
            passed = re.search(r'^' + '#' * rule['level'] + r'\s+' + re.escape(rule['text']) + r'\s*$', text, re.M) is not None
            item.update(expected=f"{'#' * rule['level']} {rule['text']}（本测试约定用 Markdown 表达层级）", actual=text)
        elif kind == 'normalized':
            passed = re.search(r'[ \t]{2,}|\n{3,}|[\x00-\x08\x0b\x0c\x0e-\x1f]', text) is None
            item.update(expected='无连续多余空格、空行和控制字符', actual=text if not passed else '已规范化')
        elif kind == 'nonempty':
            passed = bool(documents) and all(doc['text'].strip() for doc in documents)
            item.update(expected='每个输出单元均非空', actual=[len(doc['text'].strip()) for doc in documents])
        elif kind == 'cer':
            rate = character_error_rate(rule['expected'], text)
            passed = rate <= rule['max_rate']
            item.update(expected=f"字符错误率 <= {rule['max_rate']:.1%}（仅忽略空白）", actual=rate)
        else:
            raise ValueError(f'未知检查类型：{kind}')
        item['status'] = 'pass' if passed else 'fail'
        checks.append(item)
    if result['execution'] == 'error':
        checks.insert(0, dict(name='解析过程正常完成', status='fail', expected='不抛异常', actual=result['error']))
    return checks


def quality_status(checks):
    statuses = {item['status'] for item in checks}
    if 'fail' in statuses:
        return 'fail'
    if 'not_executed' in statuses:
        return 'not_executed'
    if statuses == {'not_applicable'} or not statuses:
        return 'not_applicable'
    return 'pass'


class QualityReport:
    def __init__(self, target=None):
        self.target = Path(target) if target else ARTIFACTS / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        self.target.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.created = datetime.now().isoformat(timespec='seconds')

    def save(self, case, result):
        result['checks'] = evaluate(case, result)
        result['quality'] = quality_status(result['checks'])
        directory = self.target / case['id']
        directory.mkdir(exist_ok=True)
        if case['id'] not in self.records:
            self.records[case['id']] = {'case': case, 'routes': {}, 'original': build_original_preview(case, directory)}
        record = self.records[case['id']]
        record['routes'][result['route']] = result
        (directory / f"{result['route']}.txt").write_text(result['text'], encoding='utf-8')
        (directory / f"{result['route']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        self.write_index()
        return result

    def write_index(self):
        payload = {'created': self.created, 'cases_executed': len(self.records), 'records': list(self.records.values())}
        (self.target / 'report.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        counts = {route: Counter(record['routes'][route]['quality'] for record in self.records.values() if route in record['routes'])
                  for route in ['native', 'project']}
        chunks = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>文档解析质量报告</title>',
                  '<style>body{font:16px/1.65 system-ui;margin:32px auto;max-width:1900px;padding:0 24px;color:#172b4d;background:#f5f7fb}'
                  'h1{font-size:28px}article,header{background:white;padding:24px;margin:20px 0;border-radius:10px}'
                  '.routes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.route{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:12px;max-height:500px;overflow:auto}'
                  'table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #dce3ed;padding:8px;text-align:left;vertical-align:top}'
                  '.pass{color:#18754b}.fail{color:#b91c1c}.not_executed,.not_applicable{color:#875500}small{color:#526580}'
                  '.source-preview{margin-top:16px;background:#fafbfd;border:1px solid #dce3ed;padding:12px;max-height:850px;overflow:auto}'
                  'figure{margin:0 0 18px}figcaption,.preview-note{font-size:13px;color:#526580}.source-preview img{width:100%;height:auto;display:block}'
                  '.word-paragraph,.word-heading,.word-subheading,.word-title,.word-edge{white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0}'
                  '.word-title{font-size:22px;font-weight:bold}.word-heading{font-size:19px;font-weight:bold}.word-subheading{font-size:17px;font-weight:bold}'
                  '.word-edge{font-size:13px;color:#526580}.sheet-scroll{overflow:auto}.source-table td{min-width:50px}.golden{margin:12px 0}'
                  '@media(max-width:1100px){.routes{grid-template-columns:1fr}}</style>',
                  '<header><h1>文档解析质量对照报告</h1>',
                  f'<p>生成时间：{escape(self.created)}；本次已运行 {len(self.records)} 份资料。未运行的样例不计入结果。</p>',
                  '<p>三栏对照：原始资料与标准答案 → LangChain 原生解析 → 项目增强解析。<br>'
                  '原生：直接调用 LangChain 加载器。增强：项目校验 → 解析 → 按需 OCR → 清洗，未做分块。<br>'
                  '通过仅代表本次样例的已定义检查通过，不能代替真实业务抽样。未执行 OCR 不能视为准确率合格。</p>'
                  '<p><small>字段检查只忽略空白，不转换符号。中文冒号“：”变成英文冒号“:”也会判为不通过；'
                  '数字、字母和符号的实际差异请对照原件及全文查看。</small></p>']
        for route, name in [('native', '原生解析'), ('project', '项目增强')]:
            chunks.append(f'<p><b>{name}</b>：' + '；'.join(f'{LABELS[key]} {counts[route][key]}' for key in LABELS) + '</p>')
        chunks.append('</header>')
        for record in self.records.values():
            case = record['case']
            chunks.extend([f'<article id="{escape(case["id"])}"><h2>{escape(case["id"])} · {escape(case["file"])}</h2>',
                           f'<p>{escape(case["category"])}：{escape(case["description"])}</p>',
                           f'<p><a href="{Path(case["path"]).as_uri()}">原始文件</a> · <a href="{Path(case["golden_path"]).as_uri()}">标准答案</a></p>', '<div class="routes">'])
            chunks.append(render_original_panel(case, record['original']))
            for route, name in [('native', 'LangChain 原生'), ('project', '项目增强')]:
                result = record['routes'].get(route)
                chunks.append(f'<section class="route"><h3>{name}</h3>')
                if result is None:
                    chunks.append('<p class="not_executed">本次未执行此路线</p></section>')
                    continue
                chunks.append(f'<p class="{result["quality"]}">质量：{LABELS[result["quality"]]}；耗时 {result["elapsed_seconds"]} 秒</p>')
                chunks.append(f'<p>解析过程：{"抛出异常" if result["execution"] == "error" else "正常返回"}</p>')
                if result.get('ocr_note'):
                    chunks.append(f'<p>{escape(result["ocr_note"])}</p>')
                if route == 'project' and case.get('ocr_pages'):
                    chunks.append('<p>实际完成 OCR 的页码：' + escape(str(result.get('ocr_used_pages', []))) + '</p>')
                chunks.append('<details open><summary>实际提取全文</summary><pre>' + escape(result['text'] or result['error'] or '（空内容）') + '</pre></details>')
                if result.get('effective_settings'):
                    chunks.append('<details><summary>本次设置</summary><pre>' + escape(json.dumps(result['effective_settings'], ensure_ascii=False, indent=2)) + '</pre></details>')
                chunks.append('<table><tr><th>检查项</th><th>结果</th><th>对照</th></tr>')
                for item in result['checks']:
                    expected = json.dumps(item.get('expected'), ensure_ascii=False)
                    actual = json.dumps(item.get('actual'), ensure_ascii=False, default=str)
                    chunks.append(f'<tr><td>{escape(item["name"])}</td><td class="{item["status"]}">{LABELS[item["status"]]}</td>'
                                  f'<td><details><summary>标准 / 实际</summary><pre>{escape(expected)}\n\n{escape(actual)}</pre></details></td></tr>')
                chunks.append('</table></section>')
            chunks.append('</div></article>')
        chunks.append('</html>')
        (self.target / 'report.html').write_text(''.join(chunks), encoding='utf-8')
        # latest 只是入口，本次结果独立保存，不把上一次未运行的案例混进来。
        if self.target.parent == ARTIFACTS:
            (ARTIFACTS / 'latest.html').write_text(
                '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=' +
                self.target.name + '/report.html"><a href="' + self.target.name + '/report.html">打开最近一次报告</a>', encoding='utf-8')


def assert_quality(result, report):
    failed = [item['name'] for item in result['checks'] if item['status'] == 'fail']
    assert not failed, f"{result['route']} 质量不合格：{'；'.join(failed)}。报告：{report.target / 'report.html'}"


def real_ocr_enabled():
    # 终端的临时开关优先；否则使用项目 .env 中保存的明确选择。
    override = os.environ.get('PARSING_QUALITY_RUN_OCR')
    if override is not None:
        return override.strip().lower() in {'1', 'true', 'yes', 'on'}
    from config.settings import settings
    return settings.PARSING_QUALITY_RUN_OCR
