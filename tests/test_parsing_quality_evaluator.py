"""验证尺子本身：用人为构造的错误输出，确保它确实能发现错误。"""
import json

import pytest

from tests.parsing_quality_support import (
    QualityReport, assert_quality, character_error_rate, evaluate, quality_status, real_ocr_enabled,
)


def result(text='', **kwargs):
    return dict(route='project', execution='success', error=None, text=text,
                documents=[{'text': text, 'metadata': {'page': 1}}], raw_documents=[],
                ocr_opt_in=False, ocr_required_pages=[], elapsed_seconds=0, **kwargs)


def example(tmp_path, checks):
    return dict(id='example', category='test', file='fixture.pdf', path=str(tmp_path / 'fixture.pdf'),
                golden_path=str(tmp_path / 'golden.json'), description='test', checks=checks)


@pytest.mark.parametrize('expected,actual', [
    ('编号：000108', '编号：OOO108'), ('金额：-1000.05', '金额：1000.05'),
    ('金额：1000.05', '金额：100005'), ('结论：不予报销', '结论：予报销'),
    ('编号：000108', '编号：0001089'),
])
def test_detects_critical_character_errors(tmp_path, expected, actual):
    case = example(tmp_path, [dict(kind='contains', name='字段', value=expected)])
    assert quality_status(evaluate(case, result(actual))) == 'fail'
    assert character_error_rate(expected, actual) > 0


def test_only_whitespace_is_ignored(tmp_path):
    case = example(tmp_path, [dict(kind='contains', name='字段', value='金额：-1000.05')])
    assert quality_status(evaluate(case, result('金额： -1000.05'))) == 'pass'
    assert character_error_rate('金额：-1000.05', '金额： -1000.05') == 0


def test_separate_numeric_cells_are_not_glued_together(tmp_path):
    case = example(tmp_path, [dict(kind='contains', name='编号', value='000109')])
    assert quality_status(evaluate(case, result('1000.05 000109 商品乙'))) == 'pass'


def test_reading_order_and_missing_blocks(tmp_path):
    case = example(tmp_path, [dict(kind='order', name='阅读顺序', values=['A', 'B', 'C'])])
    assert quality_status(evaluate(case, result('A C B'))) == 'fail'
    assert quality_status(evaluate(case, result('A B'))) == 'fail'
    assert quality_status(evaluate(case, result('A B C'))) == 'pass'


def test_table_values_must_belong_to_correct_row(tmp_path):
    case = example(tmp_path, [dict(kind='table', name='表格', rows=[['商品', '金额'], ['甲', '100'], ['乙', '20']])])
    assert quality_status(evaluate(case, result('| 商品 | 金额 |\n| --- | --- |\n| 甲 | 20 |\n| 乙 | 100 |'))) == 'fail'
    assert quality_status(evaluate(case, result('商品 金额 甲 100 乙 20'))) == 'fail'
    assert quality_status(evaluate(case, result('| 商品 | 金额 |\n| --- | --- |\n| 甲 | 100 |\n| 乙 | 20 |'))) == 'pass'


def test_ocr_not_executed_is_never_pass(tmp_path):
    case = example(tmp_path, [dict(kind='contains', name='OCR 金额', value='-1000.05', requires_ocr=True)])
    assert quality_status(evaluate(case, result('-1000.05'))) == 'not_executed'


def test_report_saved_before_failing_assertion_and_escapes_html(tmp_path):
    case = example(tmp_path, [dict(kind='contains', name='字段', value='正确')])
    report = QualityReport(tmp_path / 'report')
    saved = report.save(case, result('<script>坏内容</script>'))
    with pytest.raises(AssertionError):
        assert_quality(saved, report)
    data = json.loads((report.target / 'report.json').read_text(encoding='utf-8'))
    assert data['records'][0]['routes']['project']['quality'] == 'fail'
    html = (report.target / 'report.html').read_text(encoding='utf-8')
    assert '<script>坏内容</script>' not in html
    assert '&lt;script&gt;坏内容&lt;/script&gt;' in html


def test_new_report_does_not_count_previous_run(tmp_path):
    case = example(tmp_path, [])
    first = QualityReport(tmp_path / 'first')
    first.save(case, result())
    second = QualityReport(tmp_path / 'second')
    second.write_index()
    payload = json.loads((second.target / 'report.json').read_text(encoding='utf-8'))
    assert payload['cases_executed'] == 0
    assert payload['records'] == []


def test_application_limits_are_not_native_parser_requirements(tmp_path):
    case = example(tmp_path, [])
    case.update(expected_rejection=True, settings={'MAX_PDF_PAGES': 3})
    native = result('four pages')
    native['route'] = 'native'
    assert quality_status(evaluate(case, native)) == 'not_applicable'


def test_missing_dependency_cannot_count_as_rejecting_bad_document(tmp_path):
    case = example(tmp_path, [])
    case.update(expected_rejection=True, rejection_types=['ValueError'])
    broken_environment = result()
    broken_environment.update(execution='error', error='ModuleNotFoundError: parser', error_type='ModuleNotFoundError')
    assert quality_status(evaluate(case, broken_environment)) == 'fail'


def test_native_loader_bypasses_project_pipeline(monkeypatch):
    from core.document_loader import DocumentLoader
    from tests.parsing_quality_support import extract_case, load_cases

    def forbidden(*args, **kwargs):
        pytest.fail('原生基线不能经过项目加载器')

    monkeypatch.setattr(DocumentLoader, 'load_raw_file', forbidden)
    case = next(case for case in load_cases() if case['id'] == 'text_fields')
    actual = extract_case(case, 'native')
    assert actual['execution'] == 'success'
    assert 'PO-2026-000108' in actual['text']


def test_offline_scan_never_calls_real_ocr(monkeypatch):
    from core import content_processing
    from tests.parsing_quality_support import extract_case, load_cases

    def forbidden(*args, **kwargs):
        pytest.fail('普通测试不能调用真实 OCR')

    monkeypatch.setattr(content_processing, 'recognize_documents', forbidden)
    case = next(case for case in load_cases() if case['id'] == 'scan_clear')
    actual = extract_case(case, 'project', run_ocr=False)
    assert actual['execution'] == 'success'
    assert actual['ocr_required_pages'] == [1]
    assert actual['ocr_used_pages'] == []
    assert quality_status(evaluate(case, actual)) == 'not_executed'


@pytest.mark.parametrize('override,configured,expected', [
    (None, False, False), (None, True, True), ('0', True, False),
    ('1', False, True), ('false', True, False), ('TRUE', False, True),
])
def test_real_ocr_uses_local_choice_with_terminal_override(monkeypatch, override, configured, expected):
    from config.settings import settings

    monkeypatch.setattr(settings, 'PARSING_QUALITY_RUN_OCR', configured)
    if override is None:
        monkeypatch.delenv('PARSING_QUALITY_RUN_OCR', raising=False)
    else:
        monkeypatch.setenv('PARSING_QUALITY_RUN_OCR', override)
    assert real_ocr_enabled() is expected


def test_original_scan_is_shown_even_when_parser_returns_empty_text(tmp_path):
    from tests.parsing_quality_support import load_cases

    case = next(case for case in load_cases() if case['id'] == 'scan_stamp')
    report = QualityReport(tmp_path / 'report')
    saved = report.save(case, result(''))
    original = report.records[case['id']]['original']
    assert original['status'] == 'ready'
    assert (report.target / original['pages'][0]['image']).is_file()
    assert (report.target / original['source_href']).is_file()
    assert saved['text'] == ''  # 原件预览不能悄悄补入解析结果。
    html = (report.target / 'report.html').read_text(encoding='utf-8')
    assert '原始资料与标准答案' in html
    assert 'LangChain 原生' in html and '项目增强' in html
    assert '订单编号：PO-2026-000108' in html
    assert 'original-page-001.png' in html


@pytest.mark.parametrize('identifier,expected', [
    ('docx_noise', ['测试企业内部页眉', '免责声明：本文件仅供测试。', '审批    责任人']),
    ('docx_headings', ['word-title', 'word-heading', 'word-subheading', '1.1 差旅']),
])
def test_word_original_keeps_noise_and_heading_styles(tmp_path, identifier, expected):
    from tests.parsing_quality_preview import build_original_preview
    from tests.parsing_quality_support import load_cases

    case = next(case for case in load_cases() if case['id'] == identifier)
    original = build_original_preview(case, tmp_path / identifier)
    assert original['status'] == 'ready'
    assert all(value in original['html'] for value in expected)
    assert '不代表 Word 的原始分页' in original['html']


def test_excel_original_keeps_merges_sheets_and_leading_zeros(tmp_path):
    from tests.parsing_quality_preview import build_original_preview
    from tests.parsing_quality_support import load_cases

    case = next(case for case in load_cases() if case['id'] == 'excel_complex')
    original = build_original_preview(case, tmp_path / case['id'])
    assert original['status'] == 'ready'
    assert 'rowspan="2"' in original['html']
    assert '000108' in original['html'] and '000109' in original['html']
    assert '工作表：一月' in original['html'] and '工作表：二月' in original['html']
    assert '-20.50' in original['html']
