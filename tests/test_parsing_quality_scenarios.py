"""验证单项运行入口：失败也有完整报告，多个方法结束后只打开一次浏览器。"""
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import pytest

from tests.parsing_quality_support import QualityReport, load_cases

scenarios = importlib.import_module('tests.文档解析单项测试')


def _result(route, *, run_ocr=False, text=''):
    return dict(route=route, execution='success', error=None, error_type=None,
                text=text, documents=[{'text': text, 'metadata': {'page': 1}}],
                raw_documents=[], ocr_opt_in=run_ocr, ocr_required_pages=[],
                ocr_used_pages=[], elapsed_seconds=0)


def test_selected_scan_saves_both_routes_before_quality_failure(tmp_path, monkeypatch):
    calls = []

    def extract(case, route, *, run_ocr):
        calls.append((case['id'], route, run_ocr))
        result = _result(route, run_ocr=run_ocr)
        if route == 'project':
            result.update(execution='error', error_type='RuntimeError', error='OCR service unavailable')
        return result

    monkeypatch.setattr(scenarios, 'extract_case', extract)
    monkeypatch.setattr(scenarios, 'real_ocr_enabled', lambda: True)
    report = QualityReport(tmp_path / 'report')
    with pytest.raises(AssertionError) as failure:
        scenarios._compare_case('scan_stamp', report)

    assert calls == [('scan_stamp', 'native', False), ('scan_stamp', 'project', True)]
    assert 'native' in str(failure.value) and 'project' in str(failure.value)
    payload = json.loads((report.target / 'report.json').read_text(encoding='utf-8'))
    assert payload['cases_executed'] == 1
    assert payload['records'][0]['case']['id'] == 'scan_stamp'
    routes = payload['records'][0]['routes']
    assert set(routes) == {'native', 'project'}
    assert routes['project']['error'] == 'OCR service unavailable'
    html = (report.target / 'report.html').read_text(encoding='utf-8')
    assert all(title in html for title in ('原始资料与标准答案', 'LangChain 原生', '项目增强'))


@pytest.mark.parametrize('browser_result', [True, False, RuntimeError('no browser')])
def test_failed_cases_open_once_at_class_cleanup(tmp_path, monkeypatch, capsys, browser_result):
    report = QualityReport(tmp_path / '报告 with spaces')
    opened = []

    def open_browser(url, *, new):
        opened.append((url, new))
        if isinstance(browser_result, Exception):
            raise browser_result
        return browser_result

    monkeypatch.setattr(scenarios, 'QualityReport', lambda: report)
    monkeypatch.setattr(scenarios.webbrowser, 'open', open_browser)
    monkeypatch.setattr(scenarios, 'real_ocr_enabled', lambda: False)
    monkeypatch.setattr(scenarios, 'extract_case', lambda case, route, **kwargs: _result(route))
    suite = unittest.TestLoader().loadTestsFromNames([
        scenarios.__name__ + '.TestDocumentParsingByScenario.test_文字PDF关键字段',
        scenarios.__name__ + '.TestDocumentParsingByScenario.test_清晰扫描件',
    ])
    outcome = unittest.TestResult()
    suite.run(outcome)

    assert outcome.testsRun == 2
    assert len(outcome.failures) == 2
    assert outcome.errors == []
    assert set(report.records) == {'text_fields', 'scan_clear'}
    assert opened == [((report.target / 'report.html').resolve().as_uri(), 2)]
    output = capsys.readouterr().out
    assert str((report.target / 'report.html').resolve()) in output
    if browser_result is not True:
        assert '请手动打开' in output


def test_unexecuted_ocr_is_saved_and_skipped_instead_of_passed(tmp_path, monkeypatch):
    # 原生能提取这个字段，增强路线的 OCR 未执行：两份结果仍应保存，不能报通过。
    case = dict(scenarios.CASES['text_fields'], checks=[
        dict(kind='contains', name='OCR 金额', value='-1000.05', requires_ocr=True),
    ])
    monkeypatch.setattr(scenarios, 'CASES', {'text_fields': case})
    monkeypatch.setattr(scenarios, 'real_ocr_enabled', lambda: False)
    monkeypatch.setattr(scenarios, 'extract_case',
                        lambda case, route, **kwargs: _result(route, text='-1000.05'))
    report = QualityReport(tmp_path / 'report')
    with pytest.raises(unittest.SkipTest):
        scenarios._compare_case('text_fields', report)
    routes = report.records['text_fields']['routes']
    assert routes['native']['quality'] == 'pass'
    assert routes['project']['quality'] == 'not_executed'


def test_no_browser_opens_without_completed_results(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('没有结果时不应打开空报告')

    monkeypatch.setattr(scenarios.webbrowser, 'open', forbidden)
    scenarios._open_report(QualityReport(tmp_path / 'report'))


def test_named_methods_cover_each_fixture_once(monkeypatch):
    selected = []
    report = object()

    def record(case_id, actual_report):
        assert actual_report is report
        selected.append(case_id)

    monkeypatch.setattr(scenarios, '_compare_case', record)
    entry = scenarios.TestDocumentParsingByScenario()
    entry.scenario_report = report
    for name in dir(entry):
        if name.startswith('test_'):
            getattr(entry, name)()
    assert Counter(selected) == Counter(case['id'] for case in load_cases())


@pytest.mark.parametrize('runner', ['unittest', 'pytest'])
def test_both_runners_handle_pass_failure_skip_and_one_report(tmp_path, runner):
    """在独立进程真正运行入口，避免只测辅助函数而漏掉运行器兼容问题。"""
    script = '''
import importlib
import json
from pathlib import Path
import sys
import unittest

module_name, runner, directory, methods_json = sys.argv[1:]
module = importlib.import_module(module_name)
report = module.QualityReport(Path(directory))
module.QualityReport = lambda: report
module.real_ocr_enabled = lambda: False
opened = []
module.webbrowser.open = lambda url, new: opened.append((url, new)) or True

def extract(case, route, *, run_ocr):
    assert not opened, 'Browser opened before all selected cases completed'
    return dict(route=route, execution='success', error=None, error_type=None,
                text='OK', documents=[{'text': 'OK', 'metadata': {'page': 1}}],
                raw_documents=[], ocr_opt_in=run_ocr, ocr_required_pages=[],
                ocr_used_pages=[], elapsed_seconds=0)

module.extract_case = extract
for case_id in ('text_fields', 'columns', 'scan_clear'):
    rule = dict(kind='contains', name='field', value='missing' if case_id == 'columns' else 'OK')
    if case_id == 'scan_clear':
        rule['requires_ocr'] = True
    module.CASES[case_id] = dict(module.CASES[case_id], checks=[rule])
methods = json.loads(methods_json)

if runner == 'unittest':
    suite = unittest.TestLoader().loadTestsFromNames([
        module_name + '.TestDocumentParsingByScenario.' + method for method in methods
    ])
    outcome = unittest.TestResult()
    suite.run(outcome)
    assert outcome.testsRun == 3
    assert len(outcome.failures) == 1, outcome.failures
    assert len(outcome.skipped) == 1, outcome.skipped
    assert outcome.errors == [], outcome.errors
else:
    import pytest

    class Outcomes:
        def __init__(self):
            self.calls = []

        def pytest_runtest_logreport(self, report):
            if report.when == 'call':
                self.calls.append(report.outcome)

    outcomes = Outcomes()
    targets = [module.__file__ + '::TestDocumentParsingByScenario::' + method for method in methods]
    exit_code = pytest.main(targets + ['-q', '-p', 'no:cacheprovider'], plugins=[outcomes])
    assert exit_code == 1, exit_code
    assert sorted(outcomes.calls) == ['failed', 'passed', 'skipped'], outcomes.calls

payload = json.loads((report.target / 'report.json').read_text(encoding='utf-8'))
assert payload['cases_executed'] == 3
assert set(report.records) == {'text_fields', 'columns', 'scan_clear'}
assert all(set(record['routes']) == {'native', 'project'} for record in report.records.values())
assert opened == [((report.target / 'report.html').resolve().as_uri(), 2)], opened
print('COMPATIBILITY_OK')
'''
    methods = ['test_文字PDF关键字段', 'test_双栏阅读顺序', 'test_清晰扫描件']
    completed = subprocess.run(
        [sys.executable, '-c', script, scenarios.__name__, runner, str(tmp_path / 'report'),
         json.dumps(methods, ensure_ascii=False)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
        capture_output=True, text=True, encoding='utf-8', timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'COMPATIBILITY_OK' in completed.stdout
