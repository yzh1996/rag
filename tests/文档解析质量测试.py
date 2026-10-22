"""显式运行本文件会生成报告；失败不会阻止已经完成的结果保存。

这是一套严格的质量评估，不是“现有功能理应全部通过”的回归测试。
发现真实短板会失败；常规单元测试不会自动收集这个中文文件名。
"""
import pytest

from tests.parsing_quality_support import (
    QualityReport, assert_quality, extract_case, load_cases, real_ocr_enabled,
)

CASES = load_cases()


@pytest.fixture(scope='module')
def quality_report():
    report = QualityReport()
    yield report
    print(f'\n文档解析质量报告：{report.target / "report.html"}')


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
class TestDocumentParsingQuality:
    def test_native_langchain(self, case, quality_report):
        """没有项目增强的基线：直接使用 LangChain 自带加载器。"""
        result = extract_case(case, 'native')
        quality_report.save(case, result)  # 必须先写报告，后做断言。
        assert_quality(result, quality_report)

    def test_project_enhanced(self, case, quality_report):
        """同一份文件走项目当前流程，严格核对独立标准答案。"""
        result = extract_case(case, 'project', run_ocr=real_ocr_enabled())
        quality_report.save(case, result)
        assert_quality(result, quality_report)
        if result['quality'] == 'not_executed':
            pytest.skip('尚未执行真实 OCR，已提取内容和检查项仍会写入报告')
