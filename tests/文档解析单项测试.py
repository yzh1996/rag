"""在 IDE 中右键运行某个方法，只比较该场景，并自动打开三栏报告。

每个方法同时运行 LangChain 原生解析和项目增强解析。
质量不合格仍然生成并打开报告，方便对照原件找出哪里解析错了。
兼容 unittest 和 pytest，PyCharm 右键运行时不需要额外传入报告参数。
"""
import unittest
import webbrowser

from tests.parsing_quality_support import (
    QualityReport, assert_quality, extract_case, load_cases, real_ocr_enabled,
)

CASES = {case['id']: case for case in load_cases()}


def _open_report(report):
    """浏览器打不开时保留报告路径，不影响真正的质量检查结果。"""
    report_path = (report.target / 'report.html').resolve()
    if not report.records or not report_path.is_file():
        return
    print(f'\n文档解析质量报告：{report_path}')
    try:
        opened = webbrowser.open(report_path.as_uri(), new=2)
    except Exception as exc:
        print(f'无法自动打开浏览器：{exc}。请手动打开上面的报告。')
    else:
        if not opened:
            print('浏览器未能自动打开，请手动打开上面的报告。')


def _compare_case(case_id, report):
    """先收齐两条路线的结果，再检查质量，避免第一条失败挡住第二条。"""
    case = CASES[case_id]
    results = []
    for route in ('native', 'project'):
        result = extract_case(case, route, run_ocr=route == 'project' and real_ocr_enabled())
        report.save(case, result)
        results.append(result)

    failures = []
    for result in results:
        try:
            assert_quality(result, report)
        except AssertionError as exc:
            failures.append(str(exc))
    assert not failures, '\n'.join(failures)
    if any(result['quality'] == 'not_executed' for result in results):
        # 两种测试运行器都认识 SkipTest，不把未执行的 OCR 当成通过或程序错误。
        raise unittest.SkipTest('尚未执行真实 OCR，两条路线的已提取内容和检查项仍会写入报告')


class TestDocumentParsingByScenario(unittest.TestCase):
    """找到想学习的场景，在 test_ 方法上右键，用 unittest 或 pytest 运行。"""

    @classmethod
    def setUpClass(cls):
        """测试开始前准备报告，同一次运行的多个方法共用这一份。"""
        super().setUpClass()
        cls.scenario_report = QualityReport()
        # 两种运行器都会执行类的收尾动作；即使检查失败，也只在最后打开一次。
        cls.addClassCleanup(_open_report, cls.scenario_report)

    # 01 文字准确性：重点看 0/O、编号、负号、小数点、姓名和否定词。
    def test_文字PDF关键字段(self):
        _compare_case('text_fields', self.scenario_report)

    def test_清晰扫描件(self):
        _compare_case('scan_clear', self.scenario_report)

    def test_低清扫描件(self):
        _compare_case('scan_low_resolution', self.scenario_report)

    def test_倾斜扫描件(self):
        _compare_case('scan_tilted', self.scenario_report)

    def test_模拟盖章扫描件(self):
        _compare_case('scan_stamp', self.scenario_report)

    # 02 完整性：检查扫描页和正文旁边的图片内容有没有漏掉。
    def test_文字页与扫描页混合(self):
        _compare_case('mixed_pages', self.scenario_report)

    def test_同页文字与图片混合(self):
        _compare_case('mixed_same_page', self.scenario_report)

    # 03 阅读顺序：检查双栏、侧栏和跨页段落的先后关系。
    def test_双栏阅读顺序(self):
        _compare_case('columns', self.scenario_report)

    def test_侧栏与脚注(self):
        _compare_case('sidebar_footnote', self.scenario_report)

    def test_跨页段落(self):
        _compare_case('cross_page_text', self.scenario_report)

    # 04 表格关系：不能只看数字有没有提出来，还要看对应的是哪一行、哪一列。
    def test_有边框表格(self):
        _compare_case('table_bordered', self.scenario_report)

    def test_无边框表格(self):
        _compare_case('table_borderless', self.scenario_report)

    def test_PDF合并单元格(self):
        _compare_case('table_merged', self.scenario_report)

    def test_跨页表格(self):
        _compare_case('table_cross_page', self.scenario_report)

    def test_Excel基础表格(self):
        _compare_case('excel_basic', self.scenario_report)

    def test_Excel多工作表与合并单元格(self):
        _compare_case('excel_complex', self.scenario_report)

    # 05 结构来源：检查版本、页码、上传信息和标题层级。
    def test_制度版本与来源信息(self):
        _compare_case('policy_metadata', self.scenario_report)

    def test_Word标题层级(self):
        _compare_case('docx_headings', self.scenario_report)

    # 06 清洗与异常：噪声应去掉，正文应保留，异常文件应按预期拒绝。
    def test_Word噪声清洗(self):
        _compare_case('docx_noise', self.scenario_report)

    def test_重复页眉页脚(self):
        _compare_case('repeated_edges', self.scenario_report)

    def test_中间空白页(self):
        _compare_case('blank_page', self.scenario_report)

    def test_损坏PDF(self):
        _compare_case('corrupt_pdf', self.scenario_report)

    def test_加密PDF(self):
        _compare_case('encrypted_pdf', self.scenario_report)

    def test_PDF页数限制(self):
        _compare_case('page_limit', self.scenario_report)

    def test_Office解压限制(self):
        _compare_case('office_limit', self.scenario_report)
