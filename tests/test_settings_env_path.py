"""验证配置不随启动目录变化，并保留系统环境变量优先的规则。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from config.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('working_subdir,override', [
    ('', False), ('tests', False), ('tests', True),
])
def test_root_env_is_loaded_from_any_working_directory(tmp_path, working_subdir, override):
    # 复制真实配置模块到临时项目，只使用虚构凭证，不改动或输出本机 .env。
    project = tmp_path / 'project'
    config = project / 'config' / 'settings.py'
    config.parent.mkdir(parents=True)
    config.write_text((ROOT / 'config' / 'settings.py').read_text(encoding='utf-8'), encoding='utf-8')
    (project / '.env').write_text(
        'PARSING_QUALITY_RUN_OCR=1\nOCR_ENABLED=true\nOCR_PROVIDER=tencent\n'
        'TENCENT_OCR_SECRET_ID=fixture-id\nTENCENT_OCR_SECRET_KEY=fixture-key\n',
        encoding='utf-8',
    )
    working_directory = project / working_subdir
    working_directory.mkdir(exist_ok=True)
    if working_subdir:
        # 启动目录即使有另一份 .env，也应读取固定的项目配置。
        (working_directory / '.env').write_text(
            'PARSING_QUALITY_RUN_OCR=0\nOCR_ENABLED=false\nOCR_PROVIDER=tesseract\n'
            'TENCENT_OCR_SECRET_ID=wrong-id\nTENCENT_OCR_SECRET_KEY=wrong-key\n',
            encoding='utf-8',
        )

    # 清除本机业务配置对这个独立进程的影响，避免测试依赖个人凭证和路径。
    fields = {name.upper() for name in Settings.model_fields}
    environment = {name: value for name, value in os.environ.items() if name.upper() not in fields}
    environment.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    if override:
        environment.update(PARSING_QUALITY_RUN_OCR='0', OCR_ENABLED='false',
                           TENCENT_OCR_SECRET_ID='process-id', TENCENT_OCR_SECRET_KEY='process-key')
    script = '''
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location('isolated_settings', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
settings = module.settings
print(json.dumps({
    'quality_ocr': settings.PARSING_QUALITY_RUN_OCR,
    'application_ocr': settings.OCR_ENABLED,
    'provider': settings.OCR_PROVIDER,
    'root_credentials': settings.TENCENT_OCR_SECRET_ID == 'fixture-id' and settings.TENCENT_OCR_SECRET_KEY == 'fixture-key',
    'process_credentials': settings.TENCENT_OCR_SECRET_ID == 'process-id' and settings.TENCENT_OCR_SECRET_KEY == 'process-key',
}))
'''
    completed = subprocess.run(
        [sys.executable, '-c', script, str(config)], cwd=working_directory, env=environment,
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        'quality_ocr': not override, 'application_ocr': not override, 'provider': 'tencent',
        'root_credentials': not override, 'process_credentials': override,
    }
