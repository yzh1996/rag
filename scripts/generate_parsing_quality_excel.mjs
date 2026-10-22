// 只生成测试用工作簿；数据与标准答案分别在生成脚本、golden.json 中保存。
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const outputDir = process.argv[2];
if (!outputDir) throw new Error('请传入测试资源输出目录');
// 优先使用已安装的公开包，也支持通过环境变量指定运行时的 node_modules。
const resolver = createRequire(process.env.ARTIFACT_NODE_MODULES
  ? path.join(process.env.ARTIFACT_NODE_MODULES, '..', 'artifact-resolver.cjs')
  : import.meta.url);
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(resolver.resolve('@oai/artifact-tool')).href);
await fs.mkdir(outputDir, { recursive: true });

for (const complex of [false, true]) {
  const workbook = Workbook.create();
  const sheets = complex ? ['一月', '二月'] : ['销量'];
  for (const [index, name] of sheets.entries()) {
    const sheet = workbook.worksheets.add(name);
    const rows = complex ? [
      ['部门', '编号', '商品', '金额'],
      ['华东', '000108', '商品甲', 1000.05 + index],
      [null, '000109', '商品乙', -20.5],
      [null, null, null, null],
      ['华西', '000110', '商品丙', 300],
    ] : [
      ['编号', '商品', '数量', '金额'],
      ['000108', '商品甲', 10, 1000.05],
      ['000109', '商品乙', 2, -20.5],
    ];
    sheet.getRange(`A1:D${rows.length}`).values = rows;
    if (complex) sheet.mergeCells('A2:A3');
    const area = sheet.getRange(`A1:D${rows.length}`);
    area.format.font = { name: 'Microsoft YaHei', size: 12, color: '#172B4D' };
    area.format.columnWidth = 24;
    area.format.rowHeight = 32;
    sheet.getRange('A1:D1').format.fill = '#E6EEF8';
    sheet.getRange('A1:D1').format.font.bold = true;
    sheet.getRange(`D2:D${rows.length}`).setNumberFormat('0.00');
    // 值始终是字符串；六位格式也让预览器正确显示编号的前导零。
    sheet.getRange(`${complex ? 'B' : 'A'}2:${complex ? 'B' : 'A'}${rows.length}`).setNumberFormat('000000');
    const preview = await workbook.render({ sheetName: name, range: `A1:D${rows.length}`, scale: 1, format: 'png' });
    await fs.writeFile(path.join(outputDir, `${complex ? '复杂' : '基础'}-${name}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  workbook.recalculate();
  console.log((await workbook.inspect({ kind: 'sheet', include: 'id,name', maxChars: 1000 })).ndjson);
  const file = await SpreadsheetFile.exportXlsx(workbook);
  await file.save(path.join(outputDir, complex ? '多表合并空行.xlsx' : '基础表格.xlsx'));
}
