# Incubation 提案 v1

初版，23 页：18 页主讲 + 5 页附录。

- `SK_hynix_Incubation_Precision_Placement_CN.pptx`：可编辑演示稿。
- `SK_hynix_Incubation_Precision_Placement_CN.pdf`：矢量预览。
- `CONTACT_SHEET.png`、`preview/`：全稿与逐页预览。
- `SPEAKER_NOTES.md`、`SOURCES.md`：讲稿与参考来源。
- `QA_REPORT.json`：检查报告。
- `build_deck.py`：本版本生成脚本，也提供后续版本复用的排版工具与来源索引。
- `requirements.txt`：三个版本使用的 Python 依赖。
- `__pycache__/`：整理前生成的 Python 缓存。

在本目录运行 `python build_deck.py` 可重新生成 v1，结果写入本目录。安装依赖：`python -m pip install -r requirements.txt`。
