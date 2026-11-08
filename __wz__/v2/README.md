# Incubation Review V2

全新叙事与视觉设计，原版位于相邻 `../v1/`。15 页主讲 + 4 页附录。文字、拓扑图、等距内存图和曲线均为 PPT 原生可编辑对象。

- `SK_hynix_Incubation_Review_V2.pptx`：正式演示稿，含备注。
- `SK_hynix_Incubation_Review_V2.pdf`：同场景矢量预览。
- `CONTACT_SHEET_V2.png`、`preview/`：全稿及逐页预览。
- `SPEAKER_NOTES_V2.md`、`SOURCES.md`：讲稿和证据边界。
- `QA_REPORT_V2.json`：结构、尺寸、字体字符和备注检查。

重新生成：在本目录运行 `python build_review_v2.py`。依赖相邻 `../v1/build_deck.py` 和 `../v1/SOURCES.md`；生成结果写入本目录。当前环境无 Office 原生渲染，PDF/PNG 为同一测量场景生成。
