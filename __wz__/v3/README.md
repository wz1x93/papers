# Incubation Review V3

沿用 V2 视觉，新增行业发展、三星重点竞争分析、美光与海力士公开行动、项目必要性。19 页主讲 + 6 页附录。V1/V2 保留。

## 文件

- `SK_hynix_Incubation_Review_V3.pptx`：原生可编辑，全部页面含讲稿备注和对应 URL。
- `SK_hynix_Incubation_Review_V3.pdf`：同场景矢量 PDF；新增来源标注可点击。
- `INDUSTRY_COMPETITION_PREVIEW_V3.png`：第 3–7 页行业与竞争页面。
- `CONTACT_SHEET_V3.png`、`preview/`：全稿与逐页图片。
- `SPEAKER_NOTES_V3.md`：逐页中文讲稿与完整链接。
- `SOURCES_V3.md`：23 项来源；本轮新增 10 项，包含日期、原文节选与使用边界。
- `CLAIM_EVIDENCE_V3.md`：主张与证据对照。
- `QA_REPORT_V3.json`：字体、布局、结构与引用检查。

## 版本变化

以 5 页行业与竞争论证替换 V2 泛化的 why-now 页面，保留已认可的技术主线，再增加 2 页证据附录。新增内容将公司公告、研发、认证 / 演示与战略推论区分。未使用未经核实的市场份额、财务预测或竞争对手领先排名。

重新生成：在本目录运行 `python build_review_v3.py`。依赖相邻 `../v2/build_review_v2.py`、`../v1/build_deck.py` 和 `../v1/SOURCES.md`；生成结果写入本目录。当前环境无 Office 原生渲染，PDF/PNG 来自同一测量场景；Office 实际放映尚未验证。
