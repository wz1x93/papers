# 海力士 Incubation 提案

主题：面向混合大模型的混合精度与分层数据放置联合优化。

## 版本导航

| 版本 | 内容 | 演示稿 | PDF | 全稿预览 |
|---|---|---|---|---|
| v1 | 初版，23 页 | [PPTX](v1/SK_hynix_Incubation_Precision_Placement_CN.pptx) | [PDF](v1/SK_hynix_Incubation_Precision_Placement_CN.pdf) | [预览](v1/CONTACT_SHEET.png) |
| v2 | 视觉与叙事重做，19 页 | [PPTX](v2/SK_hynix_Incubation_Review_V2.pptx) | [PDF](v2/SK_hynix_Incubation_Review_V2.pdf) | [预览](v2/CONTACT_SHEET_V2.png) |
| **v3（最新）** | 增加行业与竞争背景，25 页 | [PPTX](v3/SK_hynix_Incubation_Review_V3.pptx) | [PDF](v3/SK_hynix_Incubation_Review_V3.pdf) | [预览](v3/CONTACT_SHEET_V3.png) |

各版本目录均包含演示稿、PDF、讲稿、来源、页面预览、检查报告和对应生成脚本。V3 另含 [行业与竞争页面预览](v3/INDUSTRY_COMPETITION_PREVIEW_V3.png) 和 [主张—证据对照](v3/CLAIM_EVIDENCE_V3.md)。

## 重新生成

Python 3.10+，Windows 微软雅黑。先安装公共依赖：

```powershell
python -m pip install -r "D:\WZ\2026\maddog\__wz__\v1\requirements.txt"
```

然后运行对应版本的脚本：

```powershell
python "D:\WZ\2026\maddog\__wz__\v1\build_deck.py"
python "D:\WZ\2026\maddog\__wz__\v2\build_review_v2.py"
python "D:\WZ\2026\maddog\__wz__\v3\build_review_v3.py"
```

每条命令只重新生成其对应版本。V2 复用 v1 排版工具与来源，V3 复用 v2、v1 的工具和来源；请保留三个相邻版本目录。PPTX 中图文为原生可编辑对象；PDF/PNG 来自同一版面数据，Office 原生放映尚未验证。

公开信息截至 2026-11-08。“15%”为拟议验证目标；容量算例、人员配置和三年路线为提案假设。
