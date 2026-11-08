"""Editable PPTX + vector PDF + PNG, all built from a shared scene.

All output is contained in this script's directory. No experimental results
are synthesized: illustrations and incubation targets are explicitly labeled.
"""
from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon


ROOT = Path(__file__).resolve().parent
PREVIEW = ROOT / "preview"
W, H = 1600, 900
PPI = 120
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
NAVY = "101C30"
DEEP = "0B1424"
INK = "17263B"
MUTED = "64738A"
BG = "F3F5F8"
WHITE = "FFFFFF"
ORANGE = "E85B34"
TEAL = "007F86"
BLUE = "3769AC"
LINE = "DCE2E9"
PALE_ORANGE = "FFF0E8"
PALE_TEAL = "E7F4F2"
PALE_BLUE = "EDF2FA"
AMBER = "A86A13"
SLIDES = []
WARNINGS = []

SOURCES = [
    ("S01", "NOVA: Technology-Architecture Co-Design of Near-Memory Processing for Attention-SSM-MoE Hybrid LLM Inference", "2026-08-23", "https://arxiv.org/abs/2608.22613", "https://arxiv.org/html/2608.22613v1", "混合负载的内存特征；近存计算背景。其架构收益来自仿真，不是本提案结果。"),
    ("S02", "SK hynix 2026 Future Forum", "2026-09-09", "https://news.skhynix.com/en/future-forum-2026/", "", "海力士公开提出 Full-Stack AI Memory 和系统协同设计；企业战略信号，不代表本项目已获认可。"),
    ("S03", "SK hynix at OCP Global Summit 2026", "2026-10-16", "https://news.skhynix.com/en/ocp-global-summit-2026/", "", "CXL 内存池、DRAM+SSD 混合内存、KV 复用和 CMM-Ax 的公开展示；展示不等同于全部功能量产。"),
    ("S04", "HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference", "2024-11", "https://arxiv.org/abs/2411.01433", "", "混合精度专家卸载、预取和缓存的已有工作。"),
    ("S05", "TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization", "2025-05", "https://arxiv.org/abs/2505.19586", "", "KV 量化与卸载相结合；不能将该组合本身声明为首创。"),
    ("S06", "Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems", "2025-12-04", "https://arxiv.org/abs/2512.04476", "https://arxiv.org/html/2512.04476v1", "专家 HBM/CXL-NDP 放置与位宽选择；最接近的 MoE 子问题。性能评估采用模拟器，使用多个精度副本。"),
    ("S07", "SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints for Efficient MoE Inference", "v1 2025-12; v5 2026-09", "https://arxiv.org/abs/2512.12990", "", "按位切片的混合精度专家缓存；精度切片不能作为未经区分的原创主张。"),
    ("S08", "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories", "2026-06", "https://arxiv.org/abs/2606.12556", "", "基于海力士 CMM 和 PCIe Gen5 SSD 的潜力评估及 FPGA 功能原型；最高 35.7% 是其特定基线下的报告值。"),
    ("S09", "A CXL Memory Rack for Multi-Turn LLM Serving (HyMCache)", "2026-07; revised 2026-08", "https://arxiv.org/abs/2607.18141", "", "真实 CXL-HM 原型，面向可复用 KV 的 DRAM/SSD 管理；不可直接外推到活跃 SSM 状态。"),
    ("S10", "ReplaySSM: Cache SSM Inputs, Not State", "2026-10-09", "https://arxiv.org/abs/2610.13778", "", "SSM 检查点与输入缓冲；读写和精度的专门机制。其报告的 1.20–1.47 倍加速不是本项目收益。"),
    ("S11", "StrataFlux: Memory-Hierarchy-Native Low-Bit LLM Inference", "2026-11-05", "https://arxiv.org/abs/2611.08363", "", "低位宽权重/KV 与 RAM、SSD 精度补充流；截至检索日为近期预印本。"),
    ("S12", "Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning", "2026-06-12", "https://arxiv.org/abs/2606.15007", "", "Attention/Mamba/MoE 混合路线的模型实例；不足以证明未来模型全部采用这一架构。"),
    ("S13", "PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States", "2026-10-21", "https://arxiv.org/abs/2610.29545", "", "递归状态的周期量化及混合位宽，强调误差传播和更新频率的重要性。"),
]


def rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def mplcolor(h):
    return "#" + h


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, round(size * PPI / 72))


def wrap(text, size, max_width, bold=False):
    """Pixel-measured line breaking; explicit lines preserved across renderers."""
    f = font(size, bold)
    lines = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        current = ""
        # Keep Latin words intact where possible; split long words only as needed.
        tokens = re.findall(r"[A-Za-z0-9_./%+−–—:<>≤≥=×-]+|\s+|.", para)
        for token in tokens:
            if f.getlength(current + token) <= max_width:
                current += token
            else:
                carry = ""
                if token and token[0] in "，。；：！？、）》】”’%" and current.strip():
                    current = current.rstrip()
                    tail_word = re.search(r"[A-Za-z0-9]+$", current)
                    carry = tail_word.group(0) if tail_word else current[-1]
                    current = current[:-len(carry)]
                if current.strip():
                    lines.append(current.rstrip())
                current = carry + token.lstrip()
                if f.getlength(current) > max_width:
                    fragment = ""
                    for ch in current:
                        if fragment and f.getlength(fragment + ch) > max_width:
                            lines.append(fragment)
                            fragment = ch
                        else:
                            fragment += ch
                    current = fragment
        if current:
            lines.append(current.rstrip())
    return lines


class Slide:
    def __init__(self, title, subtitle="", section="", refs="", dark=False, appendix=False):
        self.title, self.subtitle = title, subtitle
        self.section, self.refs, self.dark, self.appendix = section, refs, dark, appendix
        self.bg = DEEP if dark else BG
        self.items = []
        self.notes = ""
        self.seconds = 45
        SLIDES.append(self)
        self.number = len(SLIDES)

    def rect(self, x, y, w, h, fill=WHITE, stroke=None, radius=0):
        self.items.append(dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, radius=radius))

    def line(self, x1, y1, x2, y2, color=LINE, width=2, arrow=False):
        self.items.append(dict(kind="line", x=x1, y=y1, x2=x2, y2=y2, color=color, width=width, arrow=arrow))

    def text(self, x, y, w, h, text, size=18, color=INK, bold=False, align="left", leading=1.20):
        # YaHei has no Unicode subscript digits / letters. Use readable plain
        # notation consistently across PPTX, PDF and PNG, without missing glyphs.
        text = text.translate(str.maketrans({"₁":"1", "₂":"2", "₃":"3", "ᵢ":"_i", "ₖ":"_k", "ₜ":"_t"}))
        lines = wrap(text, size, w - 4, bold)
        line_height = size * PPI / 72 * leading
        required = len(lines) * line_height
        original_size = size
        while required > h + 1 and size > max(10.5, original_size * .84):
            size = round(size - .25, 2)
            lines = wrap(text, size, w - 4, bold)
            line_height = size * PPI / 72 * leading
            required = len(lines) * line_height
        if required > h + 1:
            raise ValueError(f"slide {self.number} text overflow: {text[:70]!r}; requires {required:.1f}, box {h}")
        self.items.append(dict(kind="text", x=x, y=y, w=w, h=h, text=text, lines=lines, size=size, color=color, bold=bold, align=align, leading=leading))

    def tag(self, x, y, text, fill=PALE_ORANGE, color=ORANGE, w=210):
        self.rect(x, y, w, 36, fill, radius=8)
        self.text(x + 12, y + 5, w - 24, 28, text, size=11.5, bold=True, color=color)

    def header(self):
        c = WHITE if self.dark else INK
        self.text(80, 44, 1440, 24, self.section.upper(), size=10.5, color="97A9C4" if self.dark else MUTED, bold=True)
        self.text(80, 91, 1440, 75, self.title, size=31, color=c, bold=True)
        if self.subtitle:
            self.text(80, 163, 1440, 42, self.subtitle, size=13.5, color="ADBBD0" if self.dark else MUTED)

    def footer(self):
        color = "8899B2" if self.dark else MUTED
        self.line(80, 837, 1520, 837, "29374B" if self.dark else LINE, 1)
        self.text(80, 851, 1030, 24, self.refs or "提案设计 / 待验证目标 · 公开信息截至 2026-11-08", size=9, color=color)
        self.text(1150, 851, 300, 24, "INCUBATION  |  " + ("APPENDIX" if self.appendix else "REVIEW DISCUSSION"), size=8.5, color=color, align="right")
        self.text(1470, 845, 50, 34, f"{self.number:02}", size=16, color=ORANGE, bold=True, align="right")

    def note(self, text, seconds=45):
        self.notes = text
        self.seconds = seconds


def card(s, x, y, w, h, label, title, body, accent=ORANGE, size=16):
    s.rect(x, y, w, h, WHITE, radius=16)
    s.rect(x, y, 6, h, accent, radius=2)
    s.text(x+26, y+22, w-52, 30, label, 11, accent, True)
    s.text(x+26, y+65, w-52, 75, title, 21, INK, True)
    s.text(x+26, y+153, w-52, h-165, body, size, MUTED)


def takeaway(s, text, y=759, fill=NAVY, color=WHITE):
    s.rect(80, y, 1440, 57, fill, radius=10)
    s.text(102, y+13, 1396, 38, text, 16, color, True)


def table(s, x, y, widths, rows, heights, header=True, font_size=14):
    yy = y
    for r, (row, hh) in enumerate(zip(rows, heights)):
        xx = x
        for c, (val, ww) in enumerate(zip(row, widths)):
            fill = NAVY if r == 0 and header else (WHITE if r % 2 else "EAF0F5")
            s.rect(xx, yy, ww-2, hh-2, fill)
            s.text(xx+16, yy+15, ww-32, hh-22, val, font_size, WHITE if r == 0 and header else INK, r == 0 or c == 0)
            xx += ww
        yy += hh


def make_slides():
    # 01 — cover
    s = Slide("", section="", dark=True)
    s.rect(0, 0, 18, 900, ORANGE)
    s.tag(82, 67, "INCUBATION PROPOSAL", fill="263246", color="FF9877", w=340)
    s.text(80, 153, 1040, 165, "精度 × 驻留\n混合大模型的联合优化", 37, WHITE, True, leading=1.15)
    s.text(84, 345, 1010, 78, "Precision–Placement Co-Optimization\nfor Hybrid LLM Inference", 20, "B6C5D9")
    s.text(84, 472, 920, 110, "KV Cache · SSM 状态 · MoE 专家权重\nHBM · CXL 附加内存 · SSD", 22, WHITE)
    s.rect(84, 652, 865, 95, "1D2B40", radius=12)
    s.text(108, 672, 817, 65, "固定资源与质量约束下，验证相对强基线\n至少 15% 的有效吞吐增益", 19, "FFB59A", True)
    # A visual joint-choice matrix, rather than a brand logo.
    for i, lab in enumerate(["HBM", "CXL", "SSD"]):
        s.text(1112+i*126, 294, 116, 35, lab, 15, "B6C5D9", True, "center")
    for r, lab in enumerate(["KV", "SSM", "MoE"]):
        s.text(1015, 365+r*104, 85, 50, lab, 17, WHITE, True)
        for c in range(3):
            selected = c == [0, 0, 1][r]
            s.rect(1112+c*126, 350+r*104, 110, 83, ORANGE if selected else "243247", radius=12)
            s.text(1122+c*126, 374+r*104, 90, 40, ["q₁", "q₂", "q₃"][r] if selected else "·", 22, WHITE if selected else "6F829F", True, "center")
    s.text(1085, 700, 430, 56, "示意：精度与位置共同决策\n图中位宽及放置不代表最优解", 11, "97A9C4", align="center")
    s.footer()
    s.note("开场：本提案研究一个明确的系统决策问题——对 KV Cache、SSM 递归状态和 MoE 专家权重，同时决定用多少精度、放在哪里。资源、质量要求和服务延迟要求固定，目标是提升满足这些要求的有效吞吐。15% 是拟议验收目标，不是已取得结果。今天申请的是先验证关键假设的 incubation 阶段。", 40)

    # 02 — executive decision
    s = Slide("建议孵化：先证明联合决策的额外价值", "战略匹配已经明确；创新机制与净收益需要用强基线验证。", "01 / Investment thesis", "来源：S02–S03；本页立项建议为提案判断。")
    s.header()
    card(s, 80, 235, 464, 400, "WHY SK HYNIX", "连接三层产品能力", "以系统效率连接 HBM、CXL 内存与 SSD。\n\n形成面向客户工作负载的配置依据与软件参考方案。", ORANGE)
    card(s, 568, 235, 464, 400, "WHAT TO INCUBATE", "统一分配精度与驻留", "三类数据共同竞争快层容量和链路带宽。\n\n优化跨对象的资源取舍，检验顺序决策留下的空间。", TEAL)
    card(s, 1056, 235, 464, 400, "DECISION REQUEST", "批准六个月验证", "设立跨职能小组与有限测试资源。\n\n通过阶段门决定是否进入 2027–2029 年产品协同路线。", BLUE)
    takeaway(s, "立项命题：同样的资源与质量，能否通过联合优化产生 ≥15% 的净有效吞吐增益？")
    s.note("先给评审结论：建议批准六个月概念验证，三年路线作为条件式扩展。战略上，海力士具备对应产品与公开布局。技术上，尚待证明的是三个数据对象共享预算时，联合决策比强顺序与交替优化多创造多少价值。申请范围包括人员、测试平台与工作负载接入，不预先承诺新芯片开发。", 50)

    # 03 — public evidence
    s = Slide("窗口已经打开：模型与内存系统正在共同变化", "公开证据支持开展验证；市场份额、未来架构占比和产品量产节奏均未作预测。", "02 / Why now", "来源：S01–S03、S12；公开信息截至 2026-11-08。")
    s.header()
    entries = [
        ("模型侧", "混合架构已有实例", "Nemotron 系列提供 Attention / Mamba / MoE 的真实混合模型场景。", "S12", ORANGE),
        ("系统侧", "容量与访问共同约束", "三类数据的生命周期、读写模式与计算强度存在明显差异。", "S01", TEAL),
        ("企业侧", "系统协同已成公开方向", "海力士展示 CXL 内存池、DRAM+SSD 混合内存及 KV 复用方案。", "S02–S03", BLUE),
    ]
    for i, (a,b,c,ref,col) in enumerate(entries):
        yy = 235+i*158
        s.rect(80, yy, 1440, 139, WHITE, radius=14)
        s.tag(103, yy+24, a, PALE_ORANGE if i==0 else PALE_TEAL if i==1 else PALE_BLUE, col, 120)
        s.text(253, yy+20, 1100, 49, b, 22, INK, True)
        s.text(253, yy+79, 1100, 52, c, 16, MUTED)
        s.text(1390, yy+25, 100, 30, ref, 11, col, True, "right")
    takeaway(s, "策略：以混合模型切入，并保留对 GQA / MLA / 其他递归状态结构的适配能力。")
    s.note("这页用三条证据说明时机。混合架构已经存在，NOVA 等研究描述了不同运算和内存需求，海力士也公开展示了跨层系统。我们不把单一架构的未来普及当成确定事实，而是以它作为高价值验证场景。平台应把对象特征与模型实现解耦，保留适配不同注意力和线性递归模块的能力。", 45)

    # 04 — shared-resource problem
    s = Slide("三个数据对象，共享同一份快层预算", "对象不同，收益和代价不同；独立优化容易忽略资源替代关系。", "03 / Core problem", "来源：S01、S10、S13；数据特征归纳。")
    s.header()
    table(s, 80, 232, [255, 400, 400, 385], [
        ["优化对象", "内存行为", "精度风险", "分层机会"],
        ["KV Cache", "随上下文与并发扩张；活跃 KV 可能每步读取", "层 / 头 / 数据块敏感性不同；长上下文需单独评估", "活跃数据与可复用前缀分开管理"],
        ["SSM 状态", "每步读写；状态大小对序列长度固定、随并发增长", "误差进入递归；影响随生成长度变化", "活跃状态靠近计算；暂停状态可迁出"],
        ["MoE 专家权重", "基本只读；路由决定工作集，热度随负载变化", "访问热度不等于量化敏感性", "驻留 / 预取 / 缺失处理 / 混合精度"],
    ], [60, 134, 134, 134], font_size=14)
    takeaway(s, "统一问题：一单位 HBM 容量，分给哪类数据、以什么精度保存，才能带来最大系统收益？")
    s.note("先把层次讲清楚：KV Cache、SSM 状态和专家权重是数据对象，Attention、SSM、MoE 是模块。容量增长规律和访问行为不同。尤其不要把旧 KV 直接当冷数据，也不要把冷专家直接当不敏感数据。三者同时存在时，共同使用 HBM 容量、互连带宽与执行资源，局部最优未必能组合成全局最优。", 55)

    # 05 — two-dimensional search
    s = Slide("联合决策空间：精度 × 位置，而非固定绑定", "候选位宽取决于算子、设备与质量验证；表中仅示意策略空间。", "04 / Proposal definition", "本页为方案设计；CXL 层指附加 DRAM / 内存池。")
    s.header()
    s.text(90, 230, 210, 36, "量化配置 qᵢ", 17, TEAL, True)
    cols = [("HBM", "低延迟 / 高带宽", ORANGE), ("CXL 内存", "容量扩展 / 共享", TEAL), ("SSD", "容量 / 持久化", BLUE)]
    for c, (a,b,col) in enumerate(cols):
        xx = 330+c*397
        s.rect(xx, 226, 375, 90, NAVY, radius=10)
        s.text(xx+15, 237, 345, 35, a, 19, WHITE, True, "center")
        s.text(xx+15, 279, 345, 28, b, 11.5, "B4C2D6", align="center")
    for r, label in enumerate(["较高精度", "中等精度", "较低精度"]):
        yy = 337+r*112
        s.text(94, yy+25, 205, 50, label, 19, INK, True)
        for c in range(3):
            xx = 330+c*397
            s.rect(xx, yy, 375, 94, WHITE, stroke=LINE, radius=10)
            s.text(xx+20, yy+20, 335, 62, ["容量与质量取舍", "传输与驻留取舍", "访问时限与预取可行性"][c], 15, MUTED, align="center")
    s.tag(90, 690, "实际候选集须通过硬件与质量筛选", PALE_TEAL, TEAL, 620)
    takeaway(s, "策略为每个对象选择可行的 (qᵢ, pᵢ)，并考虑迁移 / 预取开销与执行位置。")
    s.note("这就是提案的两个维度。不同精度可以出现在不同层，不能预设低精度只在 SSD、高精度只在 HBM。候选集不是所有对象都能任意使用 2、4、8、16 位，必须看内核支持和质量。位置决策可以包含副本或精度补充，但需要把重复占用计入预算。普通 CXL 内存和具有近数据计算能力的设备必须分别建模。", 50)

    # 06 — illustrative capacity threshold
    s = Slide("核心假设：跨对象释放容量，可能跨越性能阈值", "说明性容量算例：总 HBM 预算固定为 80 GB；不代表任一模型测量结果。", "05 / Mechanism to validate", "全部数字为机制示意；未假设量化无损，也未推导吞吐收益。")
    s.header()
    legend = [("其他与预留", "8B98AC"), ("SSM 状态", TEAL), ("KV Cache", BLUE), ("专家驻留", ORANGE)]
    for j, (lab,col) in enumerate(legend):
        x=80+j*355
        s.rect(x, 236, 22, 22, col, radius=3)
        s.text(x+33, 231, 310, 40, lab, 14, INK)
    for r, (label, vals) in enumerate([("独立配置", [20,10,20,30]), ("联合候选", [20,5,15,40])]):
        yy = 345+r*158
        s.text(80, yy+12, 220, 45, label, 21, INK, True)
        xx=320
        for (lab,col), v in zip(legend, vals):
            ww=v/80*1160
            s.rect(xx, yy, ww-2, 82, col, radius=4)
            s.text(xx+4, yy+23, ww-10, 43, str(v)+" GB", 16, WHITE, True, "center")
            xx += ww
    s.rect(80, 645, 1440, 90, PALE_TEAL, radius=12)
    s.text(104, 662, 1392, 65, "待验证链条：KV / 状态压缩 → 释放 10 GB → 更多专家驻留 → 缺失流量下降 → 有效吞吐变化", 17, TEAL, True)
    takeaway(s, "关键实验：净收益能否覆盖质量损失、解量化、迁移、内核及调度开销？")
    s.note("用容量算例解释创新机制，不能把这页当实验。上方 20+10+20+30 等于 80 GB；下方 20+5+15+40 仍等于 80 GB。压缩状态和 KV 假设释放 10 GB，换成更大的专家驻留集。它是否减少缺失、是否有 15% 的收益完全需要测量。还可能出现相反情况：压缩影响质量，或者解量化使收益消失。", 55)

    # 07 — prior art
    s = Slide("相关工作密集：创新需要落在可检验的机制上", "比较聚焦本轮核查的直接相关能力；不构成穷尽查新或专利结论。", "06 / Prior art & differentiation", "来源：S04–S11；摘要及 S06 正文核查。")
    s.header()
    table(s, 80, 230, [285, 665, 490], [
        ["相关工作", "已覆盖能力", "本提案需进一步证明"],
        ["HOBBIT / SliceMoE", "混合精度专家卸载、缓存或精度切片", "跨对象共享预算的增量价值"],
        ["TailorKV / StrataFlux", "KV 量化与卸载；权重 / KV 的跨层精度补充", "递归状态误差与驻留的耦合"],
        ["CXL GPU–NDP MoE", "HBM / CXL-NDP 专家分配 + 专家位宽", "三类对象及 SSD 的统一资源模型"],
        ["ITME / HyMCache", "CXL / DRAM / SSD，权重或前缀 KV 分层", "精度选择带来的额外优化空间"],
        ["ReplaySSM / PACE", "递归状态重构、周期更新或量化", "状态质量预算对其他对象的影响"],
    ], [55, 80, 80, 80, 80, 80], font_size=13)
    takeaway(s, "拟议贡献：统一边际收益模型 + 跨对象分配策略 + 实际在线开销与净收益验证。")
    s.note("主动承认已有工作会增加可信度。我们不声称首次做量化与卸载，也不把增加一层 SSD 本身视为突破。最接近的 MoE 研究已经在 HBM/CXL-NDP 上做专家位宽选择。需要证明的差异是三个对象在共享预算下怎样相互替代，以及运行时能否低成本实现。全部新颖性目前属于待验证判断，不能据此保证专利性。", 60)

    # 08 — formal objective
    s = Slide("优化目标：固定资源下最大化有效吞吐", "Goodput = 在规定质量与 TTFT / TPOT 服务时限内完成的输出 token / 秒。", "07 / Objective & constraints")
    s.header()
    s.rect(80, 235, 890, 170, NAVY, radius=16)
    s.text(112, 264, 826, 60, "max  Goodput(q, p, π | R, W)", 29, WHITE, True)
    s.text(112, 344, 826, 44, "q：精度配置   p：驻留 / 副本   π：迁移、预取与执行策略", 14, "B7C8DD")
    s.rect(998, 235, 522, 170, PALE_ORANGE, radius=16)
    s.text(1024, 258, 470, 45, "≥15% 是验收假设", 24, ORANGE, True)
    s.text(1024, 322, 470, 64, "R 为固定资源；W 为预先登记的负载。\n比较对象为同约束下的强基线。", 15, INK)
    blocks=[("容量约束", "HBM / CXL / SSD 容量\n副本、元数据、临时缓冲均计入"), ("链路与计算约束", "有效带宽、队列、设备可达性\n内核精度支持与转换吞吐"), ("质量与服务约束", "关键任务非劣性 / 长生成质量\nTTFT、TPOT 尾延迟与成功率")]
    for i,(a,b) in enumerate(blocks):
        xx=80+i*488
        s.rect(xx, 440, 464, 239, WHITE, radius=14)
        s.text(xx+25, 464, 414, 43, a, 20, [ORANGE,TEAL,BLUE][i], True)
        s.text(xx+25, 531, 414, 128, b, 16, MUTED)
    takeaway(s, "离线阶段求质量与成本画像；在线阶段执行有界开销的策略。")
    s.note("先固定实验对象，再定义目标，避免把每种收益混成一个 15%。主要 KPI 是满足质量和延迟约束的有效吞吐，副指标包括流量、能耗和每百万合格 token 成本。质量门槛必须按关键任务预先定义，不能只用平均困惑度。资源包括临时缓冲和多个精度副本。正式优化可用离散、分段线性或混合整数方法，不预先指定非线性求解器。", 60)

    # 09 — architecture
    s = Slide("系统方案：离线画像 + 在线联合资源管理", "第一阶段采用现有 GPU 与内存 / SSD；近存计算作为独立扩展配置评估。", "08 / Proposed architecture", "本页为拟议架构；CXL GPU 直达、DMA 与设备端计算均需按平台验证。")
    s.header()
    s.rect(80, 235, 380, 430, WHITE, radius=14)
    s.text(105, 257, 330, 48, "离线画像与校准", 22, INK, True)
    s.text(105, 328, 330, 270, "质量敏感性\n按对象 / 层 / 专家评估\n\n平台成本模型\n访问、转换、迁移与计算\n\n候选策略库\n可行精度与驻留组合", 16, MUTED)
    s.rect(510, 235, 550, 430, NAVY, radius=14)
    s.text(538, 257, 494, 48, "在线联合资源管理器", 22, WHITE, True)
    for j, t in enumerate(["观测：并发、长度、路由、队列与命中", "决策：快层容量与精度的统一分配", "执行：驻留、预取、迁移与版本切换", "保护：质量约束、最小驻留期与回退"]):
        s.rect(538, 330+j*73, 494, 56, "25374F", radius=8)
        s.text(553, 343+j*73, 464, 38, t, 14.5, WHITE)
    s.line(460, 450, 505, 450, ORANGE, 4, True)
    for j,(a,b,col) in enumerate([("HBM", "GPU 执行 / 高价值驻留", ORANGE), ("CXL 内存", "容量扩展 / 复用状态", TEAL), ("SSD", "后备容量 / 冷数据", BLUE)]):
        yy=235+j*147
        s.rect(1110, yy, 410, 136, WHITE, radius=12)
        s.text(1134, yy+20, 362, 42, a, 22, col, True)
        s.text(1134, yy+82, 362, 42, b, 14.5, MUTED)
        s.line(1060, yy+66, 1104, yy+66, col, 3, True)
    s.text(90, 697, 1410, 44, "主机 DDR / 页锁定缓冲、PCIe / CXL 拓扑和数据中转路径均纳入测量。", 16, MUTED)
    takeaway(s, "核心资产：可校准的成本模型、对象级遥测、低开销策略与推理框架连接器。")
    s.note("架构分为离线和在线。离线量化校准不在每个 token 执行，先获取质量敏感性和真实硬件成本。在线利用请求和设备遥测，以不同时间尺度调整预算、驻留和预取。第一年不需要设计 NOVA 那样的新 DRAM，也不默认 CXL 设备带计算功能。主机 DRAM 中转不能被省略；GPU 直达、CPU 复制和设备计算路径要分别测。", 55)

    # 10 — policy design
    s = Slide("拟议策略：按净边际收益分配稀缺资源", "先建立小规模可解释上界，再设计接近上界的在线策略。", "09 / Algorithm hypothesis")
    s.header()
    steps=[("01", "生成可行候选", "精度 × 驻留 × 访问路径\n过滤不满足质量与内核支持的组合"), ("02", "评估净边际收益", "收益：减少关键路径等待\n成本：占用、迁移、转换与误差"), ("03", "统一预算分配", "跨 KV / 状态 / 专家竞争容量\n显式处理带宽与队列冲突"), ("04", "有界更新与回退", "设置最小驻留期和收益门槛\n负载漂移时重规划并安全切换")]
    for i,(n,a,b) in enumerate(steps):
        xx=80+i*366
        s.rect(xx, 246, 342, 378, WHITE, radius=14)
        s.text(xx+25, 270, 292, 62, n, 34, ORANGE, True)
        s.text(xx+25, 356, 292, 74, a, 20, INK, True)
        s.text(xx+25, 455, 292, 150, b, 15, MUTED)
        if i<3: s.line(xx+342, 425, xx+361, 425, TEAL, 3, True)
    s.rect(80, 660, 1440, 73, PALE_BLUE, radius=12)
    s.text(104, 679, 1392, 49, "策略粒度建议：请求准入时定预算；阶段边界重规划；每步只执行轻量计数与预取。", 16, BLUE, True)
    takeaway(s, "验收重点：净收益相对强交替优化仍存在，并且控制开销低于其带来的节省。")
    s.note("这页是算法假设，不是已经实现的算法。净边际收益需要包含质量和性能，不应只按访问热度排序。离线小规模可以枚举或优化求上界，在线采用查表、贪心或有界搜索，具体方法由测量决定。避免每个 token 都重新量化或大规模搬迁。质量恢复必须依赖保留的原始数据、残差或有效检查点，不能把低位宽转成高位宽当成恢复。", 55)

    # 11 — different semantics
    s = Slide("统一优化框架，保留三类对象的不同语义", "质量和数据正确性首先定义可行域，再在可行域中优化性能。", "10 / Object-specific safeguards", "设计依据：S05–S07、S10–S13。")
    s.header()
    card(s, 80, 235, 464, 425, "KV CACHE", "按生命周期区分", "活跃 KV 与可复用前缀分开。\n\n稠密注意力的旧 KV 仍可能每步读取；SSD 需要预取或复用窗口。\n\n检索、长上下文质量单独验证。", BLUE, 15)
    card(s, 568, 235, 464, 425, "RECURRENT STATE", "保护递归历史", "活跃状态优先靠近执行单元。\n\n低精度误差可能反复进入更新；提高位宽不能恢复丢失历史。\n\n暂停状态与检查点允许迁出。", TEAL, 15)
    card(s, 1056, 235, 464, 425, "EXPERT WEIGHTS", "热度与敏感性分开", "推理权重只读，便于保留主副本。\n\n使用热度决定驻留价值，使用质量画像决定量化容忍度。\n\n副本与精度切片计入真实容量。", ORANGE, 15)
    takeaway(s, "统一的是资源分配目标；状态一致性、误差传播与访问时限必须分别建模。")
    s.note("统一模型不代表忽略语义。KV 前缀复用与每步读取的活跃 KV 不是同一类访问。SSM 状态是不断变化的，量化误差进入下一次更新，不能无依据迁到 SSD。专家则是只读，可以选择主副本或补充切片，但成本必须计入。要先定义哪些组合能用，再进行全局优化。", 45)

    # 12 — baseline design
    s = Slide("15% 必须来自联合决策，并经强基线验证", "所有对比保持同一质量门槛、资源预算、服务时限、预热条件与输入序列。", "11 / Baselines & attribution")
    s.header()
    table(s, 80, 231, [170, 510, 760], [
        ["编号", "比较方案", "它排除的替代解释"],
        ["B0", "固定精度 + 静态放置", "提供可复现参考，不单独用于证明创新"],
        ["B1 / B2", "只优化精度 / 只优化放置", "判断单一维度是否已解释全部收益"],
        ["B3", "双顺序 + 充分调优的交替优化", "排除仅因执行顺序或调参不足形成的优势"],
        ["B4", "最接近方法的适用实现 / 组合基线", "与相关先进方法比较，明确兼容范围及改动"],
        ["J / U", "联合策略 / 小规模离线上界", "量化可实现收益与剩余优化空间"],
    ], [55, 76, 76, 90, 90, 80], font_size=13.5)
    takeaway(s, "主要基线 B*：B1–B4 中满足同等约束的最强可复现方案；不得只选 B0。")
    s.note("这是项目能否证明创新的关键。比较不能停留在静态分层，必须包括充分调优的顺序和交替优化。相关方法要按各自适用范围接入，不能把某个只解决 KV 的系统直接作为完整系统的弱基线。对于不兼容组件需要说明实现差异或构建组合基线。Oracle 是小规模离线上界，不是生产可用系统，也不能用理想预知替代在线预测成本。", 60)

    # 13 — experiment matrix
    s = Slide("实验覆盖：模型、内存压力与负载动态", "在实验前登记主测试集合；完整报告有收益、无收益和性能退化的场景。", "12 / Evaluation plan", "模型候选：S01、S12；具体 checkpoint 与内核支持在 G0 锁定。")
    s.header()
    table(s, 80, 232, [290, 675, 475], [
        ["维度", "建议设置", "要验证的问题"],
        ["模型", "至少 2 个混合模型；1 个 Attention–MoE 对照", "收益是否依赖某单一结构？"],
        ["上下文 / 生成", "8K / 32K / 128K 输入，短与长生成；按模型支持筛选", "KV 与递归状态瓶颈如何变化？"],
        ["并发与路由", "低 / 高并发、路由漂移、多轮与暂停恢复", "在线策略是否稳定？"],
        ["资源配置", "工作集可驻留、接近阈值、明显溢出；固定各层容量", "收益来自哪里，何时消失？"],
        ["硬件路径", "真实 CXL + NVMe；GPU 访问 / 中转路径分别测量", "仿真是否与可部署路径一致？"],
    ], [55, 78, 87, 78, 87, 87], font_size=13.5)
    takeaway(s, "分组报告：容量受限场景的主 KPI，与可完全驻留场景的性能回归边界。")
    s.note("实验集合先登记，避免做完后只选择收益大的配置。以至少两个混合模型验证可迁移性，再用 Attention–MoE 模型观察状态模块的贡献。容量压力要覆盖完全装得下、接近阈值与明显溢出三种情形。若模型不支持 128K，不强行外推，替换为支持范围内配置。测真实路径，并将分析模型、仿真和实测分开报告。", 55)

    # 14 — KPI and gate
    s = Slide("预先定义验收：主指标、约束与归因同时通过", "建议标准，须在 G0 与评审共同锁定；此处没有项目实测结果。", "13 / Success criteria")
    s.header()
    s.rect(80, 235, 490, 463, NAVY, radius=16)
    s.text(112, 268, 426, 48, "PRIMARY KPI", 15, "B6C7DD", True)
    s.text(112, 343, 426, 99, "+15%", 57, "FF9D7A", True)
    s.text(112, 460, 426, 175, "相对 B* 的有效吞吐增益\n\n主测试集配对比值的几何均值 ≥1.15；95% 置信区间下界 >1.00。", 18, WHITE)
    checks=[("质量", "关键任务逐项非劣性；容差与评测集预登记。长检索、推理及长生成单列。"), ("服务", "满足约定 TTFT / TPOT 尾延迟与成功率要求；不以放宽 SLO 换吞吐。"), ("归因", "分别消融跨对象预算、精度选择、驻留和预取；控制开销完整计入。"), ("部署", "主结论以实测支持；仿真与预测单列，能耗 / TCO 仅在可测范围报告。")]
    for i,(a,b) in enumerate(checks):
        yy=235+i*119
        s.rect(602, yy, 918, 105, WHITE, radius=12)
        s.tag(623, yy+17, a, PALE_TEAL, TEAL, 94)
        s.text(740, yy+17, 751, 77, b, 15, INK)
    takeaway(s, "通过门槛：效应量、统计可信度、质量和服务约束均成立；不以单次峰值代替验证。")
    s.note("15% 是拟议目标值，统计上要重复测量。建议验收为：几何均值至少 1.15，且配对分析的 95% 置信区间下界高于 1.00。注意，这证明收益方向具有统计支持，并不表示真实增益的下界也达到 15%。正式协议在 G0 锁定。主要集合为预登记的容量受限场景，逐场景计算相对最强基线的比值；完整集合另列，不能隐去负结果。质量阈值由关键任务决定。能耗测整机或明确边界，TCO 用敏感性分析。", 60)

    # 15 — incubation gates
    s = Slide("六个月 incubation：用阶段门降低投入不确定性", "先判断优化空间，再判断可实现性，最后判断部署价值。", "14 / Incubation plan")
    s.header()
    phases=[("G0", "第 1–6 周", "锁定问题与上界", "模型 / 质量 / SLO / B* 锁定\n真实路径校准与瓶颈画像\n小规模离线上界", ORANGE), ("G1", "第 7–12 周", "验证联合机制", "可运行联合策略\n对比强顺序与交替基线\n消融与在线开销测量", TEAL), ("G2", "第 13–24 周", "验证净收益与部署", "至少 2 个混合模型\n真实 CXL / SSD 与负载漂移\n质量、尾延迟和回退验证", BLUE)]
    for i,(gate,t,a,b,col) in enumerate(phases):
        xx=80+i*488
        s.rect(xx, 240, 464, 419, WHITE, radius=16)
        s.tag(xx+24, 262, gate, PALE_ORANGE if i==0 else PALE_TEAL if i==1 else PALE_BLUE, col, 80)
        s.text(xx+126, 267, 310, 38, t, 16, MUTED)
        s.text(xx+24, 337, 416, 83, a, 23, INK, True)
        s.text(xx+24, 448, 416, 180, b, 16, MUTED)
    s.rect(80, 689, 1440, 52, PALE_ORANGE, radius=10)
    s.text(103, 700, 1394, 37, "停止 / 转向条件：上界不足、收益被开销抵消，或质量与服务约束无法同时满足。", 15, ORANGE, True)
    takeaway(s, "G2 通过后再申请产品化投入；若仅单对象有效，收缩为对应专项。")
    s.note("六个月不是研发全部产品，而是回答是否值得继续。G0 先测瓶颈和小规模上界，若充分约束后的空间小于目标，就及时调整。G1 实现算法并归因，G2 上真实设备和动态负载。若只有 KV 或专家的局部方案有效，可以转为专项，而不继续坚持三对象联合框架。分阶段决策使评审能够控制投入风险。", 55)

    # 16 — product roadmap
    s = Slide("2027–2029：从机制验证走向产品协同", "以下为拟议研发路线；后续阶段均以此前阶段证据与合作条件为前提。", "15 / Three-year strategic path", "战略背景：S02–S03；年份与交付物均为本提案建议。")
    s.header()
    for i,(year,a,b,c,col) in enumerate([
        ("2027", "证明机制", "联合策略 + 可校准模型", "输出：测评方法、参考实现、适用边界\n决策：≥15% 净增益是否可复现", ORANGE),
        ("2028", "联合原型", "设备接口 + 框架连接器", "输出：CXL / SSD 遥测与提示接口\n决策：客户工作负载与成本是否获益", TEAL),
        ("2029", "参考方案", "配置指南 + 客户验证", "输出：软硬件参考架构与部署规范\n决策：生态接入、维护成本与商业可行性", BLUE),
    ]):
        yy=237+i*157
        s.rect(80, yy, 1440, 139, WHITE, radius=14)
        s.text(105, yy+25, 200, 70, year, 38, col, True)
        s.text(330, yy+19, 375, 48, a, 24, INK, True)
        s.text(330, yy+81, 375, 37, b, 14, MUTED)
        s.text(760, yy+25, 724, 100, c, 16, INK)
    takeaway(s, "长期资产：工作负载画像、质量—成本模型、设备接口与跨平台验证能力。")
    s.note("三年规划用交付物定义，不预设量产。2027 年验证机制，2028 年与设备及推理框架协同，2029 年在客户场景形成参考方案。HBF 或带计算能力的 CXL 可以作为扩展变量，但基础方案不能依赖它们未来必然可用。商业价值还要评估客户采用、研发维护和伙伴分工，不能把性能提升直接等同于收入增长。", 45)

    # 17 — request and business
    s = Slide("资源申请：小团队、可复现平台、明确伙伴边界", "建议配置用于六个月验证；预算与设备规格在 G0 按实际可用资源核定。", "16 / Resources & ownership")
    s.header()
    card(s, 80, 235, 464, 398, "PEOPLE", "4–5 人跨职能小组", "算法 / 量化：1 人\n推理系统与内核：2 人\n平台与性能工程：1 人\n项目 / 产品协同：兼职或 1 人", ORANGE, 16)
    card(s, 568, 235, 464, 398, "PLATFORM", "有限规模真实测试", "GPU 服务器与固定 HBM 预算\n真实 CXL 内存及 NVMe SSD\n可记录带宽 / 功耗的工具链\n框架、模型与数据访问条件", TEAL, 16)
    card(s, 1056, 235, 464, 398, "PARTNERS", "对接客户与生态", "海力士：器件特性与系统模型\n加速器 / OEM：拓扑与驱动\n推理框架：接口与内核接入\n客户：负载、质量与 SLO", BLUE, 16)
    s.rect(80, 664, 1440, 72, PALE_BLUE, radius=12)
    s.text(104, 681, 1392, 53, "商业评估：每百万合格 token 成本、配置采用率与产品组合价值；不假设 HBM 需求必然增长。", 16, BLUE, True)
    takeaway(s, "本轮申请：六个月人力与验证资源 + 阶段评审机制 + 指定业务 / 技术对接人。")
    s.note("人力和设备是建议配置，不是经批准预算。算法与系统工程需要共同参与，否则无法计入真实内核和数据移动成本。海力士控制内存与 SSD 产品，并不控制 GPU 软件栈，所以要明确 OEM、加速器与推理框架伙伴。客户价值用合格 token 成本衡量，企业收益还要看采用率和产品组合，减少 HBM 使用并不自动意味着商业收益增加。", 45)

    # 18 — decision close
    s = Slide("批准孵化，验证一个明确的系统命题", "Precision–Placement Co-Optimization for Hybrid LLM Inference", "17 / Decision", dark=True)
    s.header()
    s.text(84, 260, 1430, 130, "同样的资源与质量，\n联合分配精度与位置，能否创造更高的有效吞吐？", 30, WHITE, True)
    for i,(a,b) in enumerate([("研究对象", "KV Cache / SSM 状态 / MoE 权重"), ("优化维度", "量化精度 × HBM / CXL / SSD 驻留"), ("证据标准", "相对强基线 ≥15% 净增益，满足质量与 SLO")]):
        yy=440+i*85
        s.text(90, yy, 230, 45, a, 17, "FF9D7A", True)
        s.text(335, yy, 1175, 54, b, 21, WHITE)
    s.rect(80, 727, 1440, 80, ORANGE, radius=12)
    s.text(106, 747, 1388, 55, "请求决策：批准六个月 PoC，按 G0 / G1 / G2 证据决定继续、收缩或停止。", 21, WHITE, True)
    s.footer()
    s.note("结束时重复决策请求。项目目标清楚，但性能结论尚未获得。我们申请的是六个月验证窗口，用强基线、真实路径和质量约束判断 15% 目标。通过后再扩大到产品接口和客户验证；未通过则根据证据收缩或停止。接下来进入问答，附录包含数学模型、收益阈值、实现风险和证据索引。", 35)

    # A1 — mathematical definition
    s = Slide("附录 A｜精度、驻留与执行路径的离散决策模型", "用于小规模离线分析与上界估计；在线实现不要求直接运行完整优化器。", "Appendix / Formalization", appendix=True)
    s.header()
    s.rect(80, 235, 880, 463, WHITE, radius=14)
    lines=[("决策变量", "xᵢₖ ∈ {0,1}：对象 i 是否采用候选配置 k"), ("候选配置 k", "位宽、位置、副本、计算 / 传输路径及版本"), ("容量约束", "Σᵢ Σₖ xᵢₖ · bytesᵢₖₜ ≤ capacityₜ"), ("选择约束", "Σₖ xᵢₖ = 1；副本作为候选配置整体计价"), ("目标与评估", "以执行模拟 / 实测估计服务 goodput，并验证质量")]
    for i,(a,b) in enumerate(lines):
        yy=258+i*82
        s.text(105, yy, 200, 40, a, 16, TEAL, True)
        s.text(325, yy, 605, 66, b, 15, INK)
    s.rect(990, 235, 530, 463, NAVY, radius=14)
    s.text(1018, 260, 474, 50, "非线性来自真实系统", 23, WHITE, True)
    s.text(1018, 343, 474, 320, "质量对位宽的非线性响应\n\n容量命中与溢出的阈值\n\n并发、队列与流水重叠\n\n跨对象质量与路由交互", 18, "C0CEE0")
    takeaway(s, "可用查表与分段近似求解；质量代理函数必须用端到端任务重新验证。")
    s.note("模型把一个对象的位宽、位置、副本和执行路径组成一个候选配置。容量包含所有数据副本、元数据和缓冲。质量损失不默认按对象线性相加，可先用画像做代理，再做端到端验证。延迟也不把所有传输简单相加，应使用实际流水和队列模型。小规模离线求解用于判断空间，线上可用近似策略。", 60)

    # A2 — hypothetical 15%
    s = Slide("附录 B｜何时可能超过 15%，何时不会", "简化延迟算例仅说明条件，不代表吞吐预测，也不能替代端到端实验。", "Appendix / Threshold analysis", appendix=True)
    s.header()
    s.rect(80, 237, 1440, 112, NAVY, radius=14)
    s.text(111, 267, 1378, 65, "总延迟下降比例 ≈ f × r − o", 31, WHITE, True, "center")
    s.text(100, 378, 1400, 54, "f：基线关键路径搬运等待占比   r：被消除的比例   o：新增开销占原总延迟比例", 17, MUTED, align="center")
    table(s, 80, 457, [310, 265, 265, 265, 335], [
        ["假设场景", "f", "r", "o", "延迟变化"],
        ["等待明显且可减少", "40%", "50%", "3%", "下降 17%"],
        ["搬运已经不重要", "10%", "50%", "3%", "下降 2%"],
    ], [57, 80, 80], font_size=16)
    takeaway(s, "仅在固定串行服务假设下，延迟下降 17% 对应约 20.5% 速率提升；生产 goodput 须实测。")
    s.note("f 指原本未被重叠隐藏的关键路径搬运等待，不是所有数据搬运时间。第一行 0.4×0.5−0.03 等于 0.17，第二行只有 0.02。若原系统已完全驻留或计算受限，不能预期 15%。在单请求固定串行条件下可以用 1/0.83 换算速率，但服务系统有排队、并发和 SLO，不能把这个换算直接当 goodput 结果。", 50)

    # A3 — risks
    s = Slide("附录 C｜主要技术风险与对应实验", "每个风险对应一个可观测指标和阶段内处理方式。", "Appendix / Risk register", appendix=True)
    s.header()
    table(s, 80, 231, [300, 510, 630], [
        ["风险", "识别方式", "处理 / 决策"],
        ["SSM 递归误差", "长生成、状态漂移、任务质量", "限制状态候选；检查点 / 残差方案单独计价"],
        ["CXL 路径不成立", "平台可达性、有效带宽、中转开销", "先测真实路径；不成立则改拓扑或调整范围"],
        ["SSD 暴露关键延迟", "缺失次数、P99、预取覆盖率", "限制活跃对象下沉；采用可行恢复窗口"],
        ["转换与策略开销", "内核时间、重规划时间、迁移流量", "离线候选库、收益门槛与驻留滞回"],
        ["收益只在单一场景", "全矩阵结果、消融、负载漂移", "明确适用边界；必要时转为单对象专项"],
        ["缺少商业差异", "客户负载与配置成本敏感性", "围绕设备接口、合作接入和参考配置形成资产"],
    ], [54, 70, 70, 70, 70, 70, 70], font_size=13)
    takeaway(s, "SSD 写放大、耐久度、数据版本一致性与副本成本，在实测原型中纳入评估。")
    s.note("风险登记是技术工作内容。特别需要将量化状态的不可逆误差与只读权重区分。CXL 支持不能从标准描述推断目标 GPU 路径可用。SSD 的读取队列和写耐久性需要在设备层测量。若上界或客户价值不足，阶段门允许停止，避免为形式上的完整方案持续投入。", 45)

    # A4 — source index
    s = Slide("附录 D｜证据索引：战略、模型与直接相关工作", "完整链接与证据边界见 SOURCES.md；PPT 演讲者备注包含对应链接。", "Appendix / References", "资料为公开来源；预印本结论需结合其评估方法理解。", appendix=True)
    s.header()
    refs=[
        ("S01", "NOVA", "混合负载与近存计算背景"),
        ("S02–S03", "SK hynix Future Forum / OCP 2026", "系统协同与 CXL / SSD 公开布局"),
        ("S04–S05", "HOBBIT / TailorKV", "专家与 KV 的量化 + 卸载"),
        ("S06–S07", "CXL GPU–NDP MoE / SliceMoE", "专家位宽、放置和精度切片"),
        ("S08–S09", "ITME / HyMCache", "CXL、DRAM、SSD 与 KV 复用"),
        ("S10–S11", "ReplaySSM / StrataFlux", "递归状态与跨层精度表示"),
        ("S12–S13", "Nemotron 3 Ultra / PACE", "混合模型实例与状态量化"),
    ]
    for i,(ref,a,b) in enumerate(refs):
        yy=235+i*67
        s.rect(80, yy, 1440, 58, WHITE, radius=7)
        s.text(97, yy+12, 175, 35, ref, 13, ORANGE, True)
        s.text(291, yy+12, 700, 35, a, 15, INK, True)
        s.text(1000, yy+12, 495, 35, b, 13, MUTED)
    takeaway(s, "证据边界：来源证明方向与已有能力；本项目创新、15% 收益和商业结果均需另行验证。")
    s.note("参考文献按主题分组，所有完整 URL 在 SOURCES.md。公开企业资料用于证明战略匹配，不作为项目成功的技术证据。论文之间硬件、负载、质量和基线不同，性能提升不能相加。此轮是提案查新，尚不是系统综述或专利自由实施判断。", 25)

    # A5 — review Q&A
    s = Slide("附录 E｜评审问答：把主张收敛到可证伪的假设", "建议以机制、证据和决策条件回答，避免用优化空间大小替代实验结果。", "Appendix / Anticipated questions", appendix=True)
    s.header()
    qas=[
        ("为什么不只做专家缓存？", "先证明状态 / KV 的预算调整能改变专家驻留及端到端表现；否则收缩项目。"),
        ("普通 CXL 与 CXL-NDP 是否一样？", "分别建模。基础原型不依赖设备端计算，NDP 作为独立扩展。"),
        ("15% 有保证吗？", "没有。它是相对强基线的拟议目标，通过离线上界和真实原型逐步验证。"),
        ("创新会不会只是组合？", "用强交替基线及跨对象消融检验；没有额外机制收益就不作强创新主张。"),
        ("为什么值得海力士做？", "把器件特性、软件策略与客户负载连接，形成配置依据和设备接口需求。"),
    ]
    for i,(q,a) in enumerate(qas):
        yy=234+i*96
        s.rect(80, yy, 1440, 85, WHITE, radius=10)
        s.text(102, yy+17, 420, 63, q, 16, INK, True)
        s.text(555, yy+17, 938, 64, a, 15, MUTED)
    takeaway(s, "评审原则：允许假设失败，但必须通过可复现实验清楚解释结果。")
    s.note("问答时保持主张边界。不要承诺所有对象都能放到 SSD，不默认热度等于敏感性，不把软件复合方案自动叫硬件创新。项目是否继续由证据决定。即使联合框架失败，经过校准的模型、负载画像和单对象发现也可以成为后续专项输入，但不能据此规避主 KPI 的验收。", 45)

    for s in SLIDES:
        if s.number not in (1,18):
            s.footer()


def draw_scene(s):
    image = Image.new("RGB", (W, H), rgb(s.bg))
    d = ImageDraw.Draw(image)
    for it in s.items:
        k=it["kind"]
        if k=="rect":
            box=(round(it["x"]), round(it["y"]), round(it["x"]+it["w"]), round(it["y"]+it["h"]))
            if it["radius"]:
                d.rounded_rectangle(box, radius=it["radius"], fill=rgb(it["fill"]), outline=rgb(it["stroke"]) if it["stroke"] else None, width=1)
            else:
                d.rectangle(box, fill=rgb(it["fill"]), outline=rgb(it["stroke"]) if it["stroke"] else None)
        elif k=="line":
            d.line((it["x"],it["y"],it["x2"],it["y2"]), fill=rgb(it["color"]), width=round(it["width"]))
            if it["arrow"]:
                x,y=it["x2"],it["y2"]
                d.polygon([(x,y),(x-9,y-6),(x-9,y+6)], fill=rgb(it["color"]))
        elif k=="text":
            f=font(it["size"],it["bold"])
            lh=it["size"]*PPI/72*it["leading"]
            for n,line in enumerate(it["lines"]):
                xx=it["x"]
                length=f.getlength(line)
                if it["align"]=="center": xx+=(it["w"]-length)/2
                elif it["align"]=="right": xx+=it["w"]-length
                d.text((xx,it["y"]+n*lh), line, font=f, fill=rgb(it["color"]), anchor="lt")
    return image


def build_pptx():
    prs=Presentation()
    prs.slide_width=Inches(W/PPI)
    prs.slide_height=Inches(H/PPI)
    prs.core_properties.title="精度 × 驻留：混合大模型的联合优化"
    prs.core_properties.subject="SK hynix incubation proposal — quality-constrained mixed precision and tiered placement"
    prs.core_properties.author="Incubation Proposal"
    prs.core_properties.keywords="HBM, CXL, SSD, KV Cache, SSM, MoE, mixed precision, incubation"
    for s in SLIDES:
        ps=prs.slides.add_slide(prs.slide_layouts[6])
        ps.background.fill.solid()
        ps.background.fill.fore_color.rgb=RGBColor.from_string(s.bg)
        for it in s.items:
            if it["kind"]=="rect":
                shp=ps.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if it["radius"] else MSO_SHAPE.RECTANGLE, Inches(it["x"]/PPI), Inches(it["y"]/PPI), Inches(it["w"]/PPI), Inches(it["h"]/PPI))
                if it["radius"]:
                    shp.adjustments[0]=min(.15,it["radius"]/min(it["w"],it["h"]))
                shp.fill.solid()
                shp.fill.fore_color.rgb=RGBColor.from_string(it["fill"])
                if it["stroke"]:
                    shp.line.color.rgb=RGBColor.from_string(it["stroke"])
                    shp.line.width=Pt(.6)
                else: shp.line.fill.background()
            elif it["kind"]=="line":
                # Native thin shape; arrowhead is also an editable triangle.
                xx=min(it["x"],it["x2"]); yy=min(it["y"],it["y2"])
                ww=max(abs(it["x2"]-it["x"]),it["width"])
                hh=max(abs(it["y2"]-it["y"]),it["width"])
                shp=ps.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(xx/PPI), Inches(yy/PPI), Inches(ww/PPI), Inches(hh/PPI))
                shp.fill.solid(); shp.fill.fore_color.rgb=RGBColor.from_string(it["color"]); shp.line.fill.background()
                if it["arrow"]:
                    a=ps.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches((it["x2"]-9)/PPI), Inches((it["y2"]-6)/PPI), Inches(12/PPI), Inches(12/PPI))
                    a.rotation=90
                    a.fill.solid(); a.fill.fore_color.rgb=RGBColor.from_string(it["color"]); a.line.fill.background()
            elif it["kind"]=="text":
                # Explicit lines and measured boxes preserve shared scene geometry.
                shp=ps.shapes.add_textbox(Inches(it["x"]/PPI), Inches(it["y"]/PPI), Inches(it["w"]/PPI), Inches(it["h"]/PPI))
                tf=shp.text_frame
                tf.clear()
                tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                tf.word_wrap=False
                tf.vertical_anchor=MSO_ANCHOR.TOP
                from pptx.enum.text import PP_ALIGN
                for n,line in enumerate(it["lines"]):
                    p=tf.paragraphs[0] if n==0 else tf.add_paragraph()
                    p.text=line
                    p.font.name="Microsoft YaHei"
                    p.font.size=Pt(it["size"])
                    p.font.bold=it["bold"]
                    p.font.color.rgb=RGBColor.from_string(it["color"])
                    p.space_before=p.space_after=Pt(0)
                    p.line_spacing=Pt(it["size"]*it["leading"])
                    p.alignment={"left":PP_ALIGN.LEFT,"right":PP_ALIGN.RIGHT,"center":PP_ALIGN.CENTER}[it["align"]]
                    link_map = {
                        "NOVA": SOURCES[0][3],
                        "SK hynix Future Forum / OCP 2026": SOURCES[2][3],
                        "HOBBIT / TailorKV": SOURCES[3][3],
                        "CXL GPU–NDP MoE / SliceMoE": SOURCES[5][3],
                        "ITME / HyMCache": SOURCES[7][3],
                        "ReplaySSM / StrataFlux": SOURCES[9][3],
                        "Nemotron 3 Ultra / PACE": SOURCES[11][3],
                    }
                    for run in p.runs:
                        rpr=run._r.get_or_add_rPr()
                        # pptx uses lxml, not ElementTree for mutation.
                        from lxml import etree
                        e=etree.SubElement(rpr,"{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
                        e.set("typeface","Microsoft YaHei")
                        if s.number == 22 and it["text"] in link_map:
                            run.hyperlink.address = link_map[it["text"]]
        refids=set(re.findall(r"S\d{2}", s.refs))
        for lo, hi in re.findall(r"S(\d{2})[–-]S(\d{2})", s.refs):
            refids.update(f"S{n:02}" for n in range(int(lo),int(hi)+1))
        urls="\n".join(f"[{id_}] {title}\n{url}" for id_,title,date,url,html,scope in SOURCES if id_ in refids)
        # All sources on the reference slide, including compressed ranges.
        if s.number==22:
            urls="\n\n".join(f"[{a}] {b}\n{d}" for a,b,c,d,e,f in SOURCES)
        ps.notes_slide.notes_text_frame.text=f"第 {s.number} 页｜{s.title or '精度 × 驻留：混合大模型的联合优化'}\n建议时长：{s.seconds} 秒\n\n{s.notes}\n\n{urls}\n\n证据标记：15% 为拟议验收目标；说明性算例不代表实测结果。"
    out=ROOT/"SK_hynix_Incubation_Precision_Placement_CN.pptx"
    prs.save(out)
    return out


def build_pdf():
    out=ROOT/"SK_hynix_Incubation_Precision_Placement_CN.pdf"
    # Figure point scale: 1 pixel coordinate = 0.6 point in a 13⅓-inch page.
    with PdfPages(out) as pdf:
        for s in SLIDES:
            fig=plt.figure(figsize=(W/PPI,H/PPI),dpi=PPI)
            fig.patch.set_facecolor(mplcolor(s.bg))
            ax=fig.add_axes([0,0,1,1])
            ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
            for it in s.items:
                if it["kind"]=="rect":
                    edge=mplcolor(it["stroke"]) if it["stroke"] else "none"
                    if it["radius"]:
                        p=FancyBboxPatch((it["x"],it["y"]),it["w"],it["h"],boxstyle=f"round,pad=0,rounding_size={it['radius']}",facecolor=mplcolor(it["fill"]),edgecolor=edge,linewidth=.6)
                    else:
                        p=Rectangle((it["x"],it["y"]),it["w"],it["h"],facecolor=mplcolor(it["fill"]),edgecolor=edge,linewidth=.6)
                    ax.add_patch(p)
                elif it["kind"]=="line":
                    ax.plot([it["x"],it["x2"]],[it["y"],it["y2"]],color=mplcolor(it["color"]),linewidth=it["width"]*72/PPI)
                    if it["arrow"]:
                        x,y=it["x2"],it["y2"]
                        ax.add_patch(Polygon([(x,y),(x-9,y-6),(x-9,y+6)],facecolor=mplcolor(it["color"]),edgecolor="none"))
                elif it["kind"]=="text":
                    fp=FontProperties(fname=FONT_BOLD if it["bold"] else FONT, size=it["size"])
                    lh=it["size"]*PPI/72*it["leading"]
                    xx=it["x"]+({"left":0,"center":it["w"]/2,"right":it["w"]}[it["align"]])
                    for n,line in enumerate(it["lines"]):
                        ax.text(xx,it["y"]+n*lh,line,fontproperties=fp,color=mplcolor(it["color"]),ha=it["align"],va="top",parse_math=False)
            pdf.savefig(fig,facecolor=fig.get_facecolor())
            plt.close(fig)
    return out


def build_notes_and_sources():
    intro="# 逐页讲稿｜精度 × 驻留：混合大模型联合优化\n\n主讲页 1–18，附录页 19–23。建议主讲约 15–18 分钟，附录按问答使用。\n\n15% 为项目目标；说明性数字、人员和路线均为提案假设。\n"
    for s in SLIDES:
        intro+=f"\n## {s.number:02}｜{s.title or '精度 × 驻留：混合大模型的联合优化'}\n\n建议时长：{s.seconds} 秒。\n\n{s.notes}\n\n页面来源：{s.refs or '方案设计与验证目标。'}\n"
    (ROOT/"SPEAKER_NOTES.md").write_text(intro,encoding="utf-8")
    refs="# 参考来源与证据边界\n\n公开来源核查日期：2026-11-08。来源基于本次提案前的网络检索与正文核查。链接保留原始地址，便于评审追溯。\n\n性能结果均属于对应论文的实验设置，未将其相加或当作本提案结果。本文献列表为定向查新，不能证明不存在其他相近工作，不能代替专利查新。\n"
    for id_,title,date,url,html,scope in SOURCES:
        refs+=f"\n## {id_}｜{title}\n\n- 日期：{date}\n- 原始来源：{url}\n"
        if html: refs+=f"- 正文：{html}\n"
        refs+=f"- 支持内容与边界：{scope}\n"
    refs+="""
## 本提案的独立假设与验收边界

- 至少 15% 指相对同等资源、质量和服务要求下强基线的有效吞吐增益，尚未验证。
- 80 GB 容量条形图为算例：20+10+20+30=80；20+5+15+40=80。未指向具体模型或已测收益。
- 附录的 17% 延迟下降由假设 0.4×0.5−0.03 得到；约 20.5% 速率提升只适用于固定串行服务假设。
- 六个月、4–5 人和 2027–2029 年交付物是拟议研发计划，未引用海力士内部预算或量产承诺。
- CXL 是互连协议；CXL 内存层在本提案中指附加 DRAM 或内存池。普通 CXL 扩展、SSD-backed CXL Hybrid Memory 和 CXL-NDP 应分别建模。
- 活跃 SSM 状态或稠密注意力 KV 的 SSD 放置并非预设可行；须满足访问时限、质量和带宽要求。
- 降精度造成的信息损失不能通过简单升精度逆转；权重主副本、残差、状态检查点均计入容量与执行成本。
- 新颖性主张是待验证的跨对象边际收益模型、联合分配和低开销实现。量化+分层、精度切片、专家冷热划分均有先例。
- 不承诺 15% 性能、15% 能效和 15% TCO 同时成立；主 KPI 与副指标分开。
"""
    (ROOT/"SOURCES.md").write_text(refs,encoding="utf-8")


def validate(pptx_path, pdf_path):
    errors=[]
    from fontTools.ttLib import TTFont
    cmap = TTFont(FONT, fontNumber=0).getBestCmap()
    missing = sorted({ch for s in SLIDES for it in s.items if it["kind"]=="text" for ch in it["text"] if not ch.isspace() and ord(ch) not in cmap})
    if missing:
        errors.append("Missing font glyphs: " + " ".join(missing))
    count=0
    for s in SLIDES:
        for it in s.items:
            count+=1
            if it["kind"] in ("rect","text"):
                if it["x"]<0 or it["y"]<0 or it["x"]+it["w"]>W+1 or it["y"]+it["h"]>H+1:
                    errors.append(f"slide {s.number}: bounds {it}")
            if it["kind"]=="text":
                f=font(it["size"],it["bold"])
                for line in it["lines"]:
                    if f.getlength(line)>it["w"]+1:
                        errors.append(f"slide {s.number}: line width {line}")
    with zipfile.ZipFile(pptx_path) as z:
        names=z.namelist()
        actual=[n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml",n)]
        notes=[n for n in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml",n)]
        xml_count=0
        for name in names:
            if name.endswith(".xml"):
                ET.fromstring(z.read(name)); xml_count+=1
        media=[n for n in names if n.startswith("ppt/media/")]
    check=Presentation(pptx_path)
    assert len(check.slides)==len(SLIDES)
    assert len(notes)==len(SLIDES)
    assert len(actual)==len(SLIDES)
    pdf_bytes=pdf_path.read_bytes()
    pdf_pages=len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    assert pdf_pages==len(SLIDES), (pdf_pages,len(SLIDES))
    assert not errors, errors
    report={
        "status":"PASS",
        "slides":len(SLIDES),"main_slides":18,"appendix_slides":len(SLIDES)-18,
        "main_speaking_minutes":round(sum(s.seconds for s in SLIDES[:18])/60,1),
        "scene_objects":count,"xml_parts_validated":xml_count,
        "notes_pages":len(notes),"embedded_raster_images_in_pptx":len(media),
        "pdf_pages":pdf_pages,
        "editable_native_shapes":True,"font":"Microsoft YaHei",
        "missing_font_glyphs":missing,
        "layout_checks":["all shapes within slide bounds","all text measured and wrapped explicitly","no text box height overflow","font glyph coverage","PPTX reload and XML parse","one notes page per slide","PDF page count matches PPTX"],
        "errors":errors,"warnings":WARNINGS,
        "rendering_limitation":"PNG and vector PDF rendered from shared scene, not through Microsoft PowerPoint/LibreOffice. Final Office-native rendering has not been tested.",
        "outputs":{"pptx_bytes":pptx_path.stat().st_size,"pdf_bytes":pdf_path.stat().st_size},
        "claims":{"15_percent":"proposed target, not measured","capacity_80GB":"illustrative only","three_year_plan":"proposed, not corporate commitment"}
    }
    (ROOT/"QA_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def main():
    if not Path(FONT).exists(): raise RuntimeError("Microsoft YaHei font required")
    make_slides()
    PREVIEW.mkdir(exist_ok=True)
    thumbs=[]
    for s in SLIDES:
        image=draw_scene(s)
        image.save(PREVIEW/f"slide_{s.number:02}.png")
        thumb=image.resize((480,270),Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    sheet=Image.new("RGB",(4*500+20,math.ceil(len(thumbs)/4)*310+20),rgb("E4E8EF"))
    sd=ImageDraw.Draw(sheet)
    for i,thumb in enumerate(thumbs):
        x=20+(i%4)*500; y=20+(i//4)*310
        sheet.paste(thumb,(x,y))
        sd.text((x+2,y+277),f"{i+1:02}  "+(SLIDES[i].title or "精度 × 驻留"),font=ImageFont.truetype(FONT,16),fill=rgb(INK))
    sheet.save(ROOT/"CONTACT_SHEET.png")
    pptx=build_pptx()
    pdf=build_pdf()
    build_notes_and_sources()
    report=validate(pptx,pdf)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
