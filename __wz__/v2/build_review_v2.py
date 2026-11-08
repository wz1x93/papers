"""Editorial redesign, with fully editable vector diagrams and presenter notes."""
from __future__ import annotations
import math
import json
import re
import zipfile
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1"))
import build_deck as shared
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from lxml import etree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon, Rectangle, Ellipse

ROOT = Path(__file__).resolve().parent
W,H,DPI = 1920,1080,144
shared.PPI = DPI
FONT = shared.FONT
BOLD = shared.FONT_BOLD
LATIN = "C:/Windows/Fonts/arial.ttf"
LATIN_BOLD = "C:/Windows/Fonts/arialbd.ttf"
MONO = "C:/Windows/Fonts/consola.ttf"
INK="172021"; BLACK="101718"; CREAM="F5F3ED"; WHITE="FFFFFF"
RED="E5482E"; REDPALE="FCE8DF"; GREY="6B7470"; LINE="D8DAD3"
TEAL="177A73"; TEALPALE="E0EFEB"; BLUE="405ADD"; BLUEPALE="E8ECFC"
DIM="95A39E"; DARKLINE="33403D"
SLIDES=[]
VERSION="V2"
MAIN_SLIDES=15
FILESTEM="SK_hynix_Incubation_Review_V2"

def rgb(c): return tuple(int(c[i:i+2],16) for i in (0,2,4))
def ff(size,bold=False,face=None):
    return ImageFont.truetype(face or (BOLD if bold else FONT),round(size*2))

def wrap(t,size,width,bold=False,face=None):
    # Shared CJK punctuation handling; same YaHei metrics used for all outputs.
    if not face: return shared.wrap(t,size,width,bold)
    result=[]
    f=ff(size,bold,face)
    for line in t.split("\n"):
        cur=""
        for word in line.split(" "):
            c=word if not cur else cur+" "+word
            if f.getlength(c)<=width: cur=c
            else:
                if cur: result.append(cur)
                cur=word
        result.append(cur)
    return result

class Page:
    def __init__(self,title,kicker="",sub="",dark=False,refs="",appendix=False):
        self.title=title; self.kicker=kicker; self.sub=sub; self.dark=dark
        self.refs=refs; self.appendix=appendix; self.bg=BLACK if dark else CREAM
        self.items=[]; self.notes=""; self.seconds=55
        SLIDES.append(self); self.n=len(SLIDES)
    @property
    def fg(self): return CREAM if self.dark else INK
    @property
    def muted(self): return DIM if self.dark else GREY
    def rect(self,x,y,w,h,c,stroke=None):
        self.items.append(dict(k="rect",x=x,y=y,w=w,h=h,c=c,stroke=stroke))
    def poly(self,points,c,stroke=None):
        self.items.append(dict(k="poly",points=points,c=c,stroke=stroke))
    def ellipse(self,x,y,w,h,c,stroke=None):
        self.items.append(dict(k="ellipse",x=x,y=y,w=w,h=h,c=c,stroke=stroke))
    def line(self,x,y,x2,y2,c=LINE,width=2,dash=False,arrow=False):
        self.items.append(dict(k="line",x=x,y=y,x2=x2,y2=y2,c=c,width=width,dash=dash))
        if arrow:
            dx=x2-x;dy=y2-y; ln=math.hypot(dx,dy)
            if ln:
                ux,uy=dx/ln,dy/ln;px,py=-uy,ux
                self.poly([(x2,y2),(x2-14*ux+7*px,y2-14*uy+7*py),(x2-14*ux-7*px,y2-14*uy-7*py)],c)
    def text(self,x,y,w,h,t,size=20,c=None,bold=False,align="left",face=None,leading=1.18):
        t=t.translate(str.maketrans({"₁":"1","₂":"2","₃":"3","ᵢ":"_i","ₖ":"_k","ₜ":"_t"}))
        if face and re.search(r"[\u3400-\u9fff]",t): face=None
        lines=wrap(t,size,w-3,bold,face)
        original=size
        def height():
            f=ff(size,bold,face)
            ink=max((f.getbbox(s)[3]-f.getbbox(s)[1] for s in lines if s),default=0)
            return max(0,len(lines)-1)*size*2*leading+ink
        while height()>h+1 and size>original*.91:
            size-=.25; lines=wrap(t,size,w-3,bold,face)
        if height()>h+1:
            raise ValueError(f"Page {self.n}: text height {height():.0f}>{h}: {t}")
        self.items.append(dict(k="text",x=x,y=y,w=w,h=h,t=t,lines=lines,size=size,c=c or self.fg,bold=bold,align=align,face=face,leading=leading))
    def label(self,x,y,w,t,c=None): self.text(x,y,w,34,t,12,c or self.muted,True,face=MONO)
    def head(self):
        self.label(96,56,1600,self.kicker)
        self.rect(96,107,51,6,RED)
        self.text(96,141,1740,106,self.title,35,self.fg,True)
        if self.sub:self.text(99,263,1710,60,self.sub,17,self.muted)
    def foot(self):
        self.line(96,1012,1824,1012,DARKLINE if self.dark else LINE,1)
        self.text(96,1032,1470,27,self.refs or "方案设计 / 待验证假设 · 公开资料截至 2026.11.08",9,self.muted)
        self.label(1595,1027,150,"APPENDIX" if self.appendix else "INCUBATION")
        self.text(1765,1023,60,42,f"{self.n:02}",17,RED,True,align="right",face=LATIN_BOLD)
    def note(self,t,sec=55): self.notes=t;self.seconds=sec

def iso(p,x,y,w,h,depth,c,top=None):
    """Isometric memory slab, all faces remain editable."""
    dx,dy=depth,-depth*.46
    p.rect(x,y,w,h,c)
    p.poly([(x,y),(x+dx,y+dy),(x+w+dx,y+dy),(x+w,y)],top or c)
    p.poly([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)],"263430" if p.dark else "BFC8C0")

def rule(p,x,y,w,title,body,color=RED):
    p.rect(x,y,5,123,color)
    p.text(x+26,y-2,w-26,47,title,20,p.fg,True)
    p.text(x+26,y+62,w-26,80,body,16,p.muted)

def small_number(p,x,y,num,title,body,w=510):
    p.text(x,y,150,85,num,44,RED,True,face=LATIN_BOLD)
    p.text(x+158,y+6,w-158,60,title,22,p.fg,True)
    p.text(x,y+110,w,118,body,18,p.muted)

def diagram_title(p,x,y,w,label,title,col=RED):
    p.label(x,y,w,label,col)
    p.text(x,y+48,w,64,title,24,p.fg,True)

def compose():
    # 01: editorial cover and custom isometric memory illustration.
    p=Page("精度与位置，\n一起决定推理效率",dark=True)
    p.label(96,67,1100,"SK HYNIX / INCUBATION PROPOSAL",DIM)
    p.rect(96,135,83,8,RED)
    p.text(92,197,1100,235,"精度与位置，\n一起决定推理效率",47,CREAM,True,leading=1.13)
    p.text(98,485,980,108,"面向 KV Cache、SSM 状态与 MoE 权重的\n混合精度 × 分层放置联合优化",24,"B6C1BB")
    p.label(98,677,1000,"PRECISION × PLACEMENT",RED)
    p.text(95,735,890,110,"在既定资源与质量约束下，\n寻找超过 15% 的净有效吞吐增益。",24,CREAM,True)
    # Slabs have irregular active tiles and data objects to communicate residency.
    for row,(name,col,y,width) in enumerate([("SSD",BLUE,701,536),("CXL MEMORY",TEAL,512,468),("HBM",RED,323,400)]):
        x=1172+(536-width)/2
        iso(p,x,y,width,89,90,"27342F",col)
        p.label(x+22,y+27,width-32,name,CREAM)
        for j in range(5):
            xx=x+33+j*(width-48)/5
            p.poly([(xx,y-14),(xx+25,y-26),(xx+66,y-26),(xx+41,y-14)],"C2CEC4" if j%2 else col)
    for x in [1275,1465,1645]:p.line(x,424,x,672,"51605A",2,True)
    p.label(1140,898,660,"HBM / CXL-ATTACHED DRAM / SSD",DIM)
    p.text(98,943,1600,32,"评审讨论稿  /  2026.11  /  ≥15% 为拟议验证目标，尚无本项目实测结果",11,DIM)
    p.foot();p.note("这不是三个模块分别压缩后再做分层。本项目把精度与驻留作为耦合决策，在相同资源、模型质量与服务时限下，最大化有效吞吐。今天申请一个六个月的孵化验证窗口，检验相对强基线能否获得超过 15% 的净增益。",40)

    # 02: investment thesis, large typographic KPI.
    p=Page("把内存组合，变成可验证的系统优势", "01 / THE INVESTMENT THESIS",refs="战略背景：S02–S03；15% 为拟议验收目标。")
    p.head()
    p.line(1160,343,1160,938,LINE,2)
    small_number(p,98,377,"01","资源已经齐备","HBM、CXL 内存与 SSD 构成快层到容量层的产品基础。",970)
    small_number(p,98,652,"02","系统决策仍有空间","何种数据值得占用 HBM，取决于精度、访问代价与质量敏感性。",970)
    p.label(1230,365,570,"PROPOSED VALIDATION TARGET",GREY)
    p.text(1215,429,610,193,"15%+",96,RED,True,face=LATIN_BOLD)
    p.text(1230,669,565,116,"固定资源与质量，\n相对强基线的净增益",25,INK,True)
    p.text(1230,836,565,91,"六个月验证机制，\n阶段门决定后续投入。",19,GREY)
    p.foot();p.note("评审首先要看到投资命题。海力士已经有跨层产品能力，我们要验证的是连接这些产品的工作负载感知决策。15% 不是市场宣传数字，也不是把不同论文的加速比相加，而是同约束强基线上的项目目标。",50)

    # 03: evidence-led timeline, three distinct signals.
    p=Page("模型与内存都在演进，联合优化进入验证窗口", "02 / WHY NOW",sub="已有混合模型与跨层设备可作为验证基础；未来架构占比不作确定性预测。",refs="来源：S01–S03、S12；企业展示不等同于所有功能已量产。")
    p.head()
    p.line(198,491,1718,491,INK,2)
    data=[(198,"MODEL","Attention + Mamba + MoE","Nemotron 3 Ultra 提供\n混合架构的真实模型实例。",RED,"S12"),(846,"ARCHITECTURE","三种访问行为共存","NOVA 刻画不同算术强度，\n状态、KV 与专家需求各异。",TEAL,"S01"),(1494,"ECOSYSTEM","全栈内存系统已展示","海力士公开展示 CXL 池化、\nDRAM+SSD 与 KV 复用。",BLUE,"S02–S03")]
    for x,lab,tt,body,col,src in data:
        p.ellipse(x-12,479,24,24,col)
        p.label(x-100,378,500,lab,col)
        p.text(x-100,548,505,100,tt,23,INK,True)
        p.text(x-100,699,505,134,body,19,GREY)
        p.label(x-100,882,500,src,GREY)
    p.foot();p.note("三个同时出现的条件支持现在开展项目：混合模型已经存在，内存访问行为有明确差异，真实的 CXL 和 SSD 方案已经可用于原型。仍要兼容 GQA、MLA 和其他递归状态结构，避免把战略依赖于单一模型家族。",50)

    # 04: custom access-pattern diagrams.
    p=Page("三类数据，三种访问规律，一个 HBM 预算", "03 / THE WORKLOAD",refs="特征依据：S01、S10、S13；图形为访问模式示意。")
    p.head()
    xs=[96,714,1332]
    for x in [676,1294]:p.line(x,341,x,921,LINE,1)
    diagram_title(p,96,348,540,"01 / KV CACHE","上下文越长，缓存越大",BLUE)
    for i in range(7):
        p.rect(110+i*68,530,49,87,BLUE if i<5 else "BAC6F0")
    p.line(112,657,570,657,BLUE,4,arrow=True)
    p.text(110,697,500,60,"历史序列持续扩展",18,BLUE,True)
    p.text(98,816,530,118,"活跃 KV 可能每步读取；\n可复用前缀具有不同生命周期。",19,GREY)
    diagram_title(p,714,348,540,"02 / SSM STATE","每步更新，误差可递归",TEAL)
    p.rect(880,513,184,118,TEAL)
    p.text(897,542,150,68,"STATE",21,WHITE,True,align="center",face=LATIN_BOLD)
    p.line(778,571,865,571,TEAL,4,arrow=True)
    p.line(1070,571,1169,571,TEAL,4,arrow=True)
    for seg in [(1152,645,1152,706),(1152,706,806,706),(806,706,806,645)]:p.line(*seg,TEAL,3,arrow=seg[-1]==645)
    p.text(729,816,520,118,"状态随并发增长；\n活跃状态优先靠近执行单元。",19,GREY)
    diagram_title(p,1332,348,500,"03 / MoE WEIGHTS","稀疏激活，驻留价值不均",RED)
    for r in range(3):
        for c in range(5):
            active=(r,c) in [(0,1),(1,3),(2,0)]
            p.rect(1350+c*88,523+r*64,69,45,RED if active else "E3DFD7")
    p.text(1344,735,480,44,"仅少数专家被路由激活",17,RED,True)
    p.text(1335,816,490,118,"基本只读，工作集动态变化；\n热度与量化敏感性分别测量。",19,GREY)
    p.foot();p.note("三类对象对应模型不同模块，但共享同一硬件预算。我们保留它们的语义差异：旧 KV 不等于冷数据，状态量化误差不能简单恢复，专家热度也不等于质量敏感性。统一优化的是稀缺资源如何分配，而不是用同一规则处理所有对象。",55)

    # 05: what is jointly optimized, visual comparison of paths.
    p=Page("两次局部最优，不保证组合后的系统最优", "04 / THE COUPLING",sub="精度改变大小、流量与内核代价；位置改变访问时限与驻留收益。")
    p.head()
    p.label(100,364,500,"SEQUENTIAL DECISIONS",GREY)
    for i,(lab,tt) in enumerate([("q","先确定量化"),("p","再安排放置"),("?","系统效果")]):
        xx=100+i*570
        p.rect(xx,431,450,116,WHITE)
        p.text(xx+23,449,86,69,lab,29,GREY,True,face=LATIN_BOLD)
        p.text(xx+123,459,310,61,tt,23,INK,True)
        if i<2:p.line(xx+462,489,xx+551,489,GREY,3,arrow=True)
    p.line(99,604,1819,604,LINE,1)
    p.label(100,659,620,"JOINT PRECISION–PLACEMENT",RED)
    for c,lab in enumerate(["HBM","CXL","SSD"]):
        p.label(288+c*185,715,165,lab,GREY)
    for r,lab in enumerate(["q-A","q-B","q-C"]):
        yy=763+r*65
        p.label(103,yy+12,145,lab,GREY)
        for c in range(3):
            xx=270+c*185
            chosen=(r,c) in [(0,0),(1,2),(2,1)]
            p.rect(xx,yy,165,49,RED if chosen else WHITE)
            p.text(xx+9,yy+6,147,40,"候选",14,WHITE if chosen else GREY,chosen,align="center")
    p.text(104,964,745,31,"二维候选示意：位宽与位置共同搜索，实际可行域须筛选。",12,GREY)
    p.text(970,702,820,165,"统一比较：\n一单位快层容量，\n换来多少净收益？",26,INK,True)
    p.text(974,895,817,70,"把质量、容量、传输与执行开销放进同一个决策。",17,GREY)
    p.foot();p.note("本提案的关键不是执行顺序换一下，而是改变优化目标。先确定精度再做分层，会把精度释放的容量和其他对象的驻留收益割裂。强交替优化可能弥补一部分，因此它必须是主要基线。最终差异能否保留是本项目要证伪或支持的假设。",55)

    # 06: analytical, not empirical, explanation of nonlinearity.
    p=Page("两条非线性，共同决定可行与最优的边界", "05 / WHY JOINT OPTIMIZATION",refs="概念图：曲线形状仅用于解释机制，不对应测量数据或拟合结果。")
    p.head()
    for xx in [96,1030]:
        p.line(xx+83,794,xx+728,794,INK,2,arrow=True)
        p.line(xx+83,794,xx+83,413,INK,2,arrow=True)
    p.label(98,338,780,"QUALITY RESPONSE",RED)
    p.text(100,876,790,78,"压缩更激进，质量损失可能越过容忍边界",20,INK,True)
    p.text(178,819,600,43,"压缩强度 →",15,GREY)
    p.text(122,410,500,45,"质量损失",15,GREY)
    pts=[(190+6*i,756-290*(i/90)**3) for i in range(91)]
    for a,b in zip(pts,pts[1:]):p.line(*a,*b,RED,4)
    p.line(181,615,817,615,GREY,2,True)
    p.label(550,565,300,"QUALITY LIMIT",GREY)
    p.label(1032,338,780,"CAPACITY THRESHOLD",TEAL)
    p.text(1030,876,790,78,"多释放一点容量，可能消除一整组缺失",20,INK,True)
    p.text(1111,819,600,43,"可用 HBM 容量 →",15,GREY)
    p.text(1066,410,500,45,"关键路径等待",15,GREY)
    path=[(1122,485),(1330,485),(1330,590),(1495,590),(1495,715),(1750,715)]
    for a,b in zip(path,path[1:]):p.line(*a,*b,TEAL,5)
    p.line(1495,436,1495,786,GREY,2,True)
    p.label(1510,441,300,"WORKING SET FITS",GREY)
    p.foot();p.note("双重非线性具体指质量响应和容量性能阈值。左图说明位宽下降可能先影响很小，再越过质量边界。右图说明装下一个工作集后可能减少整组缺失。两个概念图不携带数据，也没有优化器跑出来的曲面。正式模型可以用离散查表和分段近似，不必预先承诺非线性求解器。",55)

    # 07: capacity example, strong visual data blocks.
    p=Page("压缩 KV 或状态，收益可能出现在专家执行", "06 / THE MECHANISM TO TEST",sub="说明性容量算例：固定 80 GB HBM。数字用于解释跨对象取舍，不代表具体模型实测。")
    p.head()
    colors=["929C94",TEAL,BLUE,RED]; labels=["其他 / 预留","SSM 状态","KV Cache","专家驻留"]
    for i,(col,lab) in enumerate(zip(colors,labels)):
        p.rect(100+i*440,365,18,18,col);p.text(131+i*440,356,390,44,lab,16,GREY)
    for row,(lab,vals) in enumerate([("独立配置",[20,10,20,30]),("联合候选",[20,5,15,40])]):
        yy=465+row*197;p.text(98,yy+30,260,70,lab,24,INK,True)
        xx=390
        for v,col in zip(vals,colors):
            ww=v/80*1428;p.rect(xx,yy,ww-3,111,col)
            p.text(xx+5,yy+30,ww-13,61,f"{v} GB",17 if v==5 else 21,WHITE,True,align="center",face=LATIN_BOLD);xx+=ww
    p.line(390+1428*.5,615,390+1428*.5,646,BLUE,3,arrow=True)
    p.text(101,878,1710,74,"释放 10 GB → 增加专家驻留 → 降低缺失流量 → 测量最终有效吞吐",23,TEAL,True)
    p.foot();p.note("这个算例用同一 80GB 预算展示资源替代。上下两行总量相同，只是状态与 KV 假设各释放 5GB，使专家驻留增加 10GB。这里不能推断吞吐或无损质量。要测的正是驻留改变带来的收益，能否覆盖量化与执行的全部成本。",55)

    # 08: proper topology, no assumed direct GPU-CXL load path.
    p=Page("用一个资源管理器，连接模型需求与真实数据路径", "07 / PROPOSED SYSTEM",refs="提案架构：普通 CXL 扩展与 CXL-NDP 分开评估；GPU 直达能力按平台核验。")
    p.head()
    p.rect(96,350,1728,125,INK)
    p.text(126,375,890,71,"精度—驻留联合资源管理器",27,WHITE,True)
    p.text(1060,389,727,51,"离线画像   /   在线预算   /   预取与版本管理",16,"CAD4CE")
    # Host and accelerator boxes.
    p.rect(96,574,575,229,WHITE)
    p.label(122,597,500,"ACCELERATOR",RED)
    p.text(124,660,220,79,"GPU",33,INK,True,face=LATIN_BOLD)
    p.rect(385,654,250,108,RED);p.text(403,680,214,65,"HBM",27,WHITE,True,align="center",face=LATIN_BOLD)
    p.rect(771,574,514,229,WHITE)
    p.label(797,597,450,"HOST / STAGING",GREY)
    p.text(800,660,452,83,"CPU + 主机 DDR",27,INK,True)
    p.rect(1405,560,419,112,TEAL)
    p.text(1430,588,369,65,"CXL 附加内存",24,WHITE,True)
    p.rect(1405,722,419,112,BLUE)
    p.text(1430,750,369,65,"NVMe SSD",25,WHITE,True,face=LATIN_BOLD)
    for xx in [387,1027,1610]:p.line(xx,476,xx,536,GREY,2,True)
    p.line(672,700,764,700,INK,4,arrow=True)
    p.line(1290,616,1396,616,TEAL,4,arrow=True)
    p.line(1290,778,1396,778,BLUE,4,arrow=True)
    p.text(685,826,625,50,"PCIe / DMA / CXL：测量实际中转与同步",14,GREY,align="center")
    p.text(100,919,1715,58,"第一阶段复用现有硬件；近存计算可作为后续独立配置，不作为基础可行性的前提。",19,INK,True)
    p.foot();p.note("架构图明确了普通路径。GPU 与 HBM在一侧，主机与 DDR 中转、CXL 和 SSD 在另一侧。具体平台支持 GPU DMA 或直接访问时可以替换路径，但不能从 CXL 标准推断 GPU 一定可直接 load。离线做质量和平台成本画像，在线管理器执行轻量的预算、驻留、迁移与预取。",60)

    # 09: optimization objective editorial.
    p=Page("目标只有一个：约束内的有效吞吐最大化", "08 / OBJECTIVE",dark=True)
    p.head()
    p.label(100,350,1700,"QUALITY-CONSTRAINED RESOURCE ALLOCATION",DIM)
    p.text(94,419,1750,127,"max  Goodput(q, p, π)",53,CREAM,True,face=LATIN_BOLD)
    p.text(99,604,1700,76,"q 精度配置      p 驻留与副本      π 迁移、预取及执行策略",23,"C1CBC3")
    for i,(a,b) in enumerate([("固定资源","容量、带宽、计算与临时缓冲"),("质量约束","关键任务、检索与长生成非劣性"),("服务约束","TTFT / TPOT 尾延迟与成功率")]):
        x=99+i*590;p.rect(x,762,55,5,RED)
        p.text(x,794,548,58,a,24,CREAM,True)
        p.text(x,878,548,69,b,18,DIM)
    p.foot();p.note("Goodput 定义为满足事先确定的质量和时限要求的输出 token 每秒。q 是精度，p 包括放置和副本，π 处理运行时实现。副本、元数据与临时缓冲都计入预算。质量误差不预设可按对象相加，最终必须端到端验证。能耗和成本是副指标，不能混用 15%。",55)

    # 10: algorithm round-trip with no card grid.
    p=Page("离线找边界，在线做有界开销的联合决策", "09 / CONTROL STRATEGY",sub="拟议方法：先建立小规模优化上界，再决定在线近似策略。")
    p.head()
    for i,(n,a,b) in enumerate([("01","画像","质量敏感性\n设备与路径代价"),("02","分配","比较跨对象净收益\n分配精度与快层容量"),("03","执行","预取、驻留与迁移\n版本一致性与回退"),("04","反馈","热度与队列变化\n达到阈值再重规划")]):
        x=98+i*448
        p.text(x,387,394,130,n,66,RED,True,face=LATIN_BOLD)
        p.text(x,566,392,81,a,30,INK,True)
        p.text(x,694,394,155,b,21,GREY)
        if i<3:p.line(x+328,594,x+425,594,INK,3,arrow=True)
    p.line(1782,870,1782,930,GREY,2)
    p.line(1782,930,105,930,GREY,2,True,arrow=True)
    p.label(792,894,550,"FEEDBACK / HYSTERESIS",TEAL)
    p.foot();p.note("这里是拟议算法路线，不是已经完成的算法。先用小规模离线优化得到可解释的空间，再选查表、贪心或有界搜索等在线近似。在线不能每个 token 重做大规模优化。重规划要有最小驻留期和收益门槛，防止负载漂移导致反复搬迁。控制和数据转换成本完整计入。",55)

    # 11: literature scopes with visual range bars.
    p=Page("已有局部组合；待证明的是跨对象的额外价值", "10 / DIFFERENTIATION",sub="选取直接相关工作定位研究边界；不作穷尽查新或专利性结论。",refs="来源：S04–S11；具体论文的原型 / 仿真边界见来源文件。")
    p.head()
    p.label(98,357,660,"EXISTING RESEARCH",GREY)
    p.label(1006,357,805,"PROPOSED CONTRIBUTION",RED)
    prior=[("专家精度 + 驻留","HOBBIT / SliceMoE / CXL GPU–NDP MoE"),("KV 精度 + 卸载","TailorKV / StrataFlux"),("CXL + DRAM + SSD","ITME / HyMCache"),("递归状态管理","ReplaySSM / PACE")]
    for i,(a,b) in enumerate(prior):
        yy=422+i*120;p.line(98,yy+102,879,yy+102,LINE,1)
        p.text(100,yy,780,49,a,22,INK,True)
        p.text(100,yy+61,780,42,b,14,GREY)
    p.line(945,359,945,936,LINE,2)
    for i,(a,b) in enumerate([("统一净边际收益","同一容量与带宽预算下比较三类对象。"),("跨对象联合分配","量化一个对象，让另一个对象获益。"),("可执行的在线策略","强基线、完整开销与端到端质量验证。")]):
        rule(p,1006,425+i*174,800,a,b,[RED,TEAL,BLUE][i])
    p.foot();p.note("新颖性需要建立在可验证机制上。专家、KV和跨层精度已有直接先例。我们要证明统一模型与跨对象分配确实比强交替优化有额外收益，并能实现。若额外收益不存在，就不能坚持强创新主张。这里的选取不是对全部文献的穷尽调查。",60)

    # 12: hero target, baseline hierarchy, essential experiment contract.
    p=Page("15%+，相对最强可复现基线，而不是静态分层", "11 / THE EXPERIMENT CONTRACT",sub="相同资源、质量门槛、服务时限与预热条件；在实验前登记测试集合。")
    p.head()
    p.text(89,391,760,204,"15%+",110,RED,True,face=LATIN_BOLD)
    p.text(101,650,720,116,"拟议主指标：\n满足质量与时限的有效吞吐增益",23,INK,True)
    p.text(101,844,720,90,"报告重复测量与置信区间，\n完整保留无收益及退化场景。",18,GREY)
    p.line(867,362,867,945,LINE,2)
    for i,(a,b) in enumerate([("单维度最优","只优化精度 / 只优化放置"),("强交替与顺序优化","双顺序充分调参，排除执行顺序优势"),("相关先进组合基线","在适用范围内实现，明确兼容差异")]):
        yy=392+i*168
        p.text(949,yy,820,65,a,25,INK,True)
        p.text(949,yy+83,820,67,b,18,GREY)
    p.rect(947,914,855,54,INK)
    p.text(965,923,819,42,"B* = 满足同等约束的上述最强方案",17,WHITE,True)
    p.foot();p.note("15% 相对最强可复现的匹配基线。至少包含只优化精度、只优化放置、双顺序、充分调优的交替优化以及相关方法的适用组合。模型与内核不兼容的差异需要公开。建议主测试集合覆盖预登记的容量受限场景，逐场景配对比较，报告几何均值及 95% CI，完整报告其他场景。",60)

    # 13: incubation timeline.
    p=Page("六个月，用三道证据门决定是否继续", "12 / INCUBATION EXECUTION",sub="申请验证窗口；产品化投入以真实原型、净增益与客户价值证据为前提。")
    p.head()
    p.line(100,446,1815,446,INK,2)
    stages=[(100,"G0","1–6 周","瓶颈与上界","锁定模型、质量与服务门槛\n校准真实硬件与访问路径\n估计联合优化的可行上界",RED),(708,"G1","7–12 周","机制与原型","实现联合策略与强基线\n消融跨对象预算和精度选择\n计入控制、转换与迁移成本",TEAL),(1316,"G2","13–24 周","净收益与部署","至少两个混合模型\n真实 CXL / SSD 与动态负载\n验证有效吞吐、质量和尾延迟",BLUE)]
    for x,g,t,a,b,col in stages:
        p.ellipse(x-11,435,22,22,col)
        p.label(x,363,510,t,col)
        p.text(x,493,500,94,g,46,col,True,face=LATIN_BOLD)
        p.text(x,622,500,77,a,28,INK,True)
        p.text(x,743,500,170,b,18,GREY)
    p.foot();p.note("G0判断问题与空间是否成立；上界不足时应及时调整。G1验证机制与原型，并与强基线比较。G2上真实设备和动态负载。主目标不成立时可以收缩为局部专项，不以其他产出来替代主 KPI。通过后再申请更大规模产品协同投入。",55)

    # 14: portfolio strategy, architecture roadmapping not fabricated revenue.
    p=Page("三年沉淀：从优化策略，走向产品组合的系统能力", "13 / STRATEGIC OPTION",refs="战略背景：S02–S03；2027–2029 为建议路线，不代表公司量产或投资承诺。")
    p.head()
    for i,(yr,a,b,col) in enumerate([("2027","验证机制","质量—成本画像\n可复现原型与适用边界",RED),("2028","协同接口","设备遥测与策略提示\n推理框架 / OEM 联合验证",TEAL),("2029","客户参考方案","配置指南与成本模型\n持续适配模型和硬件",BLUE)]):
        xx=99+i*589
        p.text(xx,370,548,135,yr,65,col,True,face=LATIN_BOLD)
        p.line(xx,540,xx+520,540,INK,2)
        p.text(xx,590,548,82,a,29,INK,True)
        p.text(xx,731,548,136,b,21,GREY)
    p.text(99,932,1720,52,"商业验证：每百万合格 token 成本、客户采用率、产品组合价值与维护投入。",18,INK,True)
    p.foot();p.note("战略资产不是某个模型专用的一组参数，而是可校准的质量成本模型、设备接口和客户验证能力。第一年验证机制，第二年协同接口，第三年形成参考方案。客户减少 HBM 使用并不自动等于海力士利润增加，企业收益需要看采用率、组合收入和维护成本。",50)

    # 15: clear ask, strong closing visual.
    p=Page("批准六个月孵化，验证联合优化的系统价值", "14 / THE DECISION",dark=True)
    p.head()
    p.text(97,359,1230,221,"先证明净增益。\n再扩大产品投入。",49,CREAM,True,leading=1.12)
    p.rect(1466,362,330,330,RED)
    p.text(1484,416,294,130,"6",76,WHITE,True,align="center",face=LATIN_BOLD)
    p.text(1493,576,276,70,"个月 / PoC",23,WHITE,True,align="center")
    for i,(a,b) in enumerate([("人员","建议 4–5 人：量化 / 系统与内核 / 平台性能"),("平台","GPU + CXL 内存 + NVMe SSD，测量真实路径"),("决策","G0 / G1 / G2：继续、收缩或停止")]):
        yy=728+i*86
        p.label(101,yy+8,150,a,RED)
        p.text(286,yy,1490,63,b,23,CREAM,True)
    p.foot();p.note("最后明确本轮申请：六个月，建议四至五人，有限真实测试平台与业务技术对接。资源配置是提案，不是已批准预算。项目坚持在质量、服务与资源约束下验证净增益，并通过阶段门决定继续、收缩或停止。请评审批准验证窗口，而不是要求提前相信尚未产生的实验结果。",40)

    # A1: experiment protocol, one purposeful dense technical page.
    p=Page("附录 / 实验协议与归因设计", "A1 / VALIDATION PROTOCOL",appendix=True)
    p.head()
    rows=[("模型与负载","≥2 个混合模型；Attention–MoE 对照；短 / 长上下文与生成"),("资源压力","可完全驻留 / 接近容量阈值 / 明显溢出；固定容量与有效带宽"),("动态场景","低 / 高并发，路由漂移，多轮复用，暂停与恢复"),("验收建议","主集配对比值几何均值 ≥1.15；95% CI 下界 >1.00"),("质量与服务","关键任务逐项非劣性；TTFT / TPOT 尾延迟与成功率预登记"),("归因消融","移除跨对象预算、精度选择、驻留或预取，分别测量开销")]
    for i,(a,b) in enumerate(rows):
        yy=348+i*99
        p.line(99,yy+80,1818,yy+80,LINE,1)
        p.text(101,yy,370,65,a,21,TEAL,True)
        p.text(499,yy,1313,68,b,20,INK)
    p.foot();p.note("协议在 G0 锁定，避免事后选择有利场景。模型需要验证内核与上下文支持。建议均值至少 15%，并对收益方向提供统计证据；CI 下界高于 1.00 不代表真实收益的下界超过 15%。质量阈值按任务预登记，完整报告未通过情形。",55)

    # A2: feasibility, not abstract checklist.
    p=Page("附录 / 哪些配置必须从可行域排除", "A2 / FEASIBILITY & CORRECTNESS",appendix=True,refs="依据：S05–S07、S10、S13；原型必须测量具体平台能力。")
    p.head()
    for i,(a,b,c,col) in enumerate([("SSM 状态","递归误差不可逆","简单升位宽不能恢复历史；主状态 / 残差 / 检查点均计价。",TEAL),("CXL 路径","GPU 可达性须验证","普通扩展、GPU DMA、主机复制及 CXL-NDP 分别建模。",RED),("SSD 访问","容量不等于访问可行","活跃对象下沉需满足读取时限；计入排队、写放大和耐久。",BLUE),("精度副本","存储收益须真实","多精度副本、元数据与临时缓冲完整计入预算。",GREY)]):
        yy=353+i*145
        p.rect(99,yy,9,109,col)
        p.text(131,yy,380,67,a,25,INK,True)
        p.text(570,yy,1210,58,b,22,col,True)
        p.text(570,yy+72,1210,64,c,18,GREY)
    p.foot();p.note("统一优化首先需要正确的可行域。SSM 不可逆误差、CXL 可达性和 SSD 访问时限决定某些组合根本不能使用。量化副本也可能抵消容量节省，不能只按一个理想低位宽副本算账。",50)

    # A3: analytical effect size not experimental charts.
    p=Page("附录 / 超过 15% 的必要条件是什么", "A3 / SENSITIVITY",appendix=True,refs="简化说明性算例；不代表实测结果、goodput 预测或普适阈值。")
    p.head()
    p.text(98,367,1740,110,"延迟下降 ≈ f × r − o",43,RED,True)
    p.text(102,534,1700,80,"f 关键路径搬运等待占比   /   r 消除比例   /   o 新增开销占原延迟比例",20,GREY)
    for i,(a,b,c,col) in enumerate([("等待明显且可削减","40% × 50% − 3%","17%",TEAL),("搬运不是主要瓶颈","10% × 50% − 3%","2%",GREY)]):
        yy=679+i*130;p.line(99,yy+107,1818,yy+107,LINE,1)
        p.text(100,yy,590,70,a,22,INK,True)
        p.text(725,yy,660,70,b,26,col,True,face=LATIN_BOLD)
        p.text(1492,yy-8,310,93,c,39,col,True,align="right",face=LATIN_BOLD)
    p.foot();p.note("f 是未被计算重叠掩盖的搬运等待。两个例子分别得 17% 和 2%，说明优化空间必须先测量。延迟、吞吐和生产有效吞吐不能直接等同。单请求固定串行场景下 17% 延迟下降可换算约 20.5% 速率提升，但这里不拿它预测服务 goodput。",45)

    # A4: source pairs with clickable titles and readable categories.
    p=Page("附录 / 研究依据与证据边界", "A4 / EVIDENCE INDEX",appendix=True)
    p.head()
    pairs=[("S01","NOVA","混合负载与近存计算"),("S02–S03","SK hynix Future Forum / OCP","系统战略与公开展示"),("S04–S05","HOBBIT / TailorKV","量化与专家 / KV 卸载"),("S06–S07","CXL GPU–NDP MoE / SliceMoE","专家位宽与驻留"),("S08–S09","ITME / HyMCache","CXL、DRAM 与 SSD 分层"),("S10–S11","ReplaySSM / StrataFlux","递归状态与跨层精度"),("S12–S13","Nemotron 3 Ultra / PACE","模型实例与状态量化")]
    for i,(ids,title,desc) in enumerate(pairs):
        yy=350+i*78;p.label(100,yy+8,250,ids,RED)
        p.text(369,yy,912,61,title,20,INK,True)
        p.text(1340,yy+6,480,50,desc,15,GREY)
        p.line(99,yy+62,1818,yy+62,LINE,1)
    p.text(100,931,1715,55,"完整 URL 与结论边界见 SOURCES.md；文献数字不能相加，也不是本项目的实测收益。",16,GREY)
    p.foot();p.note("这些公开来源支持战略匹配和研究背景。企业展示不证明本项目已获认可；仿真或原型的论文结果也不直接外推。完整来源索引沿用 SOURCES.md，创新性仍需系统查新和实验证明。",25)


def png(page):
    im=Image.new("RGB",(W,H),rgb(page.bg));d=ImageDraw.Draw(im)
    for it in page.items:
        k=it["k"];c=rgb(it["c"])
        if k=="rect":d.rectangle((it["x"],it["y"],it["x"]+it["w"],it["y"]+it["h"]),fill=c,outline=rgb(it["stroke"]) if it["stroke"] else None)
        elif k=="ellipse":d.ellipse((it["x"],it["y"],it["x"]+it["w"],it["y"]+it["h"]),fill=c,outline=rgb(it["stroke"]) if it["stroke"] else None)
        elif k=="poly":d.polygon(it["points"],fill=c)
        elif k=="line":
            if not it["dash"]:d.line((it["x"],it["y"],it["x2"],it["y2"]),fill=c,width=round(it["width"]))
            else:
                dx=it["x2"]-it["x"];dy=it["y2"]-it["y"];ln=math.hypot(dx,dy)
                for pos in range(0,max(1,round(ln)),20):
                    end=min(pos+10,ln)
                    d.line((it["x"]+dx*pos/ln,it["y"]+dy*pos/ln,it["x"]+dx*end/ln,it["y"]+dy*end/ln),fill=c,width=round(it["width"]))
        elif k=="text":
            f=ff(it["size"],it["bold"],it["face"])
            for j,t in enumerate(it["lines"]):
                x=it["x"];tw=f.getlength(t)
                if it["align"]=="right":x+=it["w"]-tw
                elif it["align"]=="center":x+=(it["w"]-tw)/2
                d.text((x,it["y"]+j*it["size"]*2*it["leading"]),t,font=f,fill=c,anchor="lt")
    return im

def source_links(page):
    ids=set(re.findall(r"S\d{2}",page.refs))
    for lo,hi in re.findall(r"S(\d{2})–S(\d{2})",page.refs):ids.update(f"S{i:02}" for i in range(int(lo),int(hi)+1))
    if page.appendix and "EVIDENCE INDEX" in page.kicker:return shared.SOURCES[:13]
    return [s for s in shared.SOURCES if s[0] in ids]

def pptx():
    prs=Presentation();prs.slide_width=Inches(W/DPI);prs.slide_height=Inches(H/DPI)
    prs.core_properties.title=f"精度与位置，一起决定推理效率 | SK hynix Incubation Review {VERSION}"
    prs.core_properties.subject="Quality-constrained mixed precision and tiered data placement"
    prs.core_properties.author="Incubation Proposal"
    for page in SLIDES:
        slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string(page.bg)
        for it in page.items:
            k=it["k"]
            if k in ("rect","ellipse"):
                sh=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE if k=="rect" else MSO_SHAPE.OVAL, Inches(it["x"]/DPI),Inches(it["y"]/DPI),Inches(it["w"]/DPI),Inches(it["h"]/DPI))
                sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(it["c"])
                if it.get("stroke"):sh.line.color.rgb=RGBColor.from_string(it["stroke"])
                else:sh.line.fill.background()
            elif k=="poly":
                pts=it["points"];builder=slide.shapes.build_freeform(pts[0][0],pts[0][1],scale=Inches(1)/DPI)
                builder.add_line_segments(pts[1:],close=True);sh=builder.convert_to_shape();sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(it["c"]);sh.line.fill.background()
            elif k=="line":
                sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(it["x"]/DPI),Inches(it["y"]/DPI),Inches(it["x2"]/DPI),Inches(it["y2"]/DPI));sh.line.color.rgb=RGBColor.from_string(it["c"]);sh.line.width=Pt(it["width"]*.5)
                if it["dash"]:
                    from pptx.enum.dml import MSO_LINE_DASH_STYLE
                    sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
            elif k=="text":
                sh=slide.shapes.add_textbox(Inches(it["x"]/DPI),Inches(it["y"]/DPI),Inches(it["w"]/DPI),Inches(it["h"]/DPI));tf=sh.text_frame;tf.clear();tf.word_wrap=False;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
                for j,t in enumerate(it["lines"]):
                    pa=tf.paragraphs[0] if j==0 else tf.add_paragraph();pa.text=t
                    pa.font.name="Consolas" if it["face"]==MONO else "Arial" if it["face"] in (LATIN,LATIN_BOLD) else "Microsoft YaHei"
                    pa.font.size=Pt(it["size"]);pa.font.bold=it["bold"];pa.font.color.rgb=RGBColor.from_string(it["c"]);pa.space_before=pa.space_after=Pt(0);pa.line_spacing=Pt(it["size"]*it["leading"])
                    pa.alignment={"left":PP_ALIGN.LEFT,"center":PP_ALIGN.CENTER,"right":PP_ALIGN.RIGHT}[it["align"]]
                    for run in pa.runs:
                        ea=etree.SubElement(run._r.get_or_add_rPr(),"{http://schemas.openxmlformats.org/drawingml/2006/main}ea");ea.set("typeface","Microsoft YaHei")
                        if it.get("url"):
                            run.hyperlink.address=it["url"]
                        elif page.appendix and "EVIDENCE INDEX" in page.kicker:
                            # Each source-group ID links to the first reference in the group.
                            m=re.match(r"S\d{2}",t)
                            if m:
                                src=next((s for s in shared.SOURCES if s[0]==m.group()),None)
                                if src:run.hyperlink.address=src[3]
        urls="\n\n".join(f"[{s[0]}] {s[1]}\n{s[3]}\n范围：{s[5]}" for s in source_links(page))
        slide.notes_slide.notes_text_frame.text=f"{page.n:02}｜{page.title}\n建议 {page.seconds} 秒\n\n{page.notes}\n\n{urls}\n\n15% 为拟议目标；概念曲线与容量示例不代表实测。"
    file=ROOT/(FILESTEM+".pptx");prs.save(file);return file

def pdf():
    file=ROOT/(FILESTEM+".pdf")
    with PdfPages(file) as out:
        for page in SLIDES:
            fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI);fig.patch.set_facecolor("#"+page.bg)
            ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis("off")
            for it in page.items:
                k=it["k"];c="#"+it["c"]
                if k=="rect":ax.add_patch(Rectangle((it["x"],it["y"]),it["w"],it["h"],facecolor=c,edgecolor="#"+it["stroke"] if it["stroke"] else "none"))
                elif k=="ellipse":ax.add_patch(Ellipse((it["x"]+it["w"]/2,it["y"]+it["h"]/2),it["w"],it["h"],facecolor=c,edgecolor="none"))
                elif k=="poly":ax.add_patch(Polygon(it["points"],facecolor=c,edgecolor="none"))
                elif k=="line":ax.plot([it["x"],it["x2"]],[it["y"],it["y2"]],color=c,lw=it["width"]*.5,linestyle=(0,(5,5)) if it["dash"] else "-")
                elif k=="text":
                    fp=FontProperties(fname=it["face"] or (BOLD if it["bold"] else FONT),size=it["size"])
                    x=it["x"]+({"left":0,"center":it["w"]/2,"right":it["w"]}[it["align"]])
                    for j,t in enumerate(it["lines"]):ax.text(x,it["y"]+j*it["size"]*2*it["leading"],t,fontproperties=fp,color=c,va="top",ha=it["align"],parse_math=False,url=it.get("url"))
            out.savefig(fig,facecolor=fig.get_facecolor());plt.close(fig)
    return file

def qa(pptfile,pdffile):
    from fontTools.ttLib import TTFont
    cmaps={};errors=[]
    for p in SLIDES:
        for it in p.items:
            if it["k"] in ("rect","ellipse","text"):
                if it["x"]<0 or it["y"]<0 or it["x"]+it["w"]>W+1 or it["y"]+it["h"]>H+1:errors.append(f"bounds p{p.n}: {it}")
            if it["k"]=="text":
                path=it["face"] or (BOLD if it["bold"] else FONT)
                if path not in cmaps:cmaps[path]=TTFont(path,fontNumber=0).getBestCmap()
                missing={ch for ch in it["t"] if not ch.isspace() and ord(ch) not in cmaps[path]}
                if missing:errors.append(f"font p{p.n}: {missing}")
                f=ff(it["size"],it["bold"],it["face"])
                for t in it["lines"]:
                    if f.getlength(t)>it["w"]+1:errors.append(f"width p{p.n}: {t}")
    prs=Presentation(pptfile)
    assert len(prs.slides)==len(SLIDES)
    with zipfile.ZipFile(pptfile) as z:
        for name in z.namelist():
            if name.endswith(".xml"):ET.fromstring(z.read(name))
        notes=len([n for n in z.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml",n)])
        media=len([n for n in z.namelist() if n.startswith("ppt/media/")])
    npages=len(re.findall(rb"/Type\s*/Page\b",pdffile.read_bytes()))
    assert notes==npages==len(SLIDES)
    report={"version":VERSION,"status":"PASS" if not errors else "FAIL","slides":len(SLIDES),"main_slides":MAIN_SLIDES,"appendix_slides":len(SLIDES)-MAIN_SLIDES,"main_minutes":round(sum(p.seconds for p in SLIDES[:MAIN_SLIDES])/60,1),"native_editable_diagrams":True,"pptx_embedded_raster_images":media,"notes_pages":notes,"pdf_pages":npages,"objects":sum(len(p.items) for p in SLIDES),"errors":errors,"office_rendering":"Not available; PNG/PDF share the measured scene; PowerPoint-native render still unverified."}
    (ROOT/f"QA_REPORT_{VERSION}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    assert not errors,errors
    return report

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/"preview").mkdir(exist_ok=True)
    compose();thumbs=[]
    for p in SLIDES:
        im=png(p);im.save(ROOT/"preview"/f"slide_{p.n:02}.png");thumbs.append(im.resize((576,324),Image.Resampling.LANCZOS))
    sheet=Image.new("RGB",(1816,math.ceil(len(SLIDES)/3)*378+28),rgb("E7E6E0"));d=ImageDraw.Draw(sheet)
    for i,im in enumerate(thumbs):
        x=24+i%3*596;y=24+i//3*378;sheet.paste(im,(x,y));t=f"{i+1:02}  "+SLIDES[i].title.replace("\n","")
        if len(t)>38:t=t[:37]+"…"
        d.text((x+1,y+336),t,font=ImageFont.truetype(FONT,17),fill=rgb(INK))
    sheet.save(ROOT/"CONTACT_SHEET_V2.png")
    pf=pptx();df=pdf()
    notes="# V2｜逐页讲稿\n\n15 页主讲 + 4 页附录。主讲约 13–15 分钟。\n\n"
    for p in SLIDES:notes+=f"## {p.n:02}｜{p.title.replace(chr(10),' ')}\n\n建议 {p.seconds} 秒。\n\n{p.notes}\n\n来源：{p.refs or '提案设计 / 待验证目标'}\n\n"
    (ROOT/"SPEAKER_NOTES_V2.md").write_text(notes,encoding="utf-8")
    (ROOT/"SOURCES.md").write_text((ROOT.parent/"v1"/"SOURCES.md").read_text(encoding="utf-8"),encoding="utf-8")
    (ROOT/"README.md").write_text("# Incubation Review V2\n\n15 页主讲 + 4 页附录。原生可编辑的图解与评审叙事。\n\n- `SK_hynix_Incubation_Review_V2.pptx`：演示稿，含备注。\n- `SK_hynix_Incubation_Review_V2.pdf`：矢量预览。\n- `CONTACT_SHEET_V2.png`、`preview/`：全稿及逐页预览。\n- `SPEAKER_NOTES_V2.md`、`SOURCES.md`：讲稿和来源。\n- `QA_REPORT_V2.json`：检查报告。\n- `build_review_v2.py`：本版本生成脚本。\n\n重新生成：在本目录执行 `python build_review_v2.py`。依赖相邻 `../v1/build_deck.py` 的来源与排版工具；生成结果均写入本目录。PDF/PNG 与 PPT 共用版面数据，Office 原生放映尚未验证。\n",encoding="utf-8")
    print(json.dumps(qa(pf,df),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
