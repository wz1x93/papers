"""V3: industry context + Samsung/Micron developments + sourced necessity.

Reuses the approved V2 editorial design and editable scene/export engine.
New statements are tied to dated primary sources; analysis is explicitly labeled.
"""
from __future__ import annotations
import json
import math
import re
import zipfile
import sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2"))
import build_review_v2 as v

ROOT=Path(__file__).resolve().parent
INK,RED,TEAL,BLUE,GREY,LINE,WHITE=v.INK,v.RED,v.TEAL,v.BLUE,v.GREY,v.LINE,v.WHITE
CREAM,DIM=v.CREAM,v.DIM

# (ID, title, publication date, primary URL, optional HTML, scope/limitations)
NEW_SOURCES=[
 ("S14","Samsung Introduces Industry’s First Open-source Software Solution for CXL Memory Platform","2021-10-07","https://news.samsungsemiconductor.com/global/samsung-introduces-industrys-first-open-source-software-solution-for-cxl-memory-platform/","","三星 SMDK 发布公告，含 Intelligent Tiering Engine、库与 API。该历史公告当时为有限测试，开源计划不能按当时已经全面部署描述；当前仓库由 S18 补充。"),
 ("S15","Samsung Electronics Announces Successful Building of Industry-First CXL Infrastructure Certified by Red Hat","2024-06-25","https://news.samsungsemiconductor.com/global/samsung-electronics-announces-successful-building-of-industry-first-cxl-infrastructure-certified-by-red-hat/","","三星 SMRC 的 Red Hat 认证 CXL 基础设施、CMM-D 验证与 SMDK memory interleaving 演示。认证与 DLRM 演示不等于完成混合 LLM 联合量化。"),
 ("S16","Part 2. zHBM: Next-Generation Memory for the Data Movement Era","2026-10-29","https://news.samsungsemiconductor.com/global/part-2-zhbm-next-generation-memory-for-the-data-movement-era/","","三星明确称 zHBM currently under development，讨论直接 3D 堆叠、Distributed I/O 和系统协同。是研发方向，非量产公告；未用其推断本项目收益。"),
 ("S17","Samsung Electronics Announces Third Quarter 2026 Results","2026-10-29","https://news.samsungsemiconductor.com/global/samsung-electronics-announces-third-quarter-2026-results/","","仅引用 Memory Business 对 HBM4 销售扩展、服务器 SSD 需求和 AI 产品组合的定性自述。不使用财务数据、份额或厂商自称领先来比较技术能力。"),
 ("S18","OpenMPDK / SMDK — Scalable Memory Development Kit, README","访问 2026-11-08；持续更新仓库","https://github.com/OpenMPDK/SMDK","","README 描述 SMDK v2.2、Intelligent Tiering Engine、兼容与优化 API。代码公开性可验证，不证明任何特定模型性能或本提案所需 GPU/CXL 路径。"),
 ("S19","Micron in High-Volume Production of HBM4 Designed for NVIDIA Vera Rubin, PCIe Gen6 SSD and SOCAMM2","2026-03-16","https://investors.micron.com/news/press-release/2026/Micron-in-High-Volume-Production-of-HBM4-Designed-for-NVIDIA-Vera-Rubin-PCIe-Gen6-SSD-and-SOCAMM2-03-16-2026/default.aspx","","美光官方称 HBM4 36GB 12H、9650 Gen6 SSD、192GB SOCAMM2 高量产，并讨论 Vera Rubin 与 BlueField-4 STX。量产状态是公司公告，不是本项目实测或独立横评。"),
 ("S20","Micron Technology, Inc. Reports Record Fiscal Fourth-Quarter and Full-Year 2026 Results","2026-09-30","https://investors.micron.com/news/press-release/2026/Micron-Technology-Inc--Reports-Record-Fiscal-Fourth-Quarter-and-Full-Year-2026-Results","","仅使用 Product Highlights：7600 PCIe Gen5 和 9650 PCIe Gen6 SSD 已向头部客户的 KV-cache 应用出货。未使用财务数字或将出货表述为 LLM 端到端提速证据。"),
 ("S21","NVIDIA Dynamo, A Low-Latency Distributed Inference Framework for Scaling Reasoning AI Models","2025-03-18","https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/","","官方技术博客正文：PD 分离、KV-aware routing、多层 KV 管理和 NIXL。只引用架构功能，不使用其中混有预测条件的 30x 数字。NVIDIA 是平台与生态方，不在本提案中作为内存器件竞争对手排名。"),
 ("S22","Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving","v1 2024-06-24；v5 2026-10-08","https://arxiv.org/abs/2407.00079","https://arxiv.org/html/2407.00079v5","论文描述 Kimi 的 KV-centric PD 分离架构，使用 CPU、DRAM、SSD 缓存并优化 SLO 下有效吞吐。仅用架构作为行业信号，不外推其 reported throughput 到本项目。"),
 ("S23","Samsung Develops Industry’s First CXL DRAM Supporting CXL 2.0","2023-05-12","https://news.samsungsemiconductor.com/global/samsung-develops-industrys-first-cxl-dram-supporting-cxl-2-0/","","三星 128GB CXL 2.0 DRAM 开发及内存池支持的历史里程碑，当年量产为计划；不能把它当作 2026 年最新代际或现状排名。"),
]
QUOTES={
 "S14":"Leveraging a proprietary Intelligent Tiering Engine, the SMDK can identify and configure the memory type, capacity and bandwidth that are most appropriate for each use case.",
 "S15":"Elements that configure servers, from CXL-related products to software, can now be directly verified at the Samsung Memory Research Center (SMRC)",
 "S16":"Samsung zHBM, currently under development, is a next-generation AI memory architecture designed around this challenge.",
 "S17":"It significantly scaled up sales of HBM4 on the back of industry-leading product competitiveness while proactively addressing growing demand for server SSDs.",
 "S18":"The SMDK is developed for the Samsung CXL (Compute Express Link) Memory Expander devices to enable a full-stack Software-Defined Memory system.",
 "S19":"HBM4 36GB 12H in high-volume production ... / 192GB SOCAMM2 in high-volume production [节选新闻要点]",
 "S20":"Our 7600 PCIe Gen 5 and 9650 PCIe Gen 6 SSD high-performance products are now shipping to market-leading customers for KV-cache applications.",
 "S21":"KV cache offloading across different memory hierarchies to increase system throughput",
 "S22":"It also leverages the underutilized CPU, DRAM, and SSD resources of the GPU cluster to implement a disaggregated cache of KVCache.",
 "S23":"For the first time ever, CXL 2.0 supports memory pooling",
}

def getsrc(id_):return next(s for s in v.shared.SOURCES if s[0]==id_)

def cite(p,x,y,w,id_,short,date):
    p.text(x,y,w,38,f"[{id_}] {short} · {date}",11.5,p.muted)
    p.items[-1]["url"]=getsrc(id_)[3]

def new_pages():
    out=[]
    # New 03: sector context, evidence rather than a made-up market CAGR.
    p=v.Page("行业变化：模型能力扩展，带来更多状态与权重管理", " / INDUSTRY CONTEXT",sub="长上下文、稀疏专家与持续服务同时存在；内存需求由多种数据共同决定。",refs="来源：S12（2026-06）、S22（2024/2026）、S21（2025-03）；指标为特定模型 / 系统实例。")
    p.head();out.append(p)
    cols=[(100,RED,"LONG CONTEXT","1M","token 上下文","Nemotron 3 Ultra 的模型报告\n展示百万 token 上下文。\n历史 KV 的容量与访问需管理。","S12","NVIDIA / 模型论文","2026.06.12"),
          (705,TEAL,"SPARSE MODEL","550B","总参数 / 55B 激活","同一混合模型具有大量\n未同时激活的专家参数。\n计算稀疏不消除权重驻留需求。","S12","NVIDIA / 模型论文","2026.06.12"),
          (1310,BLUE,"SERVING SYSTEM","KV","成为服务调度对象","Mooncake 与 Dynamo 将\n复用、传输和分层缓存\n纳入推理服务架构。","S22","Mooncake / 论文","v5 2026.10")]
    for i,(x,col,lab,big,tt,body,src,short,date) in enumerate(cols):
        if i:p.line(x-36,355,x-36,925,LINE,1)
        p.label(x,354,535,lab,col)
        p.text(x-3,423,535,146,big,69,col,True,face=v.LATIN_BOLD)
        p.text(x,607,535,72,tt,25,INK,True)
        p.text(x,726,527,155,body,18.5,GREY)
        cite(p,x,924,535,src,short,date)
    p.foot();p.note("这一页提供行业背景，不预测市场份额或所有模型的路线。NVIDIA 的 Nemotron 3 Ultra 报告了 1M 上下文、550B 总参数与 55B 激活参数。它说明计算激活规模和需要保存的模型容量并不相同。Mooncake 与 Dynamo 则把状态复用和跨层移动提升到服务系统层面。数字属于特定模型，不能外推为行业平均。",65)

    # New 04: two dated architectural milestones plus current product signal.
    p=v.Page("跨层状态管理，已经进入服务架构与产品交付", " / INDUSTRY DEVELOPMENT",sub="从服务论文、平台软件到 SSD 产品应用，内存与存储协同具有明确的落地信号。",refs="来源：S22 / Mooncake；S21 / NVIDIA Dynamo；S20 / Micron Product Highlights。")
    p.head();out.append(p)
    # A hierarchy motif, with explicit data-lifecycle semantics.
    for i,(x,w,lab,col) in enumerate([(100,390,"GPU / HBM",RED),(665,470,"HOST / DRAM",TEAL),(1310,510,"SSD / SHARED CACHE",BLUE)]):
        p.rect(x,366,w,100,col);p.text(x+18,394,w-36,62,lab,23,WHITE,True,face=v.LATIN_BOLD)
    p.line(509,417,646,417,GREY,3,arrow=True);p.line(1154,417,1291,417,GREY,3,arrow=True)
    p.text(100,491,1720,64,"图示对象：可复用 / 非活跃的上下文数据；活跃数据的下沉仍须满足访问时限。",17,GREY)
    evidence=[(100,"2024","Mooncake","服务论文","PD 分离与 KV-centric 调度，\n利用 CPU、DRAM、SSD 缓存。","S22","Mooncake","v1 2024.06"),
              (705,"2025","NVIDIA Dynamo","官方平台功能","分布式 KV 管理与 NIXL，\n连接 GPU、主机与存储层。","S21","NVIDIA 技术博客","2025.03.18"),
              (1310,"2026","Micron SSD","公司出货自述","7600 / 9650 SSD 面向\n客户 KV-cache 应用出货。","S20","Micron FY26 Q4","2026.09.30")]
    for x,yr,tt,status,body,id_,short,date in evidence:
        p.label(x,597,540,yr,RED)
        p.text(x,650,535,65,tt,27,INK,True)
        p.text(x,739,535,49,status,15,TEAL,True)
        p.text(x,803,535,98,body,18,GREY)
        cite(p,x,935,535,id_,short,date)
    p.foot();p.note("这三条不是可直接比较的性能测试，而是不同层次的行业落地信号。Mooncake 以生产服务背景描述多层 KV；NVIDIA 官方博客把 KV 管理和 NIXL 纳入 Dynamo；美光 2026 年 9 月在产品要点中自述 SSD 已向 KV-cache 客户出货。平台和客户已经在处理跨层状态，项目应验证如何在已有生态上进一步联合精度与驻留。",65)

    # New 05: Samsung is the focal competitor, maturity made explicit.
    p=v.Page("三星：软件、器件、认证与协同研发连续推进", " / SAMSUNG DEVELOPMENT",sub="四个公开里程碑显示能力投入；历史事件用于展示路径，不代表 2026 年最新代际排名。",refs="来源：S14 / 2021-10；S23 / 2023-05；S15 / 2024-06；S16–S17 / 2026-10-29。")
    p.head();out.append(p)
    p.line(100,438,1818,438,INK,2)
    events=[(100,"2021","SMDK","软件公告","智能分层引擎 / API；\n公开仓库见 S18。","S14","Samsung SMDK","2021.10.07",RED),
            (545,"2023","CXL 2.0","器件开发","128GB CXL DRAM 开发；\n支持内存池。","S23","Samsung CXL","2023.05.12",TEAL),
            (990,"2024","Red Hat","基础设施认证","CXL 验证环境；\nCMM-D / SMDK 演示。","S15","Samsung / Red Hat","2024.06.25",BLUE),
            (1435,"2026","zHBM","研发方向","直接 3D 堆叠；\nDistributed I/O；\n与 AI 芯片协同设计。","S16","Samsung zHBM","2026.10.29",RED)]
    for x,yr,tt,status,body,id_,short,date,col in events:
        p.ellipse(x-10,428,20,20,col)
        p.text(x,350,385,75,yr,33,col,True,face=v.LATIN_BOLD)
        p.text(x,492,385,83,tt,29,INK,True)
        p.label(x,606,385,status,col)
        p.text(x,671,385,151,body,17,GREY)
        cite(p,x,860,385,id_,short,date)
    p.rect(100,925,1720,56,INK)
    p.text(121,936,1678,42,"2026 Q3 公司公告：扩大 HBM4 销售并响应服务器 SSD 需求 [S17]；zHBM 仍属研发方向 [S16]。",16.5,WHITE,True)
    p.foot();p.note("三星是本次重点竞争对手。2021 年 SMDK 软件公告包含智能分层引擎，当前公开仓库提供了代码层证据；2023 年 CXL 2.0 为历史开发里程碑；2024 年三星宣布 Red Hat 认证的 CXL 基础设施与 CMM-D 验证。2026 年 zHBM 文章明确写 currently under development。它们证明了投入方向，不证明三星已经实现与本提案相同的三对象联合优化，也不支持任何现成的跨厂商性能排名。",75)

    # New 06: competitor / company landscape, no unsupported rankings.
    p=v.Page("竞争维度扩展：产品组合与系统协同都值得比较", " / COMPETITIVE LANDSCAPE",sub="以下为公开行动的并列梳理，不是性能横评，也不对未检索到的能力作缺失判断。",refs="来源：三星 S15–S18；美光 S19–S20；海力士 S02–S03。")
    p.head();out.append(p)
    companies=[(100,"SAMSUNG",RED,"软件与协同研发","HBM4 / SSD 组合 [S17]\nCMM-D / SMDK 软件与认证\nzHBM 协同研发 [S16]","S15","官方认证公告","2024.06.25","S16","zHBM / 在研发","2026.10.29"),
               (705,"MICRON",BLUE,"平台关联的产品交付","HBM4 / Gen6 SSD\nSOCAMM2 平台关联方案\nKV-cache 应用出货自述","S19","GTC 产品量产公告","2026.03.16","S20","FY26 Q4 产品要点","2026.09.30"),
               (1310,"SK HYNIX",TEAL,"内存中心系统展示","MARS / CXL 内存池\nDRAM+SSD 混合内存\nKV Cache 共享与复用","S03","OCP 系统展示","2026.10.16","S02","Future Forum 战略","2026.09.09")]
    for i,(x,name,col,tt,body,a,at,ad,b,bt,bd) in enumerate(companies):
        if i:p.line(x-36,350,x-36,904,LINE,1)
        p.text(x-2,353,535,90,name,37,col,True,face=v.LATIN_BOLD)
        p.line(x,474,x+510,474,INK,2)
        p.text(x,514,535,85,tt,25,INK,True)
        p.text(x,650,530,133,body,18.5,GREY)
        cite(p,x,819,535,a,at,ad);cite(p,x,867,535,b,bt,bd)
    p.text(100,942,1715,45,"提案定位：在既有产品与生态上，验证“质量感知精度 × 驻留”能否创造可部署的差异化。",18,INK,True)
    p.foot();p.note("这页并列公开行动，不能用单个公告打出领先落后的分数。美光 3 月官方宣布 HBM4、Gen6 SSD 和 SOCAMM2 高量产，并联系 Vera Rubin 与 BlueField-4 STX；9 月产品要点自述 KV-cache SSD 出货。海力士 10 月展示 MARS、CXL池化及混合内存。三星组合证据包括软件认证与协同研发。三家都在建设系统能力，因此提案需要以端到端验证和设备接口建立自身价值，不能把硬件产品齐全本身当作护城河。",70)

    # New 07: necessity from evidence to a bounded investment case.
    p=v.Page("项目必要性：把更多硬件选择，变成客户可测的收益", " / WHY INCUBATE NOW",dark=True,refs="事实依据：S03、S12、S15–S16、S18–S22；右侧为提案分析与待验证目标。")
    p.head();out.append(p)
    facts=[(358,"需求端","长上下文与混合模型，\n使状态与专家资源取舍更复杂。","S12 / S21 / S22"),
           (538,"供给端","快层、扩展内存和 SSD，\n增加了配置与访问路径选择。","S03 / S19 / S20"),
           (718,"竞争端","软件接口、客户验证与协同设计，\n已成为公开投入方向。","S15 / S16 / S18")]
    for yy,a,b,ids in facts:
        p.rect(100,yy,5,136,RED)
        p.text(124,yy,230,59,a,24,CREAM,True)
        p.text(360,yy,620,111,b,21,"C1CBC3")
        p.label(360,yy+115,620,ids,DIM)
    p.line(1022,355,1022,920,v.DARKLINE,2)
    p.label(1094,355,700,"STRATEGIC INFERENCE",RED)
    p.text(1090,427,708,188,"海力士需要回答：\n同样的资源与质量，\n怎样配置更有价值？",30,CREAM,True)
    p.text(1095,700,700,143,"项目产出：质量—成本模型、\n精度与驻留策略、设备提示接口，\n以及客户负载的实测证据。",20,DIM)
    p.text(100,938,1705,48,"本轮投资判断：先用六个月验证增量收益；通过后再扩大合作与产品化投入。",20,CREAM,True)
    p.foot();p.note("必要性来自可验证的经营与技术问题。需求与硬件选项越来越多，如果没有工作负载和质量成本画像，就难以向客户量化不同配置的价值。竞争公开投入软件和验证能力，是应加快试验的信号，但不意味着这个具体优化必然有收益。右侧属于提案的战略判断，最终仍由六个月强基线实验决定继续与否。",65)
    return out

def evidence_appendices():
    out=[]
    for group,ids in [(1,["S14","S23","S15","S16","S18"]),(2,["S17","S19","S20","S21","S22"])]:
        p=v.Page(f"附录 / 行业与竞争证据索引 {group}/2",f"A{4+group} / INDUSTRY SOURCE INDEX",sub="来源 ID 可点击；完整标题、URL、原文节选与适用边界见 SOURCES_V3.md。",refs="来源："+"、".join(ids)+"；核查日期 2026-11-08。",appendix=True)
        p.head();out.append(p)
        for i,id_ in enumerate(ids):
            s=getsrc(id_);yy=350+i*120
            p.label(100,yy+2,170,id_,RED);p.items[-1]["url"]=s[3]
            short={"S14":"Samsung SMDK 软件发布","S23":"Samsung CXL 2.0 DRAM 开发","S15":"Samsung × Red Hat：CXL 认证环境","S16":"Samsung zHBM：研发中的系统协同","S18":"OpenMPDK / SMDK 公开仓库","S17":"Samsung 2026 Q3：HBM / SSD 组合","S19":"Micron：HBM4 / Gen6 SSD / SOCAMM2","S20":"Micron：KV-cache 应用 SSD 出货自述","S21":"NVIDIA Dynamo：分层 KV 与 NIXL","S22":"Mooncake：KV-centric 服务论文"}[id_]
            p.text(302,yy,1515,58,short,23,INK,True);p.items[-1]["url"]=s[3]
            p.text(305,yy+66,1515,47,s[2],14,GREY)
            p.line(100,yy+106,1818,yy+106,LINE,1)
        p.text(100,960,1710,32,"核查日期 2026.11.08。公告状态、认证 / 演示、研发方向分别解释；未引用市场份额或财务预测。",12,GREY)
        p.foot();p.note("引用均来自已核查的官方页面、公开代码或论文。点击来源可访问原始网页；来源文档还保留原文节选及证据边界。竞争公告不是独立横向测试，当前未检索到的内容不意味着竞争对手没有该能力。",25)
    return out

def strip_footer(p):
    p.items=[it for it in p.items if not (it["k"]=="line" and it["y"]==1012) and not (it["k"]=="text" and it["y"]>=1020)]

def compose_v3():
    v.ROOT=ROOT;v.VERSION="V3";v.FILESTEM="SK_hynix_Incubation_Review_V3";v.MAIN_SLIDES=19
    v.shared.SOURCES=v.shared.SOURCES[:13]+NEW_SOURCES
    v.compose();old=v.SLIDES[:]
    new=new_pages();extra=evidence_appendices()
    # Preserve V2's approved main sequence after replacing its generic why-now.
    ordered=old[:2]+new+old[3:15]+old[15:19]+extra
    v.SLIDES[:]=ordered
    for i,p in enumerate(ordered,1):
        strip_footer(p);oldk=p.kicker;p.n=i
        if i==1:
            for it in p.items:
                if it["k"]=="text" and it["t"]=="SK HYNIX / INCUBATION PROPOSAL":
                    it["t"]="SK HYNIX / INCUBATION REVIEW V3";it["lines"]=[it["t"]]
        elif not p.appendix:
            suffix=oldk.split("/",1)[-1].strip()
            p.kicker=f"{i-1:02} / {suffix}"
            for it in p.items:
                if it["k"]=="text" and it["t"]==oldk:
                    it["t"]=p.kicker;it["lines"]=[p.kicker]
        if p.appendix and "EVIDENCE INDEX" in p.kicker:
            p.sub="技术研究来源索引；行业与竞争来源另见后两页。"
        p.foot()
    return ordered

def write_sources():
    # Reuse the 13 technical references, and add the 10 freshly verified sources.
    original=(ROOT.parent/"v1"/"SOURCES.md").read_text(encoding="utf-8")
    text="# V3｜行业、竞争与技术来源\n\n核查日期：2026-11-08。S01–S13 为既有技术来源；S14–S23 为本轮新增核查。\n\n"
    text+="## 引用规范\n\n每项行业判断都区分公开事实、状态及提案分析。公司公告按公司自述标注，研发不视作量产，认证 / 演示不视作特定 LLM 性能验证。未使用未核实的市场份额、CAGR、财务数字或领先排名。\n\n"
    for s in NEW_SOURCES:
        id_,title,date,url,html,scope=s
        text+=f"## {id_}｜{title}\n\n- 发布日期：{date}\n- 核查日期：2026-11-08\n- 原始来源：{url}\n"
        if html:text+=f"- 正文：{html}\n"
        text+=f"- 使用范围与限制：{scope}\n\n原文节选：\n\n> {QUOTES[id_]}\n\n"
    text+="---\n\n# 既有技术来源与假设边界（S01–S13）\n\n"+original
    text+="\n\n## V3 补充边界\n\n- 1M token、550B 总参数、55B 激活参数是 Nemotron 3 Ultra 论文中的模型实例，非行业平均值。\n- Samsung 2021/2023/2024 事件表示历史发展路径，不作为 2026 年最新 CXL 产品代际排名。\n- zHBM 在 2026-10-29 的原文中明确为 currently under development。\n- Micron 量产与出货状态按发布方表述记录，不代替独立出货核验或 LLM 性能横评。\n- NVIDIA 为平台 / 软件生态方；未将其作为与 DRAM 厂商相同类别的竞争主体排序。\n- “需要孵化联合优化”是提案战略推论；并非引用文献已经证明本项目必要或必然成功。\n"
    (ROOT/"SOURCES_V3.md").write_text(text,encoding="utf-8")
    (ROOT/"SOURCES.md").write_text(text,encoding="utf-8")

    claims=[
      (3,"模型实例具有 1M 上下文、550B 总参数和 55B 激活参数","事实 / 模型实例","S12","不外推为行业均值或统一未来路线"),
      (4,"KV 跨层管理进入服务架构与产品应用","行业归纳","S20–S22","各证据分别是论文、平台功能和公司出货自述"),
      (5,"三星持续投入软件、CXL器件、认证与 zHBM 研发","公开行动事实","S14–S16、S18、S23","zHBM 未称量产；不声称完整实现本提案"),
      (6,"三家厂商均有系统协同公开行动","证据归纳","S02–S03、S15–S20","非性能、份额或成熟度的打分排名"),
      (7,"海力士值得孵化质量感知的联合策略","战略判断 / 待验证","S03、S12、S15–S16、S18–S22","需通过六个月实验验证净收益及商业条件"),
    ]
    md="# V3｜主张—证据核查表\n\n| 页码 | 页面主张 | 性质 | 来源 | 必须保留的边界 |\n|---|---|---|---|---|\n"
    for row in claims:md+="| "+" | ".join(map(str,row))+" |\n"
    md+="\n## 成熟度用语\n\n- 发布 / 开发：来源中公告的开发或软件发布事件。\n- 认证 / 演示：已宣告的互操作认证、基础设施或演示，不推导客户量产。\n- 量产 / 出货：注明公司公告或公司自述，不作为本项目独立核验。\n- 在研发：保留研发限定语，不画成可购买的成熟产品。\n- 战略判断：与事实分开，保留本项目阶段门与收益验证要求。\n"
    (ROOT/"CLAIM_EVIDENCE_V3.md").write_text(md,encoding="utf-8")

def audit(pptfile,report):
    valid={s[0] for s in v.shared.SOURCES};errors=[]
    for p in v.SLIDES:
        for sid in re.findall(r"S\d{2}",p.refs):
            if sid not in valid:errors.append(f"Unknown source {sid} on page {p.n}")
        if p.n in range(3,8) and not p.refs:errors.append(f"Missing source footer page {p.n}")
    with zipfile.ZipFile(pptfile) as z:
        external_links=sum(z.read(n).count(b'TargetMode="External"') for n in z.namelist() if n.endswith(".rels"))
        assert external_links>=30,external_links
    for p in v.SLIDES[2:7]:
        s=v.source_links(p)
        if not s:errors.append(f"No note references on page {p.n}")
    assert not errors,errors
    report.update({"industry_slides":[3,4],"samsung_slide":5,"competitive_landscape_slide":6,"necessity_slide":7,"new_sources":len(NEW_SOURCES),"total_sources":len(valid),"external_hyperlinks":external_links,"citation_errors":errors,"fact_vs_analysis":"company-reported production, certification, R&D and strategic inference explicitly distinguished"})
    (ROOT/"QA_REPORT_V3.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/"preview").mkdir(exist_ok=True)
    pages=compose_v3();thumbs=[]
    for p in pages:
        im=v.png(p);im.save(ROOT/"preview"/f"slide_{p.n:02}.png");thumbs.append(im.resize((576,324),Image.Resampling.LANCZOS))
    def contact(subset,filename):
        sheet=Image.new("RGB",(1816,math.ceil(len(subset)/3)*378+28),v.rgb("E7E6E0"));d=ImageDraw.Draw(sheet)
        for i,idx in enumerate(subset):
            x=24+i%3*596;y=24+i//3*378;sheet.paste(thumbs[idx],(x,y));t=f"{idx+1:02}  "+pages[idx].title.replace("\n","")
            if len(t)>38:t=t[:37]+"…"
            d.text((x+1,y+336),t,font=ImageFont.truetype(v.FONT,17),fill=v.rgb(INK))
        sheet.save(ROOT/filename)
    contact(list(range(len(pages))),"CONTACT_SHEET_V3.png")
    contact([2,3,4,5,6],"INDUSTRY_COMPETITION_PREVIEW_V3.png")
    pf=v.pptx();df=v.pdf()
    notes="# V3｜逐页讲稿\n\n19 页主讲 + 6 页附录；新增行业与竞争背景在第 3–7 页。主讲约 18 分钟，可按评审时长略讲技术附录。\n\n"
    for p in pages:
        notes+=f"## {p.n:02}｜{p.title.replace(chr(10),' ')}\n\n建议 {p.seconds} 秒。\n\n{p.notes}\n\n"
        for s in v.source_links(p):notes+=f"- [{s[0]}] {s[1]} · {s[2]}\n  {s[3]}\n"
        notes+="\n"
    (ROOT/"SPEAKER_NOTES_V3.md").write_text(notes,encoding="utf-8")
    write_sources()
    (ROOT/"README.md").write_text("# Incubation Review V3\n\n19 页主讲 + 6 页附录。沿用 V2 视觉，新增行业发展、三星与美光竞争动态及项目必要性。\n\n- `SK_hynix_Incubation_Review_V3.pptx`：原生可编辑，含讲稿备注。\n- `SK_hynix_Incubation_Review_V3.pdf`：矢量预览。\n- `INDUSTRY_COMPETITION_PREVIEW_V3.png`：新增第 3–7 页预览。\n- `CONTACT_SHEET_V3.png`、`preview/`：全稿与逐页预览。\n- `SPEAKER_NOTES_V3.md`：逐页讲稿与链接。\n- `SOURCES_V3.md`、`SOURCES.md`：23 项来源与边界。\n- `CLAIM_EVIDENCE_V3.md`：主张—证据对照。\n- `QA_REPORT_V3.json`：检查报告。\n- `build_review_v3.py`：本版本生成脚本。\n\n重新生成：在本目录执行 `python build_review_v3.py`。依赖相邻 `../v2/build_review_v2.py` 与 `../v1/build_deck.py`，以及 `../v1/SOURCES.md`；生成结果均写入本目录。PDF/PNG 与 PPT 共用版面数据，Office 原生放映尚未验证。\n",encoding="utf-8")
    print(json.dumps(audit(pf,v.qa(pf,df)),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
