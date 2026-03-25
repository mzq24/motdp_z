import sys
import subprocess
import os

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
except ImportError:
    print("Installing python-pptx...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx", "--quiet"])
    from pptx import Presentation
    from pptx.util import Inches, Pt

prs = Presentation()

# 1. 标题页
slide = prs.slides.add_slide(prs.slide_layouts[0])
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = "基于扩散模型的轨迹规划与能量引导"
subtitle.text = "MoT-DP Project Presentation\n解决 Anchor 与 GT 的 Gap 以及未来展望"

# 2. 问题背景
slide = prs.slides.add_slide(prs.slide_layouts[1])
title, body = slide.shapes.title, slide.placeholders[1]
title.text = "1. Diffusion Drive 中 Anchor 与 GT 的 Gap"
tf = body.text_frame
tf.text = "传统的 Diffusion Drive 严重依赖基于 Anchor 做截断扩散 (Truncated Diffusion)。"
p = tf.add_paragraph()
p.text = "核心痛点：预设的 Anchor 与实际 GT 轨迹之间存在固有的空间/语义 Gap。"
p.level = 1
p = tf.add_paragraph()
p.text = "这种硬性的空间绑定限制了模型的探索能力，在偏离分布 (OOD) 时容易产生巨大误差。"
p.level = 1

# 3. 数据支撑
slide = prs.slides.add_slide(prs.slide_layouts[5])
title = slide.shapes.title
title.text = "数值记录支撑 (Ablation Table)"
rows, cols = 5, 4
left = Inches(0.5); top = Inches(1.5); width = Inches(9.0); height = Inches(2.3)
table = slide.shapes.add_table(rows, cols, left, top, width, height).table

headers = ["实验配置 (模式)", "多步 DDIM (L2)", "1-step Denoise", "结论"]
for i, h in enumerate(headers): table.cell(0, i).text = h

table.cell(1, 0).text = "DD Baseline Abs 默认"
table.cell(1, 1).text = "~0.290"
table.cell(1, 2).text = "0.295"
table.cell(1, 3).text = "基线：依赖 Anchor 强大的残差兜底"

table.cell(2, 0).text = "Delta 纯差分 (脱离Anchor)"
table.cell(2, 1).text = "0.564"
table.cell(2, 2).text = "0.264"
table.cell(2, 3).text = "脱离限制后1步变强，但多步发散"

table.cell(3, 0).text = "Delta + 动态 BEV"
table.cell(3, 1).text = "0.344"
table.cell(3, 2).text = "0.268"
table.cell(3, 3).text = "动态采样仅挽回约 55% Gap"

table.cell(4, 0).text = "White Noise (Anchor-Free)"
table.cell(4, 1).text = "~0.270"
table.cell(4, 2).text = "-"
table.cell(4, 3).text = "彻底解绑Anchor，多步推理稳定高精"

tf2 = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(9.0), Inches(1.0)).text_frame
tf2.text = "结论：传统架构抛弃 Anchor 会导致多步崩溃。但纯 White Noise (Anchor-Free) 机制不仅摆脱了 Anchor 束缚，且多步去噪极其稳定 (~0.27)。"

# 4. 前沿方案：BridgeDrive
slide = prs.slides.add_slide(prs.slide_layouts[1])
title, body = slide.shapes.title, slide.placeholders[1]
title.text = "2. BridgeDrive (DDBM) 的解决思路及局限性"
tf = body.text_frame
tf.text = "布朗桥扩散 (Brownian Bridge)：恢复前向和反向对称性。"
p = tf.add_paragraph()
p.text = "两头固定：将初始点从高斯噪声，换成从 GT(起点) 桥接至 K-Means Anchor(终点)。"
p.level = 1
p = tf.add_paragraph()
p.text = "局限性剖析："
p.level = 0
p = tf.add_paragraph()
p.text = "1. 由于噪声被两端死死钉住，模型的自主探索能力极弱。"
p.level = 1
p = tf.add_paragraph()
p.text = "2. 极其依赖分类网络：一旦 Anchor 被错选，Bridge 就会把车精准地送往错误的目的地。"
p.level = 1

# 5. 我们的方案
slide = prs.slides.add_slide(prs.slide_layouts[1])
title, body = slide.shapes.title, slide.placeholders[1]
title.text = "3. 我们的方案：Anchor-Free 能量引导扩散"
tf = body.text_frame
tf.text = "动机：彻底解绑！回归真实分布探索。"
p = tf.add_paragraph()
p.text = "Anchor-Free 纯白噪声起步：不使用 Anchor 做物理截断，直接预测。"
p.level = 1
p = tf.add_paragraph()
p.text = "Cross-Attention 先验柔性注入："
p.level = 0
p = tf.add_paragraph()
p.text = "抛弃起点的硬性约束，需要参考 Anchor 时，转为 Mode Queries 在 Attention 层做柔性获取。"
p.level = 1

# 6. 对比学习能量训练
slide = prs.slides.add_slide(prs.slide_layouts[1])
title, body = slide.shapes.title, slide.placeholders[1]
title.text = "对比学习能量训练 (Contrastive Learning)"
tf = body.text_frame
tf.text = "采用标准的对比学习机制，分离构建能量判别头："
p = tf.add_paragraph()
p.text = "正样本 (Positive)：真实 GT 引入速度缩放增广 —— 物理目标：Pull 拉向低能量安全流形。"
p.level = 1
p = tf.add_paragraph()
p.text = "负样本 (Negative)：带行为禁止标签的危险 Anchor —— 物理目标：Push 推向高能量区。"
p.level = 1
p = tf.add_paragraph()
p.text = "平滑梯度：去掉二分类，改用 SmoothL1 保证能量空间可微。"
p.level = 1

# 7. Base + Correction
slide = prs.slides.add_slide(prs.slide_layouts[1])
title, body = slide.shapes.title, slide.placeholders[1]
title.text = "机制包装：1-Step Denoise + Energy Correction"
tf = body.text_frame
tf.text = "基于验证集的 L2 Error 极小差距，我们主打：直接利用精准的 1-Step Base 预测。"
p = tf.add_paragraph()
p.text = "迭代式能量重构机制 (Refine & Correct)："
p.level = 0
p = tf.add_paragraph()
p.text = "不需要传统 10步 极高的耗时，拿到单步 Clean 轨迹后，挂载三个独立连续能量头。"
p.level = 1
p = tf.add_paragraph()
p.text = "E_collision (防撞) / E_offroad (防越界) / E_target (导航偏离)"
p.level = 2
p = tf.add_paragraph()
p.text = "直接抽取能量梯度，对生成轨迹做精准精调修形保证物理安全。"
p.level = 1

# 8. Future Works
slide = prs.slides.add_slide(prs.slide_layouts[1])
title, body = slide.shapes.title, slide.placeholders[1]
title.text = "4. 未来展望 (Future Works)"
tf = body.text_frame
tf.text = "解耦条件控制 (Compositional / Decomposed Conditioning)："
p = tf.add_paragraph()
p.text = "语言模型提供 Text-Conditioned BEV 指令作为语义透镜。"
p.level = 1
p = tf.add_paragraph()
p.text = "差分 Delta 决策树式的局部能量剪枝："
p.level = 0
p = tf.add_paragraph()
p.text = "从大颗粒度绝对轨迹，转入高精细的 Delta 差分空间进行干预。"
p.level = 1
p = tf.add_paragraph()
p.text = "让能量在每一次推演 (per-step) 时像决策树剪枝一样，做局部精调和短距代价评估。"
p.level = 1

# 生成保存文件
out_path = "/media/z/data/mzq/others/MoT-DP/docs/Presentation_EnergyGuidedDiffusion.pptx"
prs.save(out_path)
print(f"PPT successfully generated at {out_path}")
