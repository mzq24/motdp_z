# 架构与开发说明书：Compositional Energy-Guided Diffusion Policy 

## 1. 核心思想 (Core Concept)
将传统的条件扩散策略（Conditional Diffusion Policy）升级为**可组合（Compositional）、可控（Controllable）的轨迹生成框架**。

* **当前痛点：** 传统的 Classifier-Free Guidance (CFG) 只能进行隐式的分布匹配，难以应对分布外（OOD）的硬性物理约束（如碰撞），且高度依赖上游感知的绝对准确性。
* **解决方案：** 引入 **Multi-head Energy Decomposition (多头能量分解)**。在扩散模型学习专家轨迹先验的基础上，显式地附加一组语义独立的能量头 $\{E_i(\tau)\}$，每个头对应一个具体的驾驶约束或偏好。通过联合优化（Joint Optimization），使模型在生成时不仅符合先验，还能主动向低能量（安全/合理）状态演化。

---

## 2. 核心架构与数学建模 (Architecture & Math Formulation)

### 2.1 基础生成模型 (Base Diffusion Model)
保持原有的 Diffusion Model 结构不变，用于学习条件轨迹的先验分布：
$$\epsilon_\theta(x_t, t, c_{\text{state}})$$
* $x_t$: $t$ 时刻的带噪轨迹。
* $c_{\text{state}}$: 环境状态条件（如 BEV 特征图、历史轨迹等）。

### 2.2 多头能量函数 (Multi-head Energy Functions)
定义 $N$ 个独立的能量评估网络 $E_i(\tau)$。为了保证梯度在轨迹空间平滑且可优化，**能量输出必须被建模为连续函数**（Regression 或 Risk Score），而非离散的二元分类。
* $E_1(\tau)$: 前向碰撞风险 (Forward Collision Risk)
* $E_2(\tau)$: 行人碰撞风险 (Pedestrian Collision Risk)
* $E_3(\tau)$: 车道偏离程度 (Lane Departure Penalty)

### 2.3 组合能量与动态权重 (Compositional Energy & Dynamic Weights)
整体能量 $E(\tau)$ 被建模为各能量头的加权组合：
$$E(\tau) = \sum_{i=1}^{N} w_i E_i(\tau)$$
* **权重的动态性：** 权重 $w_i$ 代表当前场景下该约束的优先级。这一层是接入 LLM 的核心接口（LLM 充当 Router 输出权重向量 $\mathbf{w}$），从而实现细粒度的风格和行为控制。

---

## 3. 训练与推理机制 (Training & Inference Mechanisms)

### 3.1 联合训练阶段 (Joint Optimization during Training)
为缓解 Train-test Mismatch，要求扩散模型和能量头协同演化。利用连续的 risk score 标签对能量头进行监督。

**联合损失函数 (Joint Loss)：**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \alpha \sum_{i=1}^{N} \mathcal{L}_{\text{energy\_}i} + \beta \mathcal{L}_{\text{alignment}}$$
* $\mathcal{L}_{\text{diffusion}}$: 标准的 Denoising MSE Loss，即 $||\epsilon - \epsilon_\theta(x_t, t, c)||^2$。
* $\mathcal{L}_{\text{energy\_}i}$: 各能量头本身的回归损失（预测 score 与 GT score 的 MSE）。
* $\mathcal{L}_{\text{alignment}}$: 对齐损失。将能量头作用于预测的去噪干净轨迹 $\hat{x}_0$ 上，利用 $\nabla_{\hat{x}_0} E(\hat{x}_0)$ 引导生成模型偏向低能量区域。

### 3.2 推理采样阶段 (Inference Phase)
采样时，将多头能量梯度显式注入到每一步的去噪更新中（Annealed Energy Guidance）。

在每一步 $t \to t-1$ 的更新中（概念公式）：
1. 计算组合能量相对于当前状态的梯度：
   $$\nabla_{x_t} E(x_t) = \sum_{i=1}^{N} w_i \nabla_{x_t} E_i(x_t)$$
2. 结合能量梯度更新状态：
   $$x_{t-1} = \text{DDIM\_Step}(x_t, \hat{\epsilon}_\theta) - \gamma_t \nabla_{x_t} E(x_t)$$
*(注：$\gamma_t$ 为退火系数，随时间步 $t$ 调整引导强度。)*

---

## 4. 给 Coding Agent 的实现建议 (Implementation Notes)

1. **解耦设计 (Modularity)：** 能量头网络 $E_i$ 的设计必须保持语义独立，尽量减少不同 Head 之间的特征耦合干扰（可使用独立的轻量级 MLP）。
2. **能量冲突处理：** 在实现加权和时，注意处理“硬约束”和“软约束”的冲突。若线性叠加导致死锁，可对硬约束引入非线性的 Barrier Function（如 Log-Sum-Exp 聚合）。
3. **梯度截断与稳定 (Gradient Clipping)：** 推理循环中注入 $\nabla_{x_t} E(x_t)$ 时，必须加入严格的梯度裁剪，防止能量梯度过大导致轨迹出现奇异值。
4. **外部接口预留：** 在 `inference.py` 中，请显式预留接收动态权重向量 $\mathbf{w} = [w_1, w_2, ..., w_N]$ 的参数接口，以便后续无缝接入大语言模型 (LLM)。