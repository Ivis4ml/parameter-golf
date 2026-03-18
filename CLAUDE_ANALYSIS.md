# Parameter Golf 深度分析

## 挑战本质

这不是一个"训练最大模型"的比赛，而是一个**信息压缩**问题。

目标：用不超过 16MB（代码 + 压缩权重）的预算，最大化对 FineWeb 验证集的压缩率。

$$\text{BPB} = \frac{\text{val\_loss (nats/token)}}{\ln 2} \times \frac{\text{tokens}}{\text{bytes}}$$

这个公式说明，提升 BPB 有两条路：
1. **降低 loss**（更好的模型）
2. **提高 tokens/bytes**（更高效的 tokenizer，即更大词表）

但两条路互相制约：更大词表 → embedding 更大 → 占用更多 16MB 预算 → 留给其他参数的空间更少。这就是为什么 baseline 选择了极小词表（1024），是一个设计上的战略选择，不是随意的。

---

## Baseline 设计决策逐一解析

### 1. 词表 1024 — 极端的参数节省

GPT-2 词表是 50257，标准 LLaMA 是 32000。1024 是极端缩减。

**代价**：每个 token 平均只覆盖 ~1.5 字节（BPE merge 很少），需要更多 token 才能表达同样的文本。

**收益**：embedding 矩阵只有 `1024 × 512 = 0.5M` 参数，而 GPT-2 词表下是 `50257 × 512 ≈ 25.7M`。在 16MB 预算里，这省下了巨大空间。

**Tied Embedding 的乘数效应**：输入 embedding 和 lm_head 共享权重（转置），等于把这 0.5M 参数用了两次，是免费的对称性约束。小词表时 tied embedding 几乎没有质量损失，因为 embedding 和 lm_head 本来就应该编码同样的语义空间。

### 2. 9层 U-Net — 不是普通 Transformer

```
Encoder: Block0 → Block1 → Block2 → Block3 → Block4
                  ↓         ↓         ↓         ↓
Decoder:                   Block8 ← Block7 ← Block6 ← Block5
                (skip connections, reversed)
```

每个 decoder block 在处理前先加入对应 encoder block 的输出（乘以可学习的 skip_weight）。

**为什么这有效**：
- 允许低层特征直接跳过中间层到达输出端，类似 ResNet 中的恒等映射
- 对于层数少（只有 9 层）的模型，减少了梯度消失
- encoder 层专注于特征提取，decoder 层专注于特征综合

**与标准 Transformer 的对比**：标准 Transformer 每层只有局部残差（x + attn(x)）。这里增加了两种额外的跳跃信息流：
- 局部残差（x + output）
- U-Net skip（从对称层直接加）
- 全局锚点 x0（每层都可以参考 embedding 输入，通过 resid_mix）

### 3. resid_mix — 可学习的"遗忘门"

```python
x = resid_mix[0] * x + resid_mix[1] * x0
```

每层都会与原始 embedding 输出 x0 做加权混合。初始化为 `[1, 0]`（完全保留 x，忽略 x0），但训练后模型可以学会在某些层更多地参考原始 token 信息。

这本质上是 highway network / LSTM 遗忘门的变体，让深层网络保持对原始输入的访问通道。

### 4. QK-RMSNorm — 稳定 attention 的低成本方案

在做点积之前对 Q 和 K 各做一次 RMSNorm，然后乘以可学习的 `q_gain`（初始 1.5）。

**作用**：
- 防止 QK 内积值过大导致 softmax 饱和（attention 退化成 one-hot）
- 无需调 softmax temperature，模型自己学 gain
- 比 LayerNorm 轻（无 bias，无仿射参数），比不做 norm 稳定很多

### 5. Muon 优化器 — 矩阵参数的 Riemannian 梯度下降

Adam 把梯度每个元素独立处理，忽略了矩阵的结构。Muon 的思想是：

> 矩阵参数的自然更新方向应该保持"近似正交"，而不是随 Adam 任意缩放。

Newton-Schulz 迭代（5步）将梯度矩阵正交化：

```python
for _ in range(5):
    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X
```

最终更新矩阵的奇异值被"压平"到接近 1，相当于在矩阵流形上做了一步更新，而不是在欧式空间里。对矩阵参数（Q/K/V/proj/MLP 权重）比 Adam 收敛更快、泛化更好。

**为什么只对矩阵用 Muon**：embedding 向量、bias、norm 参数等 1D 参数不是方阵，正交化无意义，所以这些继续用 Adam。

### 6. 量化导出 — int8 per-row + zlib

训练用 bf16/fp32 保持精度，但提交时：

1. **2D 矩阵**：per-row int8（每行找 99.99984th 百分位截断，然后量化到 [-127, 127]）
   - per-row 比 per-tensor 精度高得多，因为不同输出通道的值域差异很大
2. **1D 向量/标量**：per-tensor int8
3. **小张量（≤65536 元素）和控制参数**：fp16 直接保留（量化收益不大）
4. 全部序列化后 `zlib.compress(level=9)`

baseline 量化损失：BPB 从 1.2172 → 1.2244（+0.0072）。这个损失直接计入最终分数，所以**量化感知训练（QAT）** 是一个重要的优化方向。

---

## 参数预算分析

baseline 参数分布（9层 × 512维，vocab=1024）：

| 组件 | 参数量 | 说明 |
|------|--------|------|
| Embedding | 1024 × 512 = 524,288 | tied，兼作 lm_head |
| 每层 Q proj | 512 × 512 = 262,144 | |
| 每层 K proj | 256 × 512 = 131,072 | 4 KV heads |
| 每层 V proj | 256 × 512 = 131,072 | |
| 每层 O proj | 512 × 512 = 262,144 | |
| 每层 MLP fc | 512 × 1024 = 524,288 | 2× expansion |
| 每层 MLP proj | 1024 × 512 = 524,288 | |
| 每层控制参数 | ~3072 | attn_scale, mlp_scale, resid_mix |
| skip_weights | 4 × 512 = 2048 | encoder 层数 = 4 |

每层约 1.84M 参数，9层约 16.6M，加 embedding 约 17.1M 总参数。量化到 int8 后约 17MB，zlib 压缩到约 15.8MB。

**关键约束**：int8 + zlib 的压缩率大约是 `原始 bf16 大小 / 4`。这意味着 16MB 对应约 64MB bf16 参数，即约 32M float32 参数上限（实际更少，因为 zlib 对随机数据压缩率有限）。

---

## 潜在改进方向

### 方向一：更好的参数利用率

**depth recurrence（层复用）**：用 N 层的参数跑 2N 甚至更多次 forward，类似 Universal Transformer。参数量减半，但每个参数被多次利用。缺点：对 RoPE 等位置相关模块需要特殊处理；梯度流需要小心。

**低秩分解**：把 `512×512` 的矩阵分解为 `512×r + r×512`（r=64），参数量从 262K 降到 65K，再用 Muon 训练。类似 LoRA 但从头训练。在预算紧张时可能值得。

### 方向二：更好的 tokenizer

理论上增大词表可以提高 tokens/bytes 比，改善 BPB 分子以外的部分。但词表从 1024 增大到 4096 时，embedding 从 0.5M 增加到 2M 参数，占用 2MB 预算，需要从其他地方削减。

关键问题：更大词表带来的 BPB 收益能否覆盖因削减其他参数而增加的 loss？这需要实验验证。

**字节级 tokenizer**：词表只有 256，embedding 极小，但 tokens/bytes ≈ 1，BPB 等于 bits/token，对模型建模能力要求极高。可能需要更深/更宽的网络。

### 方向三：量化感知训练（QAT）

当前流程：bf16 训练 → 训后 int8 量化，量化误差 ~0.007 BPB。

QAT 在训练时模拟量化：
```python
# 前向时用量化权重计算
w_fake_quant = dequantize(quantize(w))
output = x @ w_fake_quant.T
```
梯度通过 STE（straight-through estimator）反传到原始 fp32 权重。最终模型的权重本身就适应了 int8 精度，量化损失大幅缩小。

### 方向四：更好的压缩

zlib 是通用压缩，不知道这是神经网络权重。针对性的压缩方法：

**ANS/算术编码**：如果权重的分布已知（如训练时强制 Laplace 分布），可以用接近理论熵的算术编码，比 zlib 更高效。

**权重剪枝 + 稀疏编码**：把小权重置零，然后用稀疏格式存储。稀疏矩阵的 zlib 压缩率比稠密矩阵高得多。

### 方向五：架构创新

**Mixture of Experts（MoE）**：参数总量大（提升模型容量），但每次 forward 只激活一部分（不影响计算）。关键是 MoE 的参数在压缩后会怎样——如果不同 expert 的权重相关性高，压缩效果好。

**线性注意力 / Mamba**：标准 attention 在长序列时计算 $O(n^2)$，对 1024 seq_len 影响不大，但架构本身是否有更好的参数效率是个开放问题。

**测试时计算（Test-Time Training）**：在推理时用部分验证集做短暂梯度更新。这会计入 10 分钟限制，但可能值得。

---

## 竞赛竞争格局分析

当前状态（2026-03-18）：只有 baseline（1.2244）和一个 4 小时 unlimited run（1.2074）。挑战刚开始。

**4小时 run 的意义**：用同样架构训练更多步，BPB 从 1.2244 降到 1.2074（-0.017）。说明 baseline 在 10 分钟内训练严重不足——13780 步就停了，理论上可以训到 20000 步以上。这意味着：

1. 更快的单步训练（系统优化：flash attention 算子、CUDA kernel）可以在 10 分钟内跑更多步
2. 更大的 batch size 可能提高每步的信息量（但受显存限制）
3. 更好的 LR 调度可以在有限步数内收敛更快

**低垂果实**：在不改变架构的情况下，优化训练效率（更快的步速）是最直接的改进路径，因为 4 小时 run 证明了更多步数确实有效。

---

## 代码约束

`train_gpt.py` 和 `train_gpt_mlx.py` 硬性上限 **1500 行**。竞赛提交放在 `/records/` 下，可以有自己的 `train_gpt.py`，不受根目录的修改限制。

提交要求 BPB 改善 ≥ 0.005，并需多次运行的日志证明 `p < 0.01`。
