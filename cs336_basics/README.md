# cs336_basics 使用手册

本文档覆盖完整流水线：**模块概览 → 数据准备 → 训练 → 文本生成**。

---

## 0) 模块结构概览

```
cs336_basics/
├── __init__.py
├── tokenizer.py          # BPE Tokenizer（encode / decode / from_files）
├── bpe.py                # BPE 训练算法（train_bpe, pre_tokenization）
├── prepare_tokens.py     # CLI：原始文本 → uint16 .bin token 文件
├── train.py              # CLI：模型训练主循环（含 W&B 日志）
├── generate.py           # generate_completion：自回归解码核心函数
├── generate_text.py      # CLI：加载 checkpoint + tokenizer 生成文本
├── transformer.py        # TransformerLM 模型定义
├── attention.py          # 多头自注意力（Multi-Head Self-Attention）
├── rope.py               # 旋转位置编码（Rotary Positional Embedding）
├── rmsnorm.py            # RMSNorm 层归一化
├── swiglu.py             # SwiGLU 前馈网络
├── embedding.py          # Token + 位置嵌入
├── linear.py             # 自定义 Linear 层
├── softmax.py            # Softmax 实现
├── cross_entropy.py      # 交叉熵损失
├── adamw.py              # AdamW 优化器
├── lr_scheduler.py       # 余弦退火学习率调度（warmup + cosine）
├── gradient_clipping.py  # 梯度裁剪
├── get_batch.py          # 从 memmap 采样 batch
├── checkpointing.py      # Checkpoint 保存 / 加载
└── pretokenization_example.py  # 预分词示例（参考用）
```

端到端流水线如下：

```
原始文本 ──[prepare_tokens.py]──► .bin token 文件
                                       │
                                       ▼
                              ──[train.py]──► checkpoint (.pt)
                                       │
                                       ▼
                           ──[generate_text.py]──► 生成文本
```

---

## 1) 训练前准备：先把文本转成 `.bin`

`train.py` 不会在线训练 BPE，也不会在线分词。它只读取二进制 token 文件：

- `np.memmap(..., dtype=np.uint16)`（每个 token 2 字节）
- 因此 token id 必须在 `[0, 65535]`

请先使用 `cs336_basics/prepare_tokens.py` 完成：

1. 用训练集文本训练 BPE（`train_bpe`）
2. 用训练出的 tokenizer 编码 train/valid 文本
3. 写出 `uint16` 二进制文件给 `train.py`

### 1.1 生成 token 二进制

```bash
uv run python cs336_basics/prepare_tokens.py \
  --train_text data/TinyStoriesV2-GPT4-train.txt \
  --valid_text data/TinyStoriesV2-GPT4-valid.txt \
  --train_bin_out data/tinystories_train_tokens.bin \
  --valid_bin_out data/tinystories_valid_tokens.bin \
  --vocab_size 10000 \
  --special_tokens "<|endoftext|>" \
  --num_processes 8 \
  --tokenizer_prefix data/tinystories_bpe
```

上面命令还会额外保存：

- `data/tinystories_bpe_vocab.json`
- `data/tinystories_bpe_merges.json`

用于复现 tokenizer 配置（生成文本时需要加载这两个文件）。

### 1.2 `prepare_tokens.py` 参数说明

| 参数 | 说明 |
|------|------|
| `--train_text` | BPE 训练文本路径（建议使用训练集） |
| `--valid_text` | 验证文本路径 |
| `--train_bin_out` | 训练集 token 输出 `.bin` |
| `--valid_bin_out` | 验证集 token 输出 `.bin` |
| `--vocab_size` | 最终词表大小（必须 > 0） |
| `--special_tokens` | 特殊 token 列表 |
| `--num_processes` | BPE 预分词并行进程数（建议 ≤ CPU 核心数，内存不足时调低） |
| `--imap_chunksize` | multiprocessing imap chunksize（默认 1） |
| `--tokenizer_prefix` | 若提供，保存 `*_vocab.json` 与 `*_merges.json` |
| `--flush_size` | 编码时内存缓冲大小（默认 1,000,000） |

### 1.3 与 `train.py` 的一致性约束

- `train.py --vocab_size` 必须与 `prepare_tokens.py --vocab_size` 一致。
- `train.py --train_data/--val_data` 必须指向 `prepare_tokens.py` 的 `.bin` 产物。
- 如果你更换了 tokenizer（词表或 merges），必须重新生成 `.bin`。

---

## 2) 如何启动训练

最小示例（在先生成 `.bin` 之后）：

```bash
uv run python cs336_basics/train.py \
  --train_data data/tinystories_train_tokens.bin \
  --val_data data/tinystories_valid_tokens.bin
```

带 W&B 的示例：

```bash
uv run python cs336_basics/train.py \
  --train_data data/tinystories_train_tokens.bin \
  --val_data data/tinystories_valid_tokens.bin \
  --use_wandb \
  --wandb_run_name tinystories-baseline
```

首次使用 W&B 需先登录：

```bash
uv run wandb login
```

粘贴 API key 即可（支持 `wandb_v1_...` 格式的新 token）。如果提示 key 长度不对，先升级：`uv pip install --upgrade wandb`。

---

## 3) `train.py` 训练流程

`cs336_basics/train.py` 的主流程是：

1. 解析命令行参数，补默认值（`device` 自动选择，`cosine_cycle_iters` 默认等于 `max_iters`）。
2. （可选）初始化 W&B run，并写入整套 `config`。
3. 用 `np.memmap` 打开训练/验证 token 文件（`uint16`）。
4. 构建 `TransformerLM`。
5. 构建 `AdamW` 优化器。
6. （可选）从 checkpoint 恢复（模型 + 优化器 + 起始步数）。
7. 进入训练循环：
   - 计算当前步学习率（warmup + cosine schedule）
   - 采样 batch（`get_batch`）
   - 前向、交叉熵 loss、反向传播、梯度裁剪、优化器更新
   - 按 `log_interval` 打印/记录训练指标
   - 按 `eval_interval` 计算验证损失
   - 按 `checkpoint_interval` 保存 checkpoint
8. 训练结束后保存最终 checkpoint，并结束 W&B run。

---

## 4) 参数详解（含约束）

### 4.1 模型参数

| 参数 | 默认值 | 说明 | 约束 |
|------|--------|------|------|
| `--vocab_size` | 10000 | 词表大小，决定 embedding 和输出头维度 | 必须与 token 数据生成时的 tokenizer 一致 |
| `--context_length` | 256 | 每个样本的序列长度 | 越大显存占用越高 |
| `--d_model` | 512 | 隐藏维度 | — |
| `--num_layers` | 6 | Transformer block 数量 | — |
| `--num_heads` | 8 | 注意力头数 | `d_model % num_heads == 0` |
| `--d_ff` | 1024 | 前馈层维度 | 建议 `~(8/3)*d_model` 且为 64 的倍数 |
| `--rope_theta` | 10000.0 | RoPE 位置编码的 theta | — |

### 4.2 优化器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 1e-3 | 最大学习率（schedule 顶点） |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--beta1` | 0.9 | Adam 一阶矩衰减 |
| `--beta2` | 0.999 | Adam 二阶矩衰减 |
| `--eps` | 1e-8 | Adam epsilon |
| `--max_grad_norm` | 1.0 | 梯度裁剪阈值 |

### 4.3 学习率调度参数

| 参数 | 默认值 | 说明 | 约束 |
|------|--------|------|------|
| `--warmup_iters` | 100 | 线性 warmup 步数 | — |
| `--cosine_cycle_iters` | None → `max_iters` | 余弦衰减结束步数 | 建议 > `warmup_iters` |
| `--min_lr_ratio` | 0.1 | 最小学习率比例，`min_lr = lr * min_lr_ratio` | 建议 `(0, 1]` |

### 4.4 数据与训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_data` | **必填** | 训练集 `.bin` 路径 |
| `--val_data` | **必填** | 验证集 `.bin` 路径 |
| `--batch_size` | 32 | 每步 batch 中的序列数 |
| `--max_iters` | 10000 | 总训练步数 |
| `--device` | None | None 时自动选择 `cuda`/`cpu` |

> **总处理 token 数** = `batch_size × max_iters × context_length`。experiments.md 中参考值为 327,680,000。

### 4.5 日志与 checkpoint 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--log_interval` | 100 | 训练日志打印间隔 |
| `--eval_interval` | 500 | 验证间隔 |
| `--eval_batches` | 20 | 每次验证采样 batch 数 |
| `--checkpoint_interval` | 1000 | 保存间隔 |
| `--checkpoint_dir` | `checkpoints` | checkpoint 目录 |
| `--resume` | None | 恢复训练的 checkpoint 路径 |

### 4.6 W&B 参数

| 参数 | 说明 |
|------|------|
| `--use_wandb` | 开启 W&B 日志 |
| `--wandb_run_name` | 设置 run 名（便于分组扫描/消融） |

W&B 记录的指标：`train/loss`、`eval/val_loss`、`optimizer/lr`、`system/elapsed_time_sec`、`system/iteration_time_ms`、`data/tokens_seen`。

---

## 5) 文本生成

训练完成后，使用 `generate_text.py` 从 checkpoint 生成文本。

### 5.1 基本用法

```bash
uv run python cs336_basics/generate_text.py \
  --checkpoint checkpoints/local_baseline/checkpoint_40000.pt \
  --vocab data/tinystories_bpe_vocab.json \
  --merges data/tinystories_bpe_merges.json \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --top_p 0.9 \
  --device cuda
```

模型参数必须与训练时一致，否则加载权重会报错。例如训练时用了小模型配置，需要加上：

```bash
  --vocab_size 10000 \
  --context_length 128 \
  --d_model 384 \
  --num_layers 4 \
  --num_heads 6 \
  --d_ff 1024
```

### 5.2 `generate_text.py` 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | **必填** | `.pt` checkpoint 文件路径 |
| `--vocab` | **必填** | vocab JSON 路径（`prepare_tokens.py` 产物） |
| `--merges` | **必填** | merges JSON 路径（`prepare_tokens.py` 产物） |
| `--prompt` | `"Once upon a time"` | 起始文本 |
| `--max_new_tokens` | 256 | 最多生成的新 token 数 |
| `--temperature` | 0.8 | 温度：越高越随机，越低越确定；0 = 贪心 |
| `--top_p` | 0.9 | 核采样阈值 `(0,1]`；1.0 = 关闭 top-p |
| `--device` | 自动 | `cuda` / `cpu` |
| `--output` | None | 可选，将生成文本保存到文件 |
| `--special_tokens` | `["<\|endoftext\|>"]` | 特殊 token 列表 |
| 模型参数 | 同 train.py | `--vocab_size`、`--context_length`、`--d_model`、`--num_layers`、`--num_heads`、`--d_ff`、`--rope_theta` |

### 5.3 temperature 与 top_p 对生成质量的影响

| 组合 | 效果 |
|------|------|
| `t=0.7, p=0.9` | 保守，重复性高，连贯性好 |
| `t=0.8, p=0.9` | 平衡，推荐作为默认 |
| `t=0.9, p=0.95` | 较有创意，偶尔偏题 |
| `t=1.0, p=0.95` | 高随机性，多样但可能不通顺 |
| `t=0 (贪心)` | 完全确定，会重复 |

### 5.4 批量生成示例

```bash
for prompt in "Once upon a time" "There was a little girl named"; do
  for t in 0.7 0.8 0.9 1.0; do
    for p in 0.9 0.95; do
      slug=$(echo "$prompt" | tr ' ' '_')
      uv run python cs336_basics/generate_text.py \
        --checkpoint checkpoints/local_baseline/checkpoint_40000.pt \
        --vocab data/tinystories_bpe_vocab.json \
        --merges data/tinystories_bpe_merges.json \
        --prompt "$prompt" \
        --temperature $t --top_p $p \
        --vocab_size 10000 --context_length 128 \
        --d_model 384 --num_layers 4 --num_heads 6 --d_ff 1024 \
        --device cuda \
        --output "outputs/${slug}_t${t}_p${p}.txt"
    done
  done
done
```

---

## 6) 推荐起步配置（TinyStories）

用于先跑通、再调参：

```bash
uv run python cs336_basics/train.py \
  --train_data data/tinystories_train_tokens.bin \
  --val_data data/tinystories_valid_tokens.bin \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000 \
  --batch_size 32 \
  --max_iters 5000 \
  --lr 1e-3 \
  --warmup_iters 200 \
  --min_lr_ratio 0.1 \
  --use_wandb \
  --wandb_run_name tinystories-baseline
```

如果显存不足，先按顺序减小：`batch_size` → `context_length` → `d_model`。

---

## 7) 推荐实验顺序：先本机调参，再上 H100

为减少 H100 租用成本，建议按以下顺序进行：

### 阶段 A：本机快速跑通（优先 CUDA）

- 目标：验证训练链路正确、日志完整、loss 曲线正常下降。
- 建议配置（低成本）：
  - `context_length=128`
  - `d_model=384`
  - `num_layers=4`
  - `num_heads=6`
  - `d_ff=1024`
  - `batch_size=8`（显存不足可降到 4）

### 阶段 B：本机学习率扫描

- 固定模型结构与数据，只改变 `--lr`。
- 建议扫描：`3e-4, 5e-4, 8e-4, 1e-3, 1.3e-3, 1.6e-3`。
- 记录每组的：
  - 最终 `val_loss`
  - 收敛速度（同样步数下 loss 下降快慢）
  - 是否发散（loss 爆炸或 NaN）

### 阶段 C：迁移到 H100 跑正式大配置

- 本机确定稳定学习率区间后，再上 H100。
- 使用课程建议的大配置（`context_length=256`, `d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`）和更大 token 预算跑正式实验。

这样做的好处是：把"实现错误和明显不稳定超参"尽量留在本机排掉，减少昂贵算力上的试错成本。
