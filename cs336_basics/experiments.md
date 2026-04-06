# 7 实验 (Experiments)

现在是时候将所有内容整合在一起，并在预训练数据集上训练（小型）语言模型了。

## 7.1 如何运行实验与交付物

理解 Transformer 架构组件背后原理的最佳方法是亲自对其进行修改并运行。实践经验是不可替代的。

为此，能够**快速、一致地进行实验并记录你所做的工作**至关重要。为了快速进行实验，我们将在一个小规模模型（1700万参数）和简单数据集（TinyStories）上运行大量实验。为了保持一致性，你将系统地消融（ablate）组件并更改超参数；为了保存记录，我们将要求你提交实验日志以及与每个实验相关的学习曲线。

为了能够提交损失曲线，**请务必定期评估验证集损失，并记录步数（steps）和挂钟时间（wallclock times）**。你可能会发现使用 Weights and Biases 等日志记录基础设施非常有帮助。

> **问题 (experiment_log)：实验日志记录（3分）**
> 
> 为你的训练和评估代码创建实验追踪基础设施，使其能够根据梯度步数和挂钟时间追踪你的实验和损失曲线。
> 
> **交付物**：用于你实验的日志记录基础设施代码，以及一份涵盖本节以下作业问题的实验日志（记录你尝试过的所有内容的文档）。

## 7.2 TinyStories 数据集

我们将从一个非常简单的数据集（TinyStories；Eldan 和 Li，2023）开始，模型在上面会训练得很快，并且我们可以观察到一些有趣的行为。获取该数据集的说明在第 1 节中。该数据集的示例如下所示。

> **示例 (tinystories_example)：TinyStories 的一个示例**
> 
> 很久很久以前，有一个叫本（Ben）的小男孩。本喜欢探索他周围的世界。他看到了许多令人惊奇的东西，比如商店里陈列的美丽花瓶。有一天，本在商店里走着，偶然发现了一个非常特别的花瓶。当本看到它时，他惊呆了！他说：“哇，那真是一个非常神奇的花瓶！我能买下它吗？”店主微笑着说：“当然可以。你可以把它带回家，向你所有的朋友展示它是多么神奇！”于是本把花瓶带回了家，他为它感到非常自豪！他把朋友们叫来，向他们展示了这个神奇的花瓶。他所有的朋友都认为这个花瓶很漂亮，都不敢相信本有多么幸运。就这样，本在商店里找到了一个神奇的花瓶！

**超参数调优** 我们将告诉你一些非常基础的初始超参数，并要求你找出其他效果良好的设置。

*   **vocab_size** 10000。典型的词表大小在数万到数十万之间。你应该改变这个值，观察词表和模型行为如何变化。
*   **context_length** 256。像 TinyStories 这样简单的数据集可能不需要很长的序列长度，但对于后面的 OpenWebText 数据，你可能想要改变这个值。尝试改变它，并观察其对每次迭代运行时间和最终困惑度（perplexity）的影响。
*   **d_model** 512。这比许多小型 Transformer 论文中使用的 768 维略小，但这会使运行速度更快。
*   **d_ff** 1344。这大约是 $\frac{8}{3} d_{\text{model}}$，同时是 64 的倍数，这有利于 GPU 性能。
*   **RoPE theta parameter $\Theta$** 10000。
*   **number of layers and heads**（层数和头数） 4 层，16 头。结合起来，这会产生大约 1700 万个非嵌入（non-embedding）参数，这是一个相当小的 Transformer。
*   **total tokens processed**（处理的 token 总数） 327,680,000（你的 batch size × total step count × context length 应该大约等于这个值）。

你应该进行一些反复试验（trial and error），为以下其他超参数找到良好的默认值：**学习率（learning rate）、学习率预热（learning rate warmup）、其他 AdamW 超参数（$\beta_1, \beta_2, \epsilon$）以及权重衰减（weight decay）**。你可以在 Kingma 和 Ba [2015] 中找到此类超参数的一些典型选择。

**整合起来** 现在，你可以通过获取训练好的 BPE 分词器（tokenizer）、对训练数据集进行分词，并在你编写的训练循环中运行它，将所有内容整合在一起。**重要提示**：如果你的实现是正确且高效的，上述超参数在 1 张 H100 GPU 上应需要大约 30-40 分钟的运行时间。如果你的运行时间长得多，请检查并确保你的数据加载、检查点保存或验证集损失计算代码没有成为运行时间的瓶颈，并且你的实现进行了正确的批处理（batched）。

**调试模型架构的提示与技巧** 我们强烈建议你熟练使用 IDE 的内置调试器（例如 VSCode/PyCharm），与使用 print 语句调试相比，这将节省你的时间。如果你使用文本编辑器，可以使用类似 `pdb` 的工具。调试模型架构时，其他一些良好的实践包括：
*   开发任何神经网络架构时，常见的第一步是过拟合到单个微批次（minibatch）。如果你的实现是正确的，你应该能够迅速将训练损失降至接近于零。
*   在模型的各个组件中设置调试断点，并检查中间张量（tensors）的形状，确保它们符合你的预期。
*   监控激活值、模型权重和梯度的范数（norms），以确保它们没有爆炸或消失。

> **问题 (learning_rate)：微调学习率（3分）（约需 4 个 H100 小时）**
> 
> 学习率是需要调整的最重要的超参数之一。基于你训练的基线模型，回答以下问题：
> 
> (a) 对学习率执行超参数扫描，并报告最终的损失值（如果优化器发散，请记录发散情况）。
> 
> **交付物**：与多个学习率相关的学习曲线。解释你的超参数搜索策略。
> 
> **交付物**：在 TinyStories 上的验证集损失（每个 token）最高不超过 1.45 的模型。

> **低资源/降级方案提示：在 CPU 或 Apple 芯片上训练少量步数**
> 
> 如果你在 `cpu` 或 `mps` 上运行，则应将处理的 token 总数减少到 40,000,000，这足以生成相当流畅的文本。你也可以将目标验证集损失从 1.45 增加到 2.00。
> 
> 在配备 M3 Max 芯片和 36 GB RAM 的设备上运行我们调整好学习率的解决方案代码，我们使用 batch size × total step count × context length = 32 × 5000 × 256 = 40,960,000 个 token，在 `cpu` 上耗时 1 小时 22 分钟，在 `mps` 上耗时 36 分钟。在第 5000 步时，我们达到了 1.80 的验证集损失。
> 
> 一些额外提示：
> *   当使用 $X$ 个训练步数时，我们建议调整余弦学习率衰减计划，使其精确地在第 $X$ 步终止衰减（即达到最小学习率）。
> *   当使用 `mps` 时，**请勿**使用 TF32 核心，即**不要**像在 `cuda` 设备上那样设置 `torch.set_float32_matmul_precision('high')`。我们尝试了在 `mps` 下启用 TF32 核心（torch 版本 2.6.0），发现后端会静默使用损坏的核心，从而导致训练不稳定。
> *   你可以通过使用 `torch.compile` 对模型进行 JIT 编译来加速训练。具体而言：
>     *   在 `cpu` 上，使用 `model = torch.compile(model)` 编译你的模型
>     *   在 `mps` 上，你可以使用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化反向传播。
>     截至 torch 版本 2.6.0，`mps` 尚不支持使用 Inductor 进行编译。

(b) 经验法则（Folk wisdom）认为，最好的学习率处于“稳定性的边缘（edge of stability）”。调查学习率发散的临界点与你最佳学习率之间的关系。

**交付物**：随着学习率增加的学习曲线（需包括至少一次发散的运行），以及关于这与收敛速度如何相关的分析。

现在，让我们改变批大小（batch size），看看训练会发生什么。批大小很重要——它们允许我们通过执行更大的矩阵乘法从 GPU 获得更高的效率，但我们真的总是希望批大小越大越好吗？让我们运行一些实验来找出答案。

> **问题 (batch_size_experiment)：批大小的变化（1分）（约需 2 个 H100 小时）**
> 
> 将你的批大小从 1 逐渐改变，直到达到 GPU 显存限制。尝试中间的至少几个批大小，包括 64 和 128 等典型大小。
> 
> **交付物**：具有不同批大小的运行对应的学习曲线。如果有必要，应再次优化学习率。
> 
> **交付物**：用几句话讨论你关于批大小及其对训练影响的发现。

有了解码器在手，我们现在可以生成文本了！我们将使用该模型生成文本，并看看效果如何。作为参考，你生成的输出应该至少和下面的示例一样好。

> **示例 (ts_generate_example)：TinyStories 语言模型的样本输出**
> 
> Once upon a time,there was a pretty girl named Lily. She loved to eat gum,especially the big black one. Oneday,Lily’s mom asked her to help cook dinner. Lily was so excited! She loved to help her mom.Lily’s mom made a big pot of soup for dinner. Lily was so happy and said, “Thank you, Mommy! I love you.”She helped her mom pour the soup into a big bowl.After dinner, Lily’s mom made some yummy soup.Lily love dit! She said,“Thank you, Mommy! This soup is so yummy!” Her mom smiled and said,“I’m glad you like it, Lily.”They finished cooking and continued to cook together.The end.

> **低资源/降级方案提示：在 CPU 或 Apple 芯片上生成文本**
> 
> 如果你使用的是处理了 4000 万个 token 的低资源配置，你应该会看到生成的文本仍然像英语，但不如上面的流畅。例如，我们在 4000 万个 token 上训练的 TinyStories 语言模型的样本输出如下：
> 
> Once upon a time,there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. Oneday,Sue went for a walk and met a ladybug! They became good friends and played on the path together.
> “Hey, Polly! Let’s go out!” said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!” 
> As Sue watched the sky moved,what it was. She

以下是精确的问题陈述及我们要求的内容：

> **问题 (generate)：生成文本（1分）**
> 
> 使用你的解码器和训练好的检查点（checkpoint），报告你的模型生成的文本。你可能需要调整解码器参数（温度、核采样 top-p 等）以获得流畅的输出。
> 
> **交付物**：至少 256 个 token 的文本输出记录（或者直到出现第一个 `<|endoftext|>` token），以及关于此输出流畅性的简短评论，并列出至少两个影响该输出质量好坏的因素。

## 7.3 消融实验与架构修改

理解 Transformer 的最佳方式是实际修改它并观察其行为。我们现在将进行一些简单的消融实验和修改。

**消融实验 1：层归一化** 人们常说层归一化对于 Transformer 训练的稳定性很重要。但也许我们想冒一下险。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

> **问题 (layer_norm_ablation)：移除 RMSNorm 并训练（1分）（约需 1 个 H100 小时）**
> 
> 从你的 Transformer 中移除所有的 RMSNorm 并进行训练。在之前最优的学习率下会发生什么？你能否通过使用更低的学习率来获得稳定性？
> 
> **交付物**：移除 RMSNorm 并进行训练时的学习曲线，以及对应最佳学习率的学习曲线。
> 
> **交付物**：一段简短的关于 RMSNorm 影响的评论。

现在让我们研究另一个乍看之下似乎是随意的层归一化选择。Pre-norm（前置归一化）Transformer 块的定义为：
$$z = x + \text{MultiHeadedSelfAttention}(\text{RMSNorm}(x))$$
$$y = z + \text{FFN}(\text{RMSNorm}(z))$$

这是对原始 Transformer 架构少数几个达成“共识”的修改之一，原始架构使用的是 post-norm（后置归一化）方法，如下所示：
$$z = \text{RMSNorm}(x + \text{MultiHeadedSelfAttention}(x))$$
$$y = \text{RMSNorm}(z + \text{FFN}(z))$$

让我们恢复到 post-norm 方法，看看会发生什么。

> **问题 (pre_norm_ablation)：实现 post-norm 并训练（1分）（约需 1 个 H100 小时）**
> 
> 将你的 pre-norm Transformer 实现修改为 post-norm。使用 post-norm 模型进行训练，看看会发生什么。
> 
> **交付物**：post-norm Transformer 的学习曲线，并与 pre-norm Transformer 进行比较。

我们看到，层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

**消融实验 2：位置嵌入** 接下来我们将研究位置嵌入对模型性能的影响。具体来说，我们将基础模型（包含 RoPE）与完全不包含位置嵌入（NoPE）的模型进行比较。事实证明，仅含解码器（decoder-only）的 Transformer，即我们实现的具有因果掩码（causal mask）的 Transformer，理论上可以在没有显式提供位置嵌入的情况下推断出相对或绝对位置信息 [Tsai 等，2019，Kazemnejad 等，2023]。我们现在将通过实证来测试 NoPE 与 RoPE 相比性能如何。

> **问题 (no_pos_emb)：实现 NoPE（1分）（约需 1 个 H100 小时）**
> 
> 修改你包含 RoPE 的 Transformer 实现，完全移除位置嵌入信息，看看会发生什么。
> 
> **交付物**：对比 RoPE 和 NoPE 性能的学习曲线。

**消融实验 3：SwiGLU 与 SiLU** 接下来，我们将跟随 Shazeer [2020] 的研究，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络的性能，来测试前馈网络中门控（gating）的重要性：
$$\text{FFN}_{\text{SiLU}}(x) = W_2\text{SiLU}(W_1x)$$

回顾一下，在我们的 SwiGLU 实现中，我们将内部前馈层的维度设置为大约 $d_{\text{ff}} = \frac{8}{3}d_{\text{model}}$（同时确保 $d_{\text{ff}} \bmod 64 = 0$，以便利用 GPU 张量核心）。在你的 $\text{FFN}_{\text{SiLU}}$ 实现中，你应该设置 $d_{\text{ff}} = 4 \times d_{\text{model}}$，以近似匹配 SwiGLU 前馈网络的参数量（SwiGLU 有三个权重矩阵，而不是两个）。

> **问题 (swiglu_ablation)：SwiGLU 与 SiLU（1分）（约需 1 个 H100 小时）**
> 
> **交付物**：在参数量近似匹配的情况下，比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线。
> 
> **交付物**：用几句话讨论你的发现。

> **低资源/降级方案提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改效果**
> 
> 在作业的剩余部分，我们将转向一个更大规模、噪声更多的网络数据集（OpenWebText），尝试架构修改，并（可选地）向课程排行榜提交结果。
> 
> 在 OpenWebText 上将语言模型训练至流畅需要很长时间，因此我们建议获取 GPU 资源有限的在线学生继续在 TinyStories 上测试架构修改（使用验证集损失作为评估性能的指标）。

## 7.4 在 OpenWebText 上运行

我们现在将转向一个更标准的、由网络爬虫数据创建的预训练数据集。我们也提供了一个 OpenWebText [Gokaslan 等，2019] 的小样本作为单个文本文件：如何获取该文件请参见第 1 节。

下面是 OpenWebText 的一个示例。注意该文本是如何变得更加真实、复杂和多样的。你可能想要浏览一下训练数据集，以了解网络抓取语料库的训练数据是什么样子的。

> **示例 (owt_example)：OWT 的一个示例**
> 
> 《棒球招股书》（Baseball Prospectus）技术总监 Harry Pavlidis 在聘用 Jonathan Judge 时冒了很大的风险。Pavlidis 知道，正如 Alan Schwarz 在《数字游戏》（The Numbers Game）中所写的那样，“在美国文化的任何一个角落，都没有比棒球运动员的表现被更精确地计算、更狂热地量化的地方了。”只需在这里或那里点击几下，你就能发现 Noah Syndergaard 的快球在飞向本垒的过程中每分钟旋转超过 2,100 次，Nelson Cruz 在 2016 年的合格击球员中拥有全场最高的平均击球初速，以及无数其他仿佛是从电子游戏或科幻小说中走出来的奇闻轶事。数据海洋的不断上涨，已经赋予了棒球文化中一个日益重要的角色更多力量：数据分析爱好者。
> 
> 这种赋权也带来了额外的审查——不仅针对测量数据本身，也针对数据背后的人员和出版物。在《棒球招股书》工作期间，Pavlidis 深知伴随定量数据不完美而来的强烈抵触。他也知道网站关于接球的衡量指标需要重做，并且这需要一个有学问的头脑——一个能够处理复杂统计建模问题的人——来完成这项工作。
> 
> “他把我们吓坏了。”——Harry Pavlidis
> 
> 基于 Judge 的写作以及他们在网站赞助的棒球场活动中的互动，Pavlidis 预感到 Judge “懂得其中门道”。不久之后，两人在喝酒时进行了交谈。Pavlidis 的直觉得到了证实。Judge 很适合这个职位——更好的是，他本人也很乐意接受。“我和很多人谈过，”Pavlidis 说，“他是唯一一个有勇气承担这项工作的人。” [...]

**注意**：对于这个实验，你可能需要重新调整学习率或批大小等超参数。

> **问题 (main_experiment)：在 OWT 上进行实验（2分）（约需 3 个 H100 小时）**
> 
> 使用与 TinyStories 相同的模型架构和总训练迭代次数，在 OpenWebText 上训练你的语言模型。这个模型表现如何？
> 
> **交付物**：你的语言模型在 OpenWebText 上的学习曲线。描述它与 TinyStories 损失的差异——我们应该如何解释这些损失？
> 
> **交付物**：来自 OpenWebText 语言模型生成的文本，格式与 TinyStories 的输出相同。该文本的流畅度如何？为什么即使我们在模型和计算预算上与 TinyStories 相同，其输出质量却更差？

## 7.5 你自己的修改 + 排行榜

祝贺你进展到这一步。你马上就要完成了！现在你将尝试改进 Transformer 架构，并看看你的超参数和架构设计与班上其他同学相比如何。

**排行榜规则** 除了以下限制外，没有其他任何限制：

*   **运行时间** 你的提交结果在 H100 上的运行时间最多不得超过 1.5 小时。你可以在 slurm 提交脚本中设置 `--time=01:30:00` 来强制执行此操作。
*   **数据** 你只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你可以随心所欲地自由发挥。

如果你正在寻找一些关于实现哪些想法的灵感，你可以查看以下资源：
*   最先进的开源大型语言模型（LLM）系列，例如 Llama 3 [Grattafiori 等，2024] 或 Qwen 2.5 [Yang 等，2024]。
*   NanoGPT speedrun 仓库（https://github.com/KellerJordan/modded-nanogpt），社区成员在该仓库中发布了许多有趣的修改，用于“竞速（speedrunning）”小规模语言模型预训练。例如，一个可以追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入（embeddings）的权重绑定在一起（参见 Vaswani 等 [2017]（第 3.4 节）和 Chowdhery 等 [2022]（第 2 节））。如果你确实尝试了权重绑定，你可能需要减小 embedding/LM 头初始化时的标准差。

在尝试完整的 1.5 小时运行之前，你最好在 OpenWebText 的一个小子集或 TinyStories 上测试这些修改。

需要注意的是，我们确实发现你在这个排行榜中觉得效果很好的一些修改，可能无法泛化到更大规模的预训练中。我们将在课程的缩放定律（scaling laws）单元进一步探讨这个想法。

> **问题 (leaderboard)：排行榜（6分）（约需 10 个 H100 小时）**
> 
> 你将在上述排行榜规则下训练一个模型，目标是在 1.5 个 H100-小时内使你的语言模型的验证集损失最小化。
> 
> **交付物**：记录的最终验证集损失、相关联的学习曲线（需清晰展示 x 轴的挂钟时间少于 1.5 小时），以及对你所做工作的描述。我们期望排行榜的提交结果至少能击败 5.0 损失的朴素基线（naive baseline）。在此处提交至排行榜：https://github.com/stanford-cs336/assignment1-basics-leaderboard。