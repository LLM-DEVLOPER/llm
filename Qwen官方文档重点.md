参考文档：

[https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html](https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html)





## 1.快速使用
---基于hugging face **transformers**库来加载千问模型

```java
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
```

会首先在本地cache中检索模型文件，未找到会进行下载

**流失输出**使用from transformers import TextStreamer库转化模型生成的token



---基于vllm部署千问服务，可以通过openai方式访问服务

## 2.核心概念
**现存类型：**千问现有模型类型

Qwen-Max > Plus > Turbo

**模型类型：**千问是<font style="color:rgb(0, 0, 0);">因果语言模型 (causal Language Models)，也叫因果语言模型 (causal Language Models)或者仅解码器语言模型 (decoder-only language models)：</font>

<font style="color:rgb(0, 0, 0);">它使用之前生成的 token 作为上下文，一次生成一个 token 的文本。</font>

---

**<font style="color:rgb(0, 0, 0);">预训练（Pre-training）和基模型(Base-model)</font>**

<font style="color:rgb(0, 0, 0);">基础语言模型 (base language models) 是在大量文本语料库上训练的基本模型，用于预测序列中的下一个词。它们的主要目标是捕捉语言的统计模式和结构，使它们能够生成连贯且具有上下文关联性的文本。</font>

**<font style="color:rgb(0, 0, 0);">要点：使用基础模型进行情境学习、下游微调等。</font>**

**<font style="color:rgb(0, 0, 0);">后训练 (Post-training) 和指令微调模型 (Instruction-tuned models)</font>**

<font style="color:rgb(0, 0, 0);">指令微调语言模型 (Instruction-tuned language models) 是专门设计用于理解并以对话风格执行特定指令的模型。这些模型经过微调，能准确地解释用户命令，并能以更高的准确性和一致性执行诸如摘要、翻译和问答等任务。</font>

<font style="color:rgb(0, 0, 0);">对于 Qwen 模型，指令调优模型是指带有 “-Instruct” 后缀的模型，例如 Qwen2.5-7B-Instruct 和 Qwen2.5-72B-Instruct 。</font>

**<font style="color:rgb(0, 0, 0);">要点：使用指令微调模型进行对话式的任务执行、下游微调等。</font>**

---

**<font style="color:rgb(0, 0, 0);">Tokens & Tokenization</font>**

<font style="color:rgb(0, 0, 0);">token 代表模型处理和生成的基本单位。大型语言模型通常使用复杂的 tokenization 来处理人类语言的广阔多样性，同时保持词表大小可控。Qwen 词表相对较大，有 15 1646 个 token。</font>

<font style="color:rgb(0, 0, 0);">Qwen采用了名为字节对编码（Byte Pair Encoding，简称BPE）的子词tokenization方法，这种方法试图学习能够用最少的 token 表示文本的 token 组合。</font>

<font style="color:rgb(0, 0, 0);">Qwen词表中因BPE而产生的 token 数量为 15 1643 个，这是一个适用于多种语言的大词表。一般而言，对于英语文本，1个token大约是3~4个字符；而对于中文文本，则大约是1.5~1.8个汉字。</font>

---

**<font style="color:rgb(0, 0, 0);">控制 Token 和 对话模板</font>**

<font style="color:rgb(0, 0, 0);">控制 token 和对话模板都作为指导模型行为和输出的机制。</font>

<font style="color:rgb(0, 0, 0);">从 Qwen2.5 开始，Qwen 模型家族，包括多模态和专项模型，将使用统一的词汇表，其中包含了所有子系列的控制 token 。Qwen2.5 的词汇表中有 22 个控制 token，使得词汇表的总规模达到 15 1665 。</font>

+ <font style="color:rgb(0, 0, 0);">通用 token 1个：</font>`<font style="color:rgb(0, 0, 0);"><|endoftext|></font>`
+ <font style="color:rgb(0, 0, 0);">对话 token 2个：</font>`<font style="color:rgb(0, 0, 0);"><|im_start|></font>`<font style="color:rgb(0, 0, 0);"> </font><font style="color:rgb(0, 0, 0);">和</font><font style="color:rgb(0, 0, 0);"> </font>`<font style="color:rgb(0, 0, 0);"><|im_end|></font>`
+ <font style="color:rgb(0, 0, 0);">工具调用 token 2个：</font><font style="color:rgb(0, 0, 0);"> </font>`<font style="color:rgb(0, 0, 0);"><tool_call></font>`<font style="color:rgb(0, 0, 0);"> </font><font style="color:rgb(0, 0, 0);">和</font><font style="color:rgb(0, 0, 0);"> </font>`<font style="color:rgb(0, 0, 0);"></tool_call></font>`
+ <font style="color:rgb(0, 0, 0);">视觉相关 token 11个</font>
+ <font style="color:rgb(0, 0, 0);">代码相关 token 6个</font>

<font style="color:rgb(0, 0, 0);">要点: Qwen 使用带有控制 token 的 ChatML 作为对话模板。</font>

**<font style="color:rgb(0, 0, 0);">长度限制</font>**

<font style="color:rgb(0, 0, 0);">对于Qwen2.5，在训练中的打包序列长度为 3 2768 个 token</font><font style="color:rgb(0, 0, 0);"> </font>[[4]](https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html#yarn)<font style="color:rgb(0, 0, 0);">。预训练中的最大文档长度即为此长度。而后训练中，user和assistant的最大消息长度则有所不同。一般情况下，assistant消息长度可达 8192 个 token。</font>

**<font style="color:rgb(0, 0, 0);">要点：Qwen2 模型可以处理 32K 或 128K token 长的文本，其中 8K 长度可作为输出。</font>**

## <font style="color:rgb(0, 0, 0);">3.Hugging Face transformers</font>
学会使用transformers库加载本地模型，并进行交互，包括多轮对话、批处理和流式输出

**显存占用**

<font style="color:rgb(0, 0, 0);">一般而言，模型加载所需显存可以按参数量乘二计算，例如，7B 模型需要 14GB 显存加载，其原因在于，对于大语言模型，计算所用数据类型为16位浮点数。当然，推理运行时还需要更多显存以记录激活状态。</font>

<font style="color:rgb(0, 0, 0);">对于 </font>`<font style="color:rgb(0, 0, 0);">transformers</font>`<font style="color:rgb(0, 0, 0);"> ，推荐加载时使用 </font>`<font style="color:rgb(0, 0, 0);">torch_dtype="auto"</font>`<font style="color:rgb(0, 0, 0);"> ，这样模型将以 </font>`<font style="color:rgb(0, 0, 0);">bfloat16</font>`<font style="color:rgb(0, 0, 0);"> 数据类型加载。否则，默认会以 </font>`<font style="color:rgb(0, 0, 0);">float32</font>`<font style="color:rgb(0, 0, 0);"> 数据类型加载，所需显存将翻倍。也可以显式传入 </font>`<font style="color:rgb(0, 0, 0);">torch.bfloat16</font>`<font style="color:rgb(0, 0, 0);"> 或 </font>`<font style="color:rgb(0, 0, 0);">torch.float16</font>`<font style="color:rgb(0, 0, 0);"> 作为 </font>`<font style="color:rgb(0, 0, 0);">torch_dtype</font>`<font style="color:rgb(0, 0, 0);"> 。</font>

## <font style="color:rgb(0, 0, 0);">4.本地部署</font>
### --Ollama本地部署
<font style="color:rgb(0, 0, 0);">适用于MacOS、Linux和Windows操作系统。</font>

<font style="color:rgb(0, 0, 0);">下载Ollama后，一条命令启动：</font>

`<font style="color:rgb(0, 0, 0);">ollama run qwen2.5</font>`<font style="color:rgb(0, 0, 0);">  
</font>`<font style="color:rgb(0, 0, 0);">ollama</font>`<font style="color:rgb(0, 0, 0);">并不托管基模型。即便模型标签不带instruct后缀，实际也是instruct模型。</font>

**<font style="color:rgb(0, 0, 0);">---用Ollama运行你自己的GGUF文件</font>**

<font style="color:rgb(0, 0, 0);">首先要创建一个名为Modelfile的文件，将自己的gguf文件加载</font>

---

### ---MLX-LM本地部署
可以运行在MacOS系统的本地模型平台

需要运行MLX格式的模型文件

一条命令也可以将普通模型转换为mlx格式的模型

`mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path mlx/Qwen2.5-7B-Instruct/ -q`

---

### ---llama.cpp
**1.首先clone github 上的llama-cpp项目到本地**

```java
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

2.编译llama-cpp，前提需要本地要有c++的编译环境

```java
cmake -B build
cmake --build build --config Release
```

3.下载模型或者自行转换格式

下载模型即到huggingface上直接下载对应模型的gguf格式，这里贴一个下载qwen模型的示例地址：

[https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main)

建议将文件下载到/models路径下

4.然后就可以运行llama-cli命令加载大模型

首先进入到/llama.cpp/build/bin下

1. **<font style="color:rgb(0, 0, 0);">对话模式启动（单轮对话）</font>**

```java
./llama-cli -m qwen2.5-0.5b-instruct-q5_k_m.gguf \
    -co -cnv -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
    -fa -ngl 80 -n 512
```

**<font style="color:rgb(0, 0, 0);">-m 或 –model</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">显然，这是模型路径。</font>

**<font style="color:rgb(0, 0, 0);">-co 或 –color</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">为输出着色以区分提示词、用户输入和生成的文本。提示文本为深黄色；用户文本为绿色；生成的文本为白色；错误文本为红色。</font>

**<font style="color:rgb(0, 0, 0);">-cnv 或 –conversation</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">在对话模式下运行。程序将相应地应用聊天模板。</font>

**<font style="color:rgb(0, 0, 0);">-p 或 –prompt</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">在对话模式下，它作为系统提示。</font>

**<font style="color:rgb(0, 0, 0);">-fa 或 –flash-attn</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">如果程序编译时支持 GPU，则启用Flash Attention注意力实现。</font>

**<font style="color:rgb(0, 0, 0);">-ngl 或 –n-gpu-layers</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">如果程序编译时支持 GPU，则将这么多层分配给 GPU 进行计算。</font>

**<font style="color:rgb(0, 0, 0);">-n 或 –predict</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">要预测的token数量。</font>

<font style="color:rgb(0, 0, 0);">你也可以通过以下方式探索其他选项：</font>

<font style="color:rgb(0, 0, 0);background-color:rgb(248, 248, 248);">./llama-cli -h</font>

2. **<font style="color:rgb(0, 0, 0);">互动模式访问大模型（连续对话）</font>**

```java
./llama-cli -m /home/zhangsh82/data/zsh/llama.cpp/models/qwen2.5-0.5b-instruct-q5_k_m.gguf \
    -co -sp -i -if -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" \
    --in-prefix "<|im_start|>user\n" --in-suffix "<|im_end|>\n<|im_start|>assistant\n" \
    -fa -ngl 80 -n 512
```

<font style="color:rgb(0, 0, 0);">我们在这里使用了一些新的选项：</font>

**<font style="color:rgb(0, 0, 0);">-sp 或 –special</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">显示特殊token。</font>

**<font style="color:rgb(0, 0, 0);">-i 或 –interactive</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">进入互动模式。你可以中断模型生成并添加新文本。</font>

**<font style="color:rgb(0, 0, 0);">-if 或 –interactive-first</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">立即等待用户输入。否则，模型将立即运行并根据提示生成文本。</font>

**<font style="color:rgb(0, 0, 0);">-p 或 –prompt</font>****<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">在互动模式下，这是模型续写用的上文。</font>

`**<font style="color:rgb(0, 0, 0);">--in-prefix</font>**`**<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">用户输入附加的前缀字符串。</font>

`**<font style="color:rgb(0, 0, 0);">--in-suffix</font>**`**<font style="color:rgb(0, 0, 0);">:</font>**

<font style="color:rgb(0, 0, 0);">用户输入附加的后缀字符串。</font>

3. <font style="color:rgb(0, 0, 0);">非互动模式（仅支持续写功能）</font>

```java
./llama-cli -m Qwen2.5-7b-instruct-q5_k_m.gguf \
    -co -sp -p "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\ngive me a short introduction to LLMs.<|im_end|>\n<|im_start|>assistant\n" \
    -fa -ngl 80 -n 512
```

## 5.量化
### --量化定义及概念
量化首先分为PTQ(训练后量化)和QAT（训练中量化），我们这里只介绍PTQ

量化还分为权重量化和激活量化

--权重量化即将训练后的权重（定值）量化到指定大小，这部分量化不需要校准数据

--激活值量化是模型在执行前向传播时，每一层计算出来的“中间结果”。比如：

```plain
y = ReLU(Wx + b)  ← 这里的 y 就是激活值
```

+ **激活值不是固定的，它取决于输入数据！**
+ 它在不同输入下分布范围可能不同，因此我们没法靠静态分析直接知道它的最大/最小值。
+ 所以必须要用一些真实样本 **跑一遍前向传播**，把每一层的激活值收集起来，再去统计它们的分布范围。

这就是为什么你看到 PTQ 工具会要求你提供 “校准数据” 来跑几轮推理 —— 就是为了**找出激活值的量化区间**。

---

用数据举一个量化的例子：

是的！INT4 / INT8 本身确实是字面上的意思 —— 指的是：

**使用 4 位或 8 位整数来表示原本的浮点数值（通常是 FP16 或 FP32）**。

但**它背后的映射过程**、**能表示的数值范围和精度**，才是重点。下面我们来用例子说清楚：

---

#### ✅ 一句话定义：
| 类型 | 位数 | 可表示的整数范围 |
| --- | --- | --- |
| INT8 | 8 位有符号整数 | -128 ~ +127 |
| INT4 | 4 位有符号整数 | -8 ~ +7 |


但注意！我们量化的对象（如激活或权重）**不是整数，而是浮点数**。我们是用这些整数来**近似表达**原来的浮点数。

---

#### 🎯 举个完整的量化例子（以 INT4 为例）
假设你有一组浮点数：

```python
原始数据 = [-2.5, -1.0, 0.0, 1.0, 2.0]
```

#### 🔸 步骤 1：计算激活的范围
假设我们统计出范围是 `min = -2.5`，`max = 2.0`。

🔸 步骤 2：映射到 INT4

INT4 可表示整数范围为：`[-8, 7]`（共 16 个数值）

我们要把 `[-2.5, 2.0]` 映射到 `[-8, 7]`，计算：

```python
scale = (max - min) / (qmax - qmin)
      = (2.0 - (-2.5)) / (7 - (-8)) = 4.5 / 15 = 0.3

zero_point = round(-min / scale) = round(2.5 / 0.3) = 8
```

#### 🔸 步骤 3：量化公式
```python
q(x) = round(x / scale) + zero_point
     = round(x / 0.3) + 8
```

#### 结果：
| 原始值 x | x / scale | round() | +zero_point | 量化后整数（INT4） |
| --- | --- | --- | --- | --- |
| -2.5 | -8.33 | -8 | 0 | **-8** ✅ 最小 |
| -1.0 | -3.33 | -3 | 5 | **-3** |
| 0.0 | 0.0 | 0 | 8 | **0** |
| 1.0 | 3.33 | 3 | 11 | **3** |
| 2.0 | 6.66 | 7 | 15 | **7** ✅ 最大 |


这组浮点数就被压缩成了只需要 4 位表示的整数 ✅

---

#### 🔁 反量化（dequantization）
我们可以用 `反量化公式` 把整数恢复成近似的浮点数：

```python
x = scale * (q - zero_point)
```

比如 `q = -3`：

```python
x ≈ 0.3 * (-3 - 8) = 0.3 * (-11) = -3.3
```

虽然跟原始值 `-1.0` 不一样，但差别可以接受 —— **这就是“精度-存储”的权衡**。

---

#### 📊 INT4 vs INT8 的区别
| 项目 | INT8 | INT4 |
| --- | --- | --- |
| 存储大小 | 1 字节 | 半字节（2 个参数占 1 字节） |
| 表示范围 | -128 ~ 127 | -8 ~ 7 |
| 精度 | 高 | 较低 |
| 推理速度 | 快 | 更快 |
| 适用场景 | 部署主流设备 | 极端压缩（边缘/微端/NPU） |


---

#### 🧠 总结一句话：
**INT4/INT8 是对浮点数的“有损近似表达”，以更低的位宽换取更高的推理效率和更小的内存占用。**

这个压缩过程靠 scale 和 zero_point 来实现线性映射，是量化的核心原理之一。

---

### --生成量化校准数据集（数据集格式要求不高，最终需要的是一个包含多个句子的列表）
数据集可以从huggingface或者modelscope下载，但是huggingface可能出现网络问题，可以先在hugg上找到想下载的数据集，再去modelscope下载。

举例：

[https://modelscope.cn/datasets/modelscope/wikitext/quickstart](https://modelscope.cn/datasets/modelscope/wikitext/quickstart)

```python
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('mixedbread-ai/wikipedia-data-en-2023-11')
print("加载完成")
```

下载好的数据集会保存在~./cache/modelscope/hub/datasets目录下

然后从数据集中选取200条纯文本出来作为校准数据集

```python
# 从wiki数据集中抽取200条数据的text作为量化校准数据集
dataset = MsDataset.load('mixedbread-ai/wikipedia-data-en-2023-11')  # 如果有子集，可通过额外参数指定
print(dataset['train'][0])
calibration_texts = dataset['train'].shuffle(seed=42).select(range(200))['text']
print("校准数据样本数：", len(calibration_texts))
```

calibration_texts可以直接作为校准数据集使用

### --AWQ量化：
太好了，AWQ（Activation-aware Weight Quantization）是现在非常🔥主流的一种高精度、低资源开销的**后训练量化（PTQ）方案**，特别适合 LLM（大语言模型）场景，比如 LLaMA、Qwen、ChatGLM 等。下面我来用通俗、实战的角度带你过一遍：

---

#### 🚀 AWQ 是什么？
**AWQ = Activation-aware Weight Quantization**  
是一种只量化权重（而不是激活）的轻量级量化方法，  
**但它在量化权重时，考虑了激活分布对误差的影响，从而更精确地选择量化方案**。

它属于后训练量化（PTQ），不需要再训练，不需要 Label，只需要模型和一点校准数据。

---

#### 🧠 它主要解决了什么问题？
传统的权重量化（比如对 weight 做 min-max 或 GPTQ）：

+ 只考虑了**权重本身的分布**
+ 没考虑权重和激活一起使用时，对最终输出造成的误差

而 **AWQ 的关键点是：**

在量化 weight 之前，**先分析激活的响应，找到权重中对输出影响最大的部分，再更精准地量化这些权重**。

---

#### 🎯 举个直观类比：
假设你在压缩一张图像（模型的权重），传统方法是整体压缩（比如 JPEG），  
但 AWQ 是先看你人眼最容易注意的地方（激活高响应的位置），然后这些地方用高精度压缩，其他地方可以粗一点。

===> 最终得到的压缩图像（量化后模型）在你“真正使用它的时候”（推理），更接近原图效果！

---

#### 🔍 AWQ 做了什么事（按步骤）？
1. **输入几条校准样本**（一般几十条就够）
2. 对每一层：
    - 跑前向传播，记录激活值分布
    - 分析哪些通道/神经元的激活值对输出影响大
3. 根据分析结果，对 weight 做**通道级的缩放 + 精调量化（scaling + clipping）**
4. 量化完之后，还会做一遍 **outlier-aware 权重修正（channel-wise rescaling）**

---

#### ⚙️ 技术特性总结
| 特性 | 说明 |
| --- | --- |
| 📦 是否量化激活 | ❌ 不量化激活，只量化权重 |
| 🧠 是否感知激活分布 | ✅ 感知，作为权重量化的指导依据 |
| 🔩 是否训练 | ❌ 无需训练，后训练量化 |
| 🪶 精度表现 | 🟢 接近 FP16（甚至 Q4 <1% Drop） |
| 🚀 性能兼容性 | 非常适合 llama.cpp / vllm / ggml 等推理引擎 |
| 🔄 支持 INT 格式 | 多支持 INT4（group-wise）/ INT8 |
| 🧰 工具 | [awq](https://github.com/mit-han-lab/llm-awq)<br/>、[autoawq](https://github.com/casper-hansen/AutoAWQ) |


---

#### 📈 实测表现（以 LLaMA 为例）
| 模型 | FP16 | AWQ-INT4 |
| --- | --- | --- |
| LLaMA-7B | 58.1 | 57.9 |
| LLaMA-13B | 60.6 | 60.2 |


几乎无精度损失，压缩 2-4 倍，推理加速 1.5~2.5 倍 👍

---

#### 🛠️ 实战量化工具推荐：
```bash
pip install autoawq
```

一行命令快速量化：

```bash
autoawq quantize \
  --model-path path_to_your_model \
  --quantize-bit 4 \
  --output-path path_to_quantized_model \
  --w-group-size 128 \
  --calib-samples 128
```

---

#### ✅ 总结一句话：
AWQ 是一种**只量化权重但考虑激活分布**的后训练量化方法，它在不牺牲精度的前提下大幅提升了推理效率，非常适合 LLM 部署落地场景！

---



#### 那么AWQ是怎么去找到比较重要的通道呢？
AWQ 的 **关键创新点** 就是：

在量化权重之前，**评估出哪些 weight 对输出影响大，然后更精准地量化它们。**

下面我给你深入但通俗地解释它是 _怎么做到这件事的_ 👇

---

#### 🧠 背后的想法：不是所有的 weight 都一样重要
在 Transformer 中，尤其是 QKV / Linear 层中，有些通道或者 weight：

+ 一点点的变化就会导致输出很大偏差（对输出“敏感”）
+ 而有些通道，哪怕你随便量化点，也不太影响最终结果

所以：**AWQ 不对所有权重“一刀切”地量化**，而是：

**找出对输出敏感的通道，专门对它们做更细致的 scale 调整，从而减少量化误差**。

---

#### 🧪 技术细节：如何判断“影响大”？
AWQ 的论文和实现中，主要用了一个经典且简单有效的方法：

#### ✅ 利用**激活值的范数（activation norm）**来判断通道重要性
以 Linear 层为例（y = Wx）：

+ 每个通道的输出其实是 `dot(wᵢ, x)`，其中：
    - `wᵢ` 是当前通道的权重向量
    - `x` 是输入激活向量（比如一个 token 的 embedding）
+ 如果 `x` 对这个通道来说取值特别大（比如 norm 很高），那这个通道的输出就会更大，误差就更容易放大

所以：

**激活越大的通道，意味着这个通道在推理时越“活跃”或重要，权重量化时就得小心处理**

---

#### 🛠️ 实际怎么处理的？
AWQ 在量化时做了两个动作：

#### 1️⃣ 对每个通道的激活值做统计（前向传播 N 条样本）
+ 一般只跑几十条校准样本（128 条够用了）
+ 记录每个通道的激活值（比如 mean / max / norm）

#### 2️⃣ 根据激活统计信息，分组或缩放权重
+ 把通道按重要性排序
+ 每组通道用自己的 `scale` 和量化范围（group-wise）
+ 有的实现还会 clip 掉激活太大的 outlier 通道

---

#### 📌 举个实际例子
假设某一层有 8 个通道，校准数据前向传播后，你记录到：

| 通道 | 激活范数（norm） |
| --- | --- |
| 0 | 0.2 |
| 1 | 3.1 |
| 2 | 0.9 |
| 3 | 0.1 |
| 4 | 5.3 ← 高 |
| 5 | 0.5 |
| 6 | 2.7 |
| 7 | 0.3 |


你就会认为：**通道 4、1、6 更重要**，  
它们用更细致的量化策略，或者优先分配更小的 `scale`，以减少误差。

---

#### 🧠 补充：它并没有真的量化激活
虽然叫 Activation-aware，但激活只是“被感知”，不是“被量化”。

所以 AWQ 的核心是：

+ **分析激活 → 引导权重量化策略**
+ **不改变推理结构，也不增加运行时开销**

---

#### ✅ 总结一句话：
AWQ 是通过 **统计每个通道的激活范数（norm）来评估它对输出的影响大小**，进而在量化时对重要通道进行更精细的处理，从而显著减少精度损失。

---

如果你有兴趣深入代码层看看，我可以带你读一下 AutoAWQ 或 MIT-AWQ 中这部分的实现逻辑，或者帮你做个通道激活分析的 demo 🚀 要不要一起玩玩？



### ---GPTQ量化
#### 概念
不同于上面的AWQ方式量化，GPTQ通过误差补偿和分列量化的方式进行量化

它也会需要一些校准数据，这些校准数据主要用于观察量化误差，GPTQ采用的逐列量化的方式，在量化每一列时，会去考虑因为前面列导致的误差。

GPTQ的量化过程是，针对每一列，先找到一个基准的量化scale，然后通过缩放得到一批scale，然后开始不断尝试这些scale下和原始权重的误差，找到这批sclae中误差最小的值作为最终scale，但是不可避免依然存在误差，因此GPTQ还有误差补偿的方式来弥补前面列量化过程中产生的误差。用整体权重得到的Y=WX，减去前面每一列使用量化权重得到的激活值，就是我们最终希望的目标激活值。有了目标值后，再通过上述的多scale尝试的方法找到这一列最合适的scale值。

#### 使用GPTQ量化Qwen模型（参考的官方示例）：
```plain
git clone https://github.com/AutoGPTQ/AutoGPTQ
cd AutoGPTQ
pip install -vvv --no-build-isolation -e .
```

开始量化：

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from modelscope.msdatasets import MsDataset
# Specify paths and hyperparameters for quantization
model_path = "/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct"
quant_path = "./models/Qwen2.5-0.5B-Instruct-GPTQ"
quantize_config = BaseQuantizeConfig(
    bits=8, # 4 or 8
    group_size=128,
    damp_percent=0.01,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
    model_name_or_path=None,
    model_file_base_name="model"
)
max_len = 8192

# Load your tokenizer and model with AutoGPTQ
# To learn about loading model to multiple GPUs,
# visit https://github.com/AutoGPTQ/AutoGPTQ/blob/main/docs/tutorial/02-Advanced-Model-Loading-and-Best-Practice.md
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
```

准备校准数据：

```python
# 从wiki数据集中抽取200条数据的text作为量化校准数据集
dataset = MsDataset.load('mixedbread-ai/wikipedia-data-en-2023-11')  # 如果有子集，可通过额外参数指定
calibration_texts = dataset['train'].shuffle(seed=42).select(range(200))['text']
print("校准数据样本数：", len(calibration_texts))
```

```python
#AUTO-GPTQ量化时使用的校准数据，需要使用tokenizer先转换为input_ids的形式，不能使用字符串
calib_data = [tokenizer(text) for text in calibration_texts]
```

执行量化过程：

```python
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
model.quantize(calib_data, cache_examples_on_gpu=False)
```

保存模型和分词器：

```python
model.save_quantized(quant_path, use_safetensors=True)
tokenizer.save_pretrained(quant_path)
```

使用vllm启动一个模型服务：

```python
!CUDA_VISIBLE_DEVICES=7 vllm serve ./models/Qwen2.5-0.5B-Instruct-GPTQ \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype half \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 1000
```

**注意：**使用GPTQ量化的Int4的千问模型，无法使用vllm启动，vllm目前仅支持FP8的量化模型

### --bpw
在深度学习模型的量化过程中，**bpw**（每权重位数，bits per weight）指的是表示每个模型权重所使用的平均位数。例如，传统的全精度（FP32）模型每个权重占用32位，而通过量化技术，可以将每个权重的位数减少到更低的值，如8位、4位，甚至更低。这有助于降低模型的内存占用和计算需求，从而加快推理速度并减少存储空间。[Medium](https://netraneupane.medium.com/hands-on-llms-quantization-a4c7ab1421c2?utm_source=chatgpt.com)

需要注意的是，实际的bpw值可能并非整数，因为量化方法可能使用混合精度策略，对模型的不同部分采用不同的位宽进行量化。例如，一种量化方法可能对某些层使用4位，对其他层使用5位，最终得到一个平均的bpw值。这种混合精度的量化策略可以在保持模型性能的同时，进一步减少模型大小。

在选择量化方案时，理解bpw有助于权衡模型精度与资源消耗之间的关系。较低的bpw通常意味着更小的模型尺寸和更快的推理速度，但可能会导致一定的精度下降。因此，需要根据具体应用场景和硬件限制，选择合适的bpw值和量化策略。



### --llama-cpp
#### --无校准量化
##### 1.首先将模型生成gguf文件（建议先升到f32）
```python
!python convert_hf_to_gguf.py /home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct --outtype f32 --outfile ./models/qwen2.5-0.5b-instruct-f32.gguf
```

##### 2.执行llama-quantize脚本，直接量化：
```python
./build/bin/llama-quantize ./models/qwen2.5-0.5b-instruct-f32.gguf ./models/qwen2.5-0.5b-instruct-q8_0.gguf F16
```

ps：如果是在jupyter中执行，会报错，需要写成python中可执行的形式：

```python
#无校准量化（不使用校准数据集）
import subprocess

command = [
    "./build/bin/llama-quantize",
    "./models/qwen2.5-0.5b-instruct-f32.gguf",
    "./models/qwen2.5-0.5b-instruct-f16.gguf",
    "F16"
]
subprocess.run(command)
```

#### --基于AWQ方式量化
##### 1.首先安装autoawq包
```python
!pip install autoawq
```

ps：可能需要调整transformers包版本，做好兼容性

##### 2.加载模型，确定量化参数
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from modelscope.msdatasets import MsDataset
 
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct"
quant_path = "./models/qwen2.5-0.5b-instruct-awq"

quant_config = {"zero_point" : True,"q_group_size" : 128,"w_bit" : 4, "version" : "GEMM"}

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path,device_map = "auto")
```

##### 3.重新获取校准数据集
```python
# 从wiki数据集中抽取200条数据的text作为量化校准数据集
dataset = MsDataset.load('mixedbread-ai/wikipedia-data-en-2023-11')  # 如果有子集，可通过额外参数指定
calibration_texts = dataset['train'].shuffle(seed=42).select(range(200))['text']
print("校准数据样本数：", len(calibration_texts))
```

```python
calib_data = [text for text in calibration_texts]
```

##### 4.执行量化准备操作
```python
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data,
    text_column='text',  # 指定文本字段的键名
    export_compatible=True  #这里指定为True会让模型在生成量化模型时更具兼容性，可以直接跑起来
)
```

注意：这里的量化函数执行完后，模型并未被量化，只是为量化做好了准备，准确的说是确定了量化时要采用的scale。详细解释：

代码片段中，调用 `model.quantize()` 方法时，添加参数 `export_compatible=True`，其作用并非直接对模型权重进行量化，而是对权重进行调整，使其更适合后续的量化过程。这种调整基于激活感知权重量化（Activation-aware Weight Quantization, AWQ）方法，主要步骤包括：

1. **权重调整**：通过分析模型在校准数据集上的激活分布，确定每个权重通道的重要性。然后，对重要的权重通道进行缩放，以减少量化误差。 citeturn0search8
2. **延迟量化**：在上述调整完成后，模型的权重尚未被实际量化。真正的量化步骤通常在后续使用其他工具（如 `convert-hf-to-gguf.py` 和 `llama-quantize`）时进行。 citeturn0search9

因此，代码中的 `model.quantize()` 方法在 `export_compatible=True` 参数下，主要是预处理模型权重，使其在后续的量化过程中能够更好地保留原始模型的性能和精度。

**数据举例：**

为了更清楚地理解模型量化过程中的预处理（如AWQ）与实际量化之间的区别，以下是一个具体的示例：

**1. 预处理阶段（AWQ调整）：**

假设我们有一个模型层的权重矩阵：

```plain
权重矩阵 W：
[ 0.8,  2.5, -1.2]
[-0.5,  1.0,  3.3]
```

在传统的量化方法中，直接将这些权重从浮点数转换为低比特整数可能导致较大的量化误差。AWQ方法通过以下步骤进行预处理：

+ 分析激活分布：在校准数据集上运行模型，收集每个通道的激活值分布，确定哪些通道对模型性能影响更大。
+ 调整权重（缩放）：对于重要的通道，应用缩放因子。例如，如果第一列权重对应的激活值较大，可能对该列权重应用一个缩放因子 `s = 0.5`：

```plain
调整后的权重矩阵 W'：
  [ 0.4,  2.5, -1.2]   （0.8 * 0.5 = 0.4）
  [-0.25, 1.0,  3.3]   （-0.5 * 0.5 = -0.25）
```

这种调整使得重要通道的权重值减小，从而在后续量化时减少量化误差。

**2. 量化阶段：**

在预处理完成后，进行实际的量化操作，例如将调整后的权重从浮点数转换为4位整数：

+ 确定量化范围：找到调整后权重矩阵中的最大绝对值，例如 `max_abs = 3.3`。
+ 计算缩放因子：对于4位量化，整数范围是 [-8, 7]，因此缩放因子 `scale = max_abs / 7 ≈ 0.471`。
+ 应用量化：将调整后的权重矩阵量化为整数：

```plain
量化后的权重矩阵 W_q：
  [ 1,  5, -3]   （round(0.4 / 0.471) = 1）
  [-1,  2,  7]   （round(-0.25 / 0.471) = -1）
```

通过上述步骤，预处理阶段的调整（如AWQ）使得权重在量化时能够更准确地表示，从而减少量化误差，保持模型性能。

**总结：预处理阶段（如AWQ）通过分析激活分布，对权重进行适当的缩放调整，使其更适合量化；而量化阶段则将这些调整后的权重转换为低比特表示。两者结合，有助于在降低模型复杂度的同时，尽可能保持其性能。**

****

**P.S:这里在生成./models/qwen2.5-0.5b-instruct-awq文件后，看了一下其内部文件，发现结果和qwen的instruct差不多，模型的核心文件都有。然后尝试去将这个模型加载运行起来，开始使用transformers库的AutoModelForCausalLM.from_pretrained(）方法加载模型，发现猛猛报错，调研后发现可能是权重文件格式不符合openai格式，后面改用AutoTokenizer.from_pretrained（）方法加载，发现也是报错，分析原因：可能确实权重文件的结构不符合这两个方法的格式。**

##### 5.保存确定好的scale模型
```python
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Model is quantized and saved at "{quant_path}"')
```

##### 6.再按照之前的流程使用convert_hf_to_gguf.py执行具体的量化过程
```python
!python convert_hf_to_gguf.py ./models/qwen2.5-0.5b-instruct-awq --outtype f32 --outfile ./models/qwen2.5-0.5b-instruct-f32-awq.gguf
```

```python
# AWQ量化（带校准数据）
import subprocess
command = [
    "./build/bin/llama-quantize",
    "./models/qwen2.5-0.5b-instruct-f32-awq.gguf",
    "./models/qwen2.5-0.5b-instruct-f16-awq.gguf",
    "F16"
]
subprocess.run(command)
```

#### --模型困惑度测试
`<font style="color:rgb(0, 0, 0);">llama.cpp</font>`<font style="color:rgb(0, 0, 0);">为我们提供了一个示例程序来计算困惑度，这评估了给定文本对模型而言的“不可能”程度。它主要用于比较：困惑度越低，模型对给定文本的记忆越好。</font>

<font style="color:rgb(0, 0, 0);">首先准备一个数据集：</font>

```python
#下载wikitext数据集到本地，测试模型的困惑度
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('modelscope/wikitext', subset_name='wikitext-2-raw-v1', split='test')
```

执行困惑度测试：

```python
command = [
    "./build/bin/llama-perplexity",
    "-m",
    "./models/qwen2.5-0.5b-instruct-f16-awq.gguf",
    "-f",
    "/home/zhangsh82/.cache/modelscope/hub/datasets/wikitext/wikitext-2-raw-v1/1.0.0/6280e5a53c82b20da4f99f484fa6f0ca9de738ff12f59efb0815fe7d8ae21478/wikitext-test.arrow"
]
subprocess.run(command)
```

执行结果：

```python
build: 5038 (193c3e03) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
llama_model_loader: loaded meta data with 26 key-value pairs and 290 tensors from ./models/qwen2.5-0.5b-instruct-f16-awq.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen2.5 0.5b Instruct Awq
llama_model_loader: - kv   3:                           general.finetune str              = instruct-awq
llama_model_loader: - kv   4:                           general.basename str              = qwen2.5
llama_model_loader: - kv   5:                         general.size_label str              = 0.5B
llama_model_loader: - kv   6:                          qwen2.block_count u32              = 24
llama_model_loader: - kv   7:                       qwen2.context_length u32              = 32768
llama_model_loader: - kv   8:                     qwen2.embedding_length u32              = 896
llama_model_loader: - kv   9:                  qwen2.feed_forward_length u32              = 4864
llama_model_loader: - kv  10:                 qwen2.attention.head_count u32              = 14
llama_model_loader: - kv  11:              qwen2.attention.head_count_kv u32              = 2
llama_model_loader: - kv  12:                       qwen2.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  13:     qwen2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  15:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  16:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  18:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  21:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  22:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  23:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                          general.file_type u32              = 1
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type  f16:  169 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = F16
print_info: file size   = 942.43 MiB (16.00 BPW) 
load: special tokens cache size = 22
load: token to piece cache size = 0.9310 MB
print_info: arch             = qwen2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 32768
print_info: n_embd           = 896
print_info: n_layer          = 24
print_info: n_head           = 14
print_info: n_head_kv        = 2
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 64
print_info: n_embd_head_v    = 64
print_info: n_gqa            = 7
print_info: n_embd_k_gqa     = 128
print_info: n_embd_v_gqa     = 128
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 4864
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 32768
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 1B
print_info: model params     = 494.03 M
print_info: general.name     = Qwen2.5 0.5b Instruct Awq
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151643 '<|endoftext|>'
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151643 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors:   CPU_Mapped model buffer size =   942.43 MiB
..........................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 4
llama_context: n_ctx         = 2048
llama_context: n_ctx_per_seq = 512
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (512) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     2.32 MiB
init: kv_size = 2048, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 24, can_shift = 1
init:        CPU KV buffer size =    24.00 MiB
llama_context: KV self size  =   24.00 MiB, K (f16):   12.00 MiB, V (f16):   12.00 MiB
llama_context:        CPU compute buffer size =   298.50 MiB
llama_context: graph nodes  = 894
llama_context: graph splits = 1
common_init_from_params: setting dry_penalty_last_n to ctx_size = 2048
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)

system_info: n_threads = 36 (n_threads_batch = 36) / 72 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 801.231 ms
perplexity: calculating perplexity over 617 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 4.20 seconds per pass - ETA 
10.78 minutes
[5]5.9615,[6]6.0265,[7]5.8214,[8]5.5404,[9]5.8321,[10]6.5176,[11]7.0585,[12]7.4067,[13]7.7750,[14]8.2667,[15]8.7040,[16]9.2678,[17]9.8532,[18]10.3352,[19]10.6466,[20]11.0134,[21]11.6023,[22]11.5693,[23]11.6952,[24]12.0265,[25]11.6275,[26]11.9358,[27]11.9598,[28]12.0548,[29]12.0115,[30]12.1281,[31]11.8333,[32]11.4872,[33]11.4241,[34]11.3175,[35]11.2055,[36]11.1704,[37]11.2962,[38]11.3603,[39]11.3144,[40]11.4045,[41]11.3009,[42]11.4031,[43]11.5521,[44]11.7364,[45]11.9746,[46]11.9983,[47]11.9372,[48]12.0063,[49]12.0339,[50]12.0964,[51]12.1538,[52]12.1331,[53]12.2117,[54]12.2104,[55]12.4167,[56]12.5660,[57]12.4693,[58]12.5439,[59]12.5879,[60]12.6847,[61]12.7410,[62]12.8092,[63]12.8234,[64]12.8911,[65]12.8894,[66]12.9349,[67]13.0027,[68]13.0784,[69]13.1273,[70]13.2364,[71]13.3259,[72]13.4420,[73]13.5751,[74]13.6522,[75]13.7962,[76]13.8127,[77]13.8353,[78]13.8191,[79]13.8744,[80]13.9482,[81]13.9984,[82]14.0143,[83]13.9725,[84]13.9247,[85]13.9665,[86]13.9858,[87]13.9166,[88]13.8996,[89]13.8656,[90]13.9366,[91]13.9381,[92]13.9159,[93]13.9616,[94]14.0552,[95]14.1829,[96]14.1854,[97]14.1802,[98]14.1737,[99]14.2364,[100]14.2166,[101]14.3062,[102]14.3175,[103]14.3026,[104]14.3419,[105]14.3304,[106]14.3507,[107]14.3487,[108]14.4207,[109]14.4448,[110]14.4862,[111]14.5425,[112]14.4938,[113]14.5259,[114]14.4861,[115]14.5348,[116]14.6044,[117]14.6440,[118]14.7402,[119]14.8606,[120]14.9137,[121]14.8772,[122]14.9484,[123]14.9910,[124]14.9310,[125]14.9285,[126]14.9199,[127]14.8495,[128]14.9046,[129]14.8711,[130]14.8763,[131]14.8160,[132]14.7558,[133]14.7399,[134]14.7327,[135]14.7373,[136]14.7011,[137]14.6998,[138]14.6581,[139]14.5846,[140]14.5372,[141]14.5281,[142]14.5350,[143]14.5247,[144]14.5130,[145]14.4881,[146]14.4245,[147]14.4459,[148]14.4495,[149]14.4660,[150]14.4530,[151]14.4758,[152]14.5308,[153]14.5064,[154]14.4155,[155]14.3555,[156]14.2879,[157]14.1557,[158]14.0633,[159]13.9905,[160]13.8855,[161]13.8157,[162]13.7800,[163]13.7305,[164]13.6583,[165]13.5972,[166]13.5305,[167]13.4992,[168]13.4736,[169]13.4459,[170]13.3967,[171]13.3535,[172]13.2901,[173]13.2423,[174]13.2231,[175]13.1599,[176]13.1340,[177]13.1179,[178]13.0915,[179]13.0499,[180]13.0111,[181]13.0027,[182]12.9994,[183]13.0540,[184]13.0694,[185]13.1022,[186]13.1422,[187]13.1911,[188]13.2346,[189]13.2843,[190]13.3210,[191]13.3837,[192]13.4314,[193]13.5077,[194]13.5666,[195]13.5784,[196]13.5802,[197]13.6393,[198]13.6801,[199]13.7119,[200]13.7323,[201]13.7391,[202]13.7858,[203]13.8009,[204]13.8015,[205]13.8341,[206]13.8770,[207]13.9072,[208]13.9132,[209]13.9428,[210]13.9931,[211]14.0337,[212]14.0435,[213]14.0560,[214]14.0319,[215]14.0333,[216]13.9928,[217]13.9682,[218]14.0218,[219]14.0466,[220]14.0566,[221]14.0346,[222]14.0181,[223]14.0061,[224]13.9530,[225]13.9670,[226]13.9680,[227]13.9469,[228]13.9123,[229]13.9220,[230]13.8854,[231]13.8887,[232]13.8514,[233]13.8326,[234]13.8084,[235]13.8011,[236]13.7929,[237]13.7833,[238]13.7785,[239]13.7772,[240]13.7824,[241]13.7660,[242]13.7429,[243]13.7557,[244]13.7658,[245]13.7383,[246]13.7085,[247]13.6944,[248]13.6889,[249]13.6997,[250]13.6959,[251]13.6959,[252]13.7131,[253]13.7091,[254]13.6948,[255]13.7130,[256]13.7270,[257]13.7081,[258]13.6975,[259]13.6966,[260]13.7257,[261]13.7370,[262]13.7494,[263]13.7923,[264]13.8268,[265]13.8484,[266]13.8691,[267]13.8885,[268]13.9075,[269]13.9301,[270]13.9693,[271]13.9944,[272]14.0002,[273]14.0214,[274]13.9976,[275]13.9720,[276]13.9614,[277]13.9585,[278]13.9576,[279]13.9716,[280]13.9853,[281]14.0052,[282]14.0215,[283]14.0103,[284]14.0157,[285]14.0409,[286]14.0469,[287]14.0381,[288]14.0616,[289]14.0661,[290]14.0694,[291]14.0622,[292]14.0683,[293]14.0558,[294]14.0514,[295]14.0469,[296]14.0379,[297]14.0206,[298]14.0256,[299]13.9825,[300]13.9509,[301]13.9016,[302]13.8524,[303]13.8128,[304]13.7916,[305]13.7362,[306]13.6980,[307]13.6590,[308]13.6865,[309]13.6807,[310]13.6614,[311]13.6785,[312]13.6970,[313]13.7411,[314]13.7498,[315]13.7637,[316]13.7767,[317]13.7933,[318]13.8337,[319]13.8584,[320]13.9024,[321]13.9090,[322]13.8962,[323]13.8911,[324]13.8850,[325]13.8892,[326]13.8750,[327]13.8901,[328]13.9072,[329]13.9088,[330]13.9124,[331]13.9040,[332]13.8922,[333]13.8996,[334]13.9167,[335]13.9098,[336]13.9082,[337]13.9200,[338]13.9118,[339]13.9115,[340]13.9356,[341]13.9481,[342]13.9439,[343]13.9658,[344]13.9393,[345]13.9729,[346]13.9866,[347]14.0113,[348]14.0306,[349]14.0444,[350]14.0326,[351]14.0261,[352]14.0076,[353]13.9956,[354]13.9924,[355]13.9817,[356]14.0009,[357]13.9861,[358]13.9732,[359]13.9911,[360]13.9826,[361]14.0026,[362]14.0008,[363]14.0010,[364]13.9904,[365]13.9730,[366]13.9801,[367]13.9783,[368]13.9800,[369]13.9706,[370]13.9592,[371]13.9594,[372]13.9452,[373]13.9476,[374]13.9400,[375]13.9427,[376]13.9379,[377]13.9135,[378]13.9415,[379]13.9540,[380]13.9589,[381]13.9514,[382]13.9523,[383]13.9509,[384]13.9570,[385]13.9667,[386]13.9552,[387]13.9742,[388]14.0030,[389]14.0501,[390]14.0838,[391]14.1272,[392]14.1594,[393]14.1785,[394]14.2106,[395]14.2481,[396]14.2657,[397]14.2816,[398]14.3156,[399]14.3488,[400]14.3686,[401]14.3971,[402]14.4179,[403]14.4399,[404]14.4629,[405]14.4866,[406]14.5081,[407]14.5421,[408]14.5809,[409]14.5948,[410]14.5839,[411]14.5628,[412]14.5723,[413]14.6088,[414]14.6347,[415]14.6425,[416]14.6552,[417]14.6413,[418]14.6479,[419]14.6598,[420]14.6723,[421]14.6835,[422]14.6857,[423]14.7002,[424]14.7042,[425]14.7044,[426]14.6859,[427]14.6837,[428]14.6688,[429]14.6556,[430]14.6553,[431]14.6636,[432]14.6686,[433]14.6656,[434]14.6884,[435]14.6938,[436]14.6997,[437]14.7014,[438]14.6983,[439]14.6931,[440]14.7063,[441]14.7167,[442]14.7197,[443]14.7134,[444]14.7170,[445]14.6975,[446]14.7126,[447]14.7323,[448]14.7366,[449]14.7458,[450]14.7331,[451]14.7059,[452]14.6523,[453]14.6072,[454]14.5715,[455]14.5366,[456]14.5064,[457]14.4780,[458]14.5160,[459]14.5329,[460]14.5487,[461]14.5382,[462]14.5379,[463]14.5365,[464]14.5133,[465]14.5237,[466]14.5322,[467]14.5276,[468]14.5442,[469]14.5498,[470]14.5429,[471]14.5624,[472]14.5814,[473]14.5851,[474]14.5719,[475]14.5751,[476]14.5668,[477]14.5828,[478]14.5962,[479]14.5979,[480]14.5828,[481]14.5949,[482]14.5866,[483]14.5828,[484]14.5765,[485]14.5652,[486]14.5717,[487]14.5794,[488]14.5840,[489]14.6042,[490]14.5891,[491]14.5849,[492]14.5915,[493]14.6061,[494]14.6222,[495]14.6317,[496]14.6421,[497]14.6459,[498]14.6599,[499]14.6698,[500]14.6779,[501]14.6807,[502]14.6708,[503]14.6641,[504]14.6444,[505]14.6563,[506]14.6678,[507]14.6514,[508]14.6570,[509]14.6468,[510]14.6414,[511]14.6665,[512]14.6744,[513]14.6617,[514]14.6534,[515]14.6591,[516]14.6584,[517]14.6410,[518]14.6323,[519]14.6222,[520]14.6217,[521]14.6166,[522]14.6033,[523]14.6048,[524]14.5858,[525]14.5748,[526]14.5697,[527]14.5626,[528]14.5640,[529]14.5781,[530]14.5732,[531]14.5797,[532]14.5739,[533]14.5881,[534]14.6143,[535]14.6201,[536]14.6303,[537]14.6344,[538]14.6300,[539]14.6360,[540]14.6384,[541]14.6340,[542]14.6432,[543]14.6369,[544]14.6321,[545]14.6336,[546]14.6358,[547]14.6371,[548]14.6134,[549]14.5999,[550]14.5975,[551]14.5924,[552]14.5858,[553]14.5750,[554]14.5769,[555]14.5877,[556]14.6018,[557]14.5909,[558]14.5870,[559]14.5832,[560]14.5790,[561]14.5708,[562]14.5772,[563]14.5622,[564]14.5627,[565]14.5613,[566]14.5586,[567]14.5453,[568]14.5656,[569]14.5725,[570]14.5780,[571]14.5936,[572]14.5891,[573]14.5751,[574]14.5511,[575]14.5312,[576]14.5030,[577]14.5135,[578]14.5165,[579]14.5273,[580]14.5171,[581]14.5029,[582]14.4779,[583]14.4908,[584]14.4693,[585]14.4508,[586]14.4314,[587]14.4012,[588]14.4076,[589]14.4136,[590]14.4150,[591]14.4167,[592]14.4277,[593]14.4424,[594]14.4568,[595]14.4577,[596]14.4811,[597]14.4948,[598]14.5026,[599]14.5135,[600]14.5188,[601]14.5047,[602]14.5064,[603]14.4983,[604]14.5069,[605]14.5084,[606]14.5251,[607]14.5310,[608]14.5301,[609]14.5321,[610]14.5287,[611]14.5274,[612]14.5359,[613]14.5557,[614]14.5695,[615]14.5899,[616]14.6012,[617]14.6109,

Final estimate: PPL = 14.6109 +/- 0.10800 #最终的困惑度
llama_perf_context_print:        load time =     381.43 ms
llama_perf_context_print: prompt eval time =  464665.94 ms / 315904 tokens (    1.47 ms per token,   679.85 tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time =  471321.77 ms / 315905 tokens
```

## 6.部署
### 1.离线批量推理
```python
#离线批量处理
# 注意：V100显卡不支持使用vllm直接部署awq量化版本额
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct")

# Pass the default decoding hyperparameters of Qwen2.5-0.5B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct",dtype = 'float16', quantization="awq")

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 2.基于openai的形式启动一个服务
```python
!CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct --dtype 'float16'
```

#### --请求这个服务：
```python
!curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'
```

python代码：

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
```

### 3.分部署部署
```python
#分布式部署，多卡并行（张量并行） --tensor-parallel-size参数
!CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct --dtype 'float16' --tensor-parallel-size 2
```

## 7.有监督微调
本章节训练大模型暂时基于LLaMA-Factory框架

使用SFT加lora方式训练一个chat-嬛嬛，使用公开的甄嬛数据集

1.配置分布式训练环境

```python
!pip install deepspeed
!pip install flash-attn --no-build-isolation
```

2.下载llama factory代码

```python
!pip install -e "/home/zhangsh82/data/zsh/LLaMA-Factory"
```

```python
!llamafactory-cli version
```

3.训练嬛嬛

数据集来源：[https://github.com/datawhalechina/self-llm.git](https://github.com/datawhalechina/self-llm.git)

```python
%%bash

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# 设置节点数为1
export NNODES=1
# 设置每个节点上的GPU数量为1
export GPUS_PER_NODE=1
# 设置当前节点的等级为0
export NODE_RANK=0
# 设置主节点的地址为本地IP
export MASTER_ADDR=localhost
# 设置主节点的端口为1234
export MASTER_PORT=5555

MODEL='/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct/'
OUTPUT_PATH='/home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-zhenhuan'
DATA_SET='zhenhuan'
DS_CONFIG_PATH='/home/zhangsh82/data/zsh/LLaMA-Factory/examples/deepspeed/ds_z3_config.json'


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 200 \
    --val_size 0.01 \
    --logging_dir /home/zhangsh82/data/zsh/Qwen-Learn \
    --use_fast_tokenizer \
    --flash_attn disabled \
    --model_name_or_path $MODEL \
    --dataset $DATA_SET \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.01 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --cutoff_len 4096 \
    --save_steps 200 \
    --plot_loss True\
    --num_train_epochs 3
```

**参数设置注意事项：**

--deepspeed 设置deepseek的配置文件，example：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1744945417453-9fe46d0a-63a2-48c1-afa9-2c90641179ac.png)

--stage 训练方式指定为sft指令微调

---

下面一组参数用于设置训练过程中的验证信息

 --do_eval \

 --eval_strategy steps \

 --eval_steps 200 \	#每200步验证一次

 --val_size 0.01 \    #从训练集中拆出0.01作为验证集

---

--dataset 设置数据集，注意不同的训练阶段，数据集的格式不同，本次基于sft微调，训练集示例：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1744945749468-9bd5c036-23a8-4ff3-bdf8-5d0d03b6b912.png)

数据集大小3729条，正常微调垂域小场景，这个数据量差不多够了

---

--finetuning_type lora \

--lora_target q_proj,v_proj\

设置为lora方式微调，还可以继续设置lora-rank（默认8）和alpha大小，平衡训练效果和训练成本

---

--warmup_ratio

 学习率预热比例   在最开始的 1% steps，学习率线性从 0 → 学习率峰值，防止刚开始梯度爆炸。  

 --weight_decay  

 权重衰减   防止过拟合，把参数逐渐往 0 拉（类似L2正则化），这里设置 0.1，偏高，适合大模型。  

 --per_device_train_batch_size 4  

 单卡每次训练 4个样本，如果多卡，还要乘设备数量。  

 --gradient_accumulation_steps 4  

 每训练 4 个 batch 才真正反向传播一次，相当于**有效 batch_size = 4 × 4 = 16**，显存不够时常用。  

 --ddp_timeout 9000  

分布式训练超时时间

 --cutoff_len 4096  

 文本超过 4096 token 会被截断，控制显存消耗。  

 --save_steps 200  

 每 200步保存一次 checkpoint，防止中断导致全挂。  

训练loss：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1744948101858-d6529350-8501-4c32-a5ad-69d7f4c99b05.png)

---

训练完毕后，可以直接调用微调后的lora模型，也可以微调后的模型权重合并到原模型内

合并操作：

```python
CUDA_VISIBLE_DEVICES=7 llamafactory-cli export \
    --model_name_or_path /home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct/ \
    --adapter_name_or_path /home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-zhenhuan \
    --template qwen \
    --finetuning_type lora \
    --export_dir /home/zhangsh82/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-zhenhuan-lora-merge \
    --export_size 2 \
    --export_legacy_format False
```

## 8.模型效果比较
1.精度比较

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225053048-14f9c1cc-b06b-4907-affd-06fb731d1abb.png)

2.速度比较

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225115873-33f978d5-6045-4f36-9a82-3861bcdb9e13.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225129292-b531ce1c-5313-4939-8cda-d56c92ef5c0a.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225141831-76e67636-51aa-45b5-82cf-59df098598d8.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225153632-f41a3a7e-512d-440a-80fb-9bb76bf56ed7.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225165746-01f04225-7ac2-410a-baae-f9d4757ae579.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225176841-ec4295cc-6bbf-4988-b0b9-02a6caf4a131.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225188510-d9a6d792-6279-451f-936b-c0a5a2a9a8ee.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1745225219994-7557ad59-57e2-49f4-b45a-adc633efb6f9.png)

更多请参考：

[https://qwen.readthedocs.io/zh-cn/latest/benchmark/speed_benchmark.html](https://qwen.readthedocs.io/zh-cn/latest/benchmark/speed_benchmark.html)

