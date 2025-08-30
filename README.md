<div align="center">
  <!-- 徽章 -->
  <img src="https://img.shields.io/badge/Framework-PyTorch-purple?style=for-the-badge&logo=pytorch" alt="Framework Badge">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" alt="Language Badge">
  <img src="https://img.shields.io/badge/Paradigm-Vision--Language_Model-orange?style=for-the-badge&logo=openai" alt="Paradigm Badge">
  <img src="https://img.shields.io/github/stars/cotix-ai/Aircraft-VLA?style=for-the-badge&color=gold" alt="Stars Badge">
</div>

<br>

<h1 align="center">
  Aircraft-VLA: 航空航天领域的 VLA 模型
</h1>
<p align="center">
点击播放

[![](https://i1.hdslb.com/bfs/archive/af68f2b6cef2da38a9b367fd73333a37bdf21bb2.jpg)](https://www.bilibili.com/video/BV1SPgbzzE3e/)

> *科罗廖夫十字绽开，我是卡门线上的花*
> 
</p>

<br>

>[!IMPORTANT]
> **核心提示**: 本项目旨在通过引入大规模、结构化的航空航天专业词汇表（Special Tokens），构建一个能够对飞行器进行深度识别、状态分析和操作理解的视觉语言模型（VLA）。

## 目录

- [✨ 项目简介](#-项目简介)
- [💡 核心设计理念](#-核心设计理念)
- [🧠 架构核心](#-架构核心)
- [🧩 核心组件详解](#-核心组件详解)
    - [组件一：航空航天专业词汇表 (AeroSpace Vocabulary)](#组件一航空航天专业词汇表-aerospace-vocabulary)
    - [组件二：分阶段对齐与微调框架](#组件二分阶段对齐与微调框架)
    - [组件三：高效LoRA训练策略](#组件三高效lora训练策略)
- [🔄 工作流程](#-工作流程)
- [🚀 独特优势与创新](#-独特优势与创新)
- [🛠️ 快速开始](#️-快速开始)
- [🤝 如何贡献](#-如何贡献)
- [📄 许可证](#-许可证)

<br>

---

## ✨ 项目简介

本项目介绍了 **Aircraft-VLA**，一个新颖的视觉语言模型框架，它通过将最先进的视觉编码器（如 CLIP ViT）与强大的大型语言模型（如 Qwen2-VL）相结合，并注入一个包含数百个专业术语的结构化词汇表，显著提升了在航空航天领域的视觉理解能力。

**Aircraft-VLA** 重新定义了机器对飞行器的认知方式，将其视为一个由层级化组件、动态状态和标准化操作构成的复杂系统，而非简单的像素集合。它超越了传统VLA在专业领域“看图说话”的局限性（例如：只能识别“一架飞机”，无法区分“处于最终进近状态的空客A350，襟翼全放”）。本架构协同了通用视觉模型的强大表征能力与专业离散化Token的精确性，创造出一个高度专业和精确的系统，能够解析图像中的细微差别，并进行逻辑推理。

<br>

---

## 💡 核心设计理念

**Aircraft-VLA** 不仅仅是另一个通用VLA的微调版本，它代表了将领域知识深度注入AI模型的范式转变。我们相信，专业领域的下一次AI飞跃，需要系统能够使用该领域的“语言”进行思考和交流。

> "AI理解的未来在于从模糊的自然语言描述转向精确、结构化的领域符号体系。"

本设计旨在克服通用模型在理解专业视觉信息时的固有局限性，在这些场景中，一个微小的视觉特征（如翼梢小翼的形状）就可能包含区分不同型号或状态的关键信息。

<br>

---

## 🧠 架构核心

**Special Token词汇表** 是 **Aircraft-VLA** 架构的基石，也是整个理解与生成过程的“语义骨架”。该机制将模型从对连续视觉特征的模糊映射中解放出来，赋予其使用离散、精确的符号进行思考的能力。

**核心功能:**
系统通过一个三阶段的训练流程，让模型掌握这个新的“语言”：
1.  **视觉-Token对齐**: 建立图像基本特征与专业Token之间的直接映射，例如“这个形状” → `<component:winglet>`。
2.  **视觉-语言预训练**: 将专业Token融入自然语言的语法和逻辑中，让模型理解“`<model:boeing-737-max>` 的 `<component:winglet>` 是分叉的”。
3.  **指令微调**: 训练模型在对话和问答中自如地运用专业Token进行分析和推理，成为一个真正的航空航天领域专家助手。

因此，模型对一张图片的最终理解，不再是一段泛泛的描述，而是一个结构化的、信息密度极高的专业分析报告。

<br>

---

## 🧩 核心组件详解

**Aircraft-VLA** 中的不同组件各司其职，通过明确的训练分工，共同实现一个从“看见”到“理解”的智能流程。

### 组件一：航空航天专业词汇表 (AeroSpace Vocabulary) (角色：知识基石)
*   **目标:** 提供一个全面、层级化、标准化的符号系统，用于描述航空航天领域的一切视觉元素，从飞行器型号、具体部件到操作状态和飞行事件。
*   **实现:** 在 `special_tokens_vocab.py` 中，通过Python类精心设计了超过300个专业Token，如 `<model:airbus-a350-900>`, `<op:flaps-landing>`, `<event:stage-separation>`。这个词汇表是模型专业能力的源泉。

### 组件二：分阶段对齐与微调框架 (角色：学习路径规划师)
*   **目标:** 引导模型高效、稳定地学习新知识，同时避免灾难性遗忘。
*   **实现:** 项目提供了三个独立的训练脚本 (`train_stage_1_alignment.py`, `train_stage_2_pretrain_lora.py`, `train_stage_3_instruction_lora.py`)。每个脚本对应一个训练阶段，负责不同的学习任务，从最基础的视觉对齐到最复杂的指令遵循，层层递进。

### 组件三：高效LoRA训练策略 (角色：资源优化器)
*   **目标:** 在有限的计算资源下，实现接近全参数微调的训练效果，使项目能够在消费级硬件上运行。
*   **实现:** 在阶段二和阶段三的训练中，全面采用低秩适配（LoRA）技术。通过 `peft` 库，在冻结大部分基础模型参数的同时，只训练少量可注入的LoRA模块，极大地降低了显存和计算需求。

<br>

---

## 🔄 工作流程

**Aircraft-VLA** 的训练遵循一个清晰的、迭代的流程，模拟了人类专家知识的构建过程：

1.  **准备 (Preparation):** 首先，将 `special_tokens_vocab.py` 中的所有专业Token注入到基础VLA（如Qwen2-VL）的Tokenizer和词嵌入层中。
2.  **阶段一：对齐 (Alignment):** 运行 `train_stage_1_alignment.py`。此阶段冻结LLM主体，仅训练视觉Projector和新Token的Embedding。目标是让模型将图像块与Token建立初步联系。
3.  **阶段二：预训练 (Pre-training):** 运行 `train_stage_2_pretrain_lora.py`。此阶段应用LoRA，使用包含专业Token的图文描述数据进行训练，让模型学习Token在语言环境中的用法和逻辑。
4.  **阶段三：指令微调 (Instruction Fine-tuning):** 运行 `train_stage_3_instruction_lora.py`。继续使用LoRA，在高质量的问答和对话数据上进行训练，激发模型的分析、推理和交互能力。
5.  **完成 (Completion):** 训练结束后，得到一个轻量级的LoRA适配器。将其与基础模型结合，即可获得一个强大的航空航天专业VLA。

<br>

---

## 🚀 独特优势与创新

尽管通用的VLA能够识别飞机，但它们仍然在“常识”层面运行。在**专业性、精确度和结构化理解**方面，仍有巨大的改进空间。

**这正是 Aircraft-VLA 旨在深入探索和解决的方向。**

**Aircraft-VLA** 通过其独特的 **专业词汇表注入与分阶段LoRA微调** 架构，提供了以下优势：

*   **极高的信息密度:** 一个 `<op:flaps-landing>` Token比一句“飞机机翼后部的板子放下来用于着陆”的描述更精确、无歧义。
*   **结构化理解能力:** 模型被引导去学习飞行器的组成结构和运行逻辑，而不仅仅是识别孤立的物体。
*   **可控的分析与生成:** 在推理时，模型输出的专业Token可以直接用于后续的程序化分析或数据统计，实现了从非结构化视觉到结构化数据的转换。
*   **资源友好:** 全流程基于LoRA，使得在单张24GB显存的GPU上训练一个专家级VLA成为可能，极大地降低了研究和应用的门槛。

<br>

---

## 🛠️ 快速开始

### 1. 先决条件

*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+
*   Hugging Face账户以及对Qwen2-VL模型的访问权限

### 2. 安装

```bash
# 克隆仓库
git clone https://github.com/cotix-ai/Aircraft-VLA.git
cd Aircraft-VLA

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置与数据准备

1.  确保你已登录Hugging Face CLI: `huggingface-cli login`。
2.  将你所有的训练图片放入 `images/` 文件夹。
3.  根据[数据格式说明](#数据准备)，创建三个阶段的 `.jsonl` 数据文件并放入 `data/` 文件夹。

### 4. 运行训练

按照顺序执行三个训练脚本：

```bash
# 阶段一：视觉-Token对齐
accelerate launch train_stage_1_alignment.py

# 阶段二：使用LoRA进行预训练
accelerate launch train_stage_2_pretrain_lora.py

# 阶段三：使用LoRA进行指令微调
accelerate launch train_stage_3_instruction_lora.py
```
*注意: 你可能需要根据你的硬件调整脚本中的 `per_device_train_batch_size` 和 `gradient_accumulation_steps`。*

<br>

---

## 🤝 如何贡献

我们热烈欢迎对 **Aircraft-VLA** 项目的任何贡献！如果你有改进建议、发现了Bug，或是想扩充词汇表和数据集，请随时提交 Pull Request 或创建 Issue。
