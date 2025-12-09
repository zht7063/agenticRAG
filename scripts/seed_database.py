"""
数据库填充脚本 - 生成演示数据

用于创建包含丰富示例数据的数据库，方便开发测试和演示使用。

功能：
1. 创建多篇经典论文数据
2. 组织论文到不同主题合集
3. 为论文添加笔记和批注
4. 创建实验记录
5. 记录研究灵感

使用方式：
    python scripts/seed_database.py
    或
    uv run python scripts/seed_database.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.database.connection import DatabaseConnection
from src.services.database.schema import SchemaManager
from src.services.database.repository import (
    Paper, Collection, Note, Experiment, Inspiration,
    PaperRepository, CollectionRepository, NoteRepository,
    ExperimentRepository, InspirationRepository
)
from src.utils.helpers.logger import get_logger

logger = get_logger("seed_database")


# ============================================================
# 示例数据定义
# ============================================================

SAMPLE_PAPERS = [
    # 原有 10 篇论文
    {
        "title": "Attention Is All You Need",
        "authors": "Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; et al.",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "keywords": "transformer, attention mechanism, neural networks, sequence modeling",
        "publish_date": "2017-06-12",
        "venue": "NeurIPS 2017",
        "doi": "10.48550/arXiv.1706.03762",
        "url": "https://arxiv.org/abs/1706.03762",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina",
        "abstract": "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.",
        "keywords": "BERT, pre-training, language model, transformers, NLP",
        "publish_date": "2018-10-11",
        "venue": "NAACL 2019",
        "doi": "10.48550/arXiv.1810.04805",
        "url": "https://arxiv.org/abs/1810.04805",
    },
    {
        "title": "Language Models are Few-Shot Learners",
        "authors": "Brown, Tom B.; Mann, Benjamin; Ryder, Nick; et al.",
        "abstract": "We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance. We train GPT-3, an autoregressive language model with 175 billion parameters, and test its few-shot learning abilities.",
        "keywords": "GPT-3, few-shot learning, language models, large scale, NLP",
        "publish_date": "2020-05-28",
        "venue": "NeurIPS 2020",
        "doi": "10.48550/arXiv.2005.14165",
        "url": "https://arxiv.org/abs/2005.14165",
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis, Patrick; Perez, Ethan; Piktus, Aleksandra; et al.",
        "abstract": "Large pre-trained language models have been shown to store factual knowledge in their parameters. However, their ability to access and precisely manipulate knowledge is still limited. We introduce RAG models that combine parametric memory with non-parametric retrieval.",
        "keywords": "RAG, retrieval, generation, knowledge, question answering",
        "publish_date": "2020-05-22",
        "venue": "NeurIPS 2020",
        "doi": "10.48550/arXiv.2005.11401",
        "url": "https://arxiv.org/abs/2005.11401",
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": "Wei, Jason; Wang, Xuezhi; Schuurmans, Dale; et al.",
        "abstract": "We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning.",
        "keywords": "chain-of-thought, reasoning, prompting, large language models",
        "publish_date": "2022-01-28",
        "venue": "NeurIPS 2022",
        "doi": "10.48550/arXiv.2201.11903",
        "url": "https://arxiv.org/abs/2201.11903",
    },
    {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": "Touvron, Hugo; Lavril, Thibaut; Izacard, Gautier; et al.",
        "abstract": "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models.",
        "keywords": "LLaMA, foundation models, open source, efficiency",
        "publish_date": "2023-02-27",
        "venue": "arXiv preprint",
        "doi": "10.48550/arXiv.2302.13971",
        "url": "https://arxiv.org/abs/2302.13971",
    },
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": "Yao, Shunyu; Zhao, Jeffrey; Yu, Dian; et al.",
        "abstract": "We present ReAct, a paradigm that synergizes reasoning and acting in large language models for solving diverse language reasoning and decision making tasks.",
        "keywords": "ReAct, reasoning, acting, agents, decision making",
        "publish_date": "2022-10-06",
        "venue": "ICLR 2023",
        "doi": "10.48550/arXiv.2210.03629",
        "url": "https://arxiv.org/abs/2210.03629",
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian",
        "abstract": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions.",
        "keywords": "ResNet, deep learning, image recognition, residual learning",
        "publish_date": "2015-12-10",
        "venue": "CVPR 2016",
        "doi": "10.48550/arXiv.1512.03385",
        "url": "https://arxiv.org/abs/1512.03385",
    },
    {
        "title": "Generative Adversarial Networks",
        "authors": "Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; et al.",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process. We simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data.",
        "keywords": "GAN, generative models, adversarial training, deep learning",
        "publish_date": "2014-06-10",
        "venue": "NeurIPS 2014",
        "doi": "10.48550/arXiv.1406.2661",
        "url": "https://arxiv.org/abs/1406.2661",
    },
    {
        "title": "A Survey on Retrieval-Augmented Text Generation",
        "authors": "Li, Huayang; Su, Yixuan; Cai, Deng; et al.",
        "abstract": "This paper surveys recent advances in retrieval-augmented text generation. We provide a comprehensive review of existing methods and discuss future directions.",
        "keywords": "survey, retrieval-augmented generation, text generation, NLP",
        "publish_date": "2022-02-17",
        "venue": "arXiv preprint",
        "doi": "10.48550/arXiv.2202.01110",
        "url": "https://arxiv.org/abs/2202.01110",
    },
    # 新增 20 篇论文
    {
        "title": "Scaling Laws for Neural Language Models",
        "authors": "Kaplan, Jared; McCandlish, Sam; Henighan, Tom; et al.",
        "abstract": "We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training.",
        "keywords": "scaling laws, language models, neural networks, power law",
        "publish_date": "2020-01-23",
        "venue": "arXiv preprint",
        "doi": "10.48550/arXiv.2001.08361",
        "url": "https://arxiv.org/abs/2001.08361",
    },
    {
        "title": "Training language models to follow instructions with human feedback",
        "authors": "Ouyang, Long; Wu, Jeff; Jiang, Xu; et al.",
        "abstract": "We show that reinforcement learning from human feedback can make language models more helpful, honest, and harmless. We demonstrate this with InstructGPT, which is trained to follow instructions.",
        "keywords": "RLHF, instruction following, human feedback, alignment",
        "publish_date": "2022-03-04",
        "venue": "NeurIPS 2022",
        "doi": "10.48550/arXiv.2203.02155",
        "url": "https://arxiv.org/abs/2203.02155",
    },
    {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu, Edward J.; Shen, Yelong; Wallis, Phillip; et al.",
        "abstract": "We propose Low-Rank Adaptation (LoRA), which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture.",
        "keywords": "LoRA, fine-tuning, parameter efficiency, adaptation",
        "publish_date": "2021-06-17",
        "venue": "ICLR 2022",
        "doi": "10.48550/arXiv.2106.09685",
        "url": "https://arxiv.org/abs/2106.09685",
    },
    {
        "title": "Improving Language Understanding by Generative Pre-Training",
        "authors": "Radford, Alec; Narasimhan, Karthik; Salimans, Tim; Sutskever, Ilya",
        "abstract": "We demonstrate that large gains on natural language understanding tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text.",
        "keywords": "GPT, pre-training, language understanding, generative models",
        "publish_date": "2018-06-11",
        "venue": "OpenAI Technical Report",
        "doi": "",
        "url": "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
    },
    {
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "authors": "Raffel, Colin; Shazeer, Noam; Roberts, Adam; et al.",
        "abstract": "We introduce T5, a unified framework that converts all text-based language problems into a text-to-text format. Our model and method achieve state-of-the-art results on many NLP benchmarks.",
        "keywords": "T5, transfer learning, text-to-text, encoder-decoder",
        "publish_date": "2019-10-23",
        "venue": "JMLR 2020",
        "doi": "10.48550/arXiv.1910.10683",
        "url": "https://arxiv.org/abs/1910.10683",
    },
    {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": "Bai, Yuntao; Kadavath, Saurav; Kundu, Sandipan; et al.",
        "abstract": "We propose Constitutional AI, a method for training harmless AI assistants through self-improvement, without human labels identifying harmful outputs.",
        "keywords": "Constitutional AI, AI safety, harmlessness, self-improvement",
        "publish_date": "2022-12-15",
        "venue": "arXiv preprint",
        "doi": "10.48550/arXiv.2212.08073",
        "url": "https://arxiv.org/abs/2212.08073",
    },
    {
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": "Schick, Timo; Dwivedi-Yu, Jane; Dessì, Roberto; et al.",
        "abstract": "We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.",
        "keywords": "tool use, API calling, language models, self-supervised",
        "publish_date": "2023-02-09",
        "venue": "arXiv preprint",
        "doi": "10.48550/arXiv.2302.04761",
        "url": "https://arxiv.org/abs/2302.04761",
    },
    {
        "title": "Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality",
        "authors": "Chiang, Wei-Lin; Li, Zhuohan; Lin, Zi; et al.",
        "abstract": "We introduce Vicuna-13B, an open-source chatbot trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT.",
        "keywords": "Vicuna, chatbot, open source, fine-tuning",
        "publish_date": "2023-03-30",
        "venue": "LMSYS Blog",
        "doi": "",
        "url": "https://lmsys.org/blog/2023-03-30-vicuna/",
    },
    {
        "title": "CLIP: Learning Transferable Visual Models From Natural Language Supervision",
        "authors": "Radford, Alec; Kim, Jong Wook; Hallacy, Chris; et al.",
        "abstract": "We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch.",
        "keywords": "CLIP, vision-language, contrastive learning, multimodal",
        "publish_date": "2021-02-26",
        "venue": "ICML 2021",
        "doi": "10.48550/arXiv.2103.00020",
        "url": "https://arxiv.org/abs/2103.00020",
    },
    {
        "title": "Diffusion Models Beat GANs on Image Synthesis",
        "authors": "Dhariwal, Prafulla; Nichol, Alex",
        "abstract": "We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We find that classifier guidance enables sampling from the model towards a particular class label.",
        "keywords": "diffusion models, image synthesis, generative models, guided sampling",
        "publish_date": "2021-05-11",
        "venue": "NeurIPS 2021",
        "doi": "10.48550/arXiv.2105.05233",
        "url": "https://arxiv.org/abs/2105.05233",
    },
    {
        "title": "Segment Anything",
        "authors": "Kirillov, Alexander; Mintun, Eric; Ravi, Nikhila; et al.",
        "abstract": "We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Our model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks.",
        "keywords": "SAM, segmentation, computer vision, foundation model",
        "publish_date": "2023-04-05",
        "venue": "ICCV 2023",
        "doi": "10.48550/arXiv.2304.02643",
        "url": "https://arxiv.org/abs/2304.02643",
    },
    {
        "title": "DenseNet: Densely Connected Convolutional Networks",
        "authors": "Huang, Gao; Liu, Zhuang; Van Der Maaten, Laurens; Weinberger, Kilian Q.",
        "abstract": "We propose a network architecture where each layer is directly connected to every other layer in a feed-forward fashion. DenseNets require substantially fewer parameters than traditional convolutional networks.",
        "keywords": "DenseNet, convolutional networks, dense connections, efficiency",
        "publish_date": "2016-08-25",
        "venue": "CVPR 2017",
        "doi": "10.48550/arXiv.1608.06993",
        "url": "https://arxiv.org/abs/1608.06993",
    },
    {
        "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        "authors": "Tan, Mingxing; Le, Quoc V.",
        "abstract": "We systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. We propose a new scaling method that uniformly scales all dimensions.",
        "keywords": "EfficientNet, model scaling, neural architecture, efficiency",
        "publish_date": "2019-05-28",
        "venue": "ICML 2019",
        "doi": "10.48550/arXiv.1905.11946",
        "url": "https://arxiv.org/abs/1905.11946",
    },
    {
        "title": "Vision Transformer (ViT): An Image is Worth 16x16 Words",
        "authors": "Dosovitskiy, Alexey; Beyer, Lucas; Kolesnikov, Alexander; et al.",
        "abstract": "We show that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks, matching or exceeding state-of-the-art convolutional networks.",
        "keywords": "ViT, vision transformer, image classification, attention",
        "publish_date": "2020-10-22",
        "venue": "ICLR 2021",
        "doi": "10.48550/arXiv.2010.11929",
        "url": "https://arxiv.org/abs/2010.11929",
    },
    {
        "title": "YOLO: You Only Look Once - Unified, Real-Time Object Detection",
        "authors": "Redmon, Joseph; Divvala, Santosh; Girshick, Ross; Farhadi, Ali",
        "abstract": "We present YOLO, a new approach to object detection. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation.",
        "keywords": "YOLO, object detection, real-time, computer vision",
        "publish_date": "2015-06-08",
        "venue": "CVPR 2016",
        "doi": "10.48550/arXiv.1506.02640",
        "url": "https://arxiv.org/abs/1506.02640",
    },
    {
        "title": "Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models",
        "authors": "Rombach, Robin; Blattmann, Andreas; Lorenz, Dominik; et al.",
        "abstract": "By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes.",
        "keywords": "Stable Diffusion, latent diffusion, image synthesis, text-to-image",
        "publish_date": "2021-12-20",
        "venue": "CVPR 2022",
        "doi": "10.48550/arXiv.2112.10752",
        "url": "https://arxiv.org/abs/2112.10752",
    },
    {
        "title": "Flamingo: a Visual Language Model for Few-Shot Learning",
        "authors": "Alayrac, Jean-Baptiste; Donahue, Jeff; Luc, Pauline; et al.",
        "abstract": "We introduce Flamingo, a family of Visual Language Models that can rapidly adapt to novel tasks using only a handful of annotated examples.",
        "keywords": "Flamingo, vision-language, few-shot learning, multimodal",
        "publish_date": "2022-04-29",
        "venue": "NeurIPS 2022",
        "doi": "10.48550/arXiv.2204.14198",
        "url": "https://arxiv.org/abs/2204.14198",
    },
    {
        "title": "Self-Instruct: Aligning Language Models with Self-Generated Instructions",
        "authors": "Wang, Yizhong; Kordi, Yeganeh; Mishra, Swaroop; et al.",
        "abstract": "We introduce Self-Instruct, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations.",
        "keywords": "self-instruct, instruction tuning, bootstrapping, alignment",
        "publish_date": "2022-12-20",
        "venue": "ACL 2023",
        "doi": "10.48550/arXiv.2212.10560",
        "url": "https://arxiv.org/abs/2212.10560",
    },
    {
        "title": "LangChain: Building Applications with LLMs through Composability",
        "authors": "Chase, Harrison",
        "abstract": "LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and can reason about how to answer based on provided context.",
        "keywords": "LangChain, LLM framework, composability, applications",
        "publish_date": "2022-10-25",
        "venue": "GitHub Repository",
        "doi": "",
        "url": "https://github.com/langchain-ai/langchain",
    },
    {
        "title": "GraphRAG: Knowledge Graph Enhanced Retrieval-Augmented Generation",
        "authors": "Edge, Darren; Trinh, Ha; Cheng, Newman; et al.",
        "abstract": "We present GraphRAG, which uses knowledge graphs to enhance retrieval-augmented generation by providing structured relationships between entities in retrieved documents.",
        "keywords": "GraphRAG, knowledge graph, retrieval, structured data",
        "publish_date": "2024-04-15",
        "venue": "Microsoft Research",
        "doi": "",
        "url": "https://www.microsoft.com/en-us/research/blog/graphrag/",
    },
]

SAMPLE_COLLECTIONS = [
    # 原有 5 个合集
    {
        "name": "Transformer 架构",
        "description": "自注意力机制和 Transformer 相关的经典论文",
        "tags": "transformer,attention,architecture",
    },
    {
        "name": "大语言模型",
        "description": "GPT、BERT、LLaMA 等大规模预训练语言模型",
        "tags": "LLM,pre-training,language model",
    },
    {
        "name": "检索增强生成 (RAG)",
        "description": "结合检索和生成的方法论文",
        "tags": "RAG,retrieval,generation",
    },
    {
        "name": "AI Agent",
        "description": "ReAct、Chain-of-Thought 等 Agent 相关研究",
        "tags": "agent,reasoning,planning",
    },
    {
        "name": "计算机视觉经典",
        "description": "ResNet、GAN 等 CV 领域里程碑论文",
        "tags": "computer vision,ResNet,GAN",
    },
    # 新增 10 个合集
    {
        "name": "模型训练与优化",
        "description": "模型训练技巧、优化方法和效率提升相关研究",
        "tags": "training,optimization,efficiency,scaling",
    },
    {
        "name": "指令微调与对齐",
        "description": "指令跟随、人类反馈强化学习和模型对齐技术",
        "tags": "instruction tuning,RLHF,alignment,fine-tuning",
    },
    {
        "name": "参数高效微调",
        "description": "LoRA、Adapter 等参数高效的模型适应方法",
        "tags": "PEFT,LoRA,adapter,efficiency",
    },
    {
        "name": "多模态学习",
        "description": "视觉-语言模型、跨模态理解与生成",
        "tags": "multimodal,vision-language,CLIP,cross-modal",
    },
    {
        "name": "图像生成",
        "description": "扩散模型、GAN 和图像合成技术",
        "tags": "image generation,diffusion,GAN,synthesis",
    },
    {
        "name": "计算机视觉架构",
        "description": "CNN、ViT 等视觉模型架构创新",
        "tags": "CNN,ViT,architecture,vision",
    },
    {
        "name": "目标检测与分割",
        "description": "YOLO、SAM 等检测和分割算法",
        "tags": "object detection,segmentation,YOLO,SAM",
    },
    {
        "name": "工具使用与 Agent 框架",
        "description": "模型工具调用、Agent 开发框架和应用构建",
        "tags": "tool use,agent framework,API,LangChain",
    },
    {
        "name": "AI 安全与对齐",
        "description": "AI 安全性、可解释性和伦理对齐研究",
        "tags": "AI safety,alignment,Constitutional AI,harmlessness",
    },
    {
        "name": "开源模型生态",
        "description": "开源大模型、社区贡献和模型评测",
        "tags": "open source,community,LLaMA,Vicuna",
    },
]

SAMPLE_NOTES = [
    # 原有 8 条笔记
    {
        "paper_index": 0,  # Attention Is All You Need
        "content": "核心创新：完全基于自注意力机制，摒弃了 RNN 和 CNN 结构，实现了更好的并行化训练。",
        "note_type": "highlight",
        "page_number": 1,
    },
    {
        "paper_index": 0,
        "content": "Multi-head attention 允许模型关注不同位置的不同表示子空间的信息。",
        "note_type": "comment",
        "page_number": 5,
    },
    {
        "paper_index": 1,  # BERT
        "content": "双向编码是关键：同时使用上下文信息进行预训练，优于传统的单向语言模型。",
        "note_type": "highlight",
        "page_number": 2,
    },
    {
        "paper_index": 1,
        "content": "Masked Language Model (MLM) 预训练任务是如何设计的？掩码比例为什么选择 15%？",
        "note_type": "question",
        "page_number": 4,
    },
    {
        "paper_index": 3,  # RAG
        "content": "RAG 结合了参数化记忆（模型参数）和非参数化检索（外部知识库），在知识密集型任务上效果显著。",
        "note_type": "highlight",
        "page_number": 1,
    },
    {
        "paper_index": 3,
        "content": "这种方法特别适合需要准确引用外部知识的场景，比如科研助手、问答系统。",
        "note_type": "comment",
        "page_number": 8,
    },
    {
        "paper_index": 4,  # Chain-of-Thought
        "content": "通过生成中间推理步骤，大幅提升了 LLM 在复杂推理任务上的表现。",
        "note_type": "highlight",
        "page_number": 3,
    },
    {
        "paper_index": 6,  # ReAct
        "content": "ReAct 框架展示了如何让 LLM 既能推理又能行动，与工具调用结合效果很好。",
        "note_type": "comment",
        "page_number": 2,
    },
    # 新增 16 条笔记
    {
        "paper_index": 2,  # GPT-3
        "content": "模型规模扩大到 175B 参数，展示了 few-shot learning 的惊人能力，无需梯度更新即可适应新任务。",
        "note_type": "highlight",
        "page_number": 1,
    },
    {
        "paper_index": 2,
        "content": "在上下文中提供少量示例就能让模型理解任务，这种能力随模型规模增大而显著提升。",
        "note_type": "comment",
        "page_number": 5,
    },
    {
        "paper_index": 5,  # LLaMA
        "content": "开源模型的重要里程碑：LLaMA-13B 在多数基准上超越 GPT-3(175B)，证明了训练效率的重要性。",
        "note_type": "highlight",
        "page_number": 2,
    },
    {
        "paper_index": 10,  # Scaling Laws
        "content": "模型性能与规模遵循幂律关系，为模型设计和资源分配提供了理论指导。",
        "note_type": "comment",
        "page_number": 3,
    },
    {
        "paper_index": 11,  # InstructGPT
        "content": "RLHF 是对齐 LLM 的关键技术：通过人类反馈优化模型行为，使其更有帮助、诚实且无害。",
        "note_type": "highlight",
        "page_number": 1,
    },
    {
        "paper_index": 11,
        "content": "相比单纯扩大模型规模，对齐训练对提升模型实用性的效果更显著。",
        "note_type": "comment",
        "page_number": 7,
    },
    {
        "paper_index": 12,  # LoRA
        "content": "通过低秩矩阵分解实现参数高效微调，显著降低了大模型适配的成本和门槛。",
        "note_type": "highlight",
        "page_number": 2,
    },
    {
        "paper_index": 12,
        "content": "LoRA 如何选择合适的秩 r？不同层使用不同秩是否有效？",
        "note_type": "question",
        "page_number": 5,
    },
    {
        "paper_index": 13,  # GPT-1
        "content": "生成式预训练 + 判别式微调的两阶段范式，奠定了现代预训练语言模型的基础。",
        "note_type": "highlight",
        "page_number": 1,
    },
    {
        "paper_index": 14,  # T5
        "content": "将所有 NLP 任务统一为 text-to-text 格式，简化了模型架构和训练流程。",
        "note_type": "comment",
        "page_number": 2,
    },
    {
        "paper_index": 16,  # Toolformer
        "content": "模型自主学习何时、如何调用外部工具，无需大量人工标注，展示了自监督学习的潜力。",
        "note_type": "highlight",
        "page_number": 3,
    },
    {
        "paper_index": 18,  # CLIP
        "content": "对比学习 + 大规模图文对预训练，实现了强大的零样本迁移能力。",
        "note_type": "highlight",
        "page_number": 1,
    },
    {
        "paper_index": 19,  # Diffusion Models
        "content": "扩散模型通过逐步去噪生成高质量图像，在质量上超越了 GAN，成为图像生成的新范式。",
        "note_type": "comment",
        "page_number": 4,
    },
    {
        "paper_index": 20,  # SAM
        "content": "Segment Anything 展示了视觉基础模型的可能性：通过 prompt 实现零样本分割。",
        "note_type": "highlight",
        "page_number": 2,
    },
    {
        "paper_index": 23,  # ViT
        "content": "纯 Transformer 架构应用于图像分类，挑战了 CNN 在视觉领域的统治地位。",
        "note_type": "highlight",
        "page_number": 1,
    },
    {
        "paper_index": 23,
        "content": "ViT 需要大规模数据才能发挥优势，数据效率是否能进一步提升？",
        "note_type": "question",
        "page_number": 6,
    },
]

SAMPLE_EXPERIMENTS = [
    # 原有 4 个实验
    {
        "name": "BERT 微调实验 - 文本分类",
        "description": "在自定义科研文献分类数据集上微调 BERT-base 模型",
        "parameters": '{"model": "bert-base-uncased", "learning_rate": 2e-5, "epochs": 3, "batch_size": 16}',
        "results": '{"accuracy": 0.92, "f1_score": 0.91, "training_time": "45min"}',
        "related_papers": "[2]",  # BERT 论文
        "status": "completed",
    },
    {
        "name": "RAG 检索性能测试",
        "description": "测试不同检索策略对 RAG 系统性能的影响",
        "parameters": '{"retrieval_method": "dense", "top_k": 5, "embedding_model": "sentence-transformers"}',
        "results": '{"retrieval_precision": 0.87, "answer_quality": 0.85}',
        "related_papers": "[4]",  # RAG 论文
        "status": "completed",
    },
    {
        "name": "Transformer 注意力可视化",
        "description": "分析 Transformer 各层注意力头的功能分工",
        "parameters": '{"model": "bert-base", "layers": [0, 6, 11], "visualization_tool": "bertviz"}',
        "results": "",
        "related_papers": "[1, 2]",  # Transformer & BERT
        "status": "running",
    },
    {
        "name": "多 Agent 协同实验",
        "description": "测试 Master-Worker 架构在复杂任务上的表现",
        "parameters": '{"num_workers": 4, "task_type": "literature_review", "coordination_strategy": "sequential"}',
        "results": "",
        "related_papers": "[7]",  # ReAct
        "status": "planned",
    },
    # 新增 8 个实验
    {
        "name": "LoRA 微调 LLaMA - 指令跟随",
        "description": "使用 LoRA 方法在中文指令数据集上微调 LLaMA-7B",
        "parameters": '{"base_model": "LLaMA-7B", "lora_rank": 8, "lora_alpha": 16, "learning_rate": 3e-4, "epochs": 5}',
        "results": '{"instruction_accuracy": 0.88, "response_quality": 4.2, "training_time": "3.5h"}',
        "related_papers": "[6, 13]",
        "status": "completed",
    },
    {
        "name": "CLIP 零样本图像分类",
        "description": "评估 CLIP 在自定义图像分类任务上的零样本性能",
        "parameters": '{"model": "CLIP-ViT-B/32", "dataset": "custom_dataset", "num_classes": 20}',
        "results": '{"zero_shot_accuracy": 0.76, "few_shot_accuracy": 0.84}',
        "related_papers": "[19]",
        "status": "completed",
    },
    {
        "name": "Stable Diffusion 图像生成优化",
        "description": "测试不同 guidance scale 和采样步数对生成质量的影响",
        "parameters": '{"model": "stable-diffusion-v2", "guidance_scales": [7.5, 10, 15], "steps": [20, 50, 100]}',
        "results": '{"optimal_guidance": 10, "optimal_steps": 50, "fid_score": 12.3}',
        "related_papers": "[26]",
        "status": "completed",
    },
    {
        "name": "SAM 分割 Prompt 策略研究",
        "description": "对比不同 prompt 类型（点、框、mask）在分割任务上的效果",
        "parameters": '{"model": "SAM-ViT-H", "prompt_types": ["point", "box", "mask"], "datasets": ["COCO", "ADE20K"]}',
        "results": "",
        "related_papers": "[21]",
        "status": "running",
    },
    {
        "name": "GPT-3 Few-shot 学习能力评估",
        "description": "在多个下游任务上测试 GPT-3 的 few-shot 学习性能",
        "parameters": '{"model": "GPT-3-davinci", "shot_numbers": [0, 1, 5, 10], "tasks": ["classification", "QA", "translation"]}',
        "results": "",
        "related_papers": "[3]",
        "status": "running",
    },
    {
        "name": "Vision Transformer 数据效率研究",
        "description": "分析 ViT 在不同数据规模下的性能表现",
        "parameters": '{"model": "ViT-B/16", "data_sizes": ["1K", "10K", "100K", "1M"], "augmentation": "RandAugment"}',
        "results": "",
        "related_papers": "[24]",
        "status": "planned",
    },
    {
        "name": "Constitutional AI 安全性测试",
        "description": "评估 Constitutional AI 方法在减少有害输出方面的效果",
        "parameters": '{"base_model": "LLaMA-13B", "constitution_version": "v1", "test_prompts": 1000}',
        "results": "",
        "related_papers": "[16]",
        "status": "planned",
    },
    {
        "name": "GraphRAG 知识图谱增强检索",
        "description": "对比传统 RAG 与 GraphRAG 在复杂问答任务上的表现",
        "parameters": '{"graph_type": "entity-relation", "retrieval_depth": 2, "top_k": 10}',
        "results": "",
        "related_papers": "[4, 30]",
        "status": "planned",
    },
]

SAMPLE_INSPIRATIONS = [
    # 原有 5 条灵感
    {
        "title": "多模态 RAG 系统",
        "content": "结合文本、图表、公式的多模态检索增强生成系统，特别适合科研论文理解。可以检索论文中的图表并结合文本进行综合分析。",
        "source_papers": "[4, 10]",  # RAG 相关论文
        "priority": "high",
        "status": "new",
    },
    {
        "title": "自反思 Agent 架构优化",
        "content": "在 Master Agent 中加入自我评估和迭代优化机制，根据答案质量决定是否需要重新检索或调用其他 Worker。",
        "source_papers": "[5, 7]",  # CoT & ReAct
        "priority": "high",
        "status": "exploring",
    },
    {
        "title": "论文引用关系图谱",
        "content": "构建论文之间的引用网络图，通过图算法发现研究热点、学术脉络和潜在的研究空白。",
        "source_papers": "[1, 2, 3, 4]",
        "priority": "medium",
        "status": "new",
    },
    {
        "title": "实验参数智能推荐",
        "content": "基于历史实验数据和相关论文，使用机器学习推荐最优的实验参数组合。",
        "source_papers": "[2, 3]",
        "priority": "medium",
        "status": "archived",
    },
    {
        "title": "领域特定的文献综述生成",
        "content": "针对特定研究领域，自动生成结构化的文献综述，包括研究背景、方法对比、趋势分析等章节。",
        "source_papers": "[4, 10]",
        "priority": "low",
        "status": "new",
    },
    # 新增 10 条灵感
    {
        "title": "模型规模与性能预测器",
        "content": "基于 Scaling Laws 开发工具，预测不同模型规模、数据量和计算资源下的性能表现，辅助资源分配决策。",
        "source_papers": "[11]",
        "priority": "high",
        "status": "new",
    },
    {
        "title": "混合 PEFT 方法研究",
        "content": "结合 LoRA、Adapter、Prefix-tuning 等多种参数高效微调方法，探索最优的模型适配策略。",
        "source_papers": "[13]",
        "priority": "high",
        "status": "exploring",
    },
    {
        "title": "跨语言知识迁移",
        "content": "研究如何利用英文预训练模型的知识高效地迁移到中文等其他语言，降低多语言模型训练成本。",
        "source_papers": "[2, 6, 14]",
        "priority": "medium",
        "status": "new",
    },
    {
        "title": "工具学习的自动化",
        "content": "扩展 Toolformer 思想，让模型自主发现、学习和组合新工具，提升 Agent 的自主性和适应性。",
        "source_papers": "[17, 29]",
        "priority": "high",
        "status": "new",
    },
    {
        "title": "视觉-语言模型的零样本迁移",
        "content": "研究 CLIP、Flamingo 等模型在医疗、遥感等专业领域的零样本迁移能力，探索领域适配方法。",
        "source_papers": "[19, 27]",
        "priority": "medium",
        "status": "exploring",
    },
    {
        "title": "扩散模型加速优化",
        "content": "研究扩散模型的采样加速方法，在保持生成质量的同时减少推理时间，提升实用性。",
        "source_papers": "[20, 26]",
        "priority": "medium",
        "status": "new",
    },
    {
        "title": "可解释的注意力机制",
        "content": "开发工具可视化和解释 Transformer 注意力模式，帮助理解模型决策过程，提升模型可信度。",
        "source_papers": "[1, 24]",
        "priority": "low",
        "status": "new",
    },
    {
        "title": "联邦学习与隐私保护",
        "content": "在大模型微调中引入联邦学习，保护数据隐私的同时利用分布式数据改进模型。",
        "source_papers": "[2, 13]",
        "priority": "medium",
        "status": "new",
    },
    {
        "title": "模型压缩与蒸馏",
        "content": "将大模型的知识蒸馏到小模型，在资源受限场景下部署高性能模型，平衡效果与效率。",
        "source_papers": "[6, 11]",
        "priority": "low",
        "status": "archived",
    },
    {
        "title": "Agentic 工作流编排",
        "content": "设计更灵活的 Agent 工作流编排系统，支持动态任务分解、并行执行和错误恢复，提升复杂任务处理能力。",
        "source_papers": "[7, 29]",
        "priority": "high",
        "status": "exploring",
    },
]

# 合集与论文的映射关系（合集索引: [论文索引列表]）
COLLECTION_PAPER_MAPPING = {
    0: [0, 1, 2, 14],                    # Transformer 架构: Attention, BERT, GPT-3, T5
    1: [1, 2, 5, 13, 17],                # 大语言模型: BERT, GPT-3, LLaMA, GPT-1, Vicuna
    2: [3, 9, 29],                       # RAG: RAG 论文, Survey, GraphRAG
    3: [4, 6, 16],                       # AI Agent: CoT, ReAct, Toolformer
    4: [7, 8, 21, 22],                   # CV 经典: ResNet, GAN, DenseNet, EfficientNet
    5: [10, 11, 12],                     # 模型训练与优化: Scaling Laws, InstructGPT, LoRA
    6: [11, 12, 27, 28],                 # 指令微调与对齐: InstructGPT, LoRA, Self-Instruct, Constitutional AI
    7: [12, 13],                         # 参数高效微调: LoRA, GPT-1 (demonstration)
    8: [18, 19, 26, 27],                 # 多模态学习: CLIP, Diffusion Models, Flamingo
    9: [8, 19, 20, 25],                  # 图像生成: GAN, Diffusion, SAM, Stable Diffusion
    10: [7, 21, 22, 23, 24],             # CV 架构: ResNet, DenseNet, EfficientNet, ViT, YOLO
    11: [20, 24],                        # 目标检测与分割: SAM, YOLO
    12: [6, 16, 28, 29],                 # 工具使用与框架: ReAct, Toolformer, LangChain, GraphRAG
    13: [15, 28],                        # AI 安全与对齐: Constitutional AI, (related to alignment)
    14: [5, 13, 17],                     # 开源模型生态: LLaMA, GPT-1, Vicuna
}


# ============================================================
# 数据填充函数
# ============================================================

def seed_papers(paper_repo: PaperRepository) -> list[int]:
    """填充论文数据"""
    logger.info(f"开始填充 {len(SAMPLE_PAPERS)} 篇论文...")
    
    paper_ids = []
    for paper_data in SAMPLE_PAPERS:
        paper = Paper(**paper_data)
        paper_id = paper_repo.create(paper)
        paper_ids.append(paper_id)
        logger.debug(f"  创建论文: {paper.title[:50]}... (ID={paper_id})")
    
    logger.info(f"✓ 完成论文填充，共 {len(paper_ids)} 篇")
    return paper_ids


def seed_collections(collection_repo: CollectionRepository, paper_ids: list[int]):
    """填充合集数据"""
    logger.info(f"开始填充 {len(SAMPLE_COLLECTIONS)} 个合集...")
    
    collection_ids = []
    for coll_data in SAMPLE_COLLECTIONS:
        collection = Collection(**coll_data)
        coll_id = collection_repo.create(collection)
        collection_ids.append(coll_id)
        logger.debug(f"  创建合集: {collection.name} (ID={coll_id})")
    
    # 建立合集与论文的关联
    logger.info("建立合集-论文关联关系...")
    for coll_index, paper_indices in COLLECTION_PAPER_MAPPING.items():
        coll_id = collection_ids[coll_index]
        for paper_index in paper_indices:
            paper_id = paper_ids[paper_index]
            collection_repo.add_paper(coll_id, paper_id)
            logger.debug(f"  添加论文 {paper_id} 到合集 {coll_id}")
    
    logger.info(f"✓ 完成合集填充，共 {len(collection_ids)} 个")
    return collection_ids


def seed_notes(note_repo: NoteRepository, paper_ids: list[int]):
    """填充笔记数据"""
    logger.info(f"开始填充 {len(SAMPLE_NOTES)} 条笔记...")
    
    note_ids = []
    for note_data in SAMPLE_NOTES:
        paper_index = note_data.pop("paper_index")
        paper_id = paper_ids[paper_index]
        
        note = Note(paper_id=paper_id, **note_data)
        note_id = note_repo.create(note)
        note_ids.append(note_id)
        logger.debug(f"  创建笔记: {note.note_type} (ID={note_id})")
    
    logger.info(f"✓ 完成笔记填充，共 {len(note_ids)} 条")
    return note_ids


def seed_experiments(experiment_repo: ExperimentRepository):
    """填充实验数据"""
    logger.info(f"开始填充 {len(SAMPLE_EXPERIMENTS)} 个实验...")
    
    exp_ids = []
    for exp_data in SAMPLE_EXPERIMENTS:
        experiment = Experiment(**exp_data)
        exp_id = experiment_repo.create(experiment)
        exp_ids.append(exp_id)
        logger.debug(f"  创建实验: {experiment.name} (ID={exp_id}, status={experiment.status})")
    
    logger.info(f"✓ 完成实验填充，共 {len(exp_ids)} 个")
    return exp_ids


def seed_inspirations(inspiration_repo: InspirationRepository):
    """填充灵感数据"""
    logger.info(f"开始填充 {len(SAMPLE_INSPIRATIONS)} 条灵感...")
    
    idea_ids = []
    for idea_data in SAMPLE_INSPIRATIONS:
        inspiration = Inspiration(**idea_data)
        idea_id = inspiration_repo.create(inspiration)
        idea_ids.append(idea_id)
        logger.debug(f"  创建灵感: {inspiration.title} (ID={idea_id}, priority={inspiration.priority})")
    
    logger.info(f"✓ 完成灵感填充，共 {len(idea_ids)} 条")
    return idea_ids


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：创建并填充演示数据库"""
    logger.info("=" * 60)
    logger.info("开始生成演示数据库")
    logger.info("=" * 60)
    
    # 数据库路径
    db_path = project_root / "data" / "scholar_demo.db"
    
    # 如果数据库已存在，询问是否覆盖
    if db_path.exists():
        logger.warning(f"数据库文件已存在: {db_path}")
        response = input("是否覆盖现有数据库？(y/n): ")
        if response.lower() != 'y':
            logger.info("操作已取消")
            return
        db_path.unlink()
        logger.info("已删除旧数据库文件")
    
    # 创建数据库连接
    logger.info(f"创建数据库: {db_path}")
    db = DatabaseConnection(str(db_path))
    db.connect()
    
    # 初始化 Schema
    logger.info("初始化数据库 Schema...")
    schema_manager = SchemaManager(db)
    schema_manager.initialize_schema()
    logger.info("✓ Schema 初始化完成")
    
    # 创建 Repository 实例
    paper_repo = PaperRepository(db)
    collection_repo = CollectionRepository(db)
    note_repo = NoteRepository(db)
    experiment_repo = ExperimentRepository(db)
    inspiration_repo = InspirationRepository(db)
    
    # 填充数据
    logger.info("")
    logger.info("开始填充示例数据...")
    logger.info("-" * 60)
    
    paper_ids = seed_papers(paper_repo)
    collection_ids = seed_collections(collection_repo, paper_ids)
    note_ids = seed_notes(note_repo, paper_ids)
    exp_ids = seed_experiments(experiment_repo)
    idea_ids = seed_inspirations(inspiration_repo)
    
    # 统计信息
    logger.info("")
    logger.info("=" * 60)
    logger.info("数据填充完成！")
    logger.info("=" * 60)
    logger.info(f"数据库路径: {db_path}")
    logger.info(f"")
    logger.info(f"数据统计:")
    logger.info(f"  - 论文:   {paper_repo.count()} 篇")
    logger.info(f"  - 合集:   {len(collection_ids)} 个")
    logger.info(f"  - 笔记:   {len(note_ids)} 条")
    logger.info(f"  - 实验:   {len(exp_ids)} 个")
    logger.info(f"  - 灵感:   {len(idea_ids)} 条")
    logger.info("")
    logger.info("可以使用以下代码连接数据库:")
    logger.info(f"  from src.services.database.connection import DatabaseConnection")
    logger.info(f"  db = DatabaseConnection('{db_path}')")
    logger.info(f"  db.connect()")
    logger.info("=" * 60)
    
    # 关闭连接
    db.close()


if __name__ == "__main__":
    main()

