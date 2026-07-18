# Annotation Transfer Machine (ATM)

信息检索系统的检索、标注与评估平台，支持稀疏和稠密检索模型、相关性标注及性能分析。

## 项目结构

```
Annotation-Transfer-Machine/
├── config/
│   ├── config_top200.json           # Top-200 检索配置
│   └── rerank_top200_config.json    # 重排配置
├── data/
│   ├── NQ/                          # Natural Questions 数据集
│   │   ├── beir-nq/                 # BEIR 格式 NQ 数据
│   │   ├── flashrag_nq/             # FlashRAG 格式 NQ 查询
│   │   ├── flashrag_wiki_100w/      # FlashRAG Wiki 语料
│   │   └── gold_doc.tsv             # 相关性标注数据
│   └── elec/                        # 电力领域测试语料
│       ├── IEEE_1547_2003_sentences.jsonl
│       ├── IEEE_1547_2018_sentences.jsonl
│       └── qa_2003_gg.jsonl
├── outputs/
│   └── index/                       # 索引输出目录
│       ├── BGE_INDEX/               # BGE 向量索引
│       ├── BM25_index/              # BM25 稀疏索引
│       ├── E5_index/                # E5 向量索引
│       └── E5-large_index/          # E5-large 向量索引
├── scripts/                         # 可执行脚本
│   ├── build_index.py               # 索引构建
│   ├── bm25_retriever.py            # BM25 检索器
│   ├── retrieve_top50.py            # Top-K 检索
│   ├── encoders.py                  # 向量编码器
│   ├── faiss_builder.py             # FAISS 索引构建
│   ├── rerank_and_evaluate.py       # 重排与评估
│   ├── generate_pr_curves.py        # PR 曲线生成
│   ├── auc_pr_optimization.py       # AUC-PR 优化分析
│   ├── config.json                  # 脚本配置
│   ├── config.py                    # 配置工具
│   ├── run_build.sh                 # 索引构建启动脚本
│   ├── run_retrieval.sh             # 检索启动脚本
│   └── monitor_rerank_progress.sh   # 重排进度监控
└── src/
    └── models/                      # 模型代码
```

## 主要功能

### 多模型检索
- **BM25**：基于词频的稀疏检索（Pyserini 实现）
- **BGE**：基于向量的稠密检索（FAISS 索引）
- **E5 / E5-large**：基于向量的稠密检索（FAISS 索引）
- 支持多模型并行检索

### 索引构建
- BM25 倒排索引（Pyserini / Anserini）
- BGE / E5 稠密向量索引（FAISS）
- 支持大型语料的批量索引构建

### 重排与评估
- 加载检索结果和相关性标注
- 应用不同重排策略
- 计算评估指标并生成对比报告

### 性能分析
- **PR 曲线**：精确率-召回率曲线分析与可视化
- **AUC-PR**：曲线下面积计算与优化
- **多维度对比**：不同检索器、模型的表现对比

### 相关性标注
- 基于大语言模型的自动相关性标注
- 标注结果用于评估和优化

### 领域数据
- 支持通用领域（NQ）和垂直领域（电力）的检索评估

## 快速开始

### 环境要求
```bash
pip install numpy pandas matplotlib scikit-learn tqdm
```

### 构建索引
```bash
cd scripts
./run_build.sh
```

### 执行检索
```bash
./run_retrieval.sh
```

### 重排评估
```bash
python rerank_and_evaluate.py --config ../config/config_top200.json
```

### 生成 PR 曲线
```bash
python generate_pr_curves.py
```

## 数据说明

### 标注文件 (gold_doc.tsv)
| 列名 | 说明 |
|------|------|
| flashrag_qid | FlashRAG 查询 ID |
| beir_qid | BEIR 查询 ID |
| beir_docid | BEIR 文档 ID |
| wiki_100w_id | Wiki 文档 ID |
| Relevance | 相关性标注（1=相关，0=不相关） |

### 数据集
- **NQ (Natural Questions)**：通用问答数据集，含 BEIR 和 FlashRAG 两种格式
- **Elec**：电力领域 IEEE 标准文档测试语料

## 配置

配置文件为 JSON 格式，主要参数在 `scripts/config.json` 中定义。

## 许可证

MIT
