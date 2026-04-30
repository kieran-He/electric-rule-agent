# 项目进展与规划

## 项目概述

基于 RAG 技术的电力政策知识问答系统，支持飞书机器人集成，提供多省份政策文档检索和智能问答服务。

## 已完成事项

### 1. 核心架构
- RAG混合检索系统（Vector + BM25 + BGE Reranker）
- 三阶段检索流程：向量检索 → BM25检索 → Reranker重排序

### 2. 多省份数据
已导入5个省份政策文档：
| 省份代码 | 省份名称 | 状态 |
|---------|---------|------|
| SD | 山东 | 已完成（1229条款，363页文档） |
| SN | 陕西 | 已完成 |
| SX | 山西 | 已完成 |
| GS | 甘肃 | 已完成 |
| AH | 安徽 | 已完成 |

### 3. 数据存储
- **ChromaDB** - 向量数据库已建立多个collection
- **BM25索引** - 各省份BM25索引已构建
  - `bm25_index.pkl` - 全局索引
  - `bm25_sn.pkl` - 陕西索引
  - `bm25_sx.pkl` - 山西索引
  - `bm25_gs.pkl` - 甘肃索引
  - `bm25_sd.pkl` - 山东索引
  - `bm25_ah.pkl` - 安徽索引

### 4. 核心功能模块
- Query Rewrite - LLM查询优化
- Query Expansion - 语义扩展
- Hybrid Retrieval - 混合检索
- Reranker - 重排序
- LLM Generation - 答案生成

### 5. 飞书集成
- 机器人服务已完成
- Webhook验证（Token + Signature）
- 事件去重
- 错误告警推送

### 6. 评估系统
- RAGAS评估框架已建立
- baseline实验结果已生成
- 支持指标：faithfulness, answer_relevancy, context_precision

### 7. 数据处理管道
- PDF/DOCX解析
- LLM智能分块
- 元数据提取
- 省份/类型/来源标记

## 当前项目路径

```
firstmodel/
├── app/                # 核心应用
│   ├── api/           # API路由
│   ├── core/          # 核心模块
│   ├── db/            # 数据库模型
│   ├── langchain/     # LangChain组件
│   ├── schemas/       # 数据模型
│   ├── services/      # 业务服务
│   └── utils/         # 工具函数
├── dataprocess/       # 数据处理管道
│   ├── pipeline.py    # 文档处理主流程
│   ├── parsers/       # PDF/DOCX解析器
│   └── chunkers/      # LLM智能分块
├── evaluation/        # RAGAS评估系统
│   ├── run_eval.py    # CLI评估入口
│   └── experiments/   # 实验数据
├── data/              # 数据存储
│   ├── docs/          # 原始政策文档（按省份）
│   ├── chroma/        # 向量数据库
│   ├── cache/         # BM25索引缓存
│   ├── dict/          # 字典数据
│   └── processed/     # 处理后数据
├── tools/             # 工具脚本
├── scripts/           # 辅助脚本
├── tests/             # 测试文件
└── docs/              # 文档
```

## 未来工作计划

### 短期（1-2周）
1. **数据扩展** - 添加更多省份政策文档
2. **性能优化** - 调优检索参数、降低延迟
3. **评估完善** - 建立持续评估机制，优化RAGAS指标

### 中期（1-2月）
4. **部署上线** - 生产环境配置、Docker化
5. **监控告警** - 完善指标追踪和异常告警
6. **文档完善** - API文档、运维手册

### 长期
7. **模型优化** - 探索更优的embedding和reranker模型
8. **功能扩展** - 多轮对话、知识图谱
9. **用户反馈闭环** - 建立用户反馈收集和模型迭代机制

## 技术栈

- **后端框架**: FastAPI
- **RAG框架**: LangChain
- **向量数据库**: ChromaDB
- **LLM**: GLM-4 / MINIMAX-M2.7
- **Embedding**: BAAI/bge-small-zh-v1.5
- **Reranker**: BAAI/bge-reranker-large

## 快速参考

详细使用说明请参阅 [README.md](README.md)