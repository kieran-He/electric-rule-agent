# Phase 2 Ragas Integration - 完整文档

## Phase 2 实施成果

### 已完成功能

✅ **GLM-Ragas兼容性测试** (`tests/test_glm_ragas_compatibility.py`)
- Mock Ragas evaluator验证
- GLM endpoint connectivity测试
- Ragas integration测试
- 完整错误处理

✅ **Ragas配置管理** (`evaluation/ragas_config.py`)
- `RagasConfig`数据类（支持环境变量和配置文件）
- `RagasBatchProcessor`批量处理优化器
- 配置验证和导出功能
- 进度监控集成
- 结果缓存支持

✅ **批量处理优化**
- 可配置batch_size (1-100)
- 自动分批处理大规模benchmark
- 失败批次错误恢复
- 进度日志输出

✅ **进度监控**
- 批次进度实时显示
- 平均分数即时反馈
- 错误告警日志

✅ **HTML报告Ragas可视化**
- 新增"LLM Quality Metrics"部分
- faithfulness/answer_relevancy/context_precision展示
- 目标阈值对比（≥0.85）
- 状态标识（PASS/FAIL）

✅ **CLI增强**
- `--ragas-config`配置文件路径参数
- 自动配置加载和环境导出
- 配置验证和降级机制

---

## 使用方法

### 1. 配置Ragas

**方式1：配置文件**

```bash
# 创建默认配置
python evaluation/ragas_config.py

# 编辑 evaluation/ragas_config.json
{
  "enabled": true,
  "use_mock": false,  # 或true用于测试
  "llm_endpoint": "https://your-glm-endpoint.com",
  "llm_api_key": "your-api-key",
  "batch_size": 10,
  "timeout_seconds": 60
}
```

**方式2：环境变量**

```bash
# Windows PowerShell
$env:RAGAS_ENABLED="true"
$env:RAGAS_ENDPOINT="https://glm-endpoint.com"
$env:RAGAS_API_KEY="your-key"
$env:RAGAS_BATCH_SIZE="10"

# Linux/Mac
export RAGAS_ENABLED=true
export RAGAS_ENDPOINT=https://glm-endpoint.com
export RAGAS_API_KEY=your-key
```

### 2. 运行Ragas评估

```bash
# 使用mock evaluator测试（无需配置）
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas --mock-ragas

# 使用配置文件
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas --ragas-config evaluation/ragas_config.json

# 使用环境变量配置
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas
```

### 3. 批量处理优化

**配置batch_size**：
- 小batch (1-5): 精确控制，适合调试
- 中batch (10-20): 平衡性能，推荐值
- 大batch (50-100): 高吞吐，适合大规模测试

```bash
# 小batch调试
python evaluation/run_eval.py run --ragas
# (在ragas_config.json设置batch_size=5)

# 大batch测试
# batch_size=50
```

### 4. 查看结果

**JSON报告**：
```json
{
  "metrics": {
    "faithfulness": 0.87,
    "answer_relevancy": 0.85,
    "context_precision": 0.82,
    ...
  }
}
```

**HTML报告**：
- 新增"LLM Quality Metrics"表格
- 显示3项Ragas指标及达标状态

---

## 批量处理原理

### RagasBatchProcessor工作流程

```
100条问题
  ↓
分成10批（每批10条）
  ↓
批次1: items 1-10  → Ragas evaluate → scores[1-10]
批次2: items 11-20 → Ragas evaluate → scores[11-20]
...
批次10: items 91-100 → Ragas evaluate → scores[91-100]
  ↓
合并所有分数 + 计算平均值
  ↓
返回完整结果
```

### 错误恢复机制

- 批次失败：自动填充0值，记录错误日志
- 继续处理后续批次
- 最终报告包含失败批次警告

---

## 性能优化建议

### 1. batch_size调优

**经验值**：
- GLM endpoint: batch_size=10-20
- OpenAI: batch_size=20-30
- Mock evaluator: batch_size=50+

### 2. 缓存启用

```json
{
  "cache_results": true,
  "cache_dir": "evaluation/.ragas_cache"
}
```

### 3. 并发处理（未来）

Phase 3计划添加：
- 多batch并发执行
- 异步API调用
- GPU加速（如果可用）

---

## GLM兼容性验证

### 运行兼容性测试

```bash
# Mock测试（无需配置）
python tests/test_glm_ragas_compatibility.py --mock

# GLM真实测试（需配置）
python tests/test_glm_ragas_compatibility.py
```

### 检查清单

✅ GLM endpoint可访问  
✅ API认证成功  
✅ Response格式匹配  
✅ Ragas evaluation完成  
✅ Scores在有效范围[0,1]  

### 常见问题

**问题1：GLM endpoint不兼容**
```
错误: Ragas evaluation failed: API format mismatch
解决: 使用Mock evaluator或实现自定义LLM wrapper
```

**问题2：超时**
```
错误: Batch processing timeout after 60s
解决: 增大timeout_seconds或减小batch_size
```

**问题3：分数异常**
```
错误: faithfulness score = 150 (out of range)
解决: 检查GLM response格式，可能需要自定义解析
```

---

## 配置参数详解

### RagasConfig完整参数

```python
@dataclass
class RagasConfig:
    enabled: bool = False              # 是否启用Ragas
    llm_endpoint: str = ""             # LLM API endpoint
    llm_api_key: str = ""              # LLM API key
    llm_model: str = "glm-4"           # 模型名称
    batch_size: int = 10               # 批次大小
    timeout_seconds: int = 60          # 超时秒数
    use_mock: bool = False             # 使用mock evaluator
    max_retries: int = 2               # 最大重试次数
    
    enable_progress_monitor: bool = True   # 进度监控
    cache_results: bool = True             # 结果缓存
    cache_dir: str = ".ragas_cache"        # 缓存目录
    
    metrics: List[str] = [
        "faithfulness", 
        "answer_relevancy", 
        "context_precision"
    ]
```

---

## 下一步计划

### Phase 3: 完整Benchmark与报告

1. 扩展benchmark到200+条（覆盖所有场景）
2. 实现跨文档综合问题
3. 添加多轮对话benchmark
4. 完善拒答类问题场景

### Phase 4: 流程与拒答指标

1. flow_complete_rate详细计算
2. context_continuation_rate多轮测试
3. rejection_correct_rate边界测试
4. 流程模板匹配算法

### Phase 5: 集成与可视化

1. CI/CD自动触发
2. Feishu机器人推送结果
3. 历史趋势图表
4. 失败问题聚类分析

---

## 文件清单

**新增文件**：
- `evaluation/ragas_config.py` - 配置管理
- `evaluation/ragas_config.json` - 默认配置
- `tests/test_glm_ragas_compatibility.py` - 兼容性测试
- `tests/test_evaluation_fixes.py` - 修复验证

**修改文件**：
- `evaluation/evaluator.py` - 集成batch processor
- `evaluation/run_eval.py` - 配置加载与CLI增强
- `evaluation/report_generator.py` - Ragas可视化
- `evaluation/ragas_evaluator.py` - GLM兼容性修复

---

## 总结

Phase 2 Ragas完整集成已实现：
- ✅ 配置管理（文件+环境变量）
- ✅ 批量处理优化（可配置batch_size）
- ✅ 进度监控（批次进度日志）
- ✅ 错误恢复（失败批次自动处理）
- ✅ HTML可视化（新增LLM Quality Metrics部分）
- ✅ GLM兼容性（endpoint验证+警告机制）

系统已具备生产级Ragas评估能力，可支持大规模benchmark测试。