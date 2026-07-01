# CLIPForge 0.3

面向生产环境的多模态模型与向量检索平台。CLIPForge 将图片和文本映射到统一语义空间，支持零样本分类、文搜图、图搜图、跨模态检索和批量向量化。

默认使用 deterministic mock，让完整系统无需模型权重即可启动；真实环境推荐使用 SigLIP2，也支持 OpenCLIP/EVA 系列。

## 这版高级在哪里

- **现代模型运行时**：SigLIP2、OpenCLIP/EVA、Mock 三种后端
- **多语言模型**：SigLIP2 原生支持多语言图文检索
- **推理优化**：FP32/FP16/BF16、SDPA、`torch.compile`、启动预热
- **并发隔离**：同步 PyTorch 推理移出 ASGI 事件循环，并通过推理闸门保护 GPU
- **Embedding cache**：文本与图片按 SHA-256 缓存，LRU 淘汰并暴露命中统计
- **Prompt ensemble**：零样本分类自动聚合多套提示词，不再只比较一个裸标签
- **不确定性感知检索**：使用 Top-K margin 与归一化熵识别模糊意图
- **人机协同重排**：根据用户正负反馈更新查询向量并重新排序
- **完整跨模态检索**：文本和图片都可作为查询，也都可作为被检索对象
- **持久化向量仓库**：SQLite WAL、集合管理、元数据过滤和重启恢复
- **企业隔离**：通过 `X-Tenant-ID` 实现租户级集合与任务隔离
- **后台任务**：大批量文本入库通过任务队列执行，可查询任务状态
- **服务治理**：API Key、限流、超时、请求 ID、JSON 日志、Prometheus、健康探针
- **交付能力**：Web 控制台、Docker 持久卷、非 root 容器、测试与 CI

## 模型选择

| 场景 | 推荐模型 | 后端 |
| --- | --- | --- |
| 常规生产、多语言检索 | `google/siglip2-base-patch16-224` | `siglip2` |
| 质量优先、显存充足 | `google/siglip2-so400m-patch14-384` | `siglip2` |
| 英文检索、OpenCLIP 生态 | `EVA02-L-14 / merged2b_s4b_b131k` | `openclip` |
| 开发与 CI | deterministic mock | `mock` |

SigLIP2 使用官方处理器执行小写化、64 token 固定 padding/truncation，并对输出向量进行 L2 归一化。具体模型行为参考 [Hugging Face SigLIP2 文档](https://huggingface.co/docs/transformers/model_doc/siglip2)；OpenCLIP 模型列表参考 [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)。

## 启动

需要 Python 3.11+。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[dev]"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开：

- 控制台：<http://127.0.0.1:8000>
- OpenAPI：<http://127.0.0.1:8000/docs>
- Prometheus：<http://127.0.0.1:8000/metrics>

项目目录含中文时建议不要使用 editable install（`-e`），部分 Windows/pip 组合可能把 `.pth` 路径写成错误代码页。

## 启用 SigLIP2

模型依赖和权重较大，单独安装：

```powershell
.\.venv\Scripts\python.exe -m pip install ".[ml]"
$env:CLIPFORGE_MODEL_BACKEND="siglip2"
$env:CLIPFORGE_MODEL_ID="google/siglip2-base-patch16-224"
$env:CLIPFORGE_MODEL_PRECISION="auto"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

GPU 质量优先配置：

```powershell
$env:CLIPFORGE_MODEL_BACKEND="siglip2"
$env:CLIPFORGE_MODEL_ID="google/siglip2-so400m-patch14-384"
$env:CLIPFORGE_MODEL_DEVICE="cuda"
$env:CLIPFORGE_MODEL_PRECISION="bf16"
$env:CLIPFORGE_MODEL_ATTENTION="sdpa"
$env:CLIPFORGE_MODEL_COMPILE="true"
```

首次启动需要下载权重。生产部署应在镜像构建或初始化容器阶段预取模型，并挂载 Hugging Face 缓存。

## 启用 OpenCLIP / EVA

```powershell
$env:CLIPFORGE_MODEL_BACKEND="openclip"
$env:CLIPFORGE_MODEL_NAME="EVA02-L-14"
$env:CLIPFORGE_MODEL_PRETRAINED="merged2b_s4b_b131k"
$env:CLIPFORGE_MODEL_PRECISION="bf16"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 关键 API

| API | 用途 |
| --- | --- |
| `POST /api/v1/embeddings/text` | 文本向量 |
| `POST /api/v1/embeddings/image` | 图片向量 |
| `POST /api/v1/similarity` | 图文相似度矩阵 |
| `POST /api/v1/classifications/zero-shot` | Prompt ensemble 零样本分类 |
| `POST /api/v1/collections` | 创建租户集合 |
| `POST /api/v1/collections/{name}/items/text` | 文本入库 |
| `POST /api/v1/collections/{name}/items/image` | 图片入库 |
| `POST /api/v1/collections/{name}/search` | 文搜文、文搜图、图搜图、图搜文 |
| `POST /api/v1/collections/{name}/search/interactive` | 不确定性检测、澄清与反馈重排 |
| `POST /api/v1/jobs/index/text` | 提交批量入库任务 |
| `GET /api/v1/jobs/{id}` | 查询任务状态 |
| `GET /api/v1/model` | 当前模型、设备、精度和能力 |
| `GET /api/v1/models/catalog` | 推荐模型预设 |

租户由请求头指定：

```text
X-Tenant-ID: acme
X-API-Key: your-secret-key
```

### 零样本分类

```bash
curl http://127.0.0.1:8000/api/v1/classifications/zero-shot \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["<base64>"],
    "labels": ["running shoe", "formal shoe", "hiking boot"],
    "templates": ["a product photo of {}.", "a close-up image of {}."],
    "top_k": 3
  }'
```

### 跨模态集合检索

```bash
curl http://127.0.0.1:8000/api/v1/collections/catalog/search \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d '{
    "query_type": "text",
    "query": "适合雨天登山的外套",
    "target_modality": "image",
    "limit": 10,
    "metadata_filter": {"region": "apac"}
  }'
```

### 不确定性感知的人机协同检索

交互检索首先根据 Top-K 分数分布计算 probability margin 和 normalized entropy。查询意图模糊时，API 返回两个代表性候选项；用户选择更相关的结果后，系统通过 Rocchio-style relevance feedback 更新查询向量：

```text
Query → SigLIP2 → Top-K → Uncertainty Estimation
                              ├─ confident → results
                              └─ ambiguous → clarification
                                                  ↓
                                         feedback re-ranking
```

```bash
curl http://127.0.0.1:8000/api/v1/collections/catalog/search/interactive \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme" \
  -d '{
    "query_type": "text",
    "query": "适合正式场合的鞋",
    "limit": 5,
    "feedback": {
      "positive_ids": ["formal-shoe-01"],
      "negative_ids": ["running-shoe-02"]
    }
  }'
```

响应包含 `uncertainty`、`clarification`、`feedback_applied` 和 `query_drift`，可用于构建“模型不确定时主动询问”的人机协同界面。

## 配置

复制 `.env.example` 为 `.env`。主要参数：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_BACKEND` | `mock` | `mock`、`siglip2` 或 `openclip` |
| `MODEL_ID` | SigLIP2 Base | Hugging Face 模型 ID |
| `MODEL_PRECISION` | `auto` | `fp32`、`fp16`、`bf16` |
| `MODEL_ATTENTION` | `sdpa` | Transformers attention 实现 |
| `MODEL_COMPILE` | `false` | 启用 `torch.compile` |
| `MODEL_WARMUP` | `true` | 启动时预热文本和图片编码路径 |
| `MAX_CONCURRENT_INFERENCE` | `1` | 单实例同时进入模型的请求数 |
| `INFERENCE_CACHE_SIZE` | `4096` | 每种模态的 LRU 向量缓存容量 |
| `VECTOR_STORE_PATH` | `data/clipforge.db` | SQLite 向量库位置 |
| `JOB_WORKERS` | `1` | 后台入库 worker 数 |
| `API_KEYS` | `[]` | 生产环境必须配置 |

## 架构

```text
Web / SDK
    │
FastAPI gateway
    ├── auth · tenant · rate limit · timeout · request ID
    ├── inference gate
    │      └── Mock / OpenCLIP-EVA / SigLIP2
    ├── prompt ensemble + multimodal search
    ├── background job manager
    └── SQLite WAL vector store
           └── tenant / collection / metadata / modality
```

SQLite provider 适合单机和边缘部署。多副本集群下一步应将相同存储接口接到 pgvector、Qdrant 或 Milvus，并把进程内任务执行器替换为 Kafka/Celery/Arq；HTTP 与模型契约无需改变。

## 验证

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe app
.\.venv\Scripts\python.exe -m pytest --cov=app
```

Apache-2.0
