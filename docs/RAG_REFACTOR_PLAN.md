# VeRAG 重构方案

> 实施状态：第一轮已采用 LangChain LCEL 顶层框架，保留 Verus 专用检索组件。已实现结构保留切块、索引快照、离线文档向量、独立双路召回、完整证据上下文及本地测试。当前存储为 JSONL/term counts/NumPy 向量，SQLite/FAISS、完整 AST、增量构建和验证闭环仍是后续路线。用户已确定 query 为英文，本轮不实现中文处理。当前实现以 README 和 rag/DESIGN.md 为准。

## 1. 重构目标

把当前的本地 CLI 原型升级为一个可持续维护的 Verus 证明检索系统：

- 能回答“这个错误为什么出现、应该参考哪类证明、对应代码在哪里”；
- 代码、教程、PDF、历史错误案例可以独立更新和重建；
- 检索结果可解释、可引用、可评测；
- 后续可以同时支持 CLI、HTTP API 和 IDE/Agent 调用。

当前实现已经有价值：BM25、错误分类、spec 权重、语义重排、RRF 和分组去重都存在。主要问题是职责集中、索引是 JSONL 全量扫描、代码按固定行切块、向量没有持久化，也没有把历史 Verus 错误变成可回归的评测集。已有 `RetrievedChunk` 和路径/页码字段，但缺少版本化契约、完整证据与展示摘要的区分。

设计假设：以本地、单用户或小团队使用为起点，核心任务是 Verus 调试、规范生成和证明示例检索。当前工作树统计为 876 个项目 Rust 文件、4,128 个 Markdown 文件、2 个 PDF；实际 chunk 数量需要重建后统计。当前没有 `.rag_index`，以下结论来自代码审查与轻量函数复现，没有执行完整检索性能评测。

### 已确认的问题

| 位置 | 行为与影响 | 重构要求 |
| --- | --- | --- |
| `index_builder._clean_doc_text` | 正则删除 `<...>`，实测 `Seq<int>` 变成 `Seq` | Markdown 语法感知清洗，保留代码与泛型 |
| `tutorial/verus/while.md` | 代码块还是 mdBook `#include` 占位符 | 在固定源版本下展开 include；缺失时明确标记并统计 |
| `tokenize.TOKEN_RE` | 纯中文问题产生空 token | 支持中文文本分词、多语言 embedding；保留代码符号专用分词 |
| `code_analyzer._SPEC_BLOCK_RE` | 在 `{` 处停止，带块的量词表达式会截断 | Verus 感知解析器，记录解析质量和 fallback |
| `retriever.query_index` | dense 只对 BM25 候选评分，不能独立补召回 | lexical 与 dense 独立召回后取并集 |
| `SemanticScorer.score` | 每次重新编码候选文本，未持久化文档向量 | 离线向量化，在线只编码 query |
| `retriever._trim_snippet` | 代码压为单行并截断到 480 字符，prompt 使用这个摘要 | 展示摘要与模型上下文分离，模型读取原始完整证据 |
| `retriever._load_cached_bm25` | 按路径缓存，重建同一路径后无主动失效 | 按 index generation 缓存和切换 |
| `retriever.query_index` | 实际按 RRF 排序，但输出 score 是 lexical + RRF×100；随后分组选择再次改变顺序 | 统一最终排序和分数语义，分别记录选择策略 |
| CLI/README | README 的 `Retriever` 类、`--force`、`none` 后端与代码接口不一致 | 从实际接口生成或检查使用示例 |

补充：`per_group_k` 只控制返回的 `grouped` 列表，不是最终 `results` 的分组上限；现有历史报告包含当前代码已排除的路径，不能视为当前版本评测。投影后端是特征哈希相似度，不具有训练语义模型的泛化能力，应明确标记为降级模式。

## 2. 推荐的目标架构

```text
                 ┌──────────────┐
 CLI / HTTP / IDE │ Query API    │
                 └──────┬───────┘
                        │ QueryRequest
                ┌───────▼────────┐
                │ Query Analyzer  │ 语言、代码、Verus error
                │ + Query Router  │ 错误类别、proof pattern、domain
                └───┬─────────┬───┘
                    │         │
             lexical│         │vector / metadata
                    ▼         ▼
              ┌────────┐  ┌────────────┐
              │ BM25   │  │ Vector DB  │
              │ FTS    │  │ embeddings │
              └───┬────┘  └──────┬─────┘
                  └──────┬───────┘
                         ▼
              ┌────────────────────┐
              │ Fusion + Reranker  │ RRF/cross-encoder/rules
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Evidence Selector  │ 多样性、覆盖、预算
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Context Builder    │ 引用、证据等级、prompt
              └────────────────────┘

Ingestion: source adapters -> parser -> semantic chunks -> metadata/FTS/vector index
Evaluation: benchmark cases -> retrieval metrics -> regression report
```

建议第一阶段采用模块化单体、Python 共享 service 层：SQLite 存元数据与关系、FTS5 存词法索引、FAISS 存向量与 chunk ID 映射。先用精确向量搜索建立基线，测量 chunk 数量、内存和 p95 后再决定 ANN；确需并发写入或多实例服务时再评估 Qdrant/pgvector。FTS5 的默认分词和 BM25 与当前实现不同，必须配置代码/中文分词并做召回回归，不是透明替换。

## 3. 分层模块

```text
rag/
  domain.py          # Chunk、Query、Evidence、Citation、RetrievalResult
  ingestion/
    sources.py       # projects/tutorial/pdf/git source adapters
    parsers.py       # Rust/Markdown/PDF parser
    chunkers.py      # function/spec/section/page chunking
    pipeline.py      # 增量索引、hash、版本和删除
  index/
    store.py         # SQLite/FTS5/vector repository
    embeddings.py    # embedding provider + cache
  config.py          # 统一配置与校验
  retrieval/
    analyzer.py      # code/error/query -> normalized Query
    lexical.py       # BM25/FTS
    dense.py         # vector search
    fusion.py        # RRF/weighted fusion
    rerank.py        # optional cross encoder + domain rules
    diversify.py     # group/repo/function coverage
  serving/
    service.py       # query/build/status API
    cli.py
  evaluation/
    cases.jsonl
    runner.py
```

所有模块通过 dataclass/Pydantic 的领域对象交互，避免 `retriever.py` 继续承担索引加载、打分、去重和 prompt 拼接。

## 4. 数据模型

`Document` 表示文件版本；`Chunk` 表示可检索片段；`Evidence` 表示一次检索结果。

此外保存 SourceManifest：来源 URL、commit/内容快照、许可证、Verus/toolchain 版本（未知则显式 unknown）、抓取时间。来源是已验证项目，不代表本地任意片段独立验证通过；单独存 `verification_status` 与对应运行记录，不能从项目名称推断。

```json
{
  "chunk_id": "sha256:...",
  "document_id": "sha256:...",
  "source_group": "project|tutorial|pdf|case",
  "repo": "verified-ironkv",
  "path": "projects/.../foo.rs",
  "symbol": "module::function",
  "chunk_kind": "function|spec|impl|doc_section|pdf_page|error_case",
  "line_start": 10,
  "line_end": 48,
  "page": null,
  "text": "...",
  "title": "...",
  "tags": ["invariant", "forall"],
  "error_categories": ["postcondition"],
  "proof_patterns": ["linear_search"],
  "content_hash": "...",
  "source_revision": "git sha"
}
```

`document_id` 是 source + 相对路径的稳定逻辑 ID；`document_version_id` 由内容哈希确定；`chunk_id` 包含 document version、parser/chunker 版本和结构定位。每条 evidence 关联 index generation。新增 `parent_chunk_id`、`symbol_id`、`related_symbols`、`parse_quality`，使小块召回后可展开完整函数、类型定义或 lemma。符号引用先做可解释的轻量关系表，无法解析的关系保持 unresolved，不需要先部署图数据库。

## 5. 索引策略

### 代码

优先按 Verus item/function/module 切块，保留完整签名、requires/ensures/invariant、函数体。相邻 lemma 通过引用关系按需展开。普通 Rust tree-sitter 对 `verus!` 内部、spec/proof 扩展语法不一定能解析，应先验证 Verus parser 集成或专用语法支持；初期可以采用识别字符串/注释的词法扫描与括号匹配作为边界提取器，并统计失败率。超长函数按 loop/spec block 二次切分，保留 `parent_chunk_id`。固定行窗口只作为无法解析时的显式 fallback。

### Markdown/PDF

按标题层级和段落切块，代码块不跨块截断；PDF 保留页码和原始抽取文本。对表格、代码块和公式做单独类型标记；抽取为空或质量低的页面进入失败清单，再决定是否 OCR。Markdown 保留真实原文件行号，include 展开内容单独保留源文件定位。每个 chunk 生成规范化检索文本和原始证据文本。教程和 PDF 可能是同一内容的不同表示，应跨格式去重并保留全部来源引用，避免重复证据占满上下文。

### 增量更新

文件内容哈希与 parser/chunker 版本决定是否重新切块；向量缓存 key 包含 embedding 模型及 revision、维度、编码配置、检索文本哈希。删除文件时同步移除 chunk 与向量。每次 build 在新 generation 目录构建数据库与向量文件，校验 ID/数量一致后原子切换 CURRENT 指针，失败保留上一版本；读取端固定一个 generation，避免混读。manifest 记录 schema、配置哈希、模型版本、来源快照和构建统计。新增 `build --changed-only`，支持回滚到上一版本。

## 6. 查询与排序

`QueryAnalyzer` 输出统一对象：

- `intent`: debug / explain / generate_spec / find_example；
- `normalized_text`；
- `spec_terms` 和 clause 类型；
- `error_categories`；
- `proof_patterns`、类型、API 符号；
- `filters`: source_group、repo、语言、版本。

必须先保留和解析 Verus error 中的 source span，再移除用于检索的路径噪声：利用 span 找失败函数、loop 和关联 spec。原始诊断不丢弃。多个错误按 proof obligation 分组，限制查询扩展和展开预算。中文问题保留原文，可增加英文领域术语扩展，禁止改写 API/符号名；无错误输入时退化为普通检索。

默认检索流程：

1. FTS/BM25 召回 100~300 条，保证符号、API 名和错误词命中；
2. dense 检索召回 100~300 条，覆盖表达方式不同但证明结构相同的案例；
3. RRF 融合；
4. 对前 50 条做可选 reranker；
5. 在各召回通道提前应用用户指定的硬过滤；融合后做版本兼容性处理、父子块合并与去重，对每个 repo 使用软限额；
6. 按任务选择证据：debug 优先失败义务对应的概念、相似证明和 API/lemma；explain 优先教程；find_example 优先项目代码。只有足够相关的证据才进入上下文，不强制凑齐 PDF/project/tutorial 配额，也不为了 top-k 填入低相关结果。无相关证据时允许返回空结果。

以上候选数量是实验起点。embedding 先比较多语言模型与代码模型，在中文问题、Verus 符号和量词表达三个切片上评测，再确定默认模型。reranker 采用支持问题与代码对输入的模型作为可插拔实现，只有提高证据相关性且延迟可接受时才启用。显式指定模型失败时返回明确错误；auto 降级时在响应中标出原因，不能静默替换。不要在推理热路径自动下载模型。

规则分数应配置化，放在 YAML/JSON 中，禁止继续散落在 Python 常量中。先采用 RRF 作为融合分，避免直接相加不同量纲的 BM25 和 cosine；reranker 启用后明确其最终排序地位。每个结果返回通道 rank/raw score、fusion_score、rerank_score、final_rank 与 selection_reason。启发式分数不是置信概率，拒答阈值需要用验证集校准。安装类文档应按 intent 路由，而不是在所有查询下永久排除。

## 7. Context Builder 与生成边界

检索层只负责产生 `Evidence[]`，不直接拼接最终 prompt。Context Builder 根据 token budget 选择证据，并输出：

- citation：文件、行号或 PDF 页码；
- evidence level：定义、相似实现、错误案例；
- 检索理由和分数分解；
- 去重后的代码/文档块。

生成器区分有证据支持的结论与待验证建议，引用稳定 evidence ID；无法确认时明确说明证据缺口。模型上下文按目标模型 tokenizer 计预算，优先保留完整 clause/证明块、签名及必要依赖，不能使用 UI 截断摘要。查询输入、系统指令和证据块明确分隔。解析输出后检查引用 ID 是否存在且位置有效。

可选的验证闭环是本项目的特色：生成 patch → 在隔离临时工作区、固定 Verus/依赖版本下验证 → 捕获新诊断 → 最多 N 次定向检索与修复。记录 timeout、编译错误、证明失败和 verified 等状态。验证通过必须对应实际 verifier 运行，且检查是否削弱 requires/ensures 或新增 assume、admit、external_body 等绕过；未经检查的成功不能计入修复成功率。历史修复案例先进入候选库，审核后入正式索引，避免错误建议自我强化。这部分在检索基线稳定后实施。

## 8. 服务接口

```text
POST /v1/index/build
GET  /v1/index/status
POST /v1/retrieve
POST /v1/answer              # 可选，接入 LLM
```

`POST /v1/retrieve` 请求包含 `question`、`code`、`verus_error`、`filters`、`top_k`、`token_budget`；响应包含 `query_analysis`、`evidence`、`citations`、`debug`、`index_generation` 和 `degraded_reason`。build 是异步任务，返回 job_id，由 status 查询结果。CLI 与 HTTP 共用 service，不强制启动 Web 服务才能本地查询。MVP 只实现本地 build/retrieve/evaluate，HTTP 和 answer 按实际集成需求启用。

每次请求记录 trace_id、各阶段耗时、候选数量、模型/索引版本、缓存命中与降级原因；统计 embedding、reranker 调用成本。默认不将用户完整代码和错误全文写入通用日志。空索引、无证据、模型不可用分别返回可区分状态；启动时检查向量维度和索引 manifest。

## 9. 评测与上线门槛

把现有 `rag/verus_run_results.json` 和 retrieval reports 转成初始 case 集：报告中的代码/错误可复用，但检索 top-k 不是正确答案标签。人工标注相关性等级、source revision + symbol/span、错误类别与相关证据类型。当前报告引用的 `examples/` 不在工作树中，需要从报告代码恢复测试 fixture 或补齐对应版本，记录来源。扩充跨项目、中文输入、无答案、版本不匹配与多错误案例；按项目/证明族拆 train/dev/test，同一函数近重复不得跨 split。测试任务及其修复答案不得进入检索语料，独立知识源中的一般相关 lemma 可以保留。

每次变更至少计算：Recall@k、MRR、nDCG、source-group coverage、citation validity、重复率、平均延迟和索引更新时间。单独维护 hard cases：postcondition/invariant、overflow、type mismatch、old reference、trigger、missing API。

建议先建立基线，再设门槛：固定 held-out 集的 Recall@10/nDCG 不下降、citation validity=100%、坏引用和丢失数据为零。p95 和内存门槛按机器、chunk 数量、冷热启动及是否启用 reranker 分别制定，当前尚无实测依据，不能承诺 1 秒。做消融实验：旧实现、仅 BM25、双路召回、加 reranker、加关系展开，并报告每类错误表现。启用生成后增加固定预算下的有效修复成功率、规格保持检查与无 RAG 对照。README 中的“16/16 HIGH”需要可追溯标注与脚本重新生成，不能作为独立泛化证明。

## 10. 分阶段迁移

### Phase 0：冻结基线

保留现有 CLI 输出，补充可重复 benchmark 和 JSON schema。先修正 stale index、缺失索引检查和异常可观测性。

### Phase 1：拆分领域模型与索引层

引入 `domain.py`、SQLite/FTS5 store、增量 manifest；旧 `chunks.jsonl` 做一次性导入。保持 `query_index()` 兼容。

### Phase 2：改进 chunking 与 embedding

加入 Rust AST/item chunking、文档标题 chunking、向量缓存；projection 只作为离线 fallback。代码 chunk 需要 function/module/symbol 元数据。

### Phase 3：独立 retrieval pipeline

拆分 analyzer、lexical、dense、fusion、rerank、diversify 和 context builder；增加 score breakdown 和可配置策略。

### Phase 4：服务化与集成

增加 FastAPI、build/status/retrieve 接口；CLI 调用 service 或共享 service 层；再接 VS Code/Agent。

### Phase 5：评测驱动优化

持续加入失败案例，比较不同 chunk、embedding、reranker 和规则版本，只有 benchmark 通过才更新默认配置。评测工作从 Phase 0 开始，每一阶段均执行；Phase 5 是持续机制，而非最后才补测。

## 11. 当前代码的优先修复顺序

1. 修复泛型清洗损坏、未展开 include、丢失中文查询与上下文 480 字符截断；冻结可复现基线。
2. 统一配置/数据契约，修正文档与 CLI 不一致、排序分数不一致和同路径重建后的缓存失效。
3. 给代码加入 function/spec-aware chunking，保留原始位置和父子关系；新增解析 fixtures 覆盖量词块、泛型、字符串/注释、宏和嵌套括号。
4. 使用持久化 FTS/vector 索引。当前 BM25 在进程首次访问某目录时构建并缓存，CLI 新进程需要重新构建，长驻进程则缓存不自动失效；双路独立召回和 generation 机制一起解决这些问题。
5. 引入任务驱动的 evidence selector 与 token-budget context builder，取消强制 PDF 配额。
6. 将经过审核的历史修复案例加入独立训练语料分区，held-out benchmark 与其隔离；验证错误到相似证明/修复案例的召回。
7. 基于消融结果决定 reranker、模型升级和 HTTP/验证闭环。迁移期间旧 query_index 作为适配层，先做影子查询比较，再切换默认版本；出现回归时切回旧 generation/策略。
