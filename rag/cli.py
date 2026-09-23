"""CLI for building and querying VeRAG index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .index_builder import build_index
from .prompting import render_prompt_context
from .retriever import query_index
from .evaluation import evaluate


def _load_optional_text(text: str | None, text_file: str | None) -> str:
    if text and text_file:
        raise ValueError("请二选一：直接传文本或文件路径，不能同时传。")
    if text:
        return text
    if text_file:
        return Path(text_file).read_text(encoding="utf-8")
    return ""


def cmd_build(args: argparse.Namespace) -> int:
    pdf_roots = [p.strip() for p in args.pdf_roots.split(",") if p.strip()]
    if args.force_tutorial_pdf and "tutorial" not in pdf_roots:
        pdf_roots.append("tutorial")
    meta = build_index(
        repo_root=args.repo_root,
        output_dir=args.index_dir,
        include_pdfs=not args.no_pdf,
        pdf_roots=pdf_roots if pdf_roots else None,
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
        semantic_proj_dim=args.semantic_proj_dim,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    query_text = _load_optional_text(args.query_text, args.query_file)
    code_text = _load_optional_text(args.code_text, args.code_file)
    error_text = _load_optional_text(args.error_text, args.error_file)

    result = query_index(
        index_dir=args.index_dir,
        query_text=query_text,
        code_text=code_text,
        error_text=error_text,
        top_k=args.top_k,
        per_group_k=args.per_group_k,
        min_project=args.min_project,
        min_tutorial=args.min_tutorial,
        min_pdf=args.min_pdf,
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
        semantic_proj_dim=args.semantic_proj_dim,
        semantic_candidate_k=args.semantic_candidate_k,
        lexical_rrf_weight=args.lexical_rrf_weight,
        semantic_rrf_weight=args.semantic_rrf_weight,
        max_context_chars=args.max_context_chars,
    )
    if args.prompt_out:
        context_text = render_prompt_context(result, max_items=args.prompt_items, max_chars=args.max_context_chars)
        Path(args.prompt_out).write_text(context_text, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    report = evaluate(args.index_dir, args.cases, args.top_k, args.semantic_backend)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["passed"] == report["case_count"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verag",
        description="VeRAG: 从 projects/tutorial/pdf 检索相关内容。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="构建索引")
    p_build.add_argument("--repo-root", default=".", help="知识库根目录")
    p_build.add_argument("--index-dir", default=".rag_index", help="索引输出目录")
    p_build.add_argument("--no-pdf", action="store_true", help="不处理 PDF")
    p_build.add_argument(
        "--pdf-roots",
        default=".",
        help="PDF 扫描根目录（相对 repo-root，逗号分隔），默认扫描整个仓库。",
    )
    p_build.add_argument(
        "--force-tutorial-pdf",
        action="store_true",
        help="强制把 tutorial 目录加入 PDF 扫描根目录。",
    )
    p_build.add_argument("--semantic-backend", choices=["none", "projection", "sentence-transformer", "auto"], default="projection")
    p_build.add_argument("--semantic-model", default="all-MiniLM-L6-v2")
    p_build.add_argument("--semantic-proj-dim", type=int, default=512)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="执行检索")
    p_query.add_argument("--index-dir", default=".rag_index", help="索引目录")
    p_query.add_argument("--query-text", default=None, help="问题文本")
    p_query.add_argument("--query-file", default=None, help="问题文本文件")
    p_query.add_argument("--code-text", default=None, help="代码片段文本")
    p_query.add_argument("--code-file", default=None, help="代码片段文件")
    p_query.add_argument("--error-text", default=None, help="报错文本")
    p_query.add_argument("--error-file", default=None, help="报错文本文件")
    p_query.add_argument("--top-k", type=int, default=12, help="总体返回条数")
    p_query.add_argument("--per-group-k", type=int, default=6, help="grouped 诊断列表每组最多返回条数")
    p_query.add_argument("--min-project", type=int, default=0, help="最终结果中最少 project 条数")
    p_query.add_argument("--min-tutorial", type=int, default=0, help="最终结果中最少 tutorial 条数")
    p_query.add_argument("--min-pdf", type=int, default=0, help="最终结果中最少 pdf 条数")
    p_query.add_argument(
        "--semantic-backend",
        default="auto",
        choices=["auto", "sentence-transformer", "projection", "none"],
        help="auto 使用索引记录的后端；none 禁用向量召回。",
    )
    p_query.add_argument(
        "--semantic-model",
        default=None,
        help="可选：校验模型名是否匹配索引，默认使用索引配置。",
    )
    p_query.add_argument("--semantic-proj-dim", type=int, default=None, help="可选：校验 projection 维度是否匹配索引")
    p_query.add_argument("--semantic-candidate-k", type=int, default=1200, help="独立向量召回的候选数")
    p_query.add_argument("--lexical-rrf-weight", type=float, default=1.0, help="RRF 中 lexical 权重")
    p_query.add_argument("--semantic-rrf-weight", type=float, default=None, help="RRF 权重：默认 projection=0.25，learned=0.9")
    p_query.add_argument("--prompt-out", default=None, help="可选：输出 LLM 上下文文本文件路径")
    p_query.add_argument("--prompt-items", type=int, default=10, help="写入 LLM 上下文的最大条数")
    p_query.add_argument("--max-context-chars", type=int, default=24000, help="完整证据块的上下文字符预算")
    p_query.set_defaults(func=cmd_query)
    p_eval = sub.add_parser("evaluate", help="Run local document-target retrieval smoke cases")
    p_eval.add_argument("--index-dir", default=".rag_index")
    p_eval.add_argument("--cases", default="evaluation/smoke_cases.jsonl")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--semantic-backend", choices=["auto", "none", "projection", "sentence-transformer"], default="auto")
    p_eval.add_argument("--output")
    p_eval.set_defaults(func=cmd_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
