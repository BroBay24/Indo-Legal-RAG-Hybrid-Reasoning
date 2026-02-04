#!/usr/bin/env python3
"""
CLI Runner untuk RAG Pipeline
Mendukung indexing, query, dan management via command line
"""
import argparse
import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_pipeline import create_pipeline


def cmd_index(args):
    """Index documents."""
    print("[PROCESS] Starting indexing process...")
    
    pipeline = create_pipeline(
        use_local_llm=False,  # Tidak perlu LLM untuk indexing
        use_pinecone=args.pinecone
    )
    
    stats = pipeline.index_documents(
        data_path=args.data_path,
        upload_to_pinecone=args.pinecone
    )
    
    print("\n[OK] Indexing complete!")
    print(json.dumps(stats, indent=2))


def cmd_query(args):
    """Query the RAG pipeline."""
    print(f"[SEARCH] Query: {args.question}")
    
    pipeline = create_pipeline(
        use_local_llm=True,
        use_pinecone=args.pinecone
    )
    
    response = pipeline.query(
        question=args.question,
        top_k=args.top_k,
        max_tokens=args.max_tokens
    )
    
    print(f"\n[OUTPUT] Answer:\n{response.answer}")
    print(f"\n[PROCESS] Sources:")
    for i, source in enumerate(response.sources, 1):
        print(f"   {i}. {source.get('source', 'Unknown')} (page {source.get('page', '?')})")


def cmd_chat(args):
    """Interactive chat mode."""
    print("💬 Interactive Chat Mode (type 'quit' to exit)")
    print("-" * 50)
    
    pipeline = create_pipeline(
        use_local_llm=True,
        use_pinecone=args.pinecone
    )
    
    while True:
        try:
            question = input("\n🙋 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("[STOP] Goodbye!")
                break
            
            if not question:
                continue
            
            response = pipeline.query(question, top_k=args.top_k)
            print(f"\n[LLM] AI: {response.answer}")
            
        except KeyboardInterrupt:
            print("\n[STOP] Goodbye!")
            break


def cmd_stats(args):
    """Show pipeline stats."""
    pipeline = create_pipeline(
        use_local_llm=False,
        use_pinecone=args.pinecone
    )
    
    stats = pipeline.get_stats()
    print("\n[STATS] Pipeline Statistics:")
    print(json.dumps(stats, indent=2, default=str))


def cmd_serve(args):
    """Start the API server."""
    import uvicorn
    from config import settings
    
    print(f"[INFO] Starting server at http://{args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline CLI untuk Hukum Indonesia"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--data-path", "-d", help="Path to data folder")
    index_parser.add_argument("--no-pinecone", dest="pinecone", action="store_false",
                              help="Skip Pinecone upload")
    index_parser.set_defaults(func=cmd_index)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the pipeline")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", "-k", type=int, default=5,
                              help="Number of documents to retrieve")
    query_parser.add_argument("--max-tokens", "-m", type=int, default=512,
                              help="Max tokens for response")
    query_parser.add_argument("--no-pinecone", dest="pinecone", action="store_false",
                              help="Use BM25 only")
    query_parser.set_defaults(func=cmd_query)
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument("--top-k", "-k", type=int, default=5)
    chat_parser.add_argument("--no-pinecone", dest="pinecone", action="store_false")
    chat_parser.set_defaults(func=cmd_chat)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--no-pinecone", dest="pinecone", action="store_false")
    stats_parser.set_defaults(func=cmd_stats)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port")
    serve_parser.add_argument("--reload", "-r", action="store_true",
                              help="Enable auto-reload")
    serve_parser.set_defaults(func=cmd_serve)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
