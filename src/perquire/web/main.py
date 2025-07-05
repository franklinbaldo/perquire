#!/usr/bin/env python3
"""
Launch script for Perquire Web Interface.
"""

import os
import sys
import uvicorn
import argparse
from pathlib import Path

from .app import create_app


def main():
    """Main entry point for web interface."""
    parser = argparse.ArgumentParser(description="Perquire Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--database", default="perquire.db", help="Database file path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    # Create FastAPI app
    app = create_app(database_path=args.database)
    
    print(f"🚀 Starting Perquire Web Interface...")
    print(f"📊 Database: {args.database}")
    print(f"🌐 URL: http://{args.host}:{args.port}")
    print(f"📖 API Docs: http://{args.host}:{args.port}/api/docs")
    print(f"🔍 Ready to investigate embeddings!")
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
