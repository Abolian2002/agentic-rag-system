#!/usr/bin/env python3
"""
启动 Cortex RAG FastAPI 服务
"""

import uvicorn
from api import app

if __name__ == "__main__":
    print("🧠 Cortex RAG API Server")
    print("=" * 50)
    print("API Documentation: http://localhost:8000/docs")
    print("Frontend UI: http://localhost:8000/")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式：代码改动自动重启
    )