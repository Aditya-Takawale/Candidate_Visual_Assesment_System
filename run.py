"""
CVA Demo — Entry Point
Starts the FastAPI server. Open the dashboard in your browser at http://localhost:8000
"""

import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("GLOG_minloglevel", "2")           # suppress MediaPipe C++ spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # suppress TF warnings

import uvicorn
from cva.config.settings import API_HOST, API_PORT

if __name__ == "__main__":
    browse_host = "127.0.0.1" if API_HOST == "0.0.0.0" else API_HOST
    print(f"\n  --> Dashboard: http://{browse_host}:{API_PORT}\n")
    uvicorn.run(
        "cva.api.main:app",
        host=API_HOST,
        port=API_PORT,
        workers=1,
        reload=False,
        log_level="info",
    )
