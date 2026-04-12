"""
CVA Demo — Entry Point
Starts the FastAPI server. Open the dashboard in your browser at http://localhost:8000
"""

import uvicorn
from cva.config.settings import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "cva.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
