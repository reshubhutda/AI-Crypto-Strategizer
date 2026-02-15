from fastapi import FastAPI
from app.api.routes import router as market_router
from app.api.agent_routes import router as agent_router

app = FastAPI(
    title="Optimizer Agent API",
    description="API for Binance data and AI agent",
    version="1.0.0"
)

# Include routers
app.include_router(market_router, prefix="/api/v1", tags=["market-data"])
app.include_router(agent_router, prefix="/api/v1/agent", tags=["ai-agent"])

@app.get("/")
async def root():
    return {
        "message": "Optimizer Agent API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }