"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .api import router
from .config import settings
from . import __version__

# Create FastAPI app
app = FastAPI(
    title="Talking Avatar API",
    description="AI-powered talking avatar generation with local model deployment",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["avatar"])


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    print("=" * 60)
    print(f"Talking Avatar API v{__version__}")
    print("=" * 60)
    print(f"Device: {settings.device}")
    print(f"Output FPS: {settings.output_fps}")
    print(f"Output Resolution: {settings.output_resolution}")
    print(f"Model Cache: {settings.model_cache_dir}")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    print("Shutting down Talking Avatar API...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Talking Avatar API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


def main():
    """Run the application."""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
