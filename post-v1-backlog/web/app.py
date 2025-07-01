"""
FastAPI web application for Perquire.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import json
import uuid
from pathlib import Path
import tempfile
import logging
from datetime import datetime

from .. import create_investigator, create_ensemble_investigator, investigate_embedding
from ..core import InvestigationResult
from ..exceptions import InvestigationError

logger = logging.getLogger(__name__)

# Pydantic models
class EmbeddingInput(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector as list of floats")
    provider: str = Field(default="gemini", description="LLM provider to use")
    strategy: str = Field(default="default", description="Investigation strategy")
    save_to_db: bool = Field(default=True, description="Save results to database")

class InvestigationResponse(BaseModel):
    investigation_id: str
    description: str
    final_similarity: float
    iterations: int
    strategy_name: str
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchInvestigationRequest(BaseModel):
    embeddings: List[List[float]]
    provider: str = Field(default="gemini")
    strategy: str = Field(default="default")
    use_ensemble: bool = Field(default=False)
    parallel: bool = Field(default=True)

class StatusResponse(BaseModel):
    total_investigations: int
    total_questions: int
    avg_similarity: float
    avg_iterations: float
    recent_investigations: List[Dict[str, Any]]

# Global variables for background tasks
background_tasks_status = {}

def create_app(database_path: str = "perquire.db") -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Perquire Web Interface",
        description="Reverse Embedding Search Through Systematic Questioning",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Templates setup
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    templates = Jinja2Templates(directory=str(templates_dir))
    
    # Static files setup
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Home page with investigation interface."""
        return templates.TemplateResponse("index.html", {"request": request})
    
    @app.get("/status", response_class=HTMLResponse)
    async def status_page(request: Request):
        """Status and statistics page."""
        return templates.TemplateResponse("status.html", {"request": request})
    
    @app.get("/batch", response_class=HTMLResponse)
    async def batch_page(request: Request):
        """Batch investigation page."""
        return templates.TemplateResponse("batch.html", {"request": request})
    
    # API endpoints
    @app.post("/api/investigate", response_model=InvestigationResponse)
    async def api_investigate(embedding_input: EmbeddingInput):
        """Investigate a single embedding."""
        try:
            # Convert to numpy array
            embedding = np.array(embedding_input.embedding)
            
            # Validate embedding
            if embedding.ndim != 1:
                raise HTTPException(status_code=400, detail="Embedding must be 1-dimensional")
            
            if len(embedding) == 0:
                raise HTTPException(status_code=400, detail="Embedding cannot be empty")
            
            # Create investigator
            investigator = create_investigator(
                llm_provider=embedding_input.provider,
                embedding_provider="gemini",  # Default to Gemini for embeddings
                strategy=embedding_input.strategy,
                database_path=database_path
            )
            
            # Run investigation
            result = investigator.investigate(
                target_embedding=embedding,
                save_to_database=embedding_input.save_to_db,
                verbose=False
            )
            
            # Calculate duration
            duration = None
            if result.end_time and result.start_time:
                duration = (result.end_time - result.start_time).total_seconds()
            
            return InvestigationResponse(
                investigation_id=result.investigation_id,
                description=result.description,
                final_similarity=result.final_similarity,
                iterations=result.iterations,
                strategy_name=result.strategy_name,
                duration_seconds=duration,
                metadata=result.metadata
            )
            
        except InvestigationError as e:
            raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in investigation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    @app.post("/api/investigate/file")
    async def api_investigate_file(
        file: UploadFile = File(...),
        provider: str = Form(default="gemini"),
        strategy: str = Form(default="default"),
        save_to_db: bool = Form(default=True)
    ):
        """Investigate embedding from uploaded file."""
        try:
            # Read file content
            content = await file.read()
            
            # Parse based on file extension
            if file.filename.endswith('.json'):
                embedding_data = json.loads(content.decode('utf-8'))
                embedding = np.array(embedding_data)
            elif file.filename.endswith('.npy'):
                # Save to temp file and load with numpy
                with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                    tmp.write(content)
                    tmp.flush()
                    embedding = np.load(tmp.name)
                    Path(tmp.name).unlink()  # Clean up
            else:
                # Try parsing as text/CSV
                try:
                    embedding = np.fromstring(content.decode('utf-8'), sep=',')
                except:
                    embedding = np.fromstring(content.decode('utf-8'), sep=' ')
            
            # Validate embedding
            if embedding.ndim != 1:
                raise HTTPException(status_code=400, detail="Embedding must be 1-dimensional")
            
            # Investigate using the API endpoint logic
            embedding_input = EmbeddingInput(
                embedding=embedding.tolist(),
                provider=provider,
                strategy=strategy,
                save_to_db=save_to_db
            )
            
            return await api_investigate(embedding_input)
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
    
    @app.post("/api/investigate/batch")
    async def api_investigate_batch(
        background_tasks: BackgroundTasks,
        batch_request: BatchInvestigationRequest
    ):
        """Start batch investigation (background task)."""
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Initialize task status
            background_tasks_status[task_id] = {
                "status": "started",
                "progress": 0,
                "total": len(batch_request.embeddings),
                "results": [],
                "started_at": datetime.now().isoformat(),
                "error": None
            }
            
            # Start background task
            background_tasks.add_task(
                run_batch_investigation,
                task_id,
                batch_request,
                database_path
            )
            
            return {"task_id": task_id, "status": "started", "total": len(batch_request.embeddings)}
            
        except Exception as e:
            logger.error(f"Batch investigation startup error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start batch investigation: {str(e)}")
    
    @app.get("/api/investigate/batch/{task_id}")
    async def api_batch_status(task_id: str):
        """Get batch investigation status."""
        if task_id not in background_tasks_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return background_tasks_status[task_id]
    
    @app.get("/api/status", response_model=StatusResponse)
    async def api_status():
        """Get investigation statistics."""
        try:
            from ..database.duckdb_provider import DuckDBProvider
            from ..database.base import DatabaseConfig
            
            # Connect to database
            db_config = DatabaseConfig(connection_string=database_path)
            db_provider = DuckDBProvider(db_config)
            
            # Get statistics
            stats = db_provider.get_statistics()
            recent = db_provider.get_recent_investigations(limit=10)
            
            return StatusResponse(
                total_investigations=stats.get('total_investigations', 0),
                total_questions=stats.get('total_questions', 0),
                avg_similarity=stats.get('avg_similarity', 0.0),
                avg_iterations=stats.get('avg_iterations', 0.0),
                recent_investigations=recent
            )
            
        except Exception as e:
            logger.error(f"Status retrieval error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
    
    @app.get("/api/investigations")
    async def api_get_investigations(limit: int = 50, offset: int = 0):
        """Get paginated list of investigations."""
        try:
            from ..database.duckdb_provider import DuckDBProvider
            from ..database.base import DatabaseConfig
            
            # Connect to database
            db_config = DatabaseConfig(connection_string=database_path)
            db_provider = DuckDBProvider(db_config)
            
            # Get investigations
            investigations = db_provider.get_all_investigations(limit=limit, offset=offset)
            
            return {"investigations": investigations, "limit": limit, "offset": offset}
            
        except Exception as e:
            logger.error(f"Investigation retrieval error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get investigations: {str(e)}")
    
    @app.get("/api/investigations/{investigation_id}")
    async def api_get_investigation(investigation_id: str):
        """Get specific investigation details."""
        try:
            from ..database.duckdb_provider import DuckDBProvider
            from ..database.base import DatabaseConfig
            
            # Connect to database
            db_config = DatabaseConfig(connection_string=database_path)
            db_provider = DuckDBProvider(db_config)
            
            # Get investigation
            investigation = db_provider.get_investigation_by_id(investigation_id)
            
            if not investigation:
                raise HTTPException(status_code=404, detail="Investigation not found")
            
            return investigation
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Investigation detail retrieval error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get investigation: {str(e)}")
    
    @app.delete("/api/investigations/{investigation_id}")
    async def api_delete_investigation(investigation_id: str):
        """Delete investigation."""
        try:
            from ..database.duckdb_provider import DuckDBProvider
            from ..database.base import DatabaseConfig
            
            # Connect to database
            db_config = DatabaseConfig(connection_string=database_path)
            db_provider = DuckDBProvider(db_config)
            
            # Delete investigation
            success = db_provider.delete_investigation(investigation_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Investigation not found")
            
            return {"deleted": True}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Investigation deletion error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to delete investigation: {str(e)}")
    
    return app


async def run_batch_investigation(
    task_id: str,
    batch_request: BatchInvestigationRequest,
    database_path: str
):
    """Background task for batch investigation."""
    try:
        results = []
        
        for i, embedding_data in enumerate(batch_request.embeddings):
            try:
                # Update progress
                background_tasks_status[task_id]["progress"] = i
                background_tasks_status[task_id]["status"] = "processing"
                
                # Convert to numpy array
                embedding = np.array(embedding_data)
                
                # Investigate
                if batch_request.use_ensemble:
                    investigator = create_ensemble_investigator(
                        strategies=['default', 'artistic', 'scientific'],
                        database_path=database_path
                    )
                    result = investigator.investigate(
                        target_embedding=embedding,
                        parallel=batch_request.parallel,
                        save_ensemble_result=True,
                        verbose=False
                    )
                else:
                    investigator = create_investigator(
                        llm_provider=batch_request.provider,
                        strategy=batch_request.strategy,
                        database_path=database_path
                    )
                    result = investigator.investigate(
                        target_embedding=embedding,
                        save_to_database=True,
                        verbose=False
                    )
                
                # Convert result to dict
                result_dict = {
                    "investigation_id": result.investigation_id,
                    "description": result.description,
                    "final_similarity": result.final_similarity,
                    "iterations": result.iterations,
                    "strategy_name": result.strategy_name
                }
                
                results.append(result_dict)
                background_tasks_status[task_id]["results"] = results
                
            except Exception as e:
                logger.error(f"Batch investigation item {i} failed: {str(e)}")
                # Continue with next embedding
        
        # Mark as completed
        background_tasks_status[task_id]["status"] = "completed"
        background_tasks_status[task_id]["progress"] = len(batch_request.embeddings)
        background_tasks_status[task_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"Batch investigation failed: {str(e)}")
        background_tasks_status[task_id]["status"] = "failed"
        background_tasks_status[task_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)