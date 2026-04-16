"""
api.py
- FastAPI REST wrapper around the insurance claims agent
- Session management for stateful conversations
- Endpoints for chat, claim status, image upload, etc.

Run:  python main.py --api
"""

import os
import shutil
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

from .agent import build_graph, AgentState
from .tools import RetrievalResult

load_dotenv()

# ==============================================================================
# Pydantic Models
# ==============================================================================

class ChatRequest(BaseModel):
    """Request to send a message to the agent."""
    session_id: str = Field(..., description="Unique session identifier")
    user_message: str = Field(..., description="User's message to the agent")
    image_path: Optional[str] = Field(None, description="Path to attached image (optional)")


class ChatResponse(BaseModel):
    """Response from a chat message."""
    session_id: str
    agent_response: str
    query_type: str  # "factual" | "cross_modal" | "analytical" | "conversational"
    claim_draft: dict = Field(default_factory=dict)
    messages_count: int
    timestamp: str


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: str  # "user" | "agent"
    content: str
    query_type: Optional[str] = None  # only for agent messages
    timestamp: str


class ConversationHistoryResponse(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: list[ConversationMessage]
    claim_draft: dict
    created_at: str
    last_updated: str


class ClaimDraftRequest(BaseModel):
    """Request to update claim draft."""
    session_id: str
    claim_draft: dict


class ClaimDraftResponse(BaseModel):
    """Current claim draft."""
    session_id: str
    claim_draft: dict
    last_updated: str


class ImageUploadResponse(BaseModel):
    """Response after image upload."""
    session_id: str
    filename: str
    file_path: str
    uploaded_at: str


class SessionResetResponse(BaseModel):
    """Response after session reset."""
    session_id: str
    status: str  # "cleared"
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response format."""
    error: str
    code: str


# ==============================================================================
# Session Manager
# ==============================================================================

class SessionState:
    """Stores agent state for a single session."""
    def __init__(self):
        self.agent_state: AgentState = {
            "messages": [],
            "query_type": "conversational",
            "claim_draft": {},
            "retrieved_docs": [],
            "image_path": None,
            "detected_policy_number": None,
            "detected_insurance_type": None,
        }
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.message_history = []  # Track metadata for conversation view


class SessionManager:
    """Manages multiple user sessions with automatic expiration."""
    
    def __init__(self, expiration_hours: int = 24):
        self.sessions: dict[str, SessionState] = {}
        self._agent = None  # Lazy-loaded on first use
        self.expiration_hours = expiration_hours
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
        self.upload_dir.mkdir(exist_ok=True)
    
    @property
    def agent(self):
        """Lazy-load agent on first use."""
        if self._agent is None:
            self._agent = build_graph()
        return self._agent
    
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SessionState()
        return session_id
    
    def get_session(self, session_id: str) -> SessionState:
        """Get session by ID, checking expiration."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # Check expiration
        if datetime.now() - session.last_updated > timedelta(hours=self.expiration_hours):
            del self.sessions[session_id]
            raise HTTPException(status_code=404, detail=f"Session {session_id} expired")
        
        return session
    
    def chat(self, session_id: str, user_message: str, image_path: Optional[str] = None) -> tuple[str, dict]:
        """
        Process a user message through the agent.
        Returns: (agent_response, updated_claim_draft)
        """
        session = self.get_session(session_id)
        
        # Update image path if provided
        if image_path:
            session.agent_state["image_path"] = image_path
        
        # Add user message
        session.agent_state["messages"].append(HumanMessage(content=user_message))
        
        # Run agent
        try:
            session.agent_state = self.agent.invoke(session.agent_state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
        
        # Extract agent response
        from langchain_core.messages import AIMessage
        agent_response = ""
        for msg in reversed(session.agent_state["messages"]):
            if isinstance(msg, AIMessage):
                agent_response = msg.content
                break
        
        # Track message in history
        session.message_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
        })
        session.message_history.append({
            "role": "agent",
            "content": agent_response,
            "query_type": session.agent_state.get("query_type", "conversational"),
            "timestamp": datetime.now().isoformat(),
        })
        
        # Reset image after use
        session.agent_state["image_path"] = None
        
        # Update session timestamp
        session.last_updated = datetime.now()
        
        return agent_response, session.agent_state.get("claim_draft", {})
    
    def get_conversation(self, session_id: str) -> tuple[list, dict, str, str]:
        """Get full conversation history."""
        session = self.get_session(session_id)
        return (
            session.message_history,
            session.agent_state.get("claim_draft", {}),
            session.created_at.isoformat(),
            session.last_updated.isoformat(),
        )
    
    def get_claim_draft(self, session_id: str) -> dict:
        """Get current claim draft."""
        session = self.get_session(session_id)
        return session.agent_state.get("claim_draft", {})
    
    def update_claim_draft(self, session_id: str, draft: dict) -> dict:
        """Update claim draft with user-provided data."""
        session = self.get_session(session_id)
        session.agent_state["claim_draft"].update(draft)
        session.last_updated = datetime.now()
        return session.agent_state["claim_draft"]
    
    async def upload_image(self, session_id: str, file: UploadFile) -> str:
        """Save uploaded image and return path."""
        session = self.get_session(session_id)

        # Create session-specific upload directory
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # HIGH fix (Bug 3): sanitise filename to prevent path traversal.
        # Path(name).name strips all directory components (e.g. "../../evil.sh" → "evil.sh").
        safe_filename = Path(file.filename).name
        if not safe_filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        file_path = session_dir / safe_filename

        # HIGH fix (Bug 4): use await file.read() (non-blocking) instead of
        # the synchronous file.file.read() which blocks the event loop.
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        session.last_updated = datetime.now()
        return str(file_path)
    
    def reset_session(self, session_id: str) -> None:
        """Clear all data for a session."""
        session = self.get_session(session_id)
        
        # Delete uploaded files
        session_dir = self.upload_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        # Delete session
        del self.sessions[session_id]
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions. Should be called periodically."""
        expired = []
        now = datetime.now()
        
        for session_id, session in self.sessions.items():
            if now - session.last_updated > timedelta(hours=self.expiration_hours):
                expired.append(session_id)
        
        for session_id in expired:
            # MEDIUM fix (Edge Case 3): delete uploaded files for expired sessions
            # so disk storage does not grow unboundedly.
            session_dir = self.upload_dir / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
            del self.sessions[session_id]
        
        return len(expired)


# ==============================================================================
# FastAPI Application
# ==============================================================================

def create_cleanup_lifespan(manager: SessionManager, cleanup_interval: int):
    """
    Create a lifespan context manager for periodic session cleanup.
    
    Encapsulates the background cleanup loop logic, keeping create_app focused
    on application wiring rather than cleanup details.
    
    Args:
        manager: SessionManager instance to clean up expired sessions.
        cleanup_interval: Seconds between cleanup runs (from env or default 3600).
    
    Returns:
        An asynccontextmanager (lifespan) suitable for FastAPI app initialization.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _cleanup_loop():
            while True:
                await asyncio.sleep(cleanup_interval)
                try:
                    manager.cleanup_expired_sessions()
                except Exception as exc:
                    # Log exceptions so ops has visibility if cleanup repeatedly fails.
                    # Sessions will continue to be cleaned up on next interval.
                    print(f"[cleanup] session cleanup error: {exc}", flush=True)

        task = asyncio.create_task(_cleanup_loop())
        try: 
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return lifespan


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    # Initialize session manager and cleanup configuration.
    manager = SessionManager(
        expiration_hours=int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
    )
    # HIGH improvement: background periodic cleanup so expired sessions and their
    # uploaded files are removed automatically without requiring a manual API call.
    # NOTE: cleanup_interval of 3600 s (1 hr) means expired sessions can sit up to
    # 2× their expiration_hours before being cleaned — set CLEANUP_INTERVAL_SECS
    # to a smaller value if more frequent cleanup is needed.
    cleanup_interval = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECS", "3600"))
    lifespan = create_cleanup_lifespan(manager, cleanup_interval)

    app = FastAPI(
        title="Insurance Claims Agent API",
        description="REST API for insurance claim processing agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==============================================================================
    # Endpoints
    # ==============================================================================

    @app.get("/health")
    def health() -> dict:
        """Health check endpoint."""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}
    
    
    @app.post("/api/session/create", response_model=dict)
    def create_session() -> dict:
        """Create a new session."""
        session_id = manager.create_session()
        return {"session_id": session_id, "created_at": datetime.now().isoformat()}
    
    
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        """Send a message to the agent and get a response."""
        agent_response, claim_draft = manager.chat(
            request.session_id,
            request.user_message,
            request.image_path,
        )
        
        session = manager.get_session(request.session_id)
        
        return ChatResponse(
            session_id=request.session_id,
            agent_response=agent_response,
            query_type=session.agent_state.get("query_type", "conversational"),
            claim_draft=claim_draft,
            messages_count=len(session.message_history),
            timestamp=datetime.now().isoformat(),
        )
    
    
    @app.get("/api/conversation", response_model=ConversationHistoryResponse)
    def get_conversation(session_id: str) -> ConversationHistoryResponse:
        """Get full conversation history for a session."""
        messages, claim_draft, created_at, last_updated = manager.get_conversation(session_id)
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=[ConversationMessage(**msg) for msg in messages],
            claim_draft=claim_draft,
            created_at=created_at,
            last_updated=last_updated,
        )
    
    
    @app.get("/api/claim-draft", response_model=ClaimDraftResponse)
    def get_claim_draft(session_id: str) -> ClaimDraftResponse:
        """Get current claim draft."""
        claim_draft = manager.get_claim_draft(session_id)
        session = manager.get_session(session_id)
        
        return ClaimDraftResponse(
            session_id=session_id,
            claim_draft=claim_draft,
            last_updated=session.last_updated.isoformat(),
        )
    
    
    @app.put("/api/claim-draft", response_model=ClaimDraftResponse)
    def update_claim_draft(request: ClaimDraftRequest) -> ClaimDraftResponse:
        """Update claim draft with user-provided fields."""
        claim_draft = manager.update_claim_draft(request.session_id, request.claim_draft)
        session = manager.get_session(request.session_id)
        
        return ClaimDraftResponse(
            session_id=request.session_id,
            claim_draft=claim_draft,
            last_updated=session.last_updated.isoformat(),
        )
    
    
    @app.post("/api/upload-image", response_model=ImageUploadResponse)
    async def upload_image(session_id: str = Form(...), file: UploadFile = File(...)) -> ImageUploadResponse:
        """Upload a damage photo."""
        # HIGH fix (Bug 4): await the now-async upload_image method
        file_path = await manager.upload_image(session_id, file)
        # MEDIUM fix: return sanitised filename, not raw user input.
        # The raw filename could expose dir traversal attempts (e.g. "../../evil.sh").
        safe_filename = Path(file.filename).name

        return ImageUploadResponse(
            session_id=session_id,
            filename=safe_filename,
            file_path=file_path,
            uploaded_at=datetime.now().isoformat(),
        )
    
    
    @app.delete("/api/session/reset", response_model=SessionResetResponse)
    def reset_session(session_id: str) -> SessionResetResponse:
        """Reset a session (clear all data)."""
        manager.reset_session(session_id)
        
        return SessionResetResponse(
            session_id=session_id,
            status="cleared",
            timestamp=datetime.now().isoformat(),
        )
    
    
    @app.get("/api/sessions/count", response_model=dict)
    def get_session_count() -> dict:
        """Get number of active sessions."""
        return {"active_sessions": len(manager.sessions)}
    
    
    @app.post("/api/sessions/cleanup", response_model=dict)
    def cleanup_sessions() -> dict:
        """Clean up expired sessions."""
        removed = manager.cleanup_expired_sessions()
        return {"removed_sessions": removed, "timestamp": datetime.now().isoformat()}
    
    
    # ==============================================================================
    # Exception Handlers
    # ==============================================================================
    
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "code": str(exc.status_code),
                "timestamp": datetime.now().isoformat(),
            }
        )
    
    async def general_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "code": "500",
                "timestamp": datetime.now().isoformat(),
            }
        )
    
    # Register exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    return app


# Initialize app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)
