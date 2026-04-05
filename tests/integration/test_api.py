"""
tests/integration/test_api.py
- Integration tests for FastAPI endpoints
- Tests session management, chat flow, image upload, claim draft updates
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Import API
from src.api import app, SessionManager


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def session_manager():
    """Session manager instance."""
    manager = SessionManager(expiration_hours=24)
    return manager


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "timestamp" in response.json()


class TestSessionManagement:
    """Test session creation and management."""
    
    def test_create_session(self, client):
        """Create a new session."""
        response = client.post("/api/session/create")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0
        assert "created_at" in data
    
    def test_reset_session(self, client):
        """Reset session clears all data."""
        # Create session
        create_resp = client.post("/api/session/create")
        session_id = create_resp.json()["session_id"]
        
        # Reset it
        reset_resp = client.delete("/api/session/reset", params={"session_id": session_id})
        assert reset_resp.status_code == 200
        assert reset_resp.json()["status"] == "cleared"
        
        # Trying to access should fail
        conv_resp = client.get("/api/conversation", params={"session_id": session_id})
        assert conv_resp.status_code == 404
    
    def test_get_session_count(self, client):
        """Track active session counts."""
        # Create a few sessions
        client.post("/api/session/create")
        client.post("/api/session/create")
        
        response = client.get("/api/sessions/count")
        assert response.status_code == 200
        assert response.json()["active_sessions"] >= 2


class TestChatEndpoint:
    """Test chat message endpoint."""
    
    def test_chat_single_message(self, client, skip_if_no_ollama, skip_if_no_chroma):
        """Send a single chat message."""
        # Create session
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Send message
        chat_resp = client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "What is my excess on my home insurance policy?"
        })
        
        assert chat_resp.status_code == 200
        data = chat_resp.json()
        assert data["session_id"] == session_id
        assert "agent_response" in data
        assert isinstance(data["agent_response"], str)
        assert len(data["agent_response"]) > 0
        assert "query_type" in data
        assert data["query_type"] in ["factual", "cross_modal", "analytical", "conversational"]
        assert "claim_draft" in data
        assert isinstance(data["claim_draft"], dict)
        assert "messages_count" in data
        assert data["messages_count"] >= 1
        assert "timestamp" in data
    
    def test_chat_multiple_turns(self, client, skip_if_no_ollama, skip_if_no_chroma):
        """Chat with multiple turns preserves state."""
        # Create session
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Turn 1
        resp1 = client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "I have a motor insurance policy POL-MOTOR-1042"
        })
        assert resp1.status_code == 200
        assert resp1.json()["claim_draft"].get("policy_number") == "POL-MOTOR-1042"
        
        # Turn 2
        resp2 = client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "I had a crash yesterday"
        })
        assert resp2.status_code == 200
        # Should preserve previous policy number
        claim_draft = resp2.json()["claim_draft"]
        assert claim_draft.get("policy_number") == "POL-MOTOR-1042"
    
    def test_chat_invalid_session(self, client):
        """Chat with invalid session ID should fail."""
        response = client.post("/api/chat", json={
            "session_id": "invalid-session-xyz",
            "user_message": "Hello"
        })
        assert response.status_code == 404
        assert "not found" in response.json()["error"].lower()
    
    def test_chat_memory_extraction(self, client, skip_if_no_ollama, skip_if_no_chroma):
        """Memory node should extract insurance type and policy numbers."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Test motor insurance detection
        resp = client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "I need help with my motor insurance claim"
        })
        assert resp.status_code == 200
        assert resp.json()["claim_draft"].get("insurance_type") == "motor"
        
        # Test home insurance
        session_resp2 = client.post("/api/session/create")
        session_id2 = session_resp2.json()["session_id"]
        resp2 = client.post("/api/chat", json={
            "session_id": session_id2,
            "user_message": "My home was damaged in a flood"
        })
        assert resp2.status_code == 200
        assert resp2.json()["claim_draft"].get("insurance_type") == "home"


class TestConversationHistory:
    """Test conversation history retrieval."""
    
    def test_get_empty_conversation(self, client, skip_if_no_ollama, skip_if_no_chroma):
        """Getting conversation with no messages."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Send one message first
        client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "Hello"
        })
        
        # Get conversation
        conv_resp = client.get("/api/conversation", params={"session_id": session_id})
        assert conv_resp.status_code == 200
        data = conv_resp.json()
        assert data["session_id"] == session_id
        assert len(data["messages"]) >= 2  # At least user + agent
        assert "claim_draft" in data
        assert "created_at" in data
        assert "last_updated" in data
        
        # Check message structure
        for msg in data["messages"]:
            assert "role" in msg
            assert msg["role"] in ["user", "agent"]
            assert "content" in msg
            assert "timestamp" in msg
            if msg["role"] == "agent":
                assert "query_type" in msg
    
    def test_conversation_history_multiple_turns(self, client, skip_if_no_ollama, skip_if_no_chroma):
        """Conversation history includes all turns."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Send 3 messages
        messages = [
            "What is my excess?",
            "I have a motor policy",
            "I had an accident"
        ]
        
        for msg in messages:
            client.post("/api/chat", json={
                "session_id": session_id,
                "user_message": msg
            })
        
        # Get conversation
        conv_resp = client.get("/api/conversation", params={"session_id": session_id})
        assert conv_resp.status_code == 200
        data = conv_resp.json()
        
        # Should have 3 user messages + 3 agent responses
        user_messages = [m for m in data["messages"] if m["role"] == "user"]
        assert len(user_messages) >= 3


class TestClaimDraft:
    """Test claim draft GET/PUT endpoints."""
    
    def test_get_claim_draft(self, client):
        """Get current claim draft."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Send message that updates draft
        client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "I have a home insurance policy POL-HOME-5678"
        })
        
        # Get draft
        draft_resp = client.get("/api/claim-draft", params={"session_id": session_id})
        assert draft_resp.status_code == 200
        data = draft_resp.json()
        assert data["session_id"] == session_id
        assert "claim_draft" in data
        assert data["claim_draft"].get("policy_number") == "POL-HOME-5678"
        assert "last_updated" in data
    
    def test_update_claim_draft(self, client):
        """Update claim draft with user-provided fields."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Update draft
        update_resp = client.put("/api/claim-draft", json={
            "session_id": session_id,
            "claim_draft": {
                "policy_number": "POL-HEALTH-9999",
                "claimant_name": "John Doe",
                "incident_date": "2025-01-15"
            }
        })
        
        assert update_resp.status_code == 200
        data = update_resp.json()
        assert data["claim_draft"]["policy_number"] == "POL-HEALTH-9999"
        assert data["claim_draft"]["claimant_name"] == "John Doe"
        
        # Verify with GET
        get_resp = client.get("/api/claim-draft", params={"session_id": session_id})
        assert get_resp.json()["claim_draft"]["claimant_name"] == "John Doe"
    
    def test_claim_draft_persists_across_messages(self, client):
        """Claim draft persists when sending new messages."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Set initial draft
        client.put("/api/claim-draft", json={
            "session_id": session_id,
            "claim_draft": {"claimant_name": "Jane Smith"}
        })
        
        # Send a message
        client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "I have motor insurance"
        })
        
        # Draft should still have claimant name + new insurance type
        draft_resp = client.get("/api/claim-draft", params={"session_id": session_id})
        data = draft_resp.json()
        assert data["claim_draft"]["claimant_name"] == "Jane Smith"
        assert data["claim_draft"].get("insurance_type") == "motor"


class TestImageUpload:
    """Test image upload endpoint."""
    
    def test_upload_image(self, client):
        """Upload an image and get file path."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"fake jpeg data")
            tmp_path = tmp.name
        
        try:
            # Upload
            with open(tmp_path, "rb") as f:
                upload_resp = client.post(
                    "/api/upload-image",
                    data={"session_id": session_id},
                    files={"file": f}
                )
            
            assert upload_resp.status_code == 200
            data = upload_resp.json()
            assert data["session_id"] == session_id
            assert "filename" in data
            assert "file_path" in data
            assert "uploaded_at" in data
            assert session_id in data["file_path"]
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_upload_image_creates_session_dir(self, client):
        """Upload creates session-specific directory."""
        session_resp = client.post("/api/session/create")
        session_id = session_resp.json()["session_id"]
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"fake png data")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as f:
                upload_resp = client.post(
                    "/api/upload-image",
                    data={"session_id": session_id},
                    files={"file": f}
                )
            
            assert upload_resp.status_code == 200
            file_path = upload_resp.json()["file_path"]
            
            # Check directory exists
            upload_dir = Path(file_path).parent
            assert upload_dir.exists()
            assert session_id in str(upload_dir)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_upload_image_invalid_session(self, client):
        """Upload with invalid session should fail."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"fake data")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as f:
                response = client.post(
                    "/api/upload-image",
                    data={"session_id": "invalid-session"},
                    files={"file": f}
                )
            
            assert response.status_code == 404
            assert "not found" in response.json()["error"].lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestSessionExpiration:
    """Test session expiration and cleanup."""
    
    def test_cleanup_sessions_endpoint(self, client):
        """Cleanup endpoint works."""
        response = client.post("/api/sessions/cleanup")
        assert response.status_code == 200
        assert "removed_sessions" in response.json()
        assert "timestamp" in response.json()


class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_required_field(self, client):
        """Chat without session_id should fail."""
        response = client.post("/api/chat", json={
            "user_message": "Hello"
        })
        assert response.status_code == 422  # Validation error
    
    def test_missing_user_message(self, client):
        """Chat without user_message should fail."""
        response = client.post("/api/chat", json={
            "session_id": "test-session"
        })
        assert response.status_code == 422
    
    def test_invalid_json(self, client):
        """Invalid JSON should return 422."""
        response = client.post("/api/chat", data="invalid json")
        assert response.status_code in [400, 422]


class TestIntegrationFlow:
    """Test complete user flows."""
    
    def test_full_claim_workflow(self, client, skip_if_no_ollama, skip_if_no_chroma):
        """Complete workflow: create session -> chat -> upload -> draft -> reset."""
        # 1. Create session
        create_resp = client.post("/api/session/create")
        session_id = create_resp.json()["session_id"]
        
        # 2. Chat about crash
        chat1 = client.post("/api/chat", json={
            "session_id": session_id,
            "user_message": "I had a motor insurance crash. Policy is POL-MOTOR-5555"
        })
        assert chat1.status_code == 200
        draft1 = chat1.json()["claim_draft"]
        assert draft1.get("policy_number") == "POL-MOTOR-5555"
        
        # 3. Upload damage photo
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"fake photo")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as f:
                upload = client.post(
                    "/api/upload-image",
                    data={"session_id": session_id},
                    files={"file": f}
                )
            assert upload.status_code == 200
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        # 4. Update claim draft manually
        update = client.put("/api/claim-draft", json={
            "session_id": session_id,
            "claim_draft": {"claimant_name": "Alice"}
        })
        assert update.status_code == 200
        
        # 5. Get full conversation
        conv = client.get("/api/conversation", params={"session_id": session_id})
        assert conv.status_code == 200
        assert len(conv.json()["messages"]) >= 1
        
        # 6. Reset session
        reset = client.delete("/api/session/reset", params={"session_id": session_id})
        assert reset.status_code == 200
        
        # 7. Verify session is gone
        get_conv = client.get("/api/conversation", params={"session_id": session_id})
        assert get_conv.status_code == 404
