"""
tests/integration/test_ingest_e2e.py
End-to-end tests for ingest.py with mocked models but real ChromaDB logic.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.ingest import (
    ingest_policies,
    ingest_damage_photos,
    ingest_claims,
    get_client,
)

import numpy as np


@pytest.mark.integration
class TestIngestPolicies:
    """Integration tests for policy indexing."""
    
    @patch("src.ingest.fitz.open")
    @patch("src.ingest.SentenceTransformer")
    def test_ingest_policies_basic(self, mock_model_class, mock_fitz_open, mock_chroma_client):
        """Test policy ingest with mocked PDF and embeddings."""
        # Mock PDF
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Comprehensive motor excess: $750. Windscreen: no excess."
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_fitz_open.return_value = mock_pdf
        
        # Mock embeddings
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_model_class.return_value = mock_model
        
        # Create temp policy file
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir)
            pdf_file = policy_dir / "NRMA_motor_pds.pdf"
            pdf_file.touch()
            
            with patch("src.ingest.POLICY_DIR", policy_dir):
                # Mock client
                mock_collection = MagicMock()
                mock_collection.get.return_value = {"ids": []}
                mock_collection.count.return_value = 1
                mock_client = MagicMock()
                mock_client.get_or_create_collection.return_value = mock_collection
                
                ingest_policies(mock_client, mock_model)
                
                # Verify collection.add was called
                assert mock_collection.add.called
    
    @patch("src.ingest.SentenceTransformer")
    def test_ingest_policies_no_pdfs(self, mock_model_class, capsys):
        """Test that ingest gracefully handles empty policy directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir)
            
            with patch("src.ingest.POLICY_DIR", policy_dir):
                mock_client = MagicMock()
                mock_collection = MagicMock()
                mock_collection.get.return_value = {"ids": []}
                mock_client.get_or_create_collection.return_value = mock_collection
                
                mock_model = MagicMock()
                
                ingest_policies(mock_client, mock_model)
                
                # Should print "No policy documents found" message
                captured = capsys.readouterr()
                assert "No policy documents found" in captured.out


@pytest.mark.integration
class TestIngestDamagePhotos:
    """Integration tests for damage photo indexing."""
    
    @patch("src.ingest.load_clip")
    @patch("src.ingest.generate_caption")
    def test_ingest_damage_photos_basic(
        self, mock_caption_fn, mock_load_clip_fn, mock_chroma_client
    ):
        """Test damage photo ingest with mocked CLIP and captions."""
        # Mock CLIP
        import torch
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_model.encode_image.return_value = torch.ones(1, 512) / 512.0
        mock_load_clip_fn.return_value = (mock_model, mock_preprocess, "cpu")
        
        # Mock caption generation
        mock_caption_fn.return_value = "Photo of car rear dent"
        
        # Create temp damage photo
        with tempfile.TemporaryDirectory() as tmpdir:
            damage_dir = Path(tmpdir)
            
            # Create a dummy PNG file
            import struct
            png_file = damage_dir / "car_dent_01.png"
            # Minimal valid PNG header
            png_file.write_bytes(
                b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
                b'\x00\x01\x01\x00\x05\xcc\xfb\x0b\x0b\x00\x00\x00\x00IEND\xaeB`\x82'
            )
            
            with patch("src.ingest.DAMAGE_DIR", damage_dir):
                with patch("src.ingest.Image.open"):  # Mock PIL Image
                    mock_client = MagicMock()
                    mock_collection = MagicMock()
                    mock_collection.get.return_value = {"ids": []}
                    mock_collection.count.return_value = 1
                    mock_client.get_or_create_collection.return_value = mock_collection
                    
                    ingest_damage_photos(mock_client)
                    
                    # Verify collection.add was called
                    assert mock_collection.add.called
    
    def test_ingest_damage_photos_no_images(self, capsys):
        """Test that ingest gracefully handles empty damage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            damage_dir = Path(tmpdir)
            
            with patch("src.ingest.DAMAGE_DIR", damage_dir):
                mock_client = MagicMock()
                mock_collection = MagicMock()
                mock_collection.get.return_value = {"ids": []}
                mock_client.get_or_create_collection.return_value = mock_collection
                
                ingest_damage_photos(mock_client)
                
                # Should print "No images found" message
                captured = capsys.readouterr()
                assert "No images found" in captured.out


@pytest.mark.integration
class TestIngestClaims:
    """Integration tests for claims indexing."""
    
    @patch("src.ingest.SentenceTransformer")
    def test_ingest_claims_basic(self, mock_model_class, sample_claims_data):
        """Test claims ingest with real JSON data."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384
        mock_model_class.return_value = mock_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            claims_dir = Path(tmpdir)
            claims_file = claims_dir / "claims.json"
            
            with open(claims_file, "w") as f:
                json.dump(sample_claims_data, f)
            
            with patch("src.ingest.CLAIMS_FILE", claims_file):
                mock_client = MagicMock()
                mock_model.encode.return_value = np.array([0.1] * 384)
                mock_collection = MagicMock()
                mock_collection.get.return_value = {"ids": []}
                mock_collection.count.return_value = len(sample_claims_data)
                mock_client.get_or_create_collection.return_value = mock_collection
                
                ingest_claims(mock_client, mock_model)
                
                # Verify collection.add called for each claim
                assert mock_collection.add.call_count == len(sample_claims_data)
    
    @patch("src.ingest.SentenceTransformer")
    def test_ingest_claims_missing_file(self, mock_model_class, capsys):
        """Test graceful handling of missing claims file."""
        mock_model = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            claims_file = Path(tmpdir) / "non_existent.json"
            
            with patch("src.ingest.CLAIMS_FILE", claims_file):
                mock_client = MagicMock()
                mock_collection = MagicMock()
                mock_client.get_or_create_collection.return_value = mock_collection
                
                ingest_claims(mock_client, mock_model)
                
                # Should print "not found" message
                captured = capsys.readouterr()
                assert "not found" in captured.out


@pytest.mark.integration
class TestGetClient:
    """Tests for ChromaDB client creation."""
    
    def test_get_client_returns_persistent_client(self):
        """Test that get_client returns a PersistentClient."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.ingest.CHROMA_DIR", tmpdir):
                client = get_client()
                
                # Verify it's a ChromaDB client
                assert hasattr(client, "get_or_create_collection")
                assert hasattr(client, "list_collections")
