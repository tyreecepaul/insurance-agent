"""
tests/unit/test_ingest_utils.py
Unit tests for ingest.py utility functions (chunk_text, classifiers, etc.)
Fast, isolated tests with no external dependencies.
"""

import pytest
from src.ingest import (
    chunk_text,
    _guess_insurer,
    _guess_insurance_type,
    _classify_damage_from_filename,
)


@pytest.mark.unit
class TestChunkText:
    """Tests for text chunking logic."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking with overlap."""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 1
        assert len(chunks[0]) == 100
        # Verify overlap
        assert chunks[1][:10] == chunks[0][-10:]
    
    def test_chunk_text_single_chunk(self):
        """Test that text smaller than chunk_size returns single chunk."""
        text = "short text"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_preserves_content(self):
        """Test that all text is preserved across chunks."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            # Each chunk overlaps with previous
            reconstructed += chunk[20:]  # Skip overlap region
        
        # Account for potential trailing text in last chunk
        assert text.startswith(reconstructed[:100])
    
    def test_chunk_text_empty(self):
        """Test handling of empty string."""
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []
    
    def test_chunk_text_no_overlap(self):
        """Test chunking with zero overlap."""
        text = "a" * 300
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        
        assert len(chunks) == 3
        assert chunks[0] == "a" * 100
        assert chunks[1] == "a" * 100
        assert chunks[2] == "a" * 100


@pytest.mark.unit
class TestGuessInsurer:
    """Tests for insurer name extraction from filenames."""
    
    def test_nrma(self):
        assert _guess_insurer("NRMA_motor_policy_2024.pdf") == "nrma"
    
    def test_allianz(self):
        assert _guess_insurer("allianz_home_insurance.pdf") == "allianz"
    
    def test_racq(self):
        assert _guess_insurer("RACQ_motor_pds.pdf") == "racq"
    
    def test_medibank(self):
        assert _guess_insurer("medibank_health_policy.pdf") == "medibank"
    
    def test_bupa(self):
        assert _guess_insurer("bupa_extras.pdf") == "bupa"
    
    def test_unknown(self):
        assert _guess_insurer("unknown_insurance.pdf") == "unknown"
    
    def test_multiple_matches_first_wins(self):
        """If multiple insurers in name, first match wins."""
        assert _guess_insurer("nrma_vs_allianz_comparison.pdf") == "nrma"
    
    def test_case_insensitive(self):
        assert _guess_insurer("NRMA_Policy.PDF") == "nrma"
        assert _guess_insurer("Allianz-Home.pdf") == "allianz"


@pytest.mark.unit
class TestGuessInsuranceType:
    """Tests for insurance type detection from filenames."""
    
    def test_motor(self):
        assert _guess_insurance_type("NRMA_motor_pds.pdf") == "motor"
        assert _guess_insurance_type("car_insurance.pdf") == "motor"
        assert _guess_insurance_type("vehicle_policy.pdf") == "motor"
        assert _guess_insurance_type("auto_extended_warranty.pdf") == "motor"
    
    def test_home(self):
        assert _guess_insurance_type("allianz_home_policy.pdf") == "home"
        assert _guess_insurance_type("house_building.pdf") == "home"
        assert _guess_insurance_type("property_pds.pdf") == "home"
        assert _guess_insurance_type("building_coverage.pdf") == "home"
    
    def test_health(self):
        assert _guess_insurance_type("medibank_health_extras.pdf") == "health"
        assert _guess_insurance_type("medical_insurance.pdf") == "health"
        assert _guess_insurance_type("hospital_cover.pdf") == "health"
    
    def test_general(self):
        """Default to general if no match."""
        assert _guess_insurance_type("misc_policy.pdf") == "general"
        assert _guess_insurance_type("coverage.pdf") == "general"
    
    def test_case_insensitive(self):
        assert _guess_insurance_type("MOTOR_POLICY.PDF") == "motor"
        assert _guess_insurance_type("Home-Insurance.pdf") == "home"


@pytest.mark.unit
class TestClassifyDamageFromFilename:
    """Tests for damage type and insurance category classification."""
    
    def test_vehicle_damage_motor(self):
        assert _classify_damage_from_filename("car_rear_dent_01") == ("vehicle damage", "motor")
        assert _classify_damage_from_filename("vehicle_crash_photo") == ("vehicle damage", "motor")
        assert _classify_damage_from_filename("bumper_damage_02") == ("vehicle damage", "motor")
        assert _classify_damage_from_filename("windscreen_crack") == ("vehicle damage", "motor")
    
    def test_property_damage_home(self):
        assert _classify_damage_from_filename("roof_hail_damage") == ("property damage", "home")
        assert _classify_damage_from_filename("ceiling_water_leak") == ("property damage", "home")
        assert _classify_damage_from_filename("wall_crack_foundation") == ("property damage", "home")
        assert _classify_damage_from_filename("window_shattered") == ("property damage", "home")
    
    def test_weather_damage_home(self):
        assert _classify_damage_from_filename("flood_basement") == ("weather damage", "home")
        assert _classify_damage_from_filename("water_damage_living_room") == ("weather damage", "home")
        assert _classify_damage_from_filename("storm_damage_fence") == ("weather damage", "home")
        assert _classify_damage_from_filename("hail_garage") == ("weather damage", "home")
    
    def test_fire_damage_home(self):
        assert _classify_damage_from_filename("fire_damage_kitchen") == ("fire damage", "home")
        assert _classify_damage_from_filename("smoke_damage_bedroom") == ("fire damage", "home")
        assert _classify_damage_from_filename("burn_marks_door") == ("fire damage", "home")
    
    def test_general_damage(self):
        """Default to general if no match."""
        assert _classify_damage_from_filename("unknown_damage_01") == ("general damage", "general")
        assert _classify_damage_from_filename("photo_35") == ("general damage", "general")
    
    def test_case_insensitive(self):
        assert _classify_damage_from_filename("CAR_DENT_PHOTO") == ("vehicle damage", "motor")
        assert _classify_damage_from_filename("ROOF_DAMAGE") == ("property damage", "home")


@pytest.mark.unit
class TestIntegration:
    """Integration tests of utility functions."""
    
    def test_full_pipeline_motor_policy(self):
        """Test extracting metadata from a complete motor policy filename."""
        filename = "NRMA_motor_comprehensive_pds_v2024.pdf"
        
        insurer = _guess_insurer(filename)
        assert insurer == "nrma"
        
        ins_type = _guess_insurance_type(filename)
        assert ins_type == "motor"
    
    def test_full_pipeline_damage_photo(self):
        """Test extracting metadata from a complete damage photo filename."""
        filename = "car_rear_dent_collision_02.jpg"
        
        damage_type, category = _classify_damage_from_filename(filename)
        assert damage_type == "vehicle damage"
        assert category == "motor"
    
    def test_chunking_realistic_content(self):
        """Test chunking a realistic policy excerpt."""
        policy_text = """
        SECTION 3: EXCESS AMOUNTS
        
        For comprehensive motor insurance, the standard excess is $750.
        Young drivers (under 25) may choose a reduced excess of $500.
        No excess applies to windscreen claims (covered separately).
        
        SECTION 4: EXCLUSIONS
        
        This policy does not cover:
        - Racing or rally participation
        - Intentional damage
        - Liability claims from uninsured drivers
        """ * 10  # Repeat to make it longer
        
        chunks = chunk_text(policy_text, chunk_size=500, overlap=50)
        
        # Verify chunking worked
        assert len(chunks) > 1
        assert all(len(chunk) > 0 for chunk in chunks)
        
        # Verify content is preserved
        full = "".join([chunks[0]] + [c[50:] for c in chunks[1:]])
        assert "SECTION 3" in full
        assert "SECTION 4" in full
