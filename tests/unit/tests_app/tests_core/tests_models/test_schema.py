import pytest
from pydantic import ValidationError

from app.core.models.schema import MedicalQuery, MedicalResponse, SessionInfo


class TestMedicalQuery:
    """Test class for the MedicalQuery model."""

    def test_valid_medical_query(self):
        """Test creating a valid MedicalQuery instance."""
        query = MedicalQuery(query="What is the recommended dose?", session_id="session-1")

        assert query.query == "What is the recommended dose?"
        assert query.session_id == "session-1"

    def test_missing_required_fields(self):
        """Test validation error when required MedicalQuery fields are missing."""
        with pytest.raises(ValidationError):
            MedicalQuery(query="Only query")


class TestMedicalResponse:
    """Test class for the MedicalResponse model."""

    def test_valid_medical_response(self):
        """Test creating a valid MedicalResponse instance."""
        response = MedicalResponse(response="This is a medical answer.")

        assert response.response == "This is a medical answer."

    def test_missing_response_field(self):
        """Test validation error when the response field is missing."""
        with pytest.raises(ValidationError):
            MedicalResponse()


class TestSessionInfo:
    """Test class for the SessionInfo model."""

    def test_valid_session_info(self):
        """Test creating a valid SessionInfo instance."""
        session = SessionInfo(session_id="session-abc")

        assert session.session_id == "session-abc"

    def test_missing_session_id(self):
        """Test validation error when session_id is missing."""
        with pytest.raises(ValidationError):
            SessionInfo()
