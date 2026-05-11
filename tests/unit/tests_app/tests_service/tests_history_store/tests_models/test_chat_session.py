from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from app.service.history_store.models.chat_session import ChatSessionModel, MessageRole, SourceModel, MetadataModel, MessageModel


class TestMessageRole:
    """Test class for MessageRole enum."""

    def test_human_role(self):
        """Test HUMAN role value."""
        assert MessageRole.HUMAN == "human"

    def test_ai_role(self):
        """Test AI role value."""
        assert MessageRole.AI == "ai"

    def test_enum_values(self):
        """Test all enum values."""
        assert list(MessageRole) == [MessageRole.HUMAN, MessageRole.AI]


class TestSourceModel:
    """Test class for SourceModel Pydantic model."""

    def test_valid_source_creation(self):
        """Test creating a valid SourceModel."""
        source = SourceModel(
            index=1,
            url="https://example.com",
            med_name="Test Med",
            section_title="Test Section",
            verified=True
        )
        assert source.index == 1
        assert source.url == "https://example.com"
        assert source.med_name == "Test Med"
        assert source.section_title == "Test Section"
        assert source.verified is True

    def test_source_default_verified(self):
        """Test that verified defaults to False."""
        source = SourceModel(
            index=2,
            url="https://example.com",
            med_name="Test",
            section_title="Section"
        )
        assert source.verified is False

    def test_source_invalid_extra_field(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            SourceModel(
                index=1,
                url="https://example.com",
                med_name="Test",
                section_title="Section",
                extra_field="invalid"
            )

    def test_source_missing_required_field(self):
        """Test validation error for missing required fields."""
        with pytest.raises(ValidationError):
            SourceModel(
                index=1,
                # missing url
                med_name="Test",
                section_title="Section"
            )


class TestMetadataModel:
    """Test class for MetadataModel Pydantic model."""

    def test_valid_metadata_creation(self):
        """Test creating a valid MetadataModel."""
        source = SourceModel(index=1, url="https://example.com", med_name="Test", section_title="Section")
        metadata = MetadataModel(
            sources=[source],
            total_sources_retrieved=1
        )
        assert len(metadata.sources) == 1
        assert metadata.total_sources_retrieved == 1

    def test_metadata_empty_sources(self):
        """Test MetadataModel with empty sources list."""
        metadata = MetadataModel(
            sources=[],
            total_sources_retrieved=0
        )
        assert metadata.sources == []
        assert metadata.total_sources_retrieved == 0

    def test_metadata_invalid_extra_field(self):
        """Test that extra fields are forbidden in MetadataModel."""
        with pytest.raises(ValidationError):
            MetadataModel(
                sources=[],
                total_sources_retrieved=0,
                extra_field="invalid"
            )


class TestMessageModel:
    """Test class for MessageModel Pydantic model."""

    def test_valid_human_message(self):
        """Test creating a valid human message."""
        message = MessageModel(
            role=MessageRole.HUMAN,
            content="Hello world"
        )
        assert message.role == MessageRole.HUMAN
        assert message.content == "Hello world"
        assert message.metadata is None
        assert isinstance(message.timestamp, datetime)

    def test_valid_ai_message_with_metadata(self):
        """Test creating a valid AI message with metadata."""
        source = SourceModel(index=1, url="https://example.com", med_name="Test", section_title="Section")
        metadata = MetadataModel(sources=[source], total_sources_retrieved=1)
        message = MessageModel(
            role=MessageRole.AI,
            content="AI response",
            metadata=metadata
        )
        assert message.role == MessageRole.AI
        assert message.content == "AI response"
        assert message.metadata is metadata

    def test_message_custom_timestamp(self):
        """Test MessageModel with custom timestamp."""
        custom_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        message = MessageModel(
            role=MessageRole.HUMAN,
            content="Test",
            timestamp=custom_time
        )
        assert message.timestamp == custom_time

    def test_message_invalid_role(self):
        """Test validation error for invalid role."""
        with pytest.raises(ValidationError):
            MessageModel(
                role="invalid_role",
                content="Test"
            )

    def test_message_empty_content(self):
        """Test MessageModel with empty content."""
        message = MessageModel(
            role=MessageRole.HUMAN,
            content=""
        )
        assert message.content == ""

    def test_message_invalid_extra_field(self):
        """Test that extra fields are forbidden in MessageModel."""
        with pytest.raises(ValidationError):
            MessageModel(
                role=MessageRole.HUMAN,
                content="Test",
                extra_field="invalid"
            )


class TestChatSessionModel:
    """Test class for ChatSessionModel Pydantic model."""

    def test_valid_chat_session_creation(self):
        """Test creating a valid ChatSessionModel with messages."""
        source = SourceModel(index=1, url="https://example.com", med_name="Test", section_title="Section")
        metadata = MetadataModel(sources=[source], total_sources_retrieved=1)
        human_message = MessageModel(role=MessageRole.HUMAN, content="Hello")
        ai_message = MessageModel(role=MessageRole.AI, content="Reply", metadata=metadata)

        session = ChatSessionModel(
            session_id="session-123",
            messages=[human_message, ai_message]
        )

        assert session.session_id == "session-123"
        assert len(session.messages) == 2
        assert session.messages[0].role == MessageRole.HUMAN
        assert session.created_at <= session.updated_at

    def test_chat_session_default_values(self):
        """Test ChatSessionModel default values for dates and messages."""
        session = ChatSessionModel(session_id="session-default")

        assert session.session_id == "session-default"
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.messages == []

    def test_chat_session_invalid_extra_field(self):
        """Test that extra fields are forbidden in ChatSessionModel."""
        with pytest.raises(ValidationError):
            ChatSessionModel(
                session_id="session-123",
                extra_field="invalid"
            )