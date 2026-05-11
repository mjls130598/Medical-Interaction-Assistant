from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(str, Enum):
    """
    Enum representing the role of a message in the chat session.
    """

    HUMAN = "human"
    AI = "ai"

class SourceModel(BaseModel):
    """
    Model representing the metadata of a source retrieved 
    as context for an AI response.
    """

    index: int
    url: str
    med_name: str
    section_title: str
    verified: bool = False

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "index": 1,
                "url": "https://www.ejemplo.com/ibuprofeno",
                "med_name": "Ibuprofeno",
                "section_title": "Información y Dosis",
                "verified": True
            }
        },
        extra = "forbid"
    )

class MetadataModel(BaseModel):
    """
    Model representing the metadata of an AI response message,
      including source citations and retrieval information.
    """

    sources: list[SourceModel]
    total_sources_retrieved: int

    model_config = ConfigDict(
        json_schema_extra = {
            "example":{
                "sources": [
                    {
                        "index": 1,
                        "url": "https://www.ejemplo.com/ibuprofeno",
                        "med_name": "Ibuprofeno",
                        "section_title": "Información y Dosis",
                        "verified": True
                    }
                ],
                "total_sources_retrieved": 1
            }
        },
        extra = "forbid"
    )

class MessageModel(BaseModel):
    """
    Model representing a single message in the chat session, 
    including role, content, timestamp, and optional metadata.
    """

    model_config = ConfigDict(
        json_schema_extra = {
            "examples": [
                {
                    "role": "human",
                    "content": "¿Cuál es la dosis recomendada de ibuprofeno para un adulto?",
                    "timestamp": "2024-06-01T12:00:00Z"
                },
                {
                    "role": "ai",
                    "content": "La dosis recomendada de ibuprofeno para un adulto es ...",
                    "metadata": {
                        "sources": [
                            {
                                "index": 1,
                                "url": "https://www.ejemplo.com/ibuprofeno",
                                "med_name": "Ibuprofeno",
                                "section_title": "Información y Dosis",
                                "verified": True
                            }
                        ],
                        "total_sources_retrieved": 1
                    },
                    "timestamp": "2024-06-01T12:00:05Z"
                }
            ]
        },
        extra = "forbid"
    )

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory = lambda: datetime.now(timezone.utc))
    metadata: Optional[MetadataModel] = None

class ChatSessionModel(BaseModel):
    """
    Model representing a chat session, which includes a unique session ID and a list of messages.
    """

    session_id: str
    created_at: datetime = Field(default_factory = lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory = lambda: datetime.now(timezone.utc))
    messages: list[MessageModel] = Field(default_factory = list)

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "created_at": "2024-06-01T12:00:00Z",
                "updated_at": "2024-06-01T12:00:05Z",
                "messages": [
                    {
                        "role": "human",
                        "content": "¿Cuál es la dosis recomendada de ibuprofeno para un adulto?",
                        "timestamp": "2024-06-01T12:00:00Z"
                    },
                    {
                        "role": "ai",
                        "content": "La dosis recomendada de ibuprofeno para un adulto es ...",
                        "metadata": {
                            "sources": [
                                {
                                    "index": 1,
                                    "url": "https://www.ejemplo.com/ibuprofeno",
                                    "med_name": "Ibuprofeno",
                                    "section_title": "Información y Dosis",
                                    "verified": True
                                }
                            ],
                            "total_sources_retrieved": 1
                        },
                        "timestamp": "2024-06-01T12:00:05Z"
                    }
                ]
            }
        },
        extra = "forbid",
        populate_by_name = True,
        arbitrary_types_allowed = True
    )
