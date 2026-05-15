from pydantic import BaseModel, Field


class Validate_AI_Response(BaseModel):
    res: str = Field(min_length=1, max_length=4096)


class UploadResponse(BaseModel):  # whether file is uploaded or not
    session_id: str
    event: str  # success or failure
    message: str | None = None


class User_Request(BaseModel):
    prompt: str
