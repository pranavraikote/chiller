"""Request/response models for the API."""

from pydantic import BaseModel, Field

from config import MAX_FIELD


class GradeRequest(BaseModel):
    question:    str = Field(..., max_length=MAX_FIELD)
    response:    str = Field(..., max_length=MAX_FIELD)
    max_grade:   int = Field(..., ge=1, le=20)
    temperature: float = Field(4.0, gt=0, le=20.0)
    tau:         float = Field(0.50,  ge=0.0, le=1.0)


class GradeResponse(BaseModel):
    predicted_grade: int
    max_grade:       int
    confidence:      float
    accepted:        bool
    prompt:          str   # returned so client can pass it back for attribution


class AttributionRequest(BaseModel):
    question:  str = Field(..., max_length=MAX_FIELD)
    response:  str = Field(..., max_length=MAX_FIELD)
    prompt:    str                                    # prompt from /grade response
    grade:     int
    max_grade: int = Field(..., ge=1, le=20)


class AttributionResponse(BaseModel):
    tokens: list   # [[word, score], ...]


class CorrectionRequest(BaseModel):
    question:        str = Field(..., max_length=MAX_FIELD)
    response:        str = Field(..., max_length=MAX_FIELD)
    max_grade:       int = Field(..., ge=1, le=20)
    predicted_grade: int
    human_grade:     int
    comment:         str = Field("", max_length=500)


class FeedbackRequest(BaseModel):
    question:  str = Field(..., max_length=MAX_FIELD)
    response:  str = Field(..., max_length=MAX_FIELD)
    grade:     int
    max_grade: int = Field(..., ge=1, le=20)


class FeedbackResponse(BaseModel):
    feedback:  str
    grounded:  bool  = False
    top_score: float = 0.0
    excerpts:  list  = []   # [{label, text, score, cited}, ...] — retrieved lecture passages
