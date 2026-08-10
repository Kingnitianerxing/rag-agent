import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_current_user, limiter
from app.auth.acl import can_delete_document, can_ingest, can_share_document, resolve_allowed_sources
from app.auth.db import delete_document_acl, get_document_acl, upsert_document_acl
from app.auth.models import VALID_ROLES, AuthUser
from app.config import get_settings
from app.core.factories import image_retrieval_ready
from app.core.pipeline import ingest_pipeline
from app.ingestion.validation import ALLOWED_SUFFIXES, IMAGE_SUFFIXES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    source: str

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        from app.ingestion.validation import validate_source as _validate

        return _validate(v, get_settings())


class IngestResponse(BaseModel):
    source: str
    chunks: int
    status: str = "ingested"


class ShareDocumentRequest(BaseModel):
    allowed_roles: list[str] = Field(default_factory=list)
    allowed_user_ids: list[int] = Field(default_factory=list)


@router.post("", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest(
    request: Request, body: IngestRequest, user: AuthUser = Depends(get_current_user)
):
    if not can_ingest(user):
        raise HTTPException(status_code=403, detail="Ingest requires editor or admin role")
    owner_id = None if user.is_service or user.id <= 0 else user.id
    result = await asyncio.to_thread(ingest_pipeline, body.source, False, owner_id)
    return IngestResponse(**result)


@router.post("/upload", response_model=IngestResponse)
@limiter.limit("10/minute")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    user: AuthUser = Depends(get_current_user),
):
    if not can_ingest(user):
        raise HTTPException(status_code=403, detail="Ingest requires editor or admin role")
    settings = get_settings()
    name = os.path.basename(file.filename or "")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}")
    if suffix in IMAGE_SUFFIXES and not image_retrieval_ready():
        raise HTTPException(
            status_code=503,
            detail="Image upload requires IMAGE_RETRIEVAL_ENABLED=true and DASHSCOPE_API_KEY",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB (max {settings.MAX_FILE_SIZE_MB}MB)",
        )

    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / name
    if dest.exists():
        dest = data_dir / f"{Path(name).stem}-{uuid.uuid4().hex[:8]}{suffix}"
    dest.write_bytes(contents)

    owner_id = None if user.is_service or user.id <= 0 else user.id
    result = await asyncio.to_thread(ingest_pipeline, str(dest), False, owner_id)
    return IngestResponse(**result)


@router.get("/documents")
async def list_documents(user: AuthUser = Depends(get_current_user)):
    """List ingested document records visible to the current user."""
    settings = get_settings()
    tracking_file = Path(settings.DATA_DIR) / "ingestions.json"
    if not tracking_file.exists():
        return []
    try:
        with open(tracking_file) as f:
            data = json.load(f)
        docs = []
        for k, v in data.items():
            acl = get_document_acl(k)
            docs.append(
                {
                    "id": k,
                    "source": v["source"],
                    "chunks": v["chunks"],
                    "ingested_at": v.get("ingested_at", ""),
                    "modality": v.get("modality", "text"),
                    "owner_id": acl["owner_id"] if acl else None,
                    "allowed_roles": acl["allowed_roles"] if acl else [],
                    "allowed_user_ids": acl["allowed_user_ids"] if acl else [],
                    "can_delete": can_delete_document(user, k),
                    "can_share": can_share_document(user),
                }
            )
        allowed = resolve_allowed_sources(user)
        if allowed is None:
            return docs
        allowed_set = set(allowed)
        return [d for d in docs if d["source"] in allowed_set]
    except Exception as exc:
        logger.error("Failed to read ingestion tracking: %s", exc)
        return []


@router.patch("/documents/{doc_id}/share")
async def share_document(
    doc_id: str,
    body: ShareDocumentRequest,
    user: AuthUser = Depends(get_current_user),
):
    """Admin-only: set which roles/users may read a document."""
    if not can_share_document(user):
        raise HTTPException(status_code=403, detail="Only admin can share documents")

    bad = sorted({r for r in body.allowed_roles if r not in VALID_ROLES})
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid roles: {bad}")

    roles = list(dict.fromkeys(body.allowed_roles))
    user_ids = list(dict.fromkeys(int(x) for x in body.allowed_user_ids))

    settings = get_settings()
    tracking_file = Path(settings.DATA_DIR) / "ingestions.json"
    if not tracking_file.exists():
        raise HTTPException(status_code=404, detail="No ingestion records found")
    try:
        with open(tracking_file) as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to read ingestion tracking: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if doc_id not in data:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    source = data[doc_id]["source"]
    acl = get_document_acl(doc_id)
    owner_id = int(acl["owner_id"]) if acl else user.id
    upsert_document_acl(
        source_hash=doc_id,
        source=source,
        owner_id=owner_id,
        allowed_roles=roles,
        allowed_user_ids=user_ids,
    )
    return {
        "status": "shared",
        "doc_id": doc_id,
        "source": source,
        "owner_id": owner_id,
        "allowed_roles": roles,
        "allowed_user_ids": user_ids,
    }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user: AuthUser = Depends(get_current_user)):
    """Remove a document record from tracking (owner or admin)."""
    if not can_delete_document(user, doc_id):
        raise HTTPException(status_code=403, detail="Not allowed to delete this document")
    settings = get_settings()
    tracking_file = Path(settings.DATA_DIR) / "ingestions.json"
    if not tracking_file.exists():
        raise HTTPException(status_code=404, detail="No ingestion records found")
    try:
        with open(tracking_file) as f:
            data = json.load(f)
        if doc_id not in data:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        removed = data.pop(doc_id)
        with open(tracking_file, "w") as f:
            json.dump(data, f, indent=2)
        delete_document_acl(doc_id)
        return {"status": "deleted", "doc_id": doc_id, "source": removed["source"]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete document record: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
