"""DashScope multimodal embeddings (qwen3-vl-embedding) for the image track."""
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def _image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = _MIME_BY_SUFFIX.get(suffix) or mimetypes.guess_type(str(path))[0] or "image/png"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _caption_text_from_response(resp) -> str:
    if getattr(resp, "status_code", None) not in (None, 200) and int(resp.status_code) != 200:
        raise RuntimeError(
            f"DashScope VL caption failed: status={resp.status_code} "
            f"code={getattr(resp, 'code', '')} message={getattr(resp, 'message', '')}"
        )
    output = getattr(resp, "output", None)
    choices = None
    if isinstance(output, dict):
        choices = output.get("choices")
    else:
        choices = getattr(output, "choices", None)
    if not choices:
        raise RuntimeError(f"Unexpected DashScope VL caption response: {resp!r}")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    content = None
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise RuntimeError(f"Empty DashScope VL caption content: {resp!r}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def caption_image_file(path: str | Path, model: str | None = None) -> str:
    """Describe an image with a DashScope vision model for grounded generation context."""
    import dashscope
    from dashscope import MultiModalConversation

    settings = get_settings()
    api_key = settings.DASHSCOPE_API_KEY
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is required for image captioning")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    data_uri = _image_data_uri(p)
    caption_model = model or settings.VL_CAPTION_MODEL
    dashscope.api_key = api_key
    messages = [
        {
            "role": "user",
            "content": [
                {"image": data_uri},
                {
                    "text": (
                        "用中文详细描述这张图片的主要内容，包括人物、物体、动作、场景和显著细节。"
                        "只输出描述，不要开场白。"
                    )
                },
            ],
        }
    ]
    resp = MultiModalConversation.call(
        api_key=api_key,
        model=caption_model,
        messages=messages,
    )
    text = _caption_text_from_response(resp)
    logger.info("vl_caption: path=%s model=%s chars=%d", p.name, caption_model, len(text))
    return text


def _extract_embedding(resp) -> list[float]:
    """Normalize DashScope MultiModalEmbedding response shapes."""
    if getattr(resp, "status_code", None) not in (None, 200) and int(resp.status_code) != 200:
        raise RuntimeError(
            f"DashScope VL embedding failed: status={resp.status_code} "
            f"code={getattr(resp, 'code', '')} message={getattr(resp, 'message', '')}"
        )
    output = getattr(resp, "output", None) or {}
    if isinstance(output, dict):
        embeddings = output.get("embeddings") or []
        if embeddings:
            emb = embeddings[0]
            if isinstance(emb, dict) and "embedding" in emb:
                return list(emb["embedding"])
            if isinstance(emb, list):
                return list(emb)
        # Some SDK versions nest under output["embedding"]
        if "embedding" in output:
            return list(output["embedding"])
    raise RuntimeError(f"Unexpected DashScope VL embedding response: {resp!r}")


class VLEmbedder:
    """Thin wrapper around dashscope.MultiModalEmbedding for text/image vectors."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.DASHSCOPE_API_KEY
        self.model = model or settings.VL_EMBEDDING_MODEL
        self.dimension = dimension if dimension is not None else settings.VL_EMBEDDING_DIMENSION
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for VL embeddings")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call(self, input_items: list) -> list[float]:
        import dashscope
        from dashscope import MultiModalEmbedding

        # Default Beijing DashScope endpoint; do not reuse OpenAI-compat MaaS base URL.
        dashscope.api_key = self.api_key
        resp = MultiModalEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=input_items,
            dimension=self.dimension,
        )
        return _extract_embedding(resp)

    def embed_text(self, text: str) -> list[float]:
        from dashscope.embeddings.multimodal_embedding import MultiModalEmbeddingItemText

        vec = self._call([MultiModalEmbeddingItemText(text=text, factor=1.0)])
        logger.debug("vl_embed_text: dim=%d", len(vec))
        return vec

    def embed_image_file(self, path: str | Path) -> list[float]:
        from dashscope.embeddings.multimodal_embedding import MultiModalEmbeddingItemImage

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        data_uri = _image_data_uri(p)
        vec = self._call([MultiModalEmbeddingItemImage(image=data_uri, factor=1.0)])
        logger.debug("vl_embed_image: path=%s dim=%d", p.name, len(vec))
        return vec
