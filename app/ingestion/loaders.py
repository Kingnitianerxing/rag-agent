import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "pdf"
        logger.info("Loaded PDF: %s (%d pages)", file_path, len(docs))
        return docs
    except Exception as exc:
        logger.error("Failed to load PDF %s: %s", file_path, exc)
        raise


def load_markdown(file_path: str) -> list[Document]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info("Loaded Markdown: %s", file_path)
        return [Document(page_content=content, metadata={"source": file_path, "file_type": "markdown"})]
    except Exception as exc:
        logger.error("Failed to load Markdown %s: %s", file_path, exc)
        raise


def load_image(file_path: str) -> list[Document]:
    """Load an image as a single Document for the VL embedding track.

    Retrieval embeds the raw image via qwen3-vl-embedding; ``page_content`` is a
    vision-model caption so grounded generation has textual evidence to cite.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {file_path}")
    name = path.name
    try:
        from app.ingestion.vl_embedder import caption_image_file

        caption = caption_image_file(path)
        content = f"[Image] {name}\n{caption}"
    except Exception as exc:  # noqa: BLE001 — still index the image even if caption fails
        logger.warning("image_caption_failed: path=%s error=%s", file_path, exc)
        content = f"[Image] {name}"
    logger.info("Loaded image: %s", file_path)
    return [
        Document(
            page_content=content,
            metadata={
                "source": file_path,
                "image_path": file_path,
                "file_type": "image",
                "modality": "image",
                "filename": name,
            },
        )
    ]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def load_webpage(url: str) -> list[Document]:
    loader = WebBaseLoader(
        url,
        requests_kwargs={"timeout": 15, "headers": {"User-Agent": "ProductionRAG/1.0"}},
    )
    docs = loader.load()
    total_size = sum(len(d.page_content) for d in docs)
    if total_size > 50 * 1024 * 1024:
        raise ValueError(f"Web content too large: {total_size} bytes")
    for doc in docs:
        doc.metadata["file_type"] = "web"
    return docs


def load_documents(source: str) -> list[Document]:
    from app.ingestion.validation import IMAGE_SUFFIXES

    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return load_pdf(source)
        elif suffix in (".md", ".markdown"):
            return load_markdown(source)
        elif suffix in IMAGE_SUFFIXES:
            return load_image(source)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    elif source.startswith(("http://", "https://")):
        return load_webpage(source)
    else:
        raise ValueError(f"Unsupported source: {source}")
