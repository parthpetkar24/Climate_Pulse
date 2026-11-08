"""
================================================================================
  knowledge_base.py  —  WHO PDF Evidence Retrieval
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Lightweight retrieval pipeline for the single WHO Air Quality Guidelines PDF.
Uses TF-IDF + cosine similarity for keyword-based search — no vector DB,
embedding model, or complex infrastructure.

Pipeline
--------
1. Extract text from the WHO PDF using PyMuPDF (pymupdf).
2. Split into chunks at natural section boundaries, preserving page metadata.
3. Build a TF-IDF index over the chunks.
4. Retrieve the top-k most relevant chunks given a query.

Each returned chunk carries:
    - text content
    - source document name
    - page number(s)
================================================================================
"""

import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("aqi_explainability.knowledge_base")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # aqi_explainability/
RAW_DIR  = BASE_DIR / "data" / "raw"

# The WHO PDF
WHO_PDF_FILENAME = "9789240034433-eng.pdf"
WHO_PDF_PATH     = RAW_DIR / WHO_PDF_FILENAME


# ── Chunk data structure ─────────────────────────────────────────────────────

class DocumentChunk:
    """A text chunk with source metadata."""

    def __init__(self, text: str, source: str, pages: list[int]):
        self.text   = text.strip()
        self.source = source
        self.pages  = pages            # 1-indexed page numbers

    @property
    def citation(self) -> str:
        if len(self.pages) == 1:
            return f"{self.source}, p. {self.pages[0]}"
        return f"{self.source}, pp. {self.pages[0]}–{self.pages[-1]}"

    def __repr__(self):
        return f"<Chunk({self.citation}, {len(self.text)} chars)>"


# ── PDF Extraction ───────────────────────────────────────────────────────────

def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extract text page-by-page from a PDF using PyMuPDF.

    Returns a list of (page_number_1indexed, page_text).
    """
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF extraction.  "
                "Install with:  pip install pymupdf"
            )

    doc = pymupdf.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append((i + 1, text))
    doc.close()
    log.info("Extracted %d non-empty pages from %s", len(pages), pdf_path.name)
    return pages


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_pages(
    pages: list[tuple[int, str]],
    source_name: str,
    max_chunk_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[DocumentChunk]:
    """
    Split page text into chunks, preferring paragraph/section boundaries.

    Strategy
    --------
    1. Concatenate pages with page markers.
    2. Split on double-newlines (paragraph boundaries).
    3. Merge short paragraphs up to max_chunk_chars.
    4. If a paragraph exceeds max_chunk_chars, split on sentences.
    """
    chunks = []

    for page_num, page_text in pages:
        # Clean up excessive whitespace but keep paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", page_text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_text = ""
        current_pages = [page_num]

        for para in paragraphs:
            # If adding this paragraph keeps us under the limit, merge
            if len(current_text) + len(para) + 2 <= max_chunk_chars:
                current_text = (current_text + "\n\n" + para).strip()
            else:
                # Save current chunk if non-empty
                if current_text:
                    chunks.append(DocumentChunk(
                        text=current_text,
                        source=source_name,
                        pages=list(set(current_pages)),
                    ))

                # If the paragraph itself is too large, split on sentences
                if len(para) > max_chunk_chars:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_text = ""
                    current_pages = [page_num]
                    for sent in sentences:
                        if len(current_text) + len(sent) + 1 <= max_chunk_chars:
                            current_text = (current_text + " " + sent).strip()
                        else:
                            if current_text:
                                chunks.append(DocumentChunk(
                                    text=current_text,
                                    source=source_name,
                                    pages=[page_num],
                                ))
                            current_text = sent
                            current_pages = [page_num]
                else:
                    current_text = para
                    current_pages = [page_num]

        # Don't forget the last chunk of this page
        if current_text:
            chunks.append(DocumentChunk(
                text=current_text,
                source=source_name,
                pages=list(set(current_pages)),
            ))

    log.info("Created %d chunks from %d pages", len(chunks), len(pages))

    # Deduplicate chunks with identical (or near-identical) text
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        # Normalise whitespace for comparison
        normalised = " ".join(chunk.text.split()).lower()
        if normalised not in seen_texts and len(normalised) > 30:
            seen_texts.add(normalised)
            unique_chunks.append(chunk)
    if len(unique_chunks) < len(chunks):
        log.info("Deduplicated: %d → %d chunks", len(chunks), len(unique_chunks))
    return unique_chunks


# ── TF-IDF Index ─────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Lightweight keyword-based retrieval over WHO PDF chunks using TF-IDF.
    """

    def __init__(self, pdf_path: Optional[Path] = None):
        self.pdf_path = pdf_path or WHO_PDF_PATH
        self.chunks: list[DocumentChunk] = []
        self._tfidf_matrix = None
        self._vectorizer   = None
        self._is_built     = False

    def build(self):
        """Extract, chunk, and index the PDF."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"WHO PDF not found at {self.pdf_path}.  "
                "Ensure the PDF is in data/raw/."
            )

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Step 1: Extract
        pages = _extract_pages(self.pdf_path)

        # Step 2: Chunk
        self.chunks = _chunk_pages(
            pages,
            source_name=self.pdf_path.name,
            max_chunk_chars=1500,
        )

        # Step 3: Build TF-IDF index
        texts = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        self._is_built = True
        log.info("Knowledge base built: %d chunks indexed", len(self.chunks))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int
            Number of chunks to return.

        Returns
        -------
        List of dicts with keys: text, citation, source, pages, score.
        """
        if not self._is_built:
            raise RuntimeError("Call build() first.")

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        top_idx = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 0.01:       # skip near-zero matches
                continue
            chunk = self.chunks[idx]
            results.append({
                "text":     chunk.text,
                "citation": chunk.citation,
                "source":   chunk.source,
                "pages":    chunk.pages,
                "score":    round(float(scores[idx]), 4),
            })

        log.info("Query: '%s' → %d relevant chunks", query[:60], len(results))
        return results

    def retrieve_for_prediction(
        self,
        aqi_value: float,
        aqi_category: str,
        top_pollutants: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Build a context-aware query from prediction details and retrieve.

        Parameters
        ----------
        aqi_value : float
            Predicted AQI.
        aqi_category : str
            E.g. "Moderate", "Poor".
        top_pollutants : list[str]
            Most influential pollutant feature names.
        top_k : int
            Number of chunks to return.
        """
        # Map feature names to human-readable pollutant names
        pollutant_map = {
            "pm25": "PM2.5 particulate matter",
            "pm10": "PM10 particulate matter",
            "no2":  "nitrogen dioxide NO2",
            "o3":   "ozone O3",
            "co":   "carbon monoxide CO",
            "so2":  "sulfur dioxide SO2",
        }

        pollutant_terms = [
            pollutant_map.get(p, p)
            for p in top_pollutants
            if p in pollutant_map
        ]

        query = (
            f"air quality {aqi_category} health effects risks "
            f"{' '.join(pollutant_terms)} "
            f"WHO guidelines recommendations exposure"
        )

        return self.retrieve(query, top_k=top_k)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    kb = KnowledgeBase()
    kb.build()

    # Test retrieval
    print("\n--- Test Query: 'PM2.5 health effects respiratory' ---")
    results = kb.retrieve("PM2.5 health effects respiratory", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score={r['score']:.4f}  |  {r['citation']}")
        print(f"    {r['text'][:200]}...")

    print(f"\n✓ Knowledge base ready with {len(kb.chunks)} chunks")

