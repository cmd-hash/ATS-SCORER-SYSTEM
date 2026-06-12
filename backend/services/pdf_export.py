import logging
import os
import platform

logger = logging.getLogger("ats_resume_scorer")

WEASYPRINT_INSTALLED = False
WEASYPRINT_IMPORT_ERROR = None

try:
    from weasyprint import HTML
    WEASYPRINT_INSTALLED = True
except Exception as e:
    WEASYPRINT_IMPORT_ERROR = str(e)
    logger.error(f"WeasyPrint import failed: {e}", exc_info=True)


def generate_combined_pdf(html_docs: dict[str, str]) -> bytes:
    if not WEASYPRINT_INSTALLED:
        raise RuntimeError(
            f"WeasyPrint is not installed or failed to load. "
            f"Original error: {WEASYPRINT_IMPORT_ERROR}. "
            f"This usually means system libraries (Pango, Cairo, GDK-PixBuf) "
            f"are missing. Platform: {platform.platform()}"
        )

    if not html_docs:
        raise ValueError("No HTML documents were provided for PDF generation.")

    documents = []
    for name, html_str in html_docs.items():
        try:
            doc = HTML(string=html_str).render()
            documents.append(doc)
        except Exception as e:
            logger.exception(f"Failed to render HTML document: {name}")
            raise RuntimeError(f"Failed to render PDF section '{name}': {str(e)}")

    if not documents:
        raise ValueError("No PDF documents were successfully rendered.")

    first_doc = documents[0]
    for other_doc in documents[1:]:
        first_doc.pages.extend(other_doc.pages)

    try:
        return first_doc.write_pdf()
    except Exception as e:
        logger.exception("Failed to write PDF")
        raise RuntimeError(f"Failed to generate PDF: {str(e)}")
