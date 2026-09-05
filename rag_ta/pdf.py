from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def markdown_to_pdf(title: str, content: str) -> BytesIO:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph(f"<b>{escape(title)}</b>", styles["Title"]), Spacer(1, 12)]
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            elements.append(Spacer(1, 6))
            continue
        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            elements.append(Paragraph(escape(stripped.lstrip("# ")), styles[f"Heading{level}"]))
        else:
            elements.append(Paragraph(escape(stripped.replace("**", "")), styles["Normal"]))
    doc.build(elements)
    buf.seek(0)
    return buf
