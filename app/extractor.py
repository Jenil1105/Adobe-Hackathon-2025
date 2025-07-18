import fitz  # PyMuPDF
import os
import re
import json

def is_heading(text):
    # Heuristics to discard junk
    if len(text.strip()) < 3:
        return False
    if re.fullmatch(r"[-\s.,:]+", text.strip()):
        return False
    return True

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def score_block(block):
    """Score text block for heading likelihood"""
    font_size = block.get("size", 0)
    text = block.get("text", "")
    flags = block.get("flags", 0)
    is_bold = bool(flags & 2)
    is_caps = text.strip().isupper()
    length = len(text.strip())

    score = 0
    if font_size > 16:
        score += 3
    elif font_size > 14:
        score += 2
    elif font_size > 12:
        score += 1

    if is_bold:
        score += 2
    if is_caps:
        score += 1
    if length < 50:
        score += 1
    if length > 100:
        score -= 2

    return score

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    outline = []
    title_candidate = None
    max_score = -1

    for page_number, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            for line in block.get("lines", []):
                line_text = " ".join(span["text"] for span in line["spans"]).strip()
                if not is_heading(line_text):
                    continue

                # Get max font size in the line
                max_span = max(line["spans"], key=lambda span: span["size"])
                font_size = max_span["size"]
                flags = max_span["flags"]

                block_data = {
                    "text": clean_text(line_text),
                    "size": font_size,
                    "flags": flags
                }

                score = score_block(block_data)

                # First, collect title candidate from page 1
                if page_number == 1 and score > max_score:
                    max_score = score
                    title_candidate = block_data["text"]

                # Decide heading level
                if score >= 6:
                    level = "H1"
                elif score >= 4:
                    level = "H2"
                elif score >= 2:
                    level = "H3"
                else:
                    continue  # Not a heading

                outline.append({
                    "level": level,
                    "text": block_data["text"],
                    "page": page_number
                })

    return {
        "title": title_candidate or "Untitled",
        "outline": outline
    }
