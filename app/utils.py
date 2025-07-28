import fitz  # PyMuPDF

def extract_text_lines(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                max_font_size = 0
                is_bold = False

                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    line_text += text + " "

                    # Style info
                    if span["size"] > max_font_size:
                        max_font_size = span["size"]
                    if "bold" in span["font"].lower():
                        is_bold = True

                if line_text.strip():
                    bbox = line["bbox"]
                    extracted_data.append({
                        "text": line_text.strip(),
                        "font_size": max_font_size,
                        "is_bold": is_bold,
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "page": page_num
                    })

    return extracted_data
