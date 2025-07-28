import os
import json
from utils import extract_text_lines
from classify import predict_headings

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def infer_title(lines):
    """
    Simple rule: pick the largest font-size line from the first page.
    """
    first_page_lines = [l for l in lines if l["page"] == 1]
    if not first_page_lines:
        return "Untitled"

    first_page_lines.sort(key=lambda x: x["font_size"], reverse=True)
    return first_page_lines[0]["text"]

def process_pdf(pdf_path, output_path):
    lines = extract_text_lines(pdf_path)
    if not lines:
        return

    title = infer_title(lines)
    outline = predict_headings(lines)

    result = {
        "title": title,
        "outline": outline
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Processed: {os.path.basename(pdf_path)} → {os.path.basename(output_path)}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_filename = filename.replace(".pdf", ".json")
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            process_pdf(input_path, output_path)

if __name__ == "__main__":
    main()
