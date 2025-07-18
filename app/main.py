import os
from extractor import process_pdf

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def main():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            output_filename = filename.replace(".pdf", ".json")
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            print(f"Processing: {filename}")
            data = process_pdf(pdf_path)
            with open(output_path, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved: {output_filename}")

if __name__ == "__main__":
    main()
