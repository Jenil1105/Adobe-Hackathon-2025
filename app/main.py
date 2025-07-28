import os
import json
import re
import pdfplumber
from pathlib import Path
from utils import (
    load_models,
    compute_body_font_size,
    extract_title_blocks,
    group_text_blocks,
    extract_features,
    is_valid_heading,
    determine_heading_level,
    is_header_footer,
    merge_blocks_if_continuous
)
import shutil

def process_pdf(pdf_path, models):
    title = ""
    outline = []

    with pdfplumber.open(pdf_path) as pdf:
        body_size = compute_body_font_size(pdf)

        title_blocks = []
        if pdf.pages:
            title_blocks = extract_title_blocks(pdf.pages[0])
            title = title_blocks['text']

        title_texts = title.strip()

        for page_num, page in enumerate(pdf.pages, 1):
            blocks = group_text_blocks(page)
            if not blocks:
                continue

            # Sort by reading order
            blocks.sort(key=lambda b: (b['top'], b['x0']))
            blocks = merge_blocks_if_continuous(blocks, page.height, False)
            # for block in blocks:
            #     print(block['text'])

            for i, block in enumerate(blocks):
                if is_header_footer(block, page.height):
                    continue

                if not is_valid_heading(block, page_num, body_size):
                    continue

                if page_num == 1 and block['text'].strip() in title_texts:
                    continue
                
                prev_block = blocks[i - 1] if i > 0 else None
                next_block = blocks[i + 1] if i < len(blocks) - 1 else None

                try:
                    features = extract_features(
                        block,
                        prev_block,
                        next_block,
                        body_size,
                        page.width,
                        page.height,
                        models
                    )

                    if models['classifier_head'].predict([features])[0]:
                        level = determine_heading_level(block, body_size)
                        outline.append({
                            "level": level,
                            "text": block['text'].strip(),
                            "page": page_num
                        })
                    else:
                        if re.match(r"^\d+\.", block['text']):
                            level = determine_heading_level(block, body_size)
                            outline.append({
                                "level": level,
                                "text": block['text'].strip(),
                                "page": page_num
                            })

                except Exception:
                    if re.match(r"^\d+\.", block['text']):
                        level = determine_heading_level(block, body_size)
                        outline.append({
                            "level": level,
                            "text": block['text'].strip(),
                            "page": page_num
                        })

    return {"title": title.strip(), "outline": outline}


if __name__ == "__main__":
    model_dir = Path(os.getenv('MODEL_DIR', 'app/models'))
    models = load_models(model_dir)

    input_dir = 'app/input'
    output_dir = 'app/output'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            result = process_pdf(pdf_path, models)

            output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
