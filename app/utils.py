import os
import re
import joblib
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

def load_models(model_dir):
    return {
        'minilm': SentenceTransformer(str(model_dir / 'minilm')),
        'pca': joblib.load(os.path.join(model_dir, 'pca.pkl')),
        'classifier_head': joblib.load(os.path.join(model_dir, 'classifier_head.pkl')),
        'classifier_level': joblib.load(os.path.join(model_dir, 'classifier_level.pkl'))
    }

def compute_body_font_size(pdf):
    font_counter = Counter()
    for page in pdf.pages:
        try:
            for char in page.chars:
                font_counter[char["size"]] += 1
        except Exception:
            continue
    return font_counter.most_common(1)[0][0] if font_counter else 11

def group_text_blocks(page, tolerance=3):
    if not hasattr(page, 'chars') or not page.chars:
        return []

    lines = {}
    for char in page.chars:
        key = round(char['top'] / tolerance) * tolerance
        lines.setdefault(key, []).append(char)

    blocks = []
    for y, chars in lines.items():
        chars.sort(key=lambda c: c['x0'])
        text = ''.join(c['text'] for c in chars)
        font_size = max(c['size'] for c in chars)
        x0 = min(c['x0'] for c in chars)
        top = min(c['top'] for c in chars)
        x1 = max(c['x1'] for c in chars)
        bottom = max(c['bottom'] for c in chars)

        blocks.append({
            'text': text,
            'size': font_size,
            'x0': x0,
            'top': top,
            'x1': x1,
            'bottom': bottom
        })

    return blocks

def merge_blocks_if_continuous(blocks, page_height, istitle, gap_threshold=10):
    merged = []
    i = 0
    if istitle:
        gap_threshold = page_height
    while i < len(blocks):
        current = blocks[i]
        while i + 1 < len(blocks):
            next_block = blocks[i + 1]
            is_same_size = abs(current['size'] - next_block['size']) < 0.1
            gap = next_block['top'] - current['bottom']
            has_no_interference = gap < gap_threshold

            if is_same_size and has_no_interference:
                current['text'] += ' ' + next_block['text']
                current['bottom'] = next_block['bottom']
                current['x1'] = max(current['x1'], next_block['x1'])
                i += 1
            else:
                break
        merged.append(current)
        i += 1
    return merged

def extract_title_blocks(first_page):
    blocks = group_text_blocks(first_page)
    if not blocks:
        return []

    top_blocks = [b for b in blocks if b['top'] < first_page.height * 0.2]
    if not top_blocks:
        top_blocks = blocks[:5]

    top_blocks.sort(key=lambda b: (-b['size'], b['top']))
    merged = merge_blocks_if_continuous(top_blocks, first_page.height, True)
    # print(merged)
    return merged[0]

def extract_features(block, prev_block, next_block, body_size, page_width, page_height, models):
    position_x = (block['x0'] + block['x1']) / 2 / page_width if page_width else 0.5
    position_y = block['top'] / page_height if page_height else 0.5

    features = [
        block['size'] / body_size if body_size else 1.0,
        position_x,
        position_y,
        1 if 0.4 < position_x < 0.6 else 0,
        len(block['text']),
        sum(1 for c in block['text'] if c.isupper()) / max(1, len(block['text'])),
        sum(1 for c in block['text'] if c.isdigit()) / max(1, len(block['text'])),
        1 if any(c in '.,;:!?' for c in block['text']) else 0,
        (block['top'] - prev_block['bottom']) if prev_block else 1000,
        (next_block['size'] / body_size) if next_block and body_size else 0
    ]

    embedding = models['minilm'].encode([block['text']])[0]
    full_features = features + embedding.tolist()
    return models['pca'].transform([full_features])[0]

def is_valid_heading(block, page_num, body_size):
    text = block['text'].strip()

    if not text or len(text) > 150:
        return False

    footer_patterns = [
        r"Page \d+ of \d+",
        r"Page \d+",
        r"Version \d{4}",
        r"© .+",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}/\d{1,2}/\d{4}",
        r"Copyright",
        r"Confidential"
    ]
    if any(re.search(p, text) for p in footer_patterns):
        return False

    if block['size'] / body_size < 1.0:
        return False

    if re.match(r"^\d+\.", text):
        return True

    return True

def determine_heading_level(block, body_size):
    text = block['text'].strip()
    size_ratio = block['size'] / body_size

    if re.match(r"^\d+\.\s+", text):
        return "H1"
    if re.match(r"^\d+\.\d+\s+", text):
        return "H2"
    if re.match(r"^\d+\.\d+\.\d+\s+", text):
        return "H3"

    if size_ratio > 1.8:
        return "H1"
    if size_ratio > 1.5:
        return "H2"
    return "H3"

def is_header_footer(block, page_height):
    return block['top'] < page_height * 0.05 or block['bottom'] > page_height * 0.9
