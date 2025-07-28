# generate_data.py

import os
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pdfplumber
from faker import Faker
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, PageBreak, FrameBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from bidi.algorithm import get_display

# ─── Font Registration ────────────────────────────────────────────────────────
FONT_DIR = Path(__file__).parent / "fonts"
FONT_DIR.mkdir(exist_ok=True)

# Register Noto Sans fonts for each language
pdfmetrics.registerFont(TTFont("NotoSans", str(FONT_DIR / "NotoSans-Regular.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-Bold", str(FONT_DIR / "NotoSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-JP", str(FONT_DIR / "NotoSansJP-Regular.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-JP-Bold", str(FONT_DIR / "NotoSansJP-Bold.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-Arabic", str(FONT_DIR / "NotoSansArabic-Regular.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-Arabic-Bold", str(FONT_DIR / "NotoSansArabic-Bold.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-Devanagari", str(FONT_DIR / "NotoSansDevanagari-Regular.ttf")))
pdfmetrics.registerFont(TTFont("NotoSans-Devanagari-Bold", str(FONT_DIR / "NotoSansDevanagari-Bold.ttf")))

# Language to font mapping
LANG_FONTS = {
    'en': ('NotoSans', 'NotoSans-Bold'),
    'ja': ('NotoSans-JP', 'NotoSans-JP-Bold'),
    'ar': ('NotoSans-Arabic', 'NotoSans-Arabic-Bold'),
    'hi': ('NotoSans-Devanagari', 'NotoSans-Devanagari-Bold')
}

# ─── Output Directory ────────────────────────────────────────────────────────
PDF_DIR = Path("pdf_data")
PDF_DIR.mkdir(exist_ok=True)

# ─── Multilingual Fakers ─────────────────────────────────────────────────────
fakers = {
    'en': Faker('en_US'),
    'ja': Faker('ja_JP'),
    'hi': Faker('hi_IN'),
    'ar': Faker('ar_EG'),
}

import arabic_reshaper
from bidi.algorithm import get_display

def reshape_bidi(text, lang):
    if lang == 'ar':
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    return text


# ─── Paragraph Factory ───────────────────────────────────────────────────────
def make_paragraph(text, size, lang='en', bold=False, italic=False, align=0):
    text = reshape_bidi(text, lang)
    font_family = LANG_FONTS[lang][1] if bold else LANG_FONTS[lang][0]
    if lang == 'ar':
        align = 2  # right-align for Arabic

    style = ParagraphStyle(
        name=f"P_{lang}",
        fontName=font_family,
        fontSize=size,
        alignment=align,
        textColor=colors.black,
        leading=size * 1.2,
        wordWrap='RTL' if lang == 'ar' else ('CJK' if lang == 'ja' else None),
    )
    return Paragraph(text, style)

# ─── PDF Builder ─────────────────────────────────────────────────────────────
def generate_pdf(path, elements, page_size, layout, lang):
    w, h = page_size
    
    # RTL handling for Arabic
    if lang == 'ar':
        if layout == 'multi':
            frames = [
                Frame(w - 50 - (w - 100)/2, 50, (w - 100)/2, h - 100, id='col1',
                leftPadding=10, rightPadding=10, topPadding=10, bottomPadding=10),
                Frame(w - 60 - (w - 100), 50, (w - 100)/2, h - 100, id='col2',
                leftPadding=10, rightPadding=10, topPadding=10, bottomPadding=10),
            ]
        else:
            frames = [Frame(w - 50 - (w - 100), 50, w - 100, h - 100, id='one',
                      leftPadding=20, rightPadding=20, topPadding=20, bottomPadding=20)]
    else:
        if layout == 'multi':
            frames = [
                Frame(50, 50, (w - 100)/2, h - 100, id='col1',
                     leftPadding=10, rightPadding=10, topPadding=10, bottomPadding=10),
                Frame(60 + (w - 100)/2, 50, (w - 100)/2, h - 100, id='col2',
                     leftPadding=10, rightPadding=10, topPadding=10, bottomPadding=10),
            ]
        else:
            frames = [Frame(50, 50, w - 100, h - 100, id='one',
                      leftPadding=20, rightPadding=20, topPadding=20, bottomPadding=20)]

    doc = BaseDocTemplate(str(path), pagesize=page_size)
    doc.addPageTemplates([PageTemplate('template', frames)])
    doc.build(elements)

# ─── Process PDF for Training ────────────────────────────────────────────────
def process_pdf_for_training(pdf_path, true_headings):
    """Extract training samples from PDF given ground-truth headings."""
    spans = []
    with pdfplumber.open(pdf_path) as pdf:
        for pnum, page in enumerate(pdf.pages, 1):
            for ch in page.chars:
                spans.append({
                    'text': ch['text'],
                    'size': round(ch['size'], 1),
                    'x0': ch['x0'],
                    'x1': ch['x1'],
                    'top': ch['top'],
                    'bottom': ch['bottom'],
                    'page': pnum
                })

    # Body font size (fallback to 11 if empty)
    body_size = Counter(s['size'] for s in spans).most_common(1)[0][0] if spans else 11

    # Group into lines
    lines = defaultdict(list)
    for s in spans:
        key = round(s['top'], 1)
        lines[key].append(s)

    # Create text blocks
    blocks = []
    for y, chars in lines.items():
        chars.sort(key=lambda c: c['x0'])
        text = ''.join(c['text'] for c in chars)
        sizes = [c['size'] for c in chars]
        blocks.append({
            'text': text,
            'size': max(sizes),
            'x0': min(c['x0'] for c in chars),
            'x1': max(c['x1'] for c in chars),
            'top': min(c['top'] for c in chars),
            'bottom': max(c['bottom'] for c in chars),
            'page': chars[0]['page']
        })

    # Sort and merge blocks
    blocks.sort(key=lambda b: (b['page'], b['top'], b['x0']))
    merged = []
    i = 0
    while i < len(blocks):
        cur = blocks[i].copy()
        j = i + 1
        while j < len(blocks) and \
              abs(blocks[j]['size'] - cur['size']) < 0.1 and \
              blocks[j]['top'] - cur['bottom'] < 5 * cur['size']:
            cur['text'] += ' ' + blocks[j]['text']
            cur['bottom'] = blocks[j]['bottom']
            cur['x1'] = max(cur['x1'], blocks[j]['x1'])
            j += 1
        merged.append(cur)
        i = j

    samples, y_h, y_l = [], [], []
    
    # Get page dimensions
    with pdfplumber.open(pdf_path) as pdf:
        dims = [(p.width, p.height) for p in pdf.pages]

    for idx, blk in enumerate(merged):
        # Label: 0=not heading, 1=heading
        lbl = 0
        lvl = ''
        for th in true_headings:
            if th['page'] == blk['page'] and th['text'].strip() == blk['text'].strip():
                lbl = 1
                lvl = th['level']
                break
        
        # Skip long non-heading blocks
        if not lbl and len(blk['text']) > 150:
            continue

        # Calculate features
        w, h = dims[blk['page'] - 1]
        px = (blk['x0'] + blk['x1']) / 2 / w
        py = blk['top'] / h
        centered = int(0.4 < px < 0.6)
        length = len(blk['text'])
        case_ratio = sum(c.isupper() for c in blk['text']) / max(1, length)
        digit_ratio = sum(c.isdigit() for c in blk['text']) / max(1, length)
        punct = int(bool(re.search(r'[.,;:!?\-]', blk['text'])))
        prev_blk = merged[idx - 1] if idx > 0 else None
        next_blk = merged[idx + 1] if idx < len(merged) - 1 else None
        prev_dist = blk['top'] - (prev_blk or blk)['bottom'] if prev_blk else 0
        next_ratio = (next_blk['size'] / body_size) if next_blk else 0

        samples.append({
            'text': blk['text'],
            'size_ratio': blk['size'] / body_size,
            'position_x': px,
            'position_y': py,
            'centered': centered,
            'length': length,
            'case_ratio': case_ratio,
            'digit_ratio': digit_ratio,
            'punct_present': punct,
            'prev_line_distance': prev_dist,
            'next_line_size_ratio': next_ratio
        })
        y_h.append(lbl)
        y_l.append(lvl)

    return {'samples': samples, 'is_heading': y_h, 'heading_level': y_l}

# ─── Generate Dataset ────────────────────────────────────────────────────────
def generate_dataset(n_samples=200):
    X, y_head, y_level = [], [], []

    for i in range(n_samples):
        # Choose document language
        loc = random.choice(list(fakers.keys()))
        fake = fakers[loc]
        
        # Document properties
        layout = random.choice(['single', 'multi'])
        page_sz = random.choice([letter, A4])
        pdf_path = PDF_DIR / f"doc_{i}_{loc}_{layout}.pdf"

        elems, truths = [], []

        # Title (H1)
        title = fake.sentence(nb_words=random.randint(2, 6))
        elems.append(make_paragraph(title, random.choice([26, 28, 30]), lang=loc, bold=True, align=1))
        elems.append(Spacer(1, 20))
        truths.append({'page': 1, 'text': title, 'level': 'H1'})

        # Content pages
        num_pages = random.randint(2, 4)
        for p in range(num_pages):
            if layout == 'multi':
                elems.append(FrameBreak())

            # Headings (H2/H3)
            for lvl, sizes in [('H2', [18, 20, 22]), ('H3', [14, 16, 18])]:
                if random.random() < 0.5:
                    htxt = fake.sentence(nb_words=random.randint(2, 6))
                    elems.append(make_paragraph(htxt, random.choice(sizes), lang=loc, bold=True))
                    elems.append(Spacer(1, 12))
                    truths.append({'page': p + 1, 'text': htxt, 'level': lvl})

            # Body paragraphs
            for _ in range(random.randint(2, 5)):
                btxt = fake.paragraph(nb_sentences=3)
                elems.append(make_paragraph(btxt, 11, lang=loc))
                elems.append(Spacer(1, 8))

            if p < num_pages - 1:
                elems.append(PageBreak())

        # Build PDF
        generate_pdf(pdf_path, elems, page_sz, layout, loc)

        # Extract features
        feats = process_pdf_for_training(str(pdf_path), truths)
        X.extend(feats['samples'])
        y_head.extend(feats['is_heading'])
        y_level.extend(feats['heading_level'])
        os.remove(pdf_path)

    return X, np.array(y_head), np.array(y_level)

if __name__ == '__main__':
    # Ensure fonts directory exists
    if not FONT_DIR.exists():
        print(f"Error: Font directory not found at {FONT_DIR}")
        print("Please download the following Noto Sans fonts and place them in the fonts directory:")
        print("- NotoSans-Regular.ttf, NotoSans-Bold.ttf (English)")
        print("- NotoSansJP-Regular.ttf, NotoSansJP-Bold.ttf (Japanese)")
        print("- NotoSansArabic-Regular.ttf, NotoSansArabic-Bold.ttf (Arabic)")
        print("- NotoSansDevanagari-Regular.ttf, NotoSansDevanagari-Bold.ttf (Hindi)")
        exit(1)
        
    X, yh, yl = generate_dataset(10)
    print(f"Generated {len(X)} samples, headings={sum(yh)}")