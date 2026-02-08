#!/usr/bin/env python3
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import io
import os
import json
import csv
import numpy as np
import gc
from pdf2image import pdfinfo_from_bytes, convert_from_bytes
import re
from multiprocessing import Pool
from tqdm import tqdm
from xml.sax.saxutils import escape
import shutil
import pdfplumber
from collections import defaultdict

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

# Alignment mapping
ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}

# Global variables for multiprocessing
GLOBAL_CONFIG = None
OUTPUT_DIR = None
recover = False


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Convert colors
    if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
    if 'font-color' in config and isinstance(config['font-color'], str):
        config['font-color'] = colors.HexColor(config['font-color'])
    if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
    if 'para-border-color' in config and isinstance(config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(config['para-border-color'])
    
    # Convert alignment
    if 'alignment' in config and isinstance(config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
    
    # Convert page size
    if 'page-size' in config and isinstance(config['page-size'], str):
        config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    return config


def text_to_images(text, output_dir, config_path=None, config_dict=None, unique_id=None):
    """
    Convert text to images - Inference interface
    
    Args:
        text: Input text content
        output_dir: Image output directory
        config_path: Configuration file path (optional)
        config_dict: Configuration dictionary (optional, higher priority than config_path)
        unique_id: Unique identifier (optional, auto-generated if not provided)
        
    Returns:
        list: List of generated image paths
        
    Example:
        >>> images = text_to_images(
        ...     text="Hello World",
        ...     output_dir="./output",
        ...     config_path="config.json"
        ... )
        >>> print(images)  # ['./output/xxx/page_001.png', ...]
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or config_dict")
        config = load_config(config_path)
    else:
        config = config_dict.copy()
        # Convert special fields in config
        if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
            config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
        if 'font-color' in config and isinstance(config['font-color'], str):
            config['font-color'] = colors.HexColor(config['font-color'])
        if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
            config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
        if 'para-border-color' in config and isinstance(config['para-border-color'], str):
            config['para-border-color'] = colors.HexColor(config['para-border-color'])
        if 'alignment' in config and isinstance(config['alignment'], str):
            config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
        if 'page-size' in config and isinstance(config['page-size'], str):
            config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    # Generate unique ID
    if unique_id is None:
        # Generate a truly unique, non-colliding ID per call to avoid directory reuse across concurrent jobs
        import uuid, time, hashlib
        raw = f"{uuid.uuid4().hex}_{time.time_ns()}"
        unique_id = hashlib.md5(raw.encode()).hexdigest()[:16]
    
    # Extract configuration parameters
    page_size = config.get('page-size', A4)
    margin_x = config.get('margin-x', 20)
    margin_y = config.get('margin-y', 20)
    font_path = config.get('font-path')
    assert font_path, "Must provide font-path"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"font-path not found: {font_path}")

    # Prefer a deterministic custom font name to avoid family mapping issues
    font_name = config.get('font-name') or 'CustomTTF'
    font_size = config.get('font-size', 9)
    line_height = config.get('line-height') or (font_size + 1)
    
    page_bg_color = config.get('page-bg-color', colors.HexColor('#FFFFFF'))
    font_color = config.get('font-color', colors.HexColor('#000000'))
    para_bg_color = config.get('para-bg-color', colors.HexColor('#FFFFFF'))
    para_border_color = config.get('para-border-color', colors.HexColor('#FFFFFF'))
    
    first_line_indent = config.get('first-line-indent', 0)
    left_indent = config.get('left-indent', 0)
    right_indent = config.get('right-indent', 0)
    alignment = config.get('alignment', TA_JUSTIFY)
    space_before = config.get('space-before', 0)
    space_after = config.get('space-after', 0)
    border_width = config.get('border-width', 0)
    border_padding = config.get('border-padding', 0)
    
    horizontal_scale = config.get('horizontal-scale', 1.0)
    dpi = config.get('dpi', 72)
    auto_crop_last_page = config.get('auto-crop-last-page', False)
    auto_crop_width = config.get('auto-crop-width', False)
    # newline_markup = config.get('newline-markup', '<font color="#FF0000"> \\n </font>')
    newline_markup = config.get('newline-markup', '<br/>')
    
    # Register font
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        # Optionally register a family to avoid mapping issues
        pdfmetrics.registerFontFamily(font_name, normal=font_name, bold=font_name, italic=font_name, boldItalic=font_name)
    except Exception as e:
        # Fallback to Helvetica to avoid hard crash; caller should inspect quality
        print(f"[text_to_images] registerFont failed for {font_path}: {e}. Falling back to Helvetica.")
        font_name = 'Helvetica'
    
    # Create PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=margin_y,
    )
    
    # Create paragraph style
    styles = getSampleStyleSheet()
    RE_CJK = re.compile(r'[\u4E00-\u9FFF]')
    
    custom = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=line_height,
        textColor=font_color,
        backColor=para_bg_color,
        borderColor=para_border_color,
        borderWidth=border_width,
        borderPadding=border_padding,
        firstLineIndent=first_line_indent,
        wordWrap="CJK" if RE_CJK.search(text) else None,
        leftIndent=left_indent,
        rightIndent=right_indent,
        alignment=alignment,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )
    
    # Process text
    def replace_spaces(s):
        return re.sub(r' {2,}', lambda m: '&nbsp;'*len(m.group()), s)
    
    text = text.replace('\xad', '').replace('\u200b', '')
    processed_text = replace_spaces(escape(text))
    parts = processed_text.split('\n')
    
    # Create paragraphs in batches
    story = []
    turns = 30
    for i in range(0, len(parts), turns):
        tmp_text = newline_markup.join(parts[i:i+turns])
        story.append(Paragraph(tmp_text, custom))
    
    # Build PDF
    doc.build(
        story,
        onFirstPage=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState()),
        onLaterPages=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState())
    )
    
    pdf_bytes = buf.getvalue()
    buf.close()
    
    # Create output directory
    out_root = os.path.join(output_dir, unique_id)
    os.makedirs(out_root, exist_ok=True)
    
    # Convert PDF to images
    info = pdfinfo_from_bytes(pdf_bytes)
    num_pages = total = info["Pages"]
    batch = 20
    image_paths = []
    
    for start in range(1, total + 1, batch):
        end = min(start + batch - 1, total)
        images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=start, last_page=end)
        
        for offset, img in enumerate(images, start=start):
            w, h = img.size
            
            # Horizontal scaling
            if horizontal_scale != 1.0:
                img = img.resize((int(w * horizontal_scale), h))
            
            # Adaptive cropping
            if auto_crop_width or (auto_crop_last_page and offset == num_pages):
                gray = np.array(img.convert("L"))
                # Estimate page background from corner region
                bg_gray = np.median(gray[:2, :2])
                tolerance = 5
                mask = np.abs(gray - bg_gray) > tolerance
                
                if auto_crop_width:
                    cols = np.where(mask.any(axis=0))[0]
                    if cols.size:
                        rightmost_col = cols[-1] + 1
                        right = min(img.width, rightmost_col + margin_x)
                        img = img.crop((0, 0, right, img.height))
                
                if auto_crop_last_page and offset == num_pages:
                    rows = np.where(mask.any(axis=1))[0]
                    if rows.size:
                        last_row = rows[-1]
                        lower = min(img.height, last_row + margin_y)
                        img = img.crop((0, 0, img.width, lower))
            
            out_path = os.path.join(out_root, f"page_{offset:03d}.png")
            img.save(out_path, 'PNG')
            image_paths.append(os.path.abspath(out_path))
            img.close()
        
        images.clear()
        del images
    
    # Merge all page images into a single tall image if there are multiple pages
    if len(image_paths) > 1:
        try:
            # Ensure deterministic order by filename
            image_paths = sorted(image_paths)
            imgs = [Image.open(p) for p in image_paths]
            total_width = max(im.width for im in imgs)
            total_height = sum(im.height for im in imgs)
            merged = Image.new('RGB', (total_width, total_height), color='white')
            y_offset = 0
            for im in imgs:
                x_offset = (total_width - im.width) // 2
                merged.paste(im, (x_offset, y_offset))
                y_offset += im.height
                im.close()
            merged_path = os.path.join(out_root, "merged.png")
            merged.save(merged_path, 'PNG')
            merged.close()
            # Delete original page images
            for p in image_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            image_paths = [os.path.abspath(merged_path)]
        except Exception as e:
            print(f"[text_to_images] merge pages failed: {e}")
            # Fallback: keep original image_paths list

    del pdf_bytes
    gc.collect()
    
    return image_paths


def generate_word_mapping(pdf_bytes, image_paths, dpi, horizontal_scale, page_size):
    """
    Generate word-to-pixel coordinate mapping from PDF for merged image

    Args:
        pdf_bytes: PDF content as bytes
        image_paths: List of generated image paths (will be merged)
        dpi: DPI used for image conversion
        horizontal_scale: Horizontal scaling factor
        page_size: PDF page size (width, height)

    Returns:
        dict: Word mapping data with coordinates adjusted for merged image
    """
    word_mapping = {
        "images": [],
        "words": []
    }

    try:
        # Open PDF with pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pixels_per_point = dpi / 72.0

            # First pass: collect all page dimensions and word data
            page_data = []
            total_height = 0
            max_width = 0

            for page_num, page in enumerate(pdf.pages):
                page_width = int(page.width * pixels_per_point * horizontal_scale)
                page_height = int(page.height * pixels_per_point)

                max_width = max(max_width, page_width)

                # Extract words with position info
                words = page.extract_words(
                    keep_blank_chars=True,
                    use_text_flow=True,
                    x_tolerance=1,
                    y_tolerance=1
                )

                page_data.append({
                    'page_num': page_num,
                    'width': page_width,
                    'height': page_height,
                    'words': words,
                    'y_offset': total_height  # Vertical offset in merged image
                })

                total_height += page_height

            # Second pass: process words with merged coordinates
            global_line_counter = 0

            for page_info in page_data:
                page_num = page_info['page_num']
                page_width = page_info['width']
                page_height = page_info['height']
                words = page_info['words']
                y_offset = page_info['y_offset']

                # Horizontal centering offset
                x_offset = (max_width - page_width) // 2

                # Group words by line (using top coordinate clustering)
                lines = defaultdict(list)
                for word in words:
                    # Use top coordinate for line grouping, with tolerance
                    line_key = round(word['top'], 1)  # Round to handle floating point precision
                    lines[line_key].append(word)

                # Sort lines by y-coordinate (top to bottom)
                sorted_lines = sorted(lines.items(), key=lambda x: x[0])

                # Process each line
                for line_idx, (line_y, line_words) in enumerate(sorted_lines):
                    # Sort words in line by x-coordinate (left to right)
                    line_words_sorted = sorted(line_words, key=lambda w: w['x0'])

                    for word_idx, word in enumerate(line_words_sorted):
                        # pdfplumber coordinates: origin at top-left
                        # Convert to pixel coordinates for merged image

                        # pdfplumber coordinates
                        pdf_x0 = word['x0']  # left
                        pdf_y0 = word['top']  # top
                        pdf_x1 = word['x1']  # right
                        pdf_y1 = word['bottom']  # bottom

                        # Convert to pixel coordinates
                        pixel_x0 = int(pdf_x0 * pixels_per_point * horizontal_scale) + x_offset
                        pixel_y0 = int(pdf_y0 * pixels_per_point) + y_offset
                        pixel_x1 = int(pdf_x1 * pixels_per_point * horizontal_scale) + x_offset
                        pixel_y1 = int(pdf_y1 * pixels_per_point) + y_offset

                        # Ensure coordinates are integers and within bounds
                        pixel_x0 = max(0, pixel_x0)
                        pixel_y0 = max(0, pixel_y0)
                        pixel_x1 = max(0, pixel_x1)
                        pixel_y1 = max(0, pixel_y1)

                        word_entry = {
                            "word": word['text'],
                            "image": "merged.png",  # Always use merged image name
                            "page": page_num + 1,
                            "line": global_line_counter + 1,  # Global line number across all pages
                            "word_in_line": word_idx + 1,
                            "bbox": [pixel_x0, pixel_y0, pixel_x1, pixel_y1],  # [x0, y0, x1, y1]
                            "confidence": 1.0,  # Assuming perfect extraction
                            "pdf_coords": {
                                "x0": word['x0'],
                                "top": word['top'],
                                "x1": word['x1'],
                                "bottom": word['bottom']
                            }
                        }

                        word_mapping["words"].append(word_entry)

                    global_line_counter += 1

            # Add merged image metadata
            if image_paths:
                merged_path = image_paths[0] if len(image_paths) == 1 else image_paths[0].replace('page_001.png', 'merged.png')
                merged_name = os.path.basename(merged_path)

                word_mapping["images"].append({
                    "name": merged_name,
                    "path": os.path.abspath(merged_path),
                    "width": max_width,
                    "height": total_height,
                    "dpi": dpi,
                    "horizontal_scale": horizontal_scale
                })

    except Exception as e:
        print(f"[generate_word_mapping] Error processing PDF: {e}")
        import traceback
        traceback.print_exc()

    return word_mapping


def text_to_images_evidence(text, output_dir, config_path=None, config_dict=None, unique_id=None, evidence_text=None):
    """
    Convert text to images - Inference interface
    
    Args:
        text: Input text content
        output_dir: Image output directory
        config_path: Configuration file path (optional)
        config_dict: Configuration dictionary (optional, higher priority than config_path)
        unique_id: Unique identifier (optional, auto-generated if not provided)
        evidence_text: Evidence text content
    Returns:
        list: List of generated image paths
        
    Example:
        >>> images = text_to_images(
        ...     text="Hello World",
        ...     output_dir="./output",
        ...     config_path="config.json"
        ... )
        >>> print(images)  # ['./output/xxx/page_001.png', ...]
    """
    # Load configuration
    if config_dict is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or config_dict")
        config = load_config(config_path)
    else:
        config = config_dict.copy()
        # Convert special fields in config
        if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
            config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
        if 'font-color' in config and isinstance(config['font-color'], str):
            config['font-color'] = colors.HexColor(config['font-color'])
        if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
            config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
        if 'para-border-color' in config and isinstance(config['para-border-color'], str):
            config['para-border-color'] = colors.HexColor(config['para-border-color'])
        if 'alignment' in config and isinstance(config['alignment'], str):
            config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
        if 'page-size' in config and isinstance(config['page-size'], str):
            config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    # Generate unique ID
    if unique_id is None:
        # Generate a truly unique, non-colliding ID per call to avoid directory reuse across concurrent jobs
        import uuid, time, hashlib
        raw = f"{uuid.uuid4().hex}_{time.time_ns()}"
        unique_id = hashlib.md5(raw.encode()).hexdigest()[:16]
    
    # Extract configuration parameters
    page_size = config.get('page-size', A4)
    margin_x = config.get('margin-x', 20)
    margin_y = config.get('margin-y', 20)
    font_path = config.get('font-path')
    assert font_path, "Must provide font-path"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"font-path not found: {font_path}")

    # Prefer a deterministic custom font name to avoid family mapping issues
    font_name = config.get('font-name') or 'CustomTTF'
    font_size = config.get('font-size', 9)
    line_height = config.get('line-height') or (font_size + 1)
    
    page_bg_color = config.get('page-bg-color', colors.HexColor('#FFFFFF'))
    font_color = config.get('font-color', colors.HexColor('#000000'))
    para_bg_color = config.get('para-bg-color', colors.HexColor('#FFFFFF'))
    para_border_color = config.get('para-border-color', colors.HexColor('#FFFFFF'))
    
    first_line_indent = config.get('first-line-indent', 0)
    left_indent = config.get('left-indent', 0)
    right_indent = config.get('right-indent', 0)
    alignment = config.get('alignment', TA_JUSTIFY)
    space_before = config.get('space-before', 0)
    space_after = config.get('space-after', 0)
    border_width = config.get('border-width', 0)
    border_padding = config.get('border-padding', 0)
    
    horizontal_scale = config.get('horizontal-scale', 1.0)
    dpi = config.get('dpi', 72)
    auto_crop_last_page = config.get('auto-crop-last-page', False)
    auto_crop_width = config.get('auto-crop-width', False)
    # newline_markup = config.get('newline-markup', '<font color="#FF0000"> \\n </font>')
    newline_markup = config.get('newline-markup', '<br/>')
    
    # Register font
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        # Optionally register a family to avoid mapping issues
        pdfmetrics.registerFontFamily(font_name, normal=font_name, bold=font_name, italic=font_name, boldItalic=font_name)
    except Exception as e:
        # Fallback to Helvetica to avoid hard crash; caller should inspect quality
        print(f"[text_to_images] registerFont failed for {font_path}: {e}. Falling back to Helvetica.")
        font_name = 'Helvetica'
    
    # Create PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=margin_y,
    )
    
    # Create paragraph style
    styles = getSampleStyleSheet()
    RE_CJK = re.compile(r'[\u4E00-\u9FFF]')
    
    custom = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=line_height,
        textColor=font_color,
        backColor=para_bg_color,
        borderColor=para_border_color,
        borderWidth=border_width,
        borderPadding=border_padding,
        firstLineIndent=first_line_indent,
        wordWrap="CJK" if RE_CJK.search(text) else None,
        leftIndent=left_indent,
        rightIndent=right_indent,
        alignment=alignment,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )
    
    # Process text
    def replace_spaces(s):
        return re.sub(r' {2,}', lambda m: '&nbsp;'*len(m.group()), s)
    
    # Clean text
    clean_text = text.replace('\xad', '').replace('\u200b', '')
    
    # Mark evidence text if provided
    def mark_evidence_in_text(original_text, evidence_list):
        """Mark evidence sentences in text with red color"""
        if not evidence_list:
            return original_text
        
        marked_text = original_text
        # Sort by length (longest first) to avoid partial matches
        sorted_evidence = sorted(evidence_list, key=len, reverse=True)
        
        # Find and mark each evidence sentence
        for evidence in sorted_evidence:
            if not evidence or not evidence.strip():
                continue
            # Escape special regex characters in evidence text
            escaped_evidence = re.escape(evidence)
            # Find all matches (including spaces)
            pattern = escaped_evidence
            # Replace with red marked version
            # Replace newlines in evidence with <br/> to prevent tag splitting
            marked_text = re.sub(
                pattern,
                lambda m: f'<font color="#FF0000">{escape(m.group().replace(chr(10), "<br/>").replace(chr(13), ""))}</font>',
                marked_text,
                flags=re.DOTALL
            )
        
        return marked_text
    
    # Create normal text version
    processed_text = replace_spaces(escape(clean_text))
    parts = processed_text.split('\n')
    
    # Create paragraphs in batches for normal version
    story = []
    turns = 30
    for i in range(0, len(parts), turns):
        tmp_text = newline_markup.join(parts[i:i+turns])
        story.append(Paragraph(tmp_text, custom))
    
    # Build PDF for normal version
    doc.build(
        story,
        onFirstPage=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState()),
        onLaterPages=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState())
    )
    
    pdf_bytes = buf.getvalue()
    buf.close()
    
    # Create output directory
    out_root = os.path.join(output_dir, unique_id)
    os.makedirs(out_root, exist_ok=True)
    
    # Convert PDF to images (normal version)
    info = pdfinfo_from_bytes(pdf_bytes)
    num_pages = total = info["Pages"]
    batch = 20
    image_paths = []
    
    for start in range(1, total + 1, batch):
        end = min(start + batch - 1, total)
        images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=start, last_page=end)
        
        for offset, img in enumerate(images, start=start):
            w, h = img.size
            
            # Horizontal scaling
            if horizontal_scale != 1.0:
                img = img.resize((int(w * horizontal_scale), h))
            
            # Adaptive cropping
            if auto_crop_width or (auto_crop_last_page and offset == num_pages):
                gray = np.array(img.convert("L"))
                # Estimate page background from corner region
                bg_gray = np.median(gray[:2, :2])
                tolerance = 5
                mask = np.abs(gray - bg_gray) > tolerance
                
                if auto_crop_width:
                    cols = np.where(mask.any(axis=0))[0]
                    if cols.size:
                        rightmost_col = cols[-1] + 1
                        right = min(img.width, rightmost_col + margin_x)
                        img = img.crop((0, 0, right, img.height))
                
                if auto_crop_last_page and offset == num_pages:
                    rows = np.where(mask.any(axis=1))[0]
                    if rows.size:
                        last_row = rows[-1]
                        lower = min(img.height, last_row + margin_y)
                        img = img.crop((0, 0, img.width, lower))
            
            out_path = os.path.join(out_root, f"page_{offset:03d}.png")
            img.save(out_path, 'PNG')
            image_paths.append(os.path.abspath(out_path))
            img.close()
        
        images.clear()
        del images
    
    # Merge all page images into a single tall image if there are multiple pages
    merged_normal_path = None
    if len(image_paths) > 1:
        try:
            # Ensure deterministic order by filename
            image_paths = sorted(image_paths)
            imgs = [Image.open(p) for p in image_paths]
            total_width = max(im.width for im in imgs)
            total_height = sum(im.height for im in imgs)
            merged = Image.new('RGB', (total_width, total_height), color='white')
            y_offset = 0
            for im in imgs:
                x_offset = (total_width - im.width) // 2
                merged.paste(im, (x_offset, y_offset))
                y_offset += im.height
                im.close()
            merged_path = os.path.join(out_root, "merged.png")
            # Fix PIL PNG save issue by ensuring proper mode and using explicit format
            try:
                merged.save(merged_path, 'PNG', optimize=False)
            except (AttributeError, OSError):
                # Fallback: save as RGB mode explicitly
                if merged.mode != 'RGB':
                    merged = merged.convert('RGB')
                merged.save(merged_path, 'PNG', optimize=False)
            merged.close()
            merged_normal_path = merged_path
            # Delete original page images
            for p in image_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            image_paths = [os.path.abspath(merged_path)]
        except Exception as e:
            print(f"[text_to_images] merge pages failed: {e}")
            # Fallback: keep original image_paths list
    
    # Generate evidence-marked version if evidence_text is provided
    if evidence_text and isinstance(evidence_text, list) and len(evidence_text) > 0:
        try:
            # Mark evidence in text
            marked_text = mark_evidence_in_text(clean_text, evidence_text)
            
            # Process marked text
            # Split by newlines but preserve the marked sections
            # Note: Evidence text newlines are already replaced with <br/> in mark_evidence_in_text
            marked_parts = []
            for line in marked_text.split('\n'):
                # Check if line contains marked sections
                if '<font color="#FF0000">' in line:
                    # Already marked, just process spaces
                    marked_parts.append(replace_spaces(line))
                else:
                    # Not marked, escape and process spaces
                    marked_parts.append(replace_spaces(escape(line)))
            
            # Create PDF for evidence-marked version
            buf_evidence = io.BytesIO()
            doc_evidence = SimpleDocTemplate(
                buf_evidence,
                pagesize=page_size,
                leftMargin=margin_x,
                rightMargin=margin_x,
                topMargin=margin_y,
                bottomMargin=margin_y,
            )
            
            # Create paragraphs in batches for evidence version
            story_evidence = []
            for i in range(0, len(marked_parts), turns):
                tmp_text = newline_markup.join(marked_parts[i:i+turns])
                story_evidence.append(Paragraph(tmp_text, custom))
            
            # Build PDF for evidence version
            doc_evidence.build(
                story_evidence,
                onFirstPage=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState()),
                onLaterPages=lambda c, d: (c.saveState(), c.setFillColor(page_bg_color), c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1), c.restoreState())
            )
            
            pdf_bytes_evidence = buf_evidence.getvalue()
            buf_evidence.close()
            
            # Convert PDF to images (evidence version)
            info_evidence = pdfinfo_from_bytes(pdf_bytes_evidence)
            num_pages_evidence = info_evidence["Pages"]
            total_evidence = num_pages_evidence
            image_paths_evidence = []
            
            for start in range(1, total_evidence + 1, batch):
                end = min(start + batch - 1, total_evidence)
                images_evidence = convert_from_bytes(pdf_bytes_evidence, dpi=dpi, first_page=start, last_page=end)
                
                for offset, img in enumerate(images_evidence, start=start):
                    w, h = img.size
                    
                    # Horizontal scaling
                    if horizontal_scale != 1.0:
                        img = img.resize((int(w * horizontal_scale), h))
                    
                    # Adaptive cropping
                    if auto_crop_width or (auto_crop_last_page and offset == num_pages_evidence):
                        gray = np.array(img.convert("L"))
                        bg_gray = np.median(gray[:2, :2])
                        tolerance = 5
                        mask = np.abs(gray - bg_gray) > tolerance
                        
                        if auto_crop_width:
                            cols = np.where(mask.any(axis=0))[0]
                            if cols.size:
                                rightmost_col = cols[-1] + 1
                                right = min(img.width, rightmost_col + margin_x)
                                img = img.crop((0, 0, right, img.height))
                        
                        if auto_crop_last_page and offset == num_pages_evidence:
                            rows = np.where(mask.any(axis=1))[0]
                            if rows.size:
                                last_row = rows[-1]
                                lower = min(img.height, last_row + margin_y)
                                img = img.crop((0, 0, img.width, lower))
                    
                    # Save with "evidence" suffix
                    out_path_evidence = os.path.join(out_root, f"page_{offset:03d}_evidence.png")
                    img.save(out_path_evidence, 'PNG')
                    image_paths_evidence.append(os.path.abspath(out_path_evidence))
                    img.close()
                
                images_evidence.clear()
                del images_evidence
            
            # Merge evidence images if there are multiple pages
            if len(image_paths_evidence) > 1:
                try:
                    image_paths_evidence = sorted(image_paths_evidence)
                    imgs_evidence = [Image.open(p) for p in image_paths_evidence]
                    total_width_evidence = max(im.width for im in imgs_evidence)
                    total_height_evidence = sum(im.height for im in imgs_evidence)
                    merged_evidence = Image.new('RGB', (total_width_evidence, total_height_evidence), color='white')
                    y_offset = 0
                    for im in imgs_evidence:
                        x_offset = (total_width_evidence - im.width) // 2
                        merged_evidence.paste(im, (x_offset, y_offset))
                        y_offset += im.height
                        im.close()
                    merged_evidence_path = os.path.join(out_root, "merged_evidence.png")
                    # Fix PIL PNG save issue by ensuring proper mode and using explicit format
                    try:
                        merged_evidence.save(merged_evidence_path, 'PNG', optimize=False)
                    except (AttributeError, OSError):
                        # Fallback: save as RGB mode explicitly
                        if merged_evidence.mode != 'RGB':
                            merged_evidence = merged_evidence.convert('RGB')
                        merged_evidence.save(merged_evidence_path, 'PNG', optimize=False)
                    merged_evidence.close()
                    # Delete original page images
                    for p in image_paths_evidence:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[text_to_images_evidence] merge evidence pages failed: {e}")
            
            del pdf_bytes_evidence
        except Exception as e:
            print(f"[text_to_images_evidence] generate evidence-marked images failed: {e}")

    # Generate word-to-pixel mapping for normal version
    try:
        word_mapping_normal = generate_word_mapping(pdf_bytes, image_paths, dpi, horizontal_scale, page_size)
        mapping_file_normal = os.path.join(out_root, "word_mapping.json")
        with open(mapping_file_normal, 'w', encoding='utf-8') as f:
            json.dump(word_mapping_normal, f, ensure_ascii=False, indent=2)
        print(f"[text_to_images_evidence] word mapping saved to {mapping_file_normal}")
    except Exception as e:
        print(f"[text_to_images_evidence] generate word mapping failed: {e}")

    del pdf_bytes
    gc.collect()
    
    return image_paths


def process_one(item):
    """Process single item - for batch processing"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    _id = item.get('unique_id')
    assert _id
    
    # Check recovery mode
    if recover and os.path.exists(os.path.join(OUTPUT_DIR, _id)):
        item['image_paths'] = []
        return item
    
    # Parse configuration
    item_config = item.get('config', {}) or {}
    config = {**GLOBAL_CONFIG, **item_config}
    
    # Process special fields in item config
    if 'page-size' in item_config and isinstance(item_config['page-size'], str):
        config['page-size'] = tuple(map(float, item_config['page-size'].split(',')))
    if 'page-bg-color' in item_config and isinstance(item_config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(item_config['page-bg-color'])
    if 'font-color' in item_config and isinstance(item_config['font-color'], str):
        config['font-color'] = colors.HexColor(item_config['font-color'])
    if 'para-bg-color' in item_config and isinstance(item_config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(item_config['para-bg-color'])
    if 'para-border-color' in item_config and isinstance(item_config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(item_config['para-border-color'])
    if 'alignment' in item_config and isinstance(item_config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(item_config['alignment'], TA_JUSTIFY)
    
    # Get text
    text = item.get('context', '')
    assert text
    
    # Call inference function
    image_paths = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_dict=config,
        unique_id=_id
    )
    
    item['image_paths'] = image_paths
    return item


def batch_process_to_images(json_path, output_dir, output_jsonl_path, 
                            config_path, processes=16, is_recover=False, batch_size=100):
    """Batch process JSON data to generate images"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    # Set global variables
    GLOBAL_CONFIG = load_config(config_path)
    OUTPUT_DIR = output_dir
    recover = is_recover
    
    print(f"Loaded config from: {config_path}")
    
    # Prepare output directory
    if not recover:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_jsonl_path):
            os.remove(output_jsonl_path)
    
    # Read data
    with open(json_path, 'r', encoding='utf-8') as f:
        data_to_process = json.load(f)
    
    # Get already processed IDs
    processed_ids = set()
    if recover and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_ids.add(item.get('unique_id'))
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed items")
    
    # Filter processed items
    data_to_process = [item for item in data_to_process 
                      if item.get('unique_id') not in processed_ids]
    print(f"Remaining items to process: {len(data_to_process)}")
    
    if not data_to_process:
        print("All items processed")
        return
    
    # Parallel processing
    batch_buffer = []
    
    with Pool(processes=processes) as pool:
        for result_item in tqdm(pool.imap_unordered(process_one, data_to_process, chunksize=1), 
                               total=len(data_to_process)):
            if result_item:
                batch_buffer.append(result_item)
                _id = result_item.get('unique_id', 'UNKNOWN')
                count = len(result_item.get('image_paths', []))
                tqdm.write(f"{_id}: generated {count} pages")
                
                # Batch write
                if len(batch_buffer) >= batch_size:
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                        for item in batch_buffer:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    batch_buffer = []
    
    # Write remaining items
    if batch_buffer:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            for item in batch_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("Processing complete")


