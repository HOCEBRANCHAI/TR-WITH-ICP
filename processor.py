"""
Invoice processing pipeline (synchronous).

This module is the "engine" behind the API in `app.py`.

### What this project does (high-level)
Given an uploaded invoice document (PDF/image), we produce a structured transaction record:
1) Extract text from the document (PDF text extraction or OCR)
2) Use an LLM to convert invoice text into a strict JSON schema
3) Validate and normalize the extracted fields (dates/currency/amount math)
4) Map the LLM schema into a "transaction register" entry (fields used by the frontend)
5) Classify invoice as Sales vs Purchase using our company names + context heuristics
6) Determine Dutch VAT return category + ICP reporting fields (NL-specific logic)
7) (Optional) Convert amounts into EUR using historical FX rates
8) (Optional) Build a balanced journal entry via data-driven posting rules

### Important notes / current design reality (be honest)
- This file currently contains multiple concerns (OCR, LLM calls, VAT logic, FX, posting rules).
  It works, but it is monolithic. A future refactor should split this into:
  `extraction/` (OCR + LLM) and `domain/` (classification + VAT + posting + FX).
- The pipeline depends heavily on the quality of extracted text. When OCR is weak, downstream
  classification becomes less reliable. Validation + confidence scoring tries to flag this.
"""

import io
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import PyPDF2
import boto3
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import pytesseract
import difflib  
from openai import OpenAI
import requests
from dotenv import load_dotenv
import pathlib
log = logging.getLogger("invoice-processor")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
env_paths = [
    pathlib.Path(__file__).parent / '.env',  # Same directory as processor.py
    pathlib.Path.cwd() / '.env',  # Current working directory
    pathlib.Path.home() / '.env',  # Home directory (fallback)
]
env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        log.info(f"Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    # Fallback: try loading from current directory without explicit path
    load_dotenv()
    log.info("Attempted to load .env from current directory")

# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    log.warning("WARNING: OPENAI_API_KEY not found in environment variables!")
    log.warning("Please ensure your .env file contains: OPENAI_API_KEY=your_key_here")
else:
    log.info("OpenAI API key loaded successfully")
# -------------------- Money helpers --------------------
# We treat amounts as Decimal internally for accuracy and rounding predictability.
CENT = Decimal("0.01")
RATE_PREC = Decimal("0.0001")

def q_money(x) -> Decimal:
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)

def q_rate(x) -> Decimal:
    return Decimal(str(x)).quantize(RATE_PREC, rounding=ROUND_HALF_UP)

def nearly_equal_money(a: Decimal, b: Decimal, tol: Decimal = CENT) -> bool:
    return abs(q_money(a) - q_money(b)) <= tol

# -------------------- Dates & currency --------------------
# Dates are expected in ISO format because they serialize cleanly and are unambiguous.
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def ensure_iso_date(s: Optional[str], field: str, errors: List[str]) -> Optional[date]:
    if not s or not ISO_DATE.match(s):
        errors.append(f"{field} must be YYYY-MM-DD (got {s!r}).")
        return None
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception:
        errors.append(f"{field} is not a valid calendar date (got {s!r}).")
        return None

KNOWN_CURRENCIES = {
    "EUR","USD","GBP","INR","EGP","AED","SAR","CAD","AUD","NZD",
    "JPY","CNY","DKK","SEK","NOK","CHF","PLN","CZK","HUF"
}

def normalize_currency(cur: Optional[str], errors: List[str]) -> Optional[str]:
    cur = (cur or "").strip().upper()
    if cur not in KNOWN_CURRENCIES:
        errors.append(f"Unknown or missing currency {cur!r}.")
        return None
    return cur

def _prev_business_day(d: date) -> date:
    while d.weekday() >= 5:  # 5=Sat,6=Sun
        d -= timedelta(days=1)
    return d

def get_eur_rate(invoice_date: date, ccy: str) -> Tuple[Decimal, str]:
    """
    Return (rate, rate_date_str) for 1 CCY -> EUR using exchangerate.host (ECB).
    Try direct (base=CCY&symbols=EUR) and fallback by inversion.
    Look back up to 7 business days.
    """
    ccy = (ccy or "").upper().strip()
    if ccy == "EUR":
        return Decimal("1"), invoice_date.isoformat()

    d = _prev_business_day(invoice_date)
    for _ in range(7):
        # direct
        url1 = f"https://api.exchangerate.host/{d.isoformat()}?base={ccy}&symbols=EUR"
        try:
            r1 = requests.get(url1, timeout=8)
            js1 = r1.json() if r1.content else {}
            rate = (js1.get("rates") or {}).get("EUR")
            if r1.status_code == 200 and rate:
                return q_rate(rate), d.isoformat()
            log.warning(f"FX miss (direct) {url1} status={r1.status_code}")
        except Exception as ex:
            log.warning(f"FX direct failed {url1}: {ex}")

        # invert
        url2 = f"https://api.exchangerate.host/{d.isoformat()}?base=EUR&symbols={ccy}"
        try:
            r2 = requests.get(url2, timeout=8)
            js2 = r2.json() if r2.content else {}
            base_rate = (js2.get("rates") or {}).get(ccy)
            if r2.status_code == 200 and base_rate and float(base_rate) != 0.0:
                inv = Decimal("1") / Decimal(str(base_rate))
                return q_rate(inv), d.isoformat()
            log.warning(f"FX miss (invert) {url2} status={r2.status_code}")
        except Exception as ex2:
            log.warning(f"FX invert failed {url2}: {ex2}")

        d = _prev_business_day(d - timedelta(days=1))
    
    # NEW: Better fallback than crash
    log.error(f"No EUR rate found for {ccy} near {invoice_date.isoformat()}. Using 1.0.")#not efficient 
    return Decimal("1.0"), invoice_date.isoformat()

# -------------------- OCR / text extraction (robust; supports PDFs & images) --------------------
# Goal: produce the best possible raw invoice text given a PDF or an image.
#
# Strategy:
# - For "real text PDFs" we prefer PyPDF2 (fast, no external services).
# - For scanned invoices / images we prefer AWS Textract (better at forms/tables).
# - If Textract isn't available or is still weak, fall back to Tesseract with preprocessing.
def _aws_region() -> str:
    """
    Prefer AWS_REGION if set, otherwise AWS_DEFAULT_REGION, otherwise us-east-1.
    This keeps behaviour backwards compatible but makes region selection explicit.
    """
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _preprocess_for_tesseract(pil_img: Image.Image) -> Image.Image:
    """
    Aggressive but safe preprocessing to improve OCR quality:
      - convert to grayscale
      - bilateral filter to reduce noise but keep edges
      - adaptive threshold for better contrast
    """
    img = np.array(pil_img.convert("L"))
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return Image.fromarray(img)
def _textract_analyze_image(img_bytes: bytes) -> str:
    textract = boto3.client("textract", region_name=_aws_region())
    resp = textract.analyze_document(
        Document={'Bytes': img_bytes},
        FeatureTypes=['TABLES', 'FORMS']
    )
    blocks = resp.get("Blocks", [])
    text_lines = []
    block_map = {b["Id"]: b for b in blocks}

    for b in blocks:
        if b.get("BlockType") == "LINE" and b.get("Text"):
            text_lines.append(b["Text"])

    kv_pairs = []
    for b in blocks:
        if b.get("BlockType") == "KEY_VALUE_SET" and "KEY" in (b.get("EntityTypes") or []):
            key_words, val_words = [], []
            for rel in b.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    for cid in rel.get("Ids", []):
                        w = block_map.get(cid)
                        if w and w.get("BlockType") == "WORD" and w.get("Text"):
                            key_words.append(w["Text"])
                if rel["Type"] == "VALUE":
                    for vid in rel.get("Ids", []):
                        v = block_map.get(vid)
                        if not v: continue
                        for rel2 in v.get("Relationships", []):
                            if rel2["Type"] == "CHILD":
                                for vcid in rel2.get("Ids", []):
                                    w = block_map.get(vcid)
                                    if w and w.get("BlockType") == "WORD" and w.get("Text"):
                                        val_words.append(w["Text"])
            k = " ".join(key_words).strip()
            v = " ".join(val_words).strip()
            if k or v:
                kv_pairs.append(f"{k}: {v}")

    combined = "\n".join(text_lines)
    if kv_pairs:
        combined += "\n--- Key-Value Pairs ---\n" + "\n".join(kv_pairs)
    return combined

def _tesseract_ocr(pil_img: Image.Image) -> str:
    pre = _preprocess_for_tesseract(pil_img)
    # Use English by default; --psm 6 handles block of text with uniform size.
    # This dramatically improves character accuracy compared to the default.
    return pytesseract.image_to_string(pre, config="--psm 6 -l eng")

def get_text_from_pdf(pdf_bytes: bytes, filename: str) -> str:
    """
    Best-effort text extraction for PDFs.

    Returns the first "good enough" text output, with progressive fallbacks:
    1) PyPDF2 text extraction
    2) AWS Textract detect_document_text on the raw PDF bytes
    3) Render PDF to images and run AWS Textract analyze_document per page
    4) Render PDF to images and run Tesseract OCR per page

    If everything yields only minimal text, we return the "best attempt" instead of crashing,
    because downstream systems prefer partial data over hard failure. If literally nothing can
    be extracted, we raise.
    """
    best_text: str = ""
    best_source: str = ""

    def _update_best(candidate: str, source: str) -> None:
        nonlocal best_text, best_source
        if candidate and len(candidate.strip()) > len(best_text.strip()):
            best_text = candidate
            best_source = source

    # 1) PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = "".join([(p.extract_text() or "") for p in reader.pages])
        _update_best(text, "PyPDF2")
        if len(text.strip()) > 80:
            log.info(f"[PyPDF2] {filename}")
            return text
        log.warning(f"[PyPDF2] minimal for {filename}; trying Textract detect.")
    except Exception as e:
        log.warning(f"[PyPDF2] failed for {filename}: {e}")

    # 2) Textract detect (only if AWS creds/region likely configured)
    try:
        textract = boto3.client("textract", region_name=_aws_region())
        resp = textract.detect_document_text(Document={"Bytes": pdf_bytes})
        text = "\n".join(
            [
                b.get("Text", "")
                for b in (resp.get("Blocks") or [])
                if b.get("BlockType") == "LINE"
            ]
        )
        _update_best(text, "Textract.detect")
        if len(text.strip()) > 60:
            log.info(f"[Textract.detect] {filename}")
            return text
        log.warning(f"[Textract.detect] minimal for {filename}; trying Analyze per page.")
    except Exception as e:
        log.warning(f"[Textract.detect] failed for {filename}: {e}; Analyze per page.")

    # 3) Textract Analyze per-page (images)
    try:
        images: List[Image.Image] = convert_from_bytes(pdf_bytes, dpi=300)
        texts = []
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            page_text = _textract_analyze_image(buf.getvalue())
            if page_text:
                texts.append(page_text)
                _update_best(page_text, "Textract.analyze IMG")
        combined = "\n\n--- PAGE BREAK ---\n\n".join(texts)
        if len(combined.strip()) > 60:
            log.info(f"[Textract.analyze IMG] {filename}")
            return combined
        log.warning(f"[Textract.analyze IMG] minimal; trying Tesseract.")
    except Exception as e:
        log.warning(f"[Textract.analyze IMG] failed for {filename}: {e}; Tesseract fallback.")

    # 4) Tesseract fallback
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
        ocr = []
        for im in images:
            page_text = _tesseract_ocr(im)
            if page_text:
                ocr.append(page_text)
                _update_best(page_text, "Tesseract")
        combined = "\n\n--- PAGE BREAK ---\n\n".join(ocr)
        if len(combined.strip()) > 20:
            log.info(f"[Tesseract] {filename}")
            return combined
    except Exception as e:
        log.error(f"[Tesseract] failed for {filename}: {e}")

    # If we reached this point, all strategies were "minimal".
    # For production we still prefer returning *something* over hard failure.
    if best_text.strip():
        log.error(
            f"PDF text extraction for {filename} only produced minimal text; "
            f"returning best attempt from {best_source} with length={len(best_text.strip())}."
        )
        return best_text

    # Absolute fallback – nothing at all could be read.
    raise ValueError(
        f"PDF text extraction failed for {filename}: "
        f"PyPDF2, Textract detect, Textract analyze (per image), and Tesseract all returned empty text."
    )


def get_text_from_image(image_bytes: bytes, filename: str) -> str:
    """
    Best-effort text extraction for image-based invoices (JPEG, PNG, TIFF, etc.).

    Fallback order:
    1) AWS Textract detect_document_text
    2) AWS Textract analyze_document (FORMS + TABLES)
    3) Tesseract OCR with preprocessing
    """
    best_text: str = ""
    best_source: str = ""

    def _update_best(candidate: str, source: str) -> None:
        nonlocal best_text, best_source
        if candidate and len(candidate.strip()) > len(best_text.strip()):
            best_text = candidate
            best_source = source

    # 1) Textract detect
    try:
        textract = boto3.client("textract", region_name=_aws_region())
        resp = textract.detect_document_text(Document={"Bytes": image_bytes})
        text = "\n".join(
            [
                b.get("Text", "")
                for b in (resp.get("Blocks") or [])
                if b.get("BlockType") == "LINE"
            ]
        )
        _update_best(text, "Textract.detect IMG")
        if len(text.strip()) > 40:
            log.info(f"[Textract.detect IMG] {filename}")
            return text
        log.warning(f"[Textract.detect IMG] minimal for {filename}; trying AnalyzeDocument.")
    except Exception as e:
        log.warning(f"[Textract.detect IMG] failed for {filename}: {e}; trying AnalyzeDocument.")

    # 2) Textract AnalyzeDocument (FORMS + TABLES)
    try:
        analyzed = _textract_analyze_image(image_bytes)
        _update_best(analyzed, "Textract.analyze IMG")
        if len(analyzed.strip()) > 40:
            log.info(f"[Textract.analyze IMG] {filename}")
            return analyzed
        log.warning(f"[Textract.analyze IMG] minimal for {filename}; trying Tesseract.")
    except Exception as e:
        log.warning(f"[Textract.analyze IMG helper] failed for {filename}: {e}; trying Tesseract.")

    # 3) Tesseract fallback
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        text = _tesseract_ocr(pil_img)
        _update_best(text, "Tesseract IMG")
        if len(text.strip()) > 20:
            log.info(f"[Tesseract IMG] {filename}")
            return text
    except Exception as e:
        log.error(f"[Tesseract IMG] failed for {filename}: {e}")

    # If we reached this point, only minimal text.
    if best_text.strip():
        log.error(
            f"Image text extraction for {filename} only produced minimal text; "
            f"returning best attempt from {best_source} with length={len(best_text.strip())}."
        )
        return best_text

    raise ValueError(
        f"Image text extraction failed for {filename}: "
        f"Textract detect, Textract analyze, and Tesseract all returned empty text."
    )


def get_text_from_document(file_bytes: bytes, filename: str) -> str:
    """
    Entry point for *any* uploaded document.

    We detect PDF vs image primarily via file signature/extension:
    - If PDF: run the PDF pipeline
    - If known image: run the image pipeline
    - Otherwise: try PDF first, then image as fallback
    """
    name = (filename or "").lower()
    ext = os.path.splitext(name)[1]

    # Quick PDF signature check
    is_pdf_header = file_bytes.startswith(b"%PDF-")
    if is_pdf_header or ext == ".pdf":
        return get_text_from_pdf(file_bytes, filename)

    # Image type detection via filename extension
    if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}:
        return get_text_from_image(file_bytes, filename)

    # Ambiguous: try PDF pipeline first, then fall back to image logic if that fails
    try:
        return get_text_from_pdf(file_bytes, filename)
    except Exception as e_pdf:
        log.warning(f"PDF pipeline failed for {filename} ({e_pdf}); trying image pipeline.")
        return get_text_from_image(file_bytes, filename)

# -------------------- LLM extraction --------------------
# Goal: convert messy invoice text into a strict, structured JSON object.
#
# Notes:
# - We ask OpenAI to return JSON only (response_format json_object) to reduce parsing errors.
# - This stage is extremely sensitive to input text quality (OCR) and truncation.
# - For production-grade quality, we should minimize "blind truncation" and instead
#   reduce to the most relevant regions (e.g., totals, VAT blocks, addresses, line items).
SECTION_LABELS = [
    "invoice", "total", "subtotal", "tax", "vat", "btw", "reverse charge",
    "verlegd", "omgekeerde heffing", "bill to", "payer", "customer", "vendor",
    "supplier", "line items", "description", "due", "payment terms", "amount"
]

def reduce_invoice_text(raw_text: str, window: int = 300) -> str:
    """
    Reduce long raw text to likely-relevant regions around invoice keywords.

    Why:
    - Many invoices contain long footers, legal terms, and repeated headers.
    - Blindly taking the first N characters can exclude the totals/VAT blocks.

    This helper extracts multiple windows around section labels and merges overlaps.
    """
    text = raw_text or ""
    text_low = text.lower()
    spans: List[Tuple[int, int]] = []
    for label in SECTION_LABELS:
        for m in re.finditer(re.escape(label), text_low):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            spans.append((start, end))
    if not spans:
        return text[:8000]
    spans.sort()
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e + 50:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    chunks = [text[s:e] for s, e in merged]
    reduced = "\n---\n".join(chunks)
    return reduced[:12000]

LLM_PROMPT = """
You are an expert, high-accuracy financial data extraction model. Your sole task is to extract structured data from the provided invoice text and respond with a single, minified JSON object. You must think like an accountant.

RULES:
1) JSON ONLY. Include all top-level keys even if null. Dates = YYYY-MM-DD. Numbers = floats (no symbols).
2) Use the invoice currency shown in the official TOTAL block; ignore "for reference" currencies.
3) subtotal = goods/services only; total_vat = all taxes; total_amount = subtotal + total_vat.
4) VAT category:
   - "Import-VAT" for import VAT.
   - "Reverse-Charge" if reverse-charge applies (e.g., verlegd, omgekeerde heffing).
   - "Standard" for normal % VAT charged.
   - "Zero-Rated" if 0% VAT and not reverse charge.
   - "Out-of-Scope" if outside tax scope (e.g., Article 44).
5) Line items:
   - Extract goods/services only; do NOT include taxes as a line item.
   - unit_price only if explicitly printed. Do not compute it.
6) VAT percentage:
   - If the invoice explicitly shows a single VAT rate (e.g., 21%), put that in vat_breakdown.rate.
   - If text states "VAT out of scope / Article 44 / reverse charge / 0%", use 0.0 in vat_breakdown.
   - If multiple rates exist, list them; total_vat must equal the sum of tax_amount.
7) ADDRESSES - CRITICAL FOR VAT CLASSIFICATION:
   - Extract COMPLETE addresses including street, city, postal code, and COUNTRY.
   - The country is ESSENTIAL for determining EU vs non-EU transactions.
   - Look for country names (e.g., "Netherlands", "Germany", "France") or country codes (e.g., "NL", "DE", "FR") at the end of addresses.
   - If country is not explicitly stated, INFER it from well-known city names:
     * Cities like "Galway", "Dublin", "Cork" -> Ireland
     * Cities like "London", "Manchester", "Birmingham" -> United Kingdom
     * Cities like "Amsterdam", "Rotterdam", "The Hague" -> Netherlands
     * Cities like "Paris", "Lyon", "Marseille" -> France
     * Cities like "Berlin", "Munich", "Hamburg" -> Germany
     * Cities like "Madrid", "Barcelona", "Valencia" -> Spain
     * Cities like "Rome", "Milan", "Naples" -> Italy
     * And other major EU/non-EU cities you recognize
   - Always include the inferred country in the address string (e.g., "Annaghdown, Galway, Ireland").
   - Include the full address as a single string with all address components.
   - This is critical for production-level VAT subcategory classification.
8) VAT NUMBERS - CRITICAL FOR EU B2B CLASSIFICATION:
   - Extract VAT registration numbers for both vendor and customer.
   - VAT numbers typically appear near company names or in header/footer sections.
   - Common formats: "NL123456789B01", "GB123456789", "IE1234567X", "DE123456789", etc.
   - Look for labels like "VAT", "BTW", "TVA", "IVA", "VAT Reg No", "Tax ID", "Registration No".
   - For EU customers, VAT number is REQUIRED for proper B2B classification.
   - If VAT number is not found, set to null (do not guess or invent).
9) GOODS VS SERVICES:
   - Determine if the invoice is for "goods" or "services" based on line item descriptions.
   - Look for keywords like "product", "item", "goods", "merchandise" for goods.
   - Look for keywords like "service", "consulting", "support", "maintenance", "software license" for services.
   - If unclear, default to "services" for B2B transactions, "goods" for physical products.
   - Set "goods_services_indicator" to "goods" or "services" or null if truly unclear.
10) BANK ACCOUNT DETAILS (IBAN):
   - Extract IBAN numbers for both vendor and customer if present on the invoice.
   - IBAN is typically found in payment/banking sections of invoices.
   - Format: IBAN codes start with 2-letter country code followed by 2 digits and up to 30 alphanumeric characters.
   - Extract "vendor_iban" from vendor's bank account details.
   - Extract "customer_iban" from customer's bank account details (if shown).
   - IBAN can be a signal for local vs non-local transactions affecting VAT categorization.
11) NOTES AND COMMENTS:
   - Extract any general notes, comments, or additional information from the invoice.
   - Include payment instructions, special terms, or any other relevant notes.
   - Store in "notes" field.
SCHEMA:

{
  "invoice_number": "string | null",
  "invoice_date": "YYYY-MM-DD | null",
  "due_date": "YYYY-MM-DD | null",
  "vendor_name": "string | null",
  "vendor_vat_id": "string | null",
  "vendor_address": "string | null (MUST include full address with country)",
  "customer_name": "string | null",
  "customer_vat_id": "string | null",
  "customer_address": "string | null (MUST include full address with country)",
  "currency": "string | null",
  "vat_category": "string | null",
  "subtotal": "float | null",
  "total_amount": "float | null",
  "total_vat": "float | null",
  "vat_breakdown": [
    {"rate": "float | 'import'", "base_amount": "float | null", "tax_amount": "float"}
  ],
  "line_items": [
    {"description": "string", "quantity": "float | null", "unit_price": "float | null", "line_total": "float | null"}
  ],
  "payment_terms": "string | null",
  "goods_services_indicator": "goods | services | null",
  "vendor_iban": "string | null",
  "customer_iban": "string | null",
  "notes": "string | null"
}

"""
def structure_text_with_llm(invoice_text: str, filename: str) -> dict:
    """
    Send invoice text to the LLM and parse the strict JSON response.

    Output:
    - A dict matching the schema described in LLM_PROMPT

    IMPORTANT:
    - If this returns an empty dict, downstream mapping/classification will become guessy.
      A future improvement is to return a typed result with explicit error codes so the API
      can report "extraction failed" instead of returning low-quality classifications.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    client = OpenAI(api_key=api_key)
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                # NOTE: The caller is responsible for passing a reduced, bounded text
                # (see reduce_invoice_text()). We still hard-cap as a safety net.
                {"role": "user", "content": f"INVOICE TEXT:\n{(invoice_text or '')[:12000]}"}
            ]
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        # IMPORTANT: Do not silently return {}. Bubble up so the pipeline state can fail fast
        # and downstream mapping/classification will not run.
        raise RuntimeError(f"LLM structuring failed for {filename}: {e}") from e

def _translate_to_english_if_dutch(text: str) -> str:
    """
    Best‑effort translation helper:
      - If the text appears in Dutch, translate it to English.
      - If it is already English or another language, return it unchanged.
    Uses the same OpenAI client as the main extraction. On any failure, returns
    the original text so we never break the pipeline because of translation.
    """
    if not text or not text.strip():
        return text

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # No API key -> cannot translate; keep original description.
        return text

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation helper. "
                        "If the user text is Dutch, respond with an accurate English translation. "
                        "If the text is already English or clearly not Dutch, return it exactly as-is. "
                        "Output plain text only, no explanations, no quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or text
    except Exception as ex:
        log.error(f"Description translation failed: {ex}")
        return text


# -------------------- Validation & mapping --------------------
# Validation is intentionally conservative:
# - We prefer flagging issues (errors/warnings) over silently "fixing" extracted values.
# - Confidence scoring uses validation errors + text quality to suggest manual review.
def _estimate_extraction_confidence(invoice_text: str, llm_data: Dict[str, Any], validation_errors: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    Heuristic confidence score for how reliable the extraction likely is.
    This is *not* a guarantee – just a signal to help prioritise manual review.
    
    Args:
        invoice_text: Raw extracted text from OCR/PDF
        llm_data: Structured data extracted by LLM
        validation_errors: List of validation errors from validate_extraction()
    """
    text = (invoice_text or "").strip()
    text_len = len(text)
    validation_errors = validation_errors or []

    # 1) Base on text length
    if text_len < 200:
        level = "low"
        reasons = [f"very short extracted text ({text_len} chars)"]
    elif text_len < 800:
        level = "medium"
        reasons = [f"moderate extracted text length ({text_len} chars)"]
    else:
        level = "high"
        reasons = [f"long extracted text ({text_len} chars)"]

    # 2) Check for presence of section labels
    text_low = text.lower()
    present_labels = [lbl for lbl in SECTION_LABELS if lbl in text_low]
    if len(present_labels) <= 2:
        reasons.append(f"few invoice keywords found ({len(present_labels)})")
        if level == "high":
            level = "medium"
    elif len(present_labels) <= 5 and level == "high":
        reasons.append(f"some invoice keywords found ({len(present_labels)})")

    # 3) Check for missing critical LLM fields
    critical_fields = [
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "customer_name",
        "subtotal",
        "total_vat",
        "total_amount",
    ]
    missing = [f for f in critical_fields if not llm_data.get(f)]
    if missing:
        reasons.append(f"missing critical fields from LLM: {', '.join(missing)}")
        if level == "high":
            level = "medium"

    # 4) VALIDATION ERRORS - Most important factor
    if validation_errors:
        # Any validation error significantly reduces confidence
        error_count = len(validation_errors)
        reasons.append(f"validation errors detected: {error_count} issue(s)")
        if error_count >= 3:
            level = "low"
        elif error_count >= 2:
            if level == "high":
                level = "medium"
            elif level == "medium":
                level = "low"
        else:
            if level == "high":
                level = "medium"

    reason_str = "; ".join(reasons)
    return level, reason_str

def validate_extraction(data: dict, filename: str) -> Tuple[date, str, Decimal, Decimal, Decimal, List[str]]:
    """
    Comprehensive validation of extracted invoice data.
    Returns: (invoice_date, currency, subtotal, vat_amount, total_amount, validation_errors)
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # 1. DATE VALIDATION
    inv_date = ensure_iso_date(data.get("invoice_date"), "invoice_date", errors)
    if inv_date:
        today = date.today()
        # Check if date is in the future (more than 30 days is suspicious)
        if inv_date > today:
            days_ahead = (inv_date - today).days
            if days_ahead > 30:
                errors.append(f"Invoice date is {days_ahead} days in the future")
            else:
                warnings.append(f"Invoice date is {days_ahead} days in the future")
        
        # Check if date is too old (more than 10 years is suspicious)
        if (today - inv_date).days > 3650:
            errors.append(f"Invoice date is more than 10 years old")
    
    if data.get("due_date"):
        due_date = ensure_iso_date(data.get("due_date"), "due_date", errors)
        if inv_date and due_date:
            # Due date should be after invoice date
            if due_date < inv_date:
                errors.append(f"Due date ({due_date}) is before invoice date ({inv_date})")
            # Due date shouldn't be more than 1 year after invoice date
            elif (due_date - inv_date).days > 365:
                warnings.append(f"Due date is more than 1 year after invoice date")

    # 2. CURRENCY VALIDATION
    currency = normalize_currency(data.get("currency"), errors)

    # 3. AMOUNT VALIDATION
    try:
        sub = q_money(data.get("subtotal") or 0)
        vat = q_money(data.get("total_vat") or 0)
        tot = q_money(data.get("total_amount") or 0)
        
        # Check for negative amounts (unless it's a credit note)
        if sub < 0:
            warnings.append(f"Subtotal is negative: {sub}")
        if vat < 0:
            warnings.append(f"VAT amount is negative: {vat}")
        if tot < 0:
            warnings.append(f"Total amount is negative: {tot}")
        
        # Validate: Subtotal + VAT = Total (with tolerance for rounding)
        calculated_total = sub + vat
        if not nearly_equal_money(calculated_total, tot, tol=Decimal("0.05")):
            diff = abs(calculated_total - tot)
            errors.append(f"Amount mismatch: Subtotal({sub}) + VAT({vat}) = {calculated_total}, but Total = {tot} (difference: {diff})")
        
        # Check for unreasonably large amounts (flag if > 1 million)
        if abs(tot) > Decimal("1000000"):
            warnings.append(f"Total amount is very large: {tot}")
        
    except (ValueError, TypeError, KeyError) as e:
        errors.append(f"Error parsing amounts: {e}")

    # 4. VAT CALCULATION VALIDATION
    try:
        vat_breakdown = data.get("vat_breakdown", [])
        if vat_breakdown:
            # Validate VAT breakdown sums to total VAT
            breakdown_vat_sum = Decimal("0")
            for item in vat_breakdown:
                if isinstance(item, dict):
                    tax_amt = item.get("tax_amount")
                    if tax_amt is not None:
                        try:
                            breakdown_vat_sum += q_money(tax_amt)
                        except (ValueError, TypeError):
                            pass
            
            if breakdown_vat_sum > 0:
                if not nearly_equal_money(breakdown_vat_sum, vat, tol=Decimal("0.05")):
                    errors.append(
                        f"VAT breakdown sum ({breakdown_vat_sum}) does not match total_vat ({vat})"
                    )
            
            # Validate VAT rates are reasonable
            for item in vat_breakdown:
                if isinstance(item, dict):
                    rate = item.get("rate")
                    if rate is not None and rate != "import":
                        try:
                            rate_val = float(rate)
                            # Common VAT rates: 0%, 9%, 21% (Netherlands), or other EU rates
                            common_rates = [0.0, 9.0, 21.0, 6.0, 10.0, 13.0, 19.0, 20.0, 22.0, 25.0]
                            if rate_val < 0 or rate_val > 30:
                                warnings.append(f"Unusual VAT rate: {rate_val}%")
                            elif rate_val > 0 and not any(abs(rate_val - cr) < 0.1 for cr in common_rates):
                                warnings.append(f"Non-standard VAT rate: {rate_val}%")
                        except (ValueError, TypeError):
                            pass
        
        # Validate VAT calculation: If we have subtotal and VAT rate, calculate expected VAT
        if vat_breakdown and len(vat_breakdown) == 1:
            item = vat_breakdown[0]
            if isinstance(item, dict):
                rate = item.get("rate")
                base = item.get("base_amount")
                if rate is not None and rate != "import" and base is not None:
                    try:
                        rate_val = float(rate)
                        base_val = q_money(base)
                        expected_vat = q_money(base_val * Decimal(str(rate_val)) / Decimal("100"))
                        if not nearly_equal_money(expected_vat, vat, tol=Decimal("0.05")):
                            warnings.append(
                                f"VAT calculation mismatch: {base_val} * {rate_val}% = {expected_vat}, "
                                f"but total_vat = {vat}"
                            )
                    except (ValueError, TypeError):
                        pass
    
    except Exception as e:
        warnings.append(f"Error validating VAT breakdown: {e}")

    # 5. CROSS-FIELD VALIDATION
    # Check if invoice number is present and reasonable
    inv_num = data.get("invoice_number")
    if not inv_num or not str(inv_num).strip():
        warnings.append("Invoice number is missing")
    
    # Check if vendor/customer names are present
    if not data.get("vendor_name") or not str(data.get("vendor_name", "")).strip():
        warnings.append("Vendor name is missing")
    if not data.get("customer_name") or not str(data.get("customer_name", "")).strip():
        warnings.append("Customer name is missing")

    # Log all issues
    all_issues = errors + warnings
    if all_issues:
        error_msg = " | ".join([f"ERROR: {e}" for e in errors])
        warning_msg = " | ".join([f"WARNING: {w}" for w in warnings])
        msg_parts = []
        if error_msg:
            msg_parts.append(error_msg)
        if warning_msg:
            msg_parts.append(warning_msg)
        log.warning(f"Validation issues for {filename}: {' | '.join(msg_parts)}")
    
    # Return validation errors list for confidence scoring
    return inv_date, currency, sub, vat, tot, errors

def _normalize_company_name(name: str) -> str:
    """
    Normalizes company name for matching by:
    - Converting to lowercase
    - Removing all punctuation and special characters
    - Removing common legal suffixes (B.V., BV, B.V, Ltd, Limited, etc.)
    - Normalizing spaces
    - Removing common stop words
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.casefold()
    
    # Remove common legal entity suffixes (case-insensitive)
    legal_suffixes = [
        r'\bb\.v\.?\b', r'\bbv\b', r'\bb\.v\b',
        r'\bltd\.?\b', r'\blimited\b',
        r'\binc\.?\b', r'\bincorporated\b',
        r'\bllc\b', r'\bll\.?c\.?\b',
        r'\bcorp\.?\b', r'\bcorporation\b',
        r'\bs\.a\.?\b', r'\bsa\b',
        r'\bs\.a\.?r\.?l\.?\b', r'\bsarl\b',
        r'\bgmbh\b', r'\bag\b',
        r'\bn\.?v\.?\b', r'\bnv\b',
        r'\bspa\b', r'\bsp\.?z\.?o\.?o\.?\b',
        r'\bsl\b', r'\bs\.?l\.?\b'
    ]
    
    for suffix_pattern in legal_suffixes:
        normalized = re.sub(suffix_pattern, '', normalized)
    
    # Remove common stop words that don't help with matching
    stop_words = [r'\bthe\b', r'\ba\b', r'\ban\b', r'\band\b', r'\bor\b', r'\bof\b']
    for stop_word in stop_words:
        normalized = re.sub(stop_word, '', normalized)
    
    # Remove all punctuation, special characters, and normalize spaces
    normalized = re.sub(r"[\s\-_/.,()\[\]{}'\"&]+", " ", normalized)
    
    # Remove multiple spaces and trim
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def _calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculates similarity between two normalized company names using difflib.
    Returns a score between 0.0 and 1.0.
    """
    n1 = _normalize_company_name(name1)
    n2 = _normalize_company_name(name2)
    if not n1 or not n2:
        return 0.0
    if n1 in n2 or n2 in n1:
        return 1.0
    return difflib.SequenceMatcher(None, n1, n2).ratio()

def _calculate_name_similarity_robust(name1: str, name2: str) -> float:
    """
    Token Set Ratio approach: highly robust against suffixes like 'B.V.'
    """
    n1 = _normalize_company_name(name1)
    n2 = _normalize_company_name(name2)
    if not n1 or not n2:
        return 0.0
    
    s1 = set(n1.split())
    s2 = set(n2.split())
    
    # Intersection score (how many words match?)
    intersection = s1.intersection(s2)
    if not intersection:
        return 0.0
    
    # Score based on the SHORTER string (so "Google" matches "Google Ireland Ltd")
    score = len(intersection) / min(len(s1), len(s2))
    return score
    
    return len(intersection) / len(union)

def _split_company_list(raw: str) -> List[str]:
    if not raw: return []
    return [p.strip() for p in re.split(r"[,\n;]", raw) if p and p.strip()]

# ----------------------------------------------------------------------------
# DYNAMIC HELPER 1: REGEX IBAN EXTRACTION (Fixes missing IBANs)
# ----------------------------------------------------------------------------
def _extract_ibans_via_regex(text: str) -> List[str]:
    """
    Finds IBANs in raw text even if LLM misses them.
    Looks for 2 letters, 2 digits, 10-30 alphanumeric.
    """
    if not text: return []
    clean = re.sub(r'\s+', '', text)
    # Generic IBAN: 2 letters, 2 digits, 10-30 alphanumeric
    return list(set(re.findall(r'([A-Z]{2}\d{2}[A-Z0-9]{10,30})', clean)))
    # Filter by length (valid IBANs are usually 15-34 chars)
    return list(set([m for m in matches if 15 <= len(m) <= 34]))

def _is_same_country_code(code: str, country_name: str) -> bool:
    """Helper to check if a 2-letter code matches a country name."""
    mapping = {
        "NL": "Netherlands", "DE": "Germany", "FR": "France", 
        "BE": "Belgium", "EG": "Egypt", "US": "United States",
        "GB": "United Kingdom", "UK": "United Kingdom", "ES": "Spain", "IT": "Italy"
    }
    c_name = mapping.get(code.upper(), "")
    if c_name and c_name.lower() in country_name.lower():
        return True
    return False

# ----------------------------------------------------------------------------
# EU Country Detection & Address Parsing
# ----------------------------------------------------------------------------
# We use lightweight heuristics to infer country from addresses when the LLM misses it.
# This is important because VAT box mapping depends heavily on whether a party is NL/EU/non-EU.

EU_COUNTRIES = {
    # Full country names
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", "czechia",
    "denmark", "estonia", "finland", "france", "germany", "greece", "hungary",
    "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands",
    "poland", "portugal", "romania", "slovakia", "slovenia", "spain", "sweden",
    # Country codes
    "at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr", "hu",
    "ie", "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro", "sk", "si", "es", "se",
    # Alternative names
    "nederland", "holland", "deutschland", "italia", "espana", "frankreich",
    "belgie", "belgique"
}

COMMON_COUNTRIES = {
    # EU countries (from above) - map to themselves
    **{country: country for country in EU_COUNTRIES if len(country) > 2},  # Full names only
    # Non-EU countries
    "united kingdom": "united kingdom", "uk": "united kingdom", "great britain": "united kingdom", "britain": "united kingdom",
    "united states": "united states", "usa": "united states", "us": "united states", "america": "united states",
    "switzerland": "switzerland", "norway": "norway", "iceland": "iceland",
    "china": "china", "japan": "japan", "india": "india", "canada": "canada",
    "australia": "australia", "new zealand": "new zealand", "south africa": "south africa",
    "brazil": "brazil", "mexico": "mexico", "argentina": "argentina",
    "egypt": "egypt", "uae": "united arab emirates", "united arab emirates": "united arab emirates", 
    "saudi arabia": "saudi arabia", "singapore": "singapore", "hong kong": "hong kong", 
    "south korea": "south korea", "taiwan": "taiwan", "turkey": "turkey",
    "russia": "russia", "ukraine": "ukraine"
}

def _extract_country_from_address(address: str) -> Optional[str]:
    """
    Extracts country name or code from an address string.
    Returns the country name if found, None otherwise.
    Handles both EU and non-EU countries.
    """
    if not address:
        return None
    
    address_lower = address.lower()
    
    # City-to-country mapping for well-known cities
    city_to_country = {
        # Ireland
        "galway": "Ireland", "dublin": "Ireland", "cork": "Ireland", "limerick": "Ireland",
        "waterford": "Ireland", "kilkenny": "Ireland", "belfast": "Ireland",
        # UK
        "london": "United Kingdom", "manchester": "United Kingdom", "birmingham": "United Kingdom",
        "glasgow": "United Kingdom", "edinburgh": "United Kingdom", "liverpool": "United Kingdom",
        # Other well-known cities
        "amsterdam": "Netherlands", "rotterdam": "Netherlands", "the hague": "Netherlands",
        "paris": "France", "lyon": "France", "marseille": "France",
        "berlin": "Germany", "munich": "Germany", "hamburg": "Germany",
        "madrid": "Spain", "barcelona": "Spain", "valencia": "Spain",
        "rome": "Italy", "milan": "Italy", "naples": "Italy",
    }
    
    # Check for cities first
    for city, country in city_to_country.items():
        if city in address_lower:
            return country
    
    # First, check for common country names (prioritize longer/more specific names first)
    # Sort by length descending to match "United Kingdom" before "Kingdom"
    country_names = sorted(COMMON_COUNTRIES.keys(), key=len, reverse=True)
    
    for country_name in country_names:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(country_name) + r'\b'
        if re.search(pattern, address_lower):
            # Get normalized country name
            normalized = COMMON_COUNTRIES[country_name]
            # Return properly capitalized country name
            if normalized == "united kingdom":
                return "United Kingdom"
            elif normalized == "united states":
                return "United States"
            elif normalized == "united arab emirates":
                return "United Arab Emirates"
            else:
                return normalized.title()
    
    # Check for country codes (usually at the end of address, after postal code)
    # Pattern: postal code + country code (e.g., "1234 AB NL" or "75001 FR")
    country_code_pattern = r'\b([a-z]{2})\b'
    matches = re.findall(country_code_pattern, address_lower)
    
    # Check matches against known country codes (check from end, country usually at end)
    code_to_country = {
        "at": "Austria", "be": "Belgium", "bg": "Bulgaria", "hr": "Croatia",
        "cy": "Cyprus", "cz": "Czech Republic", "dk": "Denmark", "ee": "Estonia",
        "fi": "Finland", "fr": "France", "de": "Germany", "gr": "Greece",
        "hu": "Hungary", "ie": "Ireland", "it": "Italy", "lv": "Latvia",
        "lt": "Lithuania", "lu": "Luxembourg", "mt": "Malta", "nl": "Netherlands",
        "pl": "Poland", "pt": "Portugal", "ro": "Romania", "sk": "Slovakia",
        "si": "Slovenia", "es": "Spain", "se": "Sweden",
        "gb": "United Kingdom", "uk": "United Kingdom", "us": "United States", "ch": "Switzerland", "eg": "Egypt"
    }
    
    for match in reversed(matches):
        if match in code_to_country:
            return code_to_country[match]
    
    return None

# -------------------- Dutch VAT Box Mapping --------------------
# This is deterministic domain logic (not LLM-driven). It maps a transaction to a Dutch VAT
# return box using country + goods/services + VAT rate + reverse charge signals.

DUTCH_VAT_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "1a": "Domestic sales taxed at 21%",
    "1b": "Domestic sales taxed at 9%",
    "1e": "Zero-rated exports (non-EU)",
    "2a": "Domestic reverse-charge sales",
    "3a": "Intra-EU supply of goods (B2B, 0%)",
    "3b": "Intra-EU supply of services (B2B, 0%)",
    "4a": "Intra-EU acquisition of goods",
    "4b": "Intra-EU acquisition of services",
    "5a": "Dutch input VAT on domestic purchases",
    "NO_TURNOVER_BOX": "Non-EU purchase of services (reverse charge; no turnover box)",
    "OUTSIDE_SCOPE": "Import of goods handled via customs (outside VAT turnover boxes)",
    "UNDETERMINED": "Unable to classify – missing or ambiguous data",
}

def _determine_dutch_vat_return_category(
    invoice_type: str,
    vendor_country: Optional[str],
    customer_country: Optional[str],
    vat_percentage: Optional[float],
    vat_amount: Optional[float],
    goods_services_indicator: Optional[str],
    reverse_charge_applied: bool,
    customer_vat_id: Optional[str],
    vendor_vat_id: Optional[str],
    vat_category: Optional[str],
) -> Tuple[str, str]:
    """
    Map an invoice to a Dutch VAT return box using  rule-based logic.

    Returns:
        (vat_box, reasoning)    
        vat_box ∈ {"1a","1b","1e","2a","3a","3b","4a","4b","5a","UNDETERMINED"}
    """
    try:
        inv_type = (invoice_type or "").strip().lower()
        supplier_country = vendor_country  # vendor is the supplier in our register
        customer_country_norm = customer_country
        
        log.debug(f"VAT mapping: type={inv_type}, vendor_country={vendor_country}, customer_country={customer_country}, vat_pct={vat_percentage}, goods_services={goods_services_indicator}")

        # Normalise helpers
        def _approx_rate(rate: Optional[float], target: float) -> bool:
            if rate is None:
                return False
            try:
                return abs(float(rate) - target) < 0.2
            except Exception:
                return False

        def _has_vat_number(v: Optional[str]) -> bool:
            return bool(str(v or "").strip())

        # SALES LOGIC ----------------------------------------------------------
        if inv_type == "sales":
            # Guard: we need a customer country
            if not customer_country:
                return "UNDETERMINED", "Sales invoice without customer country"

            is_cust_nl = _is_nl_country(customer_country)
            is_cust_eu = _is_eu_country(customer_country) and not is_cust_nl
            is_cust_non_eu = not _is_eu_country(customer_country) and not is_cust_nl
            rate_is_0 = _approx_rate(vat_percentage, 0.0)

            # 1a – Domestic sales taxed at 21%
            if is_cust_nl and _approx_rate(vat_percentage, 21.0) and not reverse_charge_applied:
                return "1a", "Sales to NL customer, 21% VAT, no reverse charge"

            # 1b – Domestic sales taxed at 9%
            if is_cust_nl and _approx_rate(vat_percentage, 9.0) and not reverse_charge_applied:
                return "1b", "Sales to NL customer, 9% VAT, no reverse charge"

            # 1e – Zero-rated exports (non‑EU)
            if is_cust_non_eu and rate_is_0:
                return "1e", "Sales to non‑EU customer with 0% VAT"

            # 2a – Domestic reverse‑charge sales
            if is_cust_nl and reverse_charge_applied:
                return "2a", "Sales to NL customer with reverse charge applied"

            # 3a / 3b – Intra‑EU B2B supplies
            if is_cust_eu:
                has_vat = _has_vat_number(customer_vat_id)
                indicator = (goods_services_indicator or "").lower()
                
                # Classify based on goods/services and rate, even if VAT number is missing
                if indicator == "goods" and rate_is_0:
                    if has_vat:
                        return "3a", "Intra‑EU supply of goods (B2B, 0%)"
                    else:
                        return "3a", "Intra‑EU supply of goods (B2B, 0%) - WARNING: Customer VAT number missing but transaction appears to be EU B2B"
                
                if indicator == "services" and rate_is_0:
                    if has_vat:
                        return "3b", "Intra‑EU supply of services (B2B, 0%)"
                    else:
                        return "3b", "Intra‑EU supply of services (B2B, 0%) - WARNING: Customer VAT number missing but transaction appears to be EU B2B"
                
                # EU but unclear goods/services or VAT rate
                if not has_vat:
                    return "UNDETERMINED", "EU sales but missing customer VAT number and unclear goods/services or rate"
                return "UNDETERMINED", "EU sales with VAT ID but unclear goods/services or rate"

            # Special edge case: NL customer, 0% VAT and no reverse‑charge → ambiguous
            if is_cust_nl and rate_is_0 and not reverse_charge_applied:
                return "UNDETERMINED", "Sales to NL customer with 0% VAT and no reverse charge"

            # Anything else on the sales side cannot be safely boxed
            return "UNDETERMINED", "Sales invoice does not meet any box criteria"

        # PURCHASE LOGIC -------------------------------------------------------
        if inv_type == "purchase":
            if not supplier_country:
                return "UNDETERMINED", "Purchase invoice without supplier country"

            is_suppl_nl = _is_nl_country(supplier_country)
            is_suppl_eu = _is_eu_country(supplier_country) and not is_suppl_nl
            indicator = (goods_services_indicator or "").lower()

            # 5a – Dutch input VAT on domestic purchases
            try:
                vat_amt_val = float(vat_amount or 0.0)
            except Exception:
                vat_amt_val = 0.0
            
            # Domestic purchases from NL suppliers (Box 5a)
            if is_suppl_nl:
                if vat_amt_val > 0.0:
                    return "5a", "Domestic purchase from NL supplier with input VAT"
                else:
                    # Domestic services/goods with 0% VAT (exempt or zero-rated)
                    # Still goes to Box 5a, but with 0 VAT amount
                    return "5a", "Domestic purchase from NL supplier with 0% VAT (exempt or zero-rated)"

            # 4a – Intra‑EU acquisition of goods (or non-EU goods via customs)
            if not is_suppl_nl and indicator == "goods":
                if is_suppl_eu:
                    return "4a", "Purchase of goods from EU supplier (non‑NL)"
                else:
                    # Non-EU goods handled via customs
                    return (
                        "OUTSIDE_SCOPE",
                        "Import of goods handled via customs (import VAT), not via Dutch VAT turnover boxes."
                    )

            # 4a – Non-EU service imports (Dynamic check, not hardcoded)
            # This must be checked BEFORE the general 4b logic to ensure Non-EU services
            # are correctly categorized as "Import of Services from Non-EU Country" in Box 4a
            # Uses _is_eu_country() helper to ensure future compliance if EU membership changes
            if not is_suppl_nl and indicator == "services":
                # Dynamic check: Is supplier country NOT an EU member?
                is_suppl_non_eu = not _is_eu_country(supplier_country)
                
                if is_suppl_non_eu:
                    # Non-EU service import - reverse charge applies (Article 196)
                    # Map to Box 4a as per requirement
                    return (
                        "4a",
                        f"Import of Services from Non-EU Country ({supplier_country}) - reverse charge (Article 196) - Box 4a"
                    )

            # 4b – Services from EU suppliers (non-NL) - reverse charge
            # According to Dutch VAT rules: "Diensten uit het buitenland aan u verricht"
            # This applies ONLY to EU (non-NL) services when reverse charge applies
            if not is_suppl_nl and indicator == "services":
                rate_is_0 = _approx_rate(vat_percentage, 0.0)
                if rate_is_0 or vat_amt_val == 0.0:
                    if is_suppl_eu:
                        return "4b", "Purchase of services from EU supplier (non‑NL) - reverse charge"

            # Everything else cannot be safely and legally inferred
            return "UNDETERMINED", "Purchase invoice could not be classified with available data"

        # Any other type → cannot classify
        return "UNDETERMINED", f"Unsupported invoice_type '{invoice_type}'"
    
    except Exception as e:
        log.error(f"VAT category determination error: {e}", exc_info=True)
        return "UNDETERMINED", f"Error during VAT classification: {str(e)}"
    
    # Final safety net - should never reach here, but ensures we always return a tuple
    log.error("VAT mapping function reached end without returning - this should never happen")
    return "UNDETERMINED", "Internal error: function did not return a value"

def _is_eu_country(country: Optional[str]) -> bool:
    """
    Returns True if the given country is an EU member, based on a
    normalised (stripped, lower-cased) country name or 2‑letter code.
    """
    if not country:
        return False
    return country.strip().lower() in EU_COUNTRIES

def _is_nl_country(country: Optional[str]) -> bool:
    """
   Returns True if the country represents the Netherlands (NL),
   with robust normalisation to tolerate extra spaces or variants.
    """
    if not country:
        return False
    country_lower = country.strip().lower()
    return (
        country_lower == "netherlands"
        or country_lower == "nl"
        or country_lower == "nederland"
        or country_lower == "holland"
    )

def _determine_goods_services_indicator(llm_data: Dict[str, Any], invoice_text: str = "") -> Optional[str]:
    # First check if LLM extracted it
    indicator = llm_data.get("goods_services_indicator")
    if indicator and indicator.lower() in ["goods", "services"]:
        return indicator.lower()
    
    # Fallback: analyze line items and description
    text_lower = (invoice_text or "").lower()
    description = ""
    if llm_data.get("line_items"):
        descriptions = [item.get("description", "") for item in llm_data.get("line_items", [])]
        description = " ".join(descriptions).lower()
    
    combined_text = (text_lower + " " + description).lower()
    
    # Goods indicators
    goods_keywords = [
        "product", "products", "item", "items", "goods", "merchandise",
        "physical", "tangible", "shipment", "delivery", "warehouse",
        "stock", "inventory", "material", "materials", "equipment",
        "hardware", "component", "parts", "supplies"
    ]
    
    # Services indicators
    services_keywords = [
        "service", "services", "consulting", "consultancy", "support",
        "maintenance", "repair", "installation", "training", "advice",
        "advisory", "software license", "licensing", "subscription",
        "professional", "expertise", "expert", "assistance", "help",
        "management", "administration", "processing", "handling"
    ]
    
    goods_count = sum(1 for keyword in goods_keywords if keyword in combined_text)
    services_count = sum(1 for keyword in services_keywords if keyword in combined_text)
    
    if goods_count > services_count and goods_count > 0:
        return "goods"
    elif services_count > goods_count and services_count > 0:
        return "services"
    
    # Default to services for B2B if unclear
    return "services"

def _determine_invoice_subcategory(
    invoice_type: str,
    vendor_address: Optional[str],
    customer_address: Optional[str],
    vat_percentage: Optional[float],
    invoice_text: str = ""
) -> str:
    """
    Determines the invoice subcategory based on:
    - Invoice type (Sales/Purchase)
    - Vendor and customer addresses (EU vs non-EU)
    - VAT rate (21%, 9%, etc.)
    """
    # Extract countries from addresses
    vendor_country = _extract_country_from_address(vendor_address) if vendor_address else None
    customer_country = _extract_country_from_address(customer_address) if customer_address else None
    
    # Check for import VAT indicators in invoice text
    import_vat_keywords = [
        "import vat", "importvat", "import btw", "importbtw",
        "invoer btw", "invoerbtw", "import tax", "customs"
    ]
    is_import_vat = False
    if invoice_text:
        text_lower = invoice_text.lower()
        is_import_vat = any(keyword in text_lower for keyword in import_vat_keywords)
    
    # Classify based on invoice type (case-safe comparison)
    if invoice_type.lower() == "sales":
        # For sales invoices, check customer country
        if customer_country:
            is_customer_eu = _is_eu_country(customer_country)
            
            # Check VAT rate for subcategory
            if vat_percentage is not None:
                vat_rate = float(vat_percentage)
                # Standard 21% rate
                if abs(vat_rate - 21.0) < 0.1 or (20.0 <= vat_rate <= 22.0):
                    if is_customer_eu:
                        return "Standard 21% - Sales to EU Countries"
                    else:
                        return "Standard 21% - Sales to Non-EU Countries"
                # Reduced 9% rate
                elif abs(vat_rate - 9.0) < 0.1 or (8.0 <= vat_rate <= 10.0):
                    if is_customer_eu:
                        return "Reduced Rate 9% - Sales to EU Countries"
                    else:
                        return "Reduced Rate 9% - Sales to Non-EU Countries"
                # Other rates
                else:
                    if is_customer_eu:
                        return f"VAT {vat_rate}% - Sales to EU Countries"
                    else:
                        return f"VAT {vat_rate}% - Sales to Non-EU Countries"
            else:
                # No VAT rate, classify by country only
                if is_customer_eu:
                    return "Sales to EU Countries"
                else:
                    return "Sales to Non-EU Countries"
        else:
            # No customer country found, use VAT rate if available
            if vat_percentage is not None:
                vat_rate = float(vat_percentage)
                if abs(vat_rate - 21.0) < 0.1 or (20.0 <= vat_rate <= 22.0):
                    return "Standard 21%"
                elif abs(vat_rate - 9.0) < 0.1 or (8.0 <= vat_rate <= 10.0):
                    return "Reduced Rate 9%"
                else:
                    return f"VAT {vat_rate}%"
            return "Sales - Country Unknown"
    
    elif invoice_type.lower() == "purchase":
        # For purchase invoices, check vendor country
        if vendor_country:
            is_vendor_eu = _is_eu_country(vendor_country)
            is_vendor_nl = _is_nl_country(vendor_country)
            
            if is_vendor_eu and not is_vendor_nl:
                return "Purchase from EU Countries"
            elif is_vendor_nl:
                return "Purchase from NL Countries"
            else:
                # Non-EU purchase - check if it's import VAT
                if is_import_vat or (vat_percentage is not None and vat_percentage > 0):
                    return "Purchase from Non-EU Countries (Import VAT)"
                else:
                    return "Purchase from Non-EU Countries"
        else:
            # No vendor country found, check for import VAT indicators
            if is_import_vat:
                return "Purchase from Non-EU Countries (Import VAT)"
            return "Purchase - Country Unknown"
    
    # Unclassified invoices
    return "Unclassified"



def _set_icp_fields_for_nl(register_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sets ICP (Intra-Community Procedure) reporting fields for Dutch VAT returns.
    
    ICP reporting is REQUIRED ONLY for:
    - Sales invoices (Type = "Sales")
    - To EU customers (non-NL)
    - With 0% VAT (Intra-EU B2B transactions)
    - With customer VAT number (B2B requirement)
    - VAT boxes 3a (goods) or 3b (services)
    
    ICP is NOT required for:
    - Sales with VAT charged (e.g., 21% VAT to EU customer = Box 1a, not ICP)
    - Sales to NL customers (domestic)
    - Sales to non-EU customers (exports)
    - Purchase invoices
    """
    # Defaults
    register_entry["ICP Return Required"] = "No"
    register_entry["ICP Reporting Category"] = ""

    invoice_type = register_entry.get("Type")
    if invoice_type != "Sales":
        return register_entry

    counterparty_country = register_entry.get("Customer Country")
    counterparty_vat_number = register_entry.get("Customer VAT ID")
    vat_box = register_entry.get("Dutch VAT Return Category", "")

    if not counterparty_country:
        return register_entry

    # Counterparty must be in EU but not Netherlands
    if not _is_eu_country(counterparty_country) or _is_nl_country(counterparty_country):
        return register_entry

    # B2B only – require a non‑empty VAT number
    if not counterparty_vat_number or not str(counterparty_vat_number).strip():
        return register_entry

    # CRITICAL FIX: ICP is only required for VAT boxes 3a (goods) and 3b (services)
    # These boxes represent Intra-EU B2B transactions with 0% VAT
    # If VAT box is 1a, 1b, 1e, 2a, or anything else, ICP is NOT required
    if vat_box == "3a":
        # Intra-EU supply of goods (B2B, 0% VAT)
        register_entry["ICP Return Required"] = "Yes"
        register_entry["ICP Reporting Category"] = "Intra-EU supply of goods (B2B)"
        return register_entry
    elif vat_box == "3b":
        # Intra-EU supply of services (B2B, 0% VAT, reverse charge)
        register_entry["ICP Return Required"] = "Yes"
        register_entry["ICP Reporting Category"] = "Intra-EU supply of services (B2B, reverse charge)"
        return register_entry
    
    # If VAT box is not 3a or 3b, ICP is NOT required
    # This handles cases like:
    # - Box 1a: Sales to EU customer with 21% VAT (domestic sales, not ICP)
    # - Box 1b: Sales to EU customer with 9% VAT (domestic sales, not ICP)
    # - Box 1e: Exports to non-EU (not ICP)
    # - Box 2a: Domestic reverse charge (not ICP)
    # - UNDETERMINED: Cannot determine, so no ICP

    return register_entry

def _derive_vat_rate_percent(llm_data: Dict[str, Any]) -> Optional[float]:
    # look in vat_breakdown
    for v in (llm_data.get("vat_breakdown") or []):
        r = v.get("rate")
        if isinstance(r, (int, float)):
            try:
                return float(r)
            except Exception:
                pass
    # fallback based on category
    vcat = (llm_data.get("vat_category") or "").strip().lower()
    if vcat in {"reverse-charge", "out-of-scope", "zero-rated"}:
        return 0.0
    return None

def _is_credit_note_simple(llm_data: Dict[str, Any], invoice_text: str) -> bool:
    """
    Detect whether the document is a credit note.
    logic:
    1. If LLM explicitly says document_type contains 'credit' → treat as credit note.
    2. Otherwise, look for common credit‑note keywords in the raw text.
    """
    doc_type = (llm_data.get("document_type") or "").strip().lower()
    if "credit" in doc_type:
        return True
                                                                                                                                        
    text = (invoice_text or "").lower()
    if not text:
        return False
                                                                                                                                                       
    keywords = [
        "credit note",
        "creditnote",
        "credit memo",
        "creditmemo",
        "crediteurnota",
        "avoir",
    ]
    return any(kw in text for kw in keywords)

def _map_llm_output_to_register_entry(llm_data: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Map the raw LLM schema into the register-entry shape used by the frontend and downstream logic.

    This is the main "boundary" between extraction output and domain logic.
    Key behaviors:
    - Sets amounts (and negates for credit notes)
    - Derives VAT rate and reverse charge flags
    - Attempts to infer countries from extracted addresses
    - Adds an extraction confidence signal for UI/review workflows
    """
    description = ""
    if llm_data.get("line_items"):
        raw_desc = llm_data["line_items"][0].get("description", "") or ""
        description = _translate_to_english_if_dutch(raw_desc)

    vat_percentage = _derive_vat_rate_percent(llm_data)
    invoice_text = llm_data.get("_invoice_text", "")
    is_credit = _is_credit_note_simple(llm_data, invoice_text)

    # Heuristic confidence score (includes validation errors if available)
    validation_errors = llm_data.get("_validation_errors", [])
    extraction_confidence_level, extraction_confidence_reason = _estimate_extraction_confidence(
        invoice_text, llm_data, validation_errors
    )
    
    goods_services_indicator = _determine_goods_services_indicator(llm_data, invoice_text)
    
    vat_category = (llm_data.get("vat_category") or "").strip().lower()
    invoice_text_lower = (invoice_text or "").lower()
    reverse_charge_keywords = [
        "reverse charge", "reverse-charge", "reversecharge",
        "btw verlegd", "btwverlegd", "vat verlegd", "vatverlegd",
        "omgekeerde heffing", "omgekeerdeheffing"
    ]
    reverse_charge_applied = (
        vat_category == "reverse-charge" or 
        any(keyword in invoice_text_lower for keyword in reverse_charge_keywords)
    )

    vendor_address = llm_data.get("vendor_address")
    customer_address = llm_data.get("customer_address")
    vendor_country = _extract_country_from_address(vendor_address) if vendor_address else None
    customer_country = _extract_country_from_address(customer_address) if customer_address else None
    
    reverse_charge_note = None
    if reverse_charge_applied and invoice_text:
        reverse_charge_patterns = [
            r"(?i)(?:reverse\s+charge|btw\s+verlegd|vat\s+verlegd|omgekeerde\s+heffing)[^.]*",
            r"(?i)vat[^.]*(?:verlegd|reverse)[^.]*",
        ]
        for pattern in reverse_charge_patterns:
            match = re.search(pattern, invoice_text)
            if match:
                reverse_charge_note = match.group(0).strip()
                break

    # DYNAMIC INSERTION: IBAN Fallback
    vendor_iban = llm_data.get("vendor_iban")
    if not vendor_iban:
        ibans = _extract_ibans_via_regex(invoice_text)
        if ibans:
            vendor_iban = ibans[0] # Best guess from regex
            log.info(f"LLM missed IBAN. Regex recovered: {vendor_iban}")

    # Core amounts (ensure credit notes are negative)
    nett_amount = float(q_money(llm_data.get("subtotal") or 0.0))
    vat_amount = float(q_money(llm_data.get("total_vat") or 0.0))
    gross_amount = float(q_money(llm_data.get("total_amount") or 0.0))

    if is_credit:
        if nett_amount > 0:
            nett_amount = -nett_amount
        if vat_amount > 0:
            vat_amount = -vat_amount
        if gross_amount > 0:
            gross_amount = -gross_amount

    return {
        "Date": llm_data.get("invoice_date"),
        "Invoice Number": llm_data.get("invoice_number"),
        "Type": "Unclassified",  # Will be set by _classify_type
        "Vendor Name": llm_data.get("vendor_name"),
        "Vendor VAT ID": llm_data.get("vendor_vat_id"),
        "Vendor Country": vendor_country,
        "Vendor Address": vendor_address,
        "Vendor IBAN": vendor_iban,  # Updated with regex fallback
        "Customer Name": llm_data.get("customer_name"),
        "Customer VAT ID": llm_data.get("customer_vat_id"),
        "Customer Country": customer_country,
        "Customer Address": customer_address,
        "Customer IBAN": llm_data.get("customer_iban"),
        "Description": description,
        "Nett Amount": nett_amount,
        "VAT %": vat_percentage,
        "VAT Amount": vat_amount,
        "Gross Amount": gross_amount,
        "Currency": (llm_data.get("currency") or "EUR"),
        "VAT Category": llm_data.get("vat_category"),
        "Reverse Charge Applied": reverse_charge_applied,
        "Reverse Charge Note": reverse_charge_note,
        "Goods Services Indicator": goods_services_indicator,
        "Subcategory": "Unclassified",
        "Dutch VAT Return Category": None,
        "ICP Return Required": "No",
        "ICP Reporting Category": "",
        "Extraction Confidence": extraction_confidence_level,
        "Extraction Confidence Reason": extraction_confidence_reason,
        "Due Date": llm_data.get("due_date"),
        "Payment Terms": llm_data.get("payment_terms"),
        "Notes": llm_data.get("notes"),
        "_filename": filename,  # Store filename for classification checks
        "Full_Extraction_Data": llm_data
    }

def _sanity_fix_vendor_customer(register_entry: Dict[str, Any], our_companies_list: List[str]) -> None:
    """
    Sanity check function. 
    Previously used to swap fields, but now disabled to avoid aggressive flipping.
    """
    pass 

def _check_foreign_iban(register_entry: Dict[str, Any], our_companies_list: List[str]) -> bool:
    """
    Returns True if a Non-NL IBAN is found in the raw text.
    This is the 'Nuclear Option' for catching Import invoices misclassified as Sales.
    """
    # Only apply if we are actually a Dutch company
    our_norms = [_normalize_company_name(x) for x in our_companies_list if x]
    is_we_dutch = any("dutch" in x or "nl" in x for x in our_norms)
    if not is_we_dutch:
        return False

    raw_text = register_entry.get("Full_Extraction_Data", {}).get("_invoice_text", "")
    found_ibans = _extract_ibans_via_regex(raw_text)
    
    for iban in found_ibans:
        clean = re.sub(r'[^A-Z0-9]', '', iban.upper())
        if len(clean) < 15:
            continue
        country = clean[:2]
        # Ignore NL IBANs. Ignore potential false positives (too short).
        if country != 'NL' and country.isalpha():
            log.info(f"Foreign IBAN detected: {clean} ({country})")
            return True
    return False

def _check_filename_context(register_entry: Dict[str, Any], our_companies_list: List[str]) -> Optional[str]:
    """
    Checks filename for patterns indicating we are the issuer (Sales invoice).
    Examples: "Invoice DFS- 002-2025" or "Invoice Dutch Food Solutions"
    Returns 'Sales' if filename suggests we issued the invoice, None otherwise.
    
    IMPORTANT: Only returns Sales if we can find our company name/initials in the filename.
    Just having "Sales" in the filename is NOT enough - that could be someone else's sales invoice to us.
    """
    filename = register_entry.get("_filename", "") or register_entry.get("Full_Extraction_Data", {}).get("_filename", "")
    if not filename:
        return None
    
    filename_lower = filename.lower()
    
    # Check if filename starts with "invoice" followed by our company name/initials
    for company in our_companies_list:
        comp_norm = _normalize_company_name(company)
        comp_words = comp_norm.split()
        
        # Extract initials (first letters of each word)
        initials = "".join([w[0] for w in comp_words if w])
        
        # Pattern 1: "Invoice [Company Name]" or "Invoice [Company Initials]"
        # Example: "Invoice DFS- 002-2025" or "Invoice Dutch Food Solutions"
        if filename_lower.startswith("invoice") or "factuur" in filename_lower:
            # Check if company name or initials appear early in filename
            filename_early = filename_lower[:100]  # First 100 chars
            if comp_norm in filename_early or (initials and initials.lower() in filename_early):
                log.info(f"Filename indicates Sales: '{filename}' contains our company -> Sales")
                return "Sales"
            
            # Also check for common patterns like "Invoice DFS-" or "Invoice DFS "
            if initials and len(initials) >= 2:
                initials_pattern = f"invoice {initials.lower()}"
                if initials_pattern in filename_early:
                    log.info(f"Filename indicates Sales: '{filename}' contains our initials '{initials}' -> Sales")
                    return "Sales"
                # Also check for "Factuur" (Dutch for Invoice)
                factuur_pattern = f"factuur {initials.lower()}"
                if factuur_pattern in filename_early:
                    log.info(f"Filename indicates Sales: '{filename}' contains our initials '{initials}' after 'factuur' -> Sales")
                    return "Sales"
    
    # If filename contains "sales" but NOT our company, this is suspicious - could be someone else's sales invoice
    # Don't trust it as a Sales indicator
    if "sales" in filename_lower:
        found_our_company = False
        for company in our_companies_list:
            comp_norm = _normalize_company_name(company)
            if comp_norm in filename_lower:
                found_our_company = True
                break
        if not found_our_company:
            log.info(f"Filename contains 'sales' but not our company - ignoring as false positive: '{filename}'")
    
    return None

def _check_document_issuer_signal(register_entry: Dict[str, Any], our_companies_list: List[str]) -> Optional[str]:
    """
    Checks if document number or filename contains our company name/initials.
    This is a STRONG signal that we issued the document (Sales).
    Examples:
    - Document Number: "DFS 25002104770" -> We issued it (Sales)
    - Filename: "Debit Note - DFS 25002104770" -> We issued it (Sales)
    Returns 'Sales' if strong signal found, None otherwise.
    """
    # Check document number first (strongest signal)
    doc_number = register_entry.get("Document Number") or ""
    if doc_number:
        doc_lower = doc_number.lower()
        for company in our_companies_list:
            comp_norm = _normalize_company_name(company).lower()
            comp_words = comp_norm.split()
            initials = "".join([w[0] for w in comp_words if w]).lower()
            
            # Check if company name or initials appear in document number
            if comp_norm in doc_lower or (initials and len(initials) >= 2 and initials in doc_lower):
                log.info(f"Document Number '{doc_number}' contains our company/initials -> Strong Sales signal")
                return "Sales"
    
    # Check filename as secondary signal
    filename = register_entry.get("_filename", "") or register_entry.get("Full_Extraction_Data", {}).get("_filename", "")
    if filename:
        filename_lower = filename.lower()
        for company in our_companies_list:
            comp_norm = _normalize_company_name(company).lower()
            comp_words = comp_norm.split()
            initials = "".join([w[0] for w in comp_words if w]).lower()
            
            # Check for patterns like "Debit Note - DFS" or "Invoice DFS-"
            # Look in first 150 chars of filename
            filename_early = filename_lower[:150]
            if comp_norm in filename_early or (initials and len(initials) >= 2 and initials in filename_early):
                # Additional check: filename should contain invoice-like keywords
                invoice_keywords = ["invoice", "factuur", "debit", "credit", "note", "nota"]
                if any(kw in filename_early for kw in invoice_keywords):
                    log.info(f"Filename '{filename}' contains our company/initials with invoice keywords -> Strong Sales signal")
                    return "Sales"
    
    return None

def _check_keyword_context(register_entry: Dict[str, Any], our_companies_list: List[str]) -> Optional[str]:
    """
    Infer Sales vs Purchase by scanning the raw OCR/PDF text for *role-labelled* mentions
    of our company name.

    Why this exists:
    - Vendor/Customer fields can be swapped or noisy depending on OCR/LLM extraction.
    - A naive "character window" around the company name is brittle because OCR often
      interleaves unrelated blocks (e.g., "Bill To" text appears close to "Supplier" block).

    Approach:
    - Split text into lines, normalize both labels and company names into a simplified form
      (lowercase, punctuation->spaces, whitespace collapsed).
    - Count role signals only when a role label is on the *same line* as our company
      OR the label is immediately above the line that contains our company (common layout).

    Returns:
    - "Sales" if we are strongly identified as Supplier/Seller/Vendor/Issuer
    - "Purchase" if we are strongly identified as Buyer/Customer/Bill To/Ship To/Invoice To
    - None if no strong signal is found
    """
    raw_text = register_entry.get("Full_Extraction_Data", {}).get("_invoice_text", "")
    if not raw_text:
        return None

    def _simplify_for_roles(s: str) -> str:
        s = (s or "").casefold()
        # Replace punctuation with spaces but keep line structure outside this function.
        s = re.sub(r"[\t\r\f\v]+", " ", s)
        s = re.sub(r"[\\/_.,()\[\]{}'\"&:;|<>]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Role labels (already "simplified" form: lowercase + spaces)
    customer_labels = [
        "buyer",
        "customer",
        "bill to",
        "invoice to",
        "ship to",
        "deliver to",
        "consignee",
        "debtor",
        "client",
        "afnemer",
        "klant",
    ]
    vendor_labels = [
        "supplier",
        "seller",
        "vendor",
        "issuer",
        "invoice from",
        "bill from",
        "remit to",
        "remittance",
        "leverancier",
        "verkoper",
    ]

    # Prepare normalized lines; ignore very short/noisy lines
    raw_lines = raw_text.splitlines()
    lines = []
    for ln in raw_lines:
        simp = _simplify_for_roles(ln)
        if simp and len(simp) >= 3:
            lines.append(simp)

    if not lines:
        return None

    def _label_near_company(line: str, label: str, comp: str, max_gap: int = 40) -> bool:
        """True if label appears before company on same line within a small distance."""
        if label not in line or comp not in line:
            return False
        # Check proximity using first occurrence positions (good enough for scoring)
        li = line.find(label)
        ci = line.find(comp)
        if li == -1 or ci == -1:
            return False
        # Prevent false positives like "... Acme ... from ..."
        if li > ci:
            return False
        return (ci - li) <= max_gap

    def _prev_line_has_label(prev_line: str, labels: List[str]) -> bool:
        """
        Stricter than substring search to reduce OCR noise:
        - match if line equals the label ("bill to")
        - or starts with the label ("bill to address", "supplier details")
        """
        if not prev_line:
            return False
        for lab in labels:
            if prev_line == lab:
                return True
            if prev_line.startswith(lab + " "):
                return True
        return False

    sales_signals = 0
    purchase_signals = 0

    for company in our_companies_list:
        comp_norm = _normalize_company_name(company)
        if not comp_norm:
            continue

        # Look for the normalized company name in the simplified lines
        for i, line in enumerate(lines):
            if comp_norm not in line:
                continue

            prev1 = lines[i - 1] if i - 1 >= 0 else ""
            prev2 = lines[i - 2] if i - 2 >= 0 else ""

            # Strong signal: label and company on same line, and close together
            if any(_label_near_company(line, lab, comp_norm) for lab in vendor_labels):
                sales_signals += 2
            if any(_label_near_company(line, lab, comp_norm) for lab in customer_labels):
                purchase_signals += 2

            # Medium signal: label on the line above, company on this line
            if prev1:
                if _prev_line_has_label(prev1, vendor_labels):
                    # Common layout: "Bill To:" / "Supplier:" on its own line, party name on next line
                    sales_signals += 2
                if _prev_line_has_label(prev1, customer_labels):
                    purchase_signals += 2
            if prev2:
                if _prev_line_has_label(prev2, vendor_labels):
                    sales_signals += 1
                if _prev_line_has_label(prev2, customer_labels):
                    purchase_signals += 1

    # Require a margin to avoid flipping on weak/noisy evidence
    if purchase_signals >= sales_signals + 2:
        log.info(f"Deep Context (role lines): Found {purchase_signals} Purchase vs {sales_signals} Sales -> Purchase")
        return "Purchase"
    if sales_signals >= purchase_signals + 2:
        log.info(f"Deep Context (role lines): Found {sales_signals} Sales vs {purchase_signals} Purchase -> Sales")
        return "Sales"

    return None

def _classify_type(register_entry: Dict[str, Any], our_companies_list: List[str]) -> str:
    """
    Decide whether an invoice is a Sales invoice (we issued it) or a Purchase invoice (we received it).

    Classification signals (rough priority order):
    - Strong issuer signals from document number / filename (if they contain our name/initials)
    - Vendor/customer name match vs our company list (robust token-set similarity)
    - Keyword context around mentions of our company ("bill to", "supplier", etc.)
    - Country-based fallback (NL vs non-NL)

    NOTE:
    - This logic depends on the LLM extracting vendor/customer fields correctly.
      If extraction is missing/incorrect, we may still return Unclassified.
    """
    """
    Decides Sales vs Purchase.
    PRIORITY: Position-based logic FIRST, then keyword context as tie-breaker.
    """
    vendor_name = register_entry.get("Vendor Name") or ""
    customer_name = register_entry.get("Customer Name") or ""

    # Score-based matching (more accurate than a single boolean threshold).
    # This reduces false flips when both names partially match (or both don't match well).
    def best_us_score(target_raw: str) -> Tuple[float, Optional[str]]:
        best = 0.0
        best_our = None
        for our_raw in our_companies_list:
            score = _calculate_name_similarity_robust(our_raw, target_raw)
            if score > best:
                best = score
                best_our = our_raw
        return best, best_our

    vendor_score, vendor_best = best_us_score(vendor_name)
    customer_score, customer_best = best_us_score(customer_name)

    # Keep the previous boolean interpretation, but derived from scores.
    us_is_vendor = vendor_score >= 0.67
    us_is_customer = customer_score >= 0.67

    # Attach debug info for troubleshooting (kept internal under Full_Extraction_Data).
    try:
        fed = register_entry.get("Full_Extraction_Data")
        if isinstance(fed, dict):
            fed["_type_debug"] = {
                "vendor_name": vendor_name,
                "customer_name": customer_name,
                "vendor_score": vendor_score,
                "customer_score": customer_score,
                "vendor_best_match": vendor_best,
                "customer_best_match": customer_best,
            }
    except Exception:
        pass

    # 0. DOCUMENT ISSUER SIGNAL (STRONGEST - Overrides position if document number/filename indicates we issued it)
    # This catches cases where LLM swapped vendor/customer but document number shows we issued it
    issuer_signal = _check_document_issuer_signal(register_entry, our_companies_list)
    if issuer_signal == "Sales":
        if isinstance(register_entry.get("Full_Extraction_Data"), dict):
            register_entry["Full_Extraction_Data"]["_type_debug"]["issuer_signal"] = "Sales"
        # If document number/filename shows we issued it, but position says we're customer -> likely field swap
        if us_is_customer and not us_is_vendor:
            log.warning(f"Document number/filename indicates Sales (we issued it), but position shows us as Customer -> Likely field swap. Forcing Sales.")
            return "Sales"
        # If we're already vendor, this confirms it
        if us_is_vendor:
            return "Sales"

    # If issuer signal didn't decide it, use score margins before the old position heuristics.
    # This reduces wrong labels when the LLM swapped roles or when one side matches us much better.
    context_type: Optional[str] = None
    score_gap = abs(vendor_score - customer_score)

    def _set_debug(k: str, v: Any) -> None:
        fed = register_entry.get("Full_Extraction_Data")
        if isinstance(fed, dict) and isinstance(fed.get("_type_debug"), dict):
            fed["_type_debug"][k] = v

    _set_debug("score_gap", score_gap)

    # Strong wins
    if vendor_score >= 0.80 and customer_score <= 0.50:
        context_type = _check_keyword_context(register_entry, our_companies_list)
        _set_debug("context_type", context_type)
        # Only override strong vendor match if context is a strong opposite signal and vendor isn't near-certain.
        if context_type == "Purchase" and vendor_score < 0.95:
            _set_debug("decision", "Purchase (context overrides strong vendor match)")
            return "Purchase"
        _set_debug("decision", "Sales (strong vendor match)")
        return "Sales"

    if customer_score >= 0.80 and vendor_score <= 0.50:
        context_type = _check_keyword_context(register_entry, our_companies_list)
        _set_debug("context_type", context_type)
        if context_type == "Sales" and customer_score < 0.95:
            _set_debug("decision", "Sales (context overrides strong customer match)")
            return "Sales"
        _set_debug("decision", "Purchase (strong customer match)")
        return "Purchase"

    # Moderate wins if there is a clear gap and at least one side is reasonably confident.
    if score_gap >= 0.20 and max(vendor_score, customer_score) >= 0.70:
        chosen = "Sales" if vendor_score > customer_score else "Purchase"
        _set_debug("decision", f"{chosen} (score gap)")
        return chosen

    # 1. POSITION LOGIC (PRIMARY - Most reliable when LLM extracts correctly)
    if us_is_vendor and not us_is_customer:
        # We are clearly the vendor -> Sales invoice
        # Cross-check against role-labelled raw-text context to catch vendor/customer swaps.
        # This is intentionally strict: _check_keyword_context() only returns a result when
        # it finds a strong label+our-company pairing, reducing false flips.
        recheck_context = _check_keyword_context(register_entry, our_companies_list)
        if recheck_context == "Purchase":
            log.info("Position says Sales (we are Vendor), but role-labelled text context says Purchase -> Purchase (likely field swap)")
            return "Purchase"
        
        # Foreign IBAN override: If we're vendor but there's a foreign IBAN, it's likely an import (Purchase)
        if _check_foreign_iban(register_entry, our_companies_list):
            return "Purchase"
        return "Sales"

    if us_is_customer and not us_is_vendor:
        # We are clearly the customer -> Purchase invoice
        # BUT: Check if document number suggests we issued it (strong override for field swaps)
        if issuer_signal == "Sales":
            log.warning(f"Position says Purchase (we are customer), but document number/filename indicates we issued it -> Likely field swap. Forcing Sales.")
            return "Sales"
        # Cross-check against role-labelled raw-text context to catch vendor/customer swaps.
        recheck_context = _check_keyword_context(register_entry, our_companies_list)
        if recheck_context == "Sales":
            log.info("Position says Purchase (we are Customer), but role-labelled text context says Sales -> Sales (likely field swap)")
            return "Sales"
        return "Purchase"

    # 2. AMBIGUOUS CASES (Both match or neither match) - Use keyword context
    if us_is_vendor and us_is_customer:
        # Both positions match - use keyword context to decide
        context_type = _check_keyword_context(register_entry, our_companies_list)
        _set_debug("context_type", context_type)
        if context_type:
            _set_debug("decision", f"{context_type} (both match; context)")
            return context_type
        # Default: If we're vendor, it's more likely Sales
        _set_debug("decision", "Sales (both match; default)")
        return "Sales"

    # 3. NEITHER MATCHES - Use keyword context and fallback logic
    context_type = _check_keyword_context(register_entry, our_companies_list)
    _set_debug("context_type", context_type)
    if context_type:
        _set_debug("decision", f"{context_type} (neither match; context)")
        return context_type

    # 4. FALLBACK LOGIC (Country-based)
    vendor_address = register_entry.get("Vendor Address") or ""
    customer_address = register_entry.get("Customer Address") or ""
    v_country = _extract_country_from_address(vendor_address)
    c_country = _extract_country_from_address(customer_address)

    if v_country and c_country:
        is_v_nl = _is_nl_country(v_country)
        is_c_nl = _is_nl_country(c_country)
        
        if is_v_nl and not is_c_nl:
            return "Sales"
        if is_c_nl and not is_v_nl:
            return "Purchase"
        if is_v_nl and is_c_nl:
            # Both NL - can't determine from country alone, return Unclassified
            return "Unclassified"

    return "Unclassified"

def _enforce_role_consistency(register_entry: Dict[str, Any], our_companies_list: List[str]):
    """
    If Type=Sales, We MUST be Vendor.
    If Type=Purchase, We MUST be Customer.
    Swap fields if this is violated.
    """
    invoice_type = register_entry.get("Type")
    if invoice_type not in ("Sales", "Purchase"):
        return

    vendor_name = register_entry.get("Vendor Name") or ""
    customer_name = register_entry.get("Customer Name") or ""
    
    def is_us(target_raw):
        best = 0.0
        for our_raw in our_companies_list:
            score = _calculate_name_similarity_robust(our_raw, target_raw)
            best = max(best, score)
        return best >= 0.67

    us_is_vendor = is_us(vendor_name)
    us_is_customer = is_us(customer_name)

    # If Sales, we must be Vendor. If we are Customer instead, SWAP.
    if invoice_type == "Sales" and us_is_customer and not us_is_vendor:
        log.warning("Role Fix: Type is Sales but we are Customer. Swapping.")
        _swap_vendor_customer(register_entry)

    # If Purchase, we must be Customer. If we are Vendor instead, SWAP.
    if invoice_type == "Purchase" and us_is_vendor and not us_is_customer:
        log.warning("Role Fix: Type is Purchase but we are Vendor. Swapping.")
        _swap_vendor_customer(register_entry)

def _swap_vendor_customer(entry: Dict[str, Any]):
    """Swap all vendor and customer fields."""
    entry["Vendor Name"], entry["Customer Name"] = entry["Customer Name"], entry["Vendor Name"]
    entry["Vendor VAT ID"], entry["Customer VAT ID"] = entry["Customer VAT ID"], entry["Vendor VAT ID"]
    entry["Vendor Address"], entry["Customer Address"] = entry["Customer Address"], entry["Vendor Address"]
    entry["Vendor IBAN"], entry["Customer IBAN"] = entry["Customer IBAN"], entry["Vendor IBAN"]
    entry["Vendor Country"], entry["Customer Country"] = entry["Customer Country"], entry["Vendor Country"]

def _classify_and_set_subcategory(register_entry: Dict[str, Any], our_companies_list: List[str]) -> Dict[str, Any]:
    """
     -End-to-end enrichment step for the register entry:
    - Classify invoice type (Sales/Purchase/Unclassified)
    - Enforce role consistency (swap vendor/customer fields if needed)
    - Determine subcategory + Dutch VAT box + ICP fields
    - Attach VAT reasoning for auditability
    """
    # 1. Classify
    invoice_type = _classify_type(register_entry, our_companies_list)
    register_entry["Type"] = invoice_type
    
    # 2. Enforce Consistency (Swap if needed)
    _enforce_role_consistency(register_entry, our_companies_list)
    
    # Update type variable after potential swaps/logic
    invoice_type = register_entry["Type"]
    
    # NORMALIZE once – enforce canonical values (Sales / Purchase / Unclassified)
    if invoice_type:
        invoice_type = invoice_type.capitalize()  # Sales / Purchase / Unclassified
        register_entry["Type"] = invoice_type
    
    # Defensive assertion to catch future regressions
    assert register_entry["Type"] in ("Sales", "Purchase", "Unclassified"), \
        f"Invalid invoice type: {register_entry['Type']}"
    
    # 3. Fallback inference – ONLY if still Unclassified
    # Do NOT override if already classified as Sales or Purchase
    if register_entry["Type"] == "Unclassified":
        vendor_address = register_entry.get("Vendor Address") or ""
        customer_address = register_entry.get("Customer Address") or ""
        vendor_country = _extract_country_from_address(vendor_address)
        customer_country = _extract_country_from_address(customer_address)
        
        if vendor_country and customer_country:
            is_vendor_nl = _is_nl_country(vendor_country)
            is_customer_nl = _is_nl_country(customer_country)
            
            if is_vendor_nl and not is_customer_nl:
                invoice_type = "Sales"
            elif is_customer_nl and not is_vendor_nl:
                invoice_type = "Purchase"
            # If both NL, do NOT guess (this was flipping many domestic Sales invoices).
            # Leave as Unclassified and let name/context signals drive the decision.
            
            # Normalize fallback result
            if invoice_type:
                invoice_type = invoice_type.capitalize()
            register_entry["Type"] = invoice_type
    
    # Ensure invoice_type variable is in sync with register_entry for downstream use
    invoice_type = register_entry["Type"]
    
    # 3. Determine Subcategory
    vendor_address = register_entry.get("Vendor Address")
    customer_address = register_entry.get("Customer Address")
    vat_percentage = register_entry.get("VAT %")
    full_data = register_entry.get("Full_Extraction_Data", {})
    invoice_text = full_data.get("_invoice_text", "") if isinstance(full_data, dict) else ""
    
    subcategory = _determine_invoice_subcategory(
        invoice_type=invoice_type,
        vendor_address=vendor_address,
        customer_address=customer_address,
        vat_percentage=vat_percentage,
        invoice_text=invoice_text
    )
    register_entry["Subcategory"] = subcategory
    
    # 4. Determine Dutch VAT Return Category
    vendor_country = register_entry.get("Vendor Country")
    customer_country = register_entry.get("Customer Country")
    if not vendor_country and vendor_address:
        vendor_country = _extract_country_from_address(vendor_address)
        if vendor_country:
            register_entry["Vendor Country"] = vendor_country  # Save back to register_entry
            log.debug(f"Extracted vendor country '{vendor_country}' from address: {vendor_address}")
    if not customer_country and customer_address:
        customer_country = _extract_country_from_address(customer_address)
        if customer_country:
            register_entry["Customer Country"] = customer_country  # Save back to register_entry
            log.debug(f"Extracted customer country '{customer_country}' from address: {customer_address}")
        else:
            log.warning(f"Could not extract customer country from address: {customer_address}")
    
    vat_amount = register_entry.get("VAT Amount")
    goods_services_indicator = register_entry.get("Goods Services Indicator")
    reverse_charge_applied = register_entry.get("Reverse Charge Applied", False)
    vat_category = register_entry.get("VAT Category")
    customer_vat_id = register_entry.get("Customer VAT ID")
    vendor_vat_id = register_entry.get("Vendor VAT ID")
    
    # Automatic reverse charge detection based on transaction characteristics
    # Reverse charge applies when:
    # - B2B cross-border services (supplier/customer from different countries)
    # - 0% VAT charged (VAT shifts to customer)
    # - Customer has VAT number (B2B transaction)
    if not reverse_charge_applied:  # Only detect if not already set by keywords
        is_services = (goods_services_indicator or "").lower() == "services"
        vat_rate_is_zero = vat_percentage is not None and abs(float(vat_percentage)) < 0.01
        vat_amount_is_zero = vat_amount is None or abs(float(vat_amount or 0.0)) < 0.01
        has_customer_vat = bool(customer_vat_id and str(customer_vat_id).strip())
        
        if invoice_type.lower() == "purchase":
            # Purchase: Supplier from outside Netherlands, services, 0% VAT, B2B
            is_supplier_nl = _is_nl_country(vendor_country) if vendor_country else False
            if (not is_supplier_nl and vendor_country and 
                is_services and 
                (vat_rate_is_zero or vat_amount_is_zero) and
                has_customer_vat):
                reverse_charge_applied = True
                if not register_entry.get("Reverse Charge Note"):
                    register_entry["Reverse Charge Note"] = (
                        f"Reverse charge applies: B2B cross-border service from {vendor_country}. "
                        f"VAT must be self-accounted in Netherlands."
                    )
        
        elif invoice_type.lower() == "sales":
            # Sales: Customer from outside Netherlands, services, 0% VAT, B2B
            is_customer_nl = _is_nl_country(customer_country) if customer_country else False
            if (not is_customer_nl and customer_country and
                is_services and
                (vat_rate_is_zero or vat_amount_is_zero) and
                has_customer_vat):
                reverse_charge_applied = True
                if not register_entry.get("Reverse Charge Note"):
                    register_entry["Reverse Charge Note"] = (
                        f"Reverse charge applies: B2B cross-border service to {customer_country}. "
                        f"Customer must self-account VAT in their country."
                    )
    
    # Update register_entry with detected reverse charge
    register_entry["Reverse Charge Applied"] = reverse_charge_applied
    
    # Defensive call – ensure we always get a (box, reason) tuple
    try:
        result = _determine_dutch_vat_return_category(
        invoice_type=invoice_type,
        vendor_country=vendor_country,
        customer_country=customer_country,
        vat_percentage=vat_percentage,
        vat_amount=vat_amount,
        goods_services_indicator=goods_services_indicator,
        reverse_charge_applied=reverse_charge_applied,
        customer_vat_id=customer_vat_id,
        vendor_vat_id=vendor_vat_id,
        vat_category=vat_category,
    )
        log.debug(f"VAT mapping result type: {type(result)}, value: {result}")
        if isinstance(result, tuple) and len(result) == 2:
            vat_box, vat_reason = result
        else:
            log.error(f"VAT mapping returned invalid result: type={type(result)}, value={result}")
            vat_box, vat_reason = "UNDETERMINED", f"Internal error: VAT mapping returned invalid result (type={type(result).__name__}, value={result})"
    except Exception as e:
        log.error(f"Dutch VAT mapping failed: {e}")
        vat_box, vat_reason = "UNDETERMINED", f"Internal error during VAT mapping: {e}"
    register_entry["Dutch VAT Return Category"] = vat_box or ""
    
    # Set description dynamically for Box 4a based on invoice type
    if vat_box == "4a":
        # Box 4a can be either EU goods or non-EU services - determine which one
        goods_services = (register_entry.get("Goods Services Indicator") or "").lower()
        vendor_country = register_entry.get("Vendor Country")
        is_vendor_eu = _is_eu_country(vendor_country) if vendor_country else False
        
        if goods_services == "services" and not is_vendor_eu:
            # Non-EU purchase of services – reverse charge import of services (Box 4a)
            register_entry["Dutch VAT Return Category Description"] = "Import of Services from Non-EU Country"
            register_entry["Internal Tax Category"] = "IMPORT_SERVICE_NON_EU"
        else:
            # EU acquisition of goods (Box 4a)
            register_entry["Dutch VAT Return Category Description"] = "Intra-EU acquisition of goods"
            register_entry["Internal Tax Category"] = "EU_ACQUISITION_GOODS"
    else:
        # For all other categories, use the standard description
        register_entry["Dutch VAT Return Category Description"] = DUTCH_VAT_CATEGORY_DESCRIPTIONS.get(
            vat_box, ""
        )
    
    register_entry["Dutch VAT Return Category Reason"] = vat_reason

    # Internal tax category flags for downstream posting / VAT calculation
    if vat_box == "NO_TURNOVER_BOX":
        # Non‑EU purchase of services – reverse charge import of services
        register_entry["Internal Tax Category"] = "NON_EU_SERVICE_REVERSE_CHARGE"
    elif vat_box == "OUTSIDE_SCOPE":
        # Non‑EU purchase of goods – import via customs
        register_entry["Internal Tax Category"] = "NON_EU_GOODS_IMPORT"

    # 5. ICP
    register_entry = _set_icp_fields_for_nl(register_entry)
    
    return register_entry

def _convert_to_eur_fields(entry: dict, conversion_enabled: bool = True) -> dict:
    """
    Convert amounts from invoice currency to EUR.

    - Uses exchangerate.host (ECB reference) with date lookback.
    - Writes EUR fields alongside original currency fields.
    - Does not raise for FX errors; it annotates the entry with `FX Error`.
    """
    if not conversion_enabled:
        entry["FX Rate (ccy->EUR)"] = None
        entry["Nett Amount (EUR)"] = None
        entry["VAT Amount (EUR)"] = None
        entry["Gross Amount (EUR)"] = None
        entry["FX Conversion Note"] = "Currency conversion disabled"
        return entry
    try:
        ccy = (entry.get("Currency") or "").upper().strip()
        inv_date_str = entry.get("Date")

        if not inv_date_str or not ccy:
            entry["FX Rate (ccy->EUR)"] = None
            entry["Nett Amount (EUR)"] = None
            entry["VAT Amount (EUR)"] = None
            entry["Gross Amount (EUR)"] = None
            entry["FX Error"] = "Missing date or currency for conversion"
            return entry

        if ccy == "EUR":
            entry["FX Rate (ccy->EUR)"] = "1.0000"
            entry["FX Rate Date"] = inv_date_str
            entry["Nett Amount (EUR)"] = round(float(entry.get("Nett Amount", 0) or 0), 2)
            entry["VAT Amount (EUR)"] = round(float(entry.get("VAT Amount", 0) or 0), 2)
            entry["Gross Amount (EUR)"] = round(float(entry.get("Gross Amount", 0) or 0), 2)
            return entry

        inv_dt = date.fromisoformat(inv_date_str)
        rate, used_date = get_eur_rate(inv_dt, ccy)
        entry["FX Rate (ccy->EUR)"] = str(rate)
        entry["FX Rate Date"] = used_date

        for k_src, k_dst in [
            ("Nett Amount", "Nett Amount (EUR)"),
            ("VAT Amount",  "VAT Amount (EUR)"),
            ("Gross Amount","Gross Amount (EUR)")
        ]:
            amt = Decimal(str(entry.get(k_src, 0) or 0))
            converted = q_money(amt * rate)
            entry[k_dst] = round(float(converted), 2)

        return entry
    except Exception as ex:
        log.error(f"EUR conversion failed for entry: {ex}")
        entry["FX Rate (ccy->EUR)"] = None
        entry["Nett Amount (EUR)"] = None
        entry["VAT Amount (EUR)"] = None
        entry["Gross Amount (EUR)"] = None
        entry["FX Error"] = f"EUR conversion failed: {str(ex)}"
        return entry
# -------------------- Main pipeline --------------------
# `robust_invoice_processor` is the extraction "entrypoint" used by `app.py`.
# It now returns a **state object** that captures stage outputs, errors, and guard results.
# This makes extraction deterministic, debuggable, and prevents downstream logic from running
# when extraction quality is weak.

# Guard defaults (tuneable, but deterministic)
MIN_OCR_TEXT_CHARS = 200
MAX_RAW_TEXT_STORE_CHARS = 20000  # stored for debugging/traceability
# If validation finds multiple hard errors, do not proceed (prevents wrong classification).
MAX_VALIDATION_ERRORS_FOR_OK = 1

def _init_extraction_state(filename: str) -> Dict[str, Any]:
    return {
        "status": "in_progress",          # in_progress | ok | partial | failed
        "filename": filename,
        "stage": None,                    # ocr | llm | validation
        "errors": [],                     # list[str]
        "warnings": [],                   # list[str]
        "guards": {},                     # stage->guard results
        "outputs": {                      # stage outputs
            "ocr_text": None,
            "reduced_text": None,
            "llm_data": None,
            "validation_errors": None,
        },
    }

def _guard_min_ocr_text(state: Dict[str, Any]) -> bool:
    text = (state["outputs"].get("ocr_text") or "").strip()
    ok = len(text) >= MIN_OCR_TEXT_CHARS
    state["guards"]["min_ocr_text_chars"] = {"ok": ok, "len": len(text), "min": MIN_OCR_TEXT_CHARS}
    if not ok:
        state["errors"].append(f"OCR text too short ({len(text)} chars). Minimum required is {MIN_OCR_TEXT_CHARS}.")
    return ok

def _guard_nonempty_llm_output(state: Dict[str, Any]) -> bool:
    llm_data = state["outputs"].get("llm_data")
    ok = isinstance(llm_data, dict) and bool(llm_data)
    state["guards"]["nonempty_llm_output"] = {"ok": ok}
    if not ok:
        state["errors"].append("LLM output is empty or invalid (expected non-empty JSON object).")
    return ok

def _missing_critical_fields(llm_data: Dict[str, Any]) -> List[str]:
    critical = ["invoice_date", "vendor_name", "customer_name", "subtotal", "total_vat", "total_amount"]
    missing = []
    for k in critical:
        v = llm_data.get(k) if isinstance(llm_data, dict) else None
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(k)
    return missing

def _guard_critical_fields(state: Dict[str, Any]) -> bool:
    llm_data = state["outputs"].get("llm_data") or {}
    missing = _missing_critical_fields(llm_data)
    ok = len(missing) == 0
    state["guards"]["critical_fields_present"] = {"ok": ok, "missing": missing}
    if not ok:
        state["errors"].append(f"Missing critical fields: {', '.join(missing)}")
    return ok

def _guard_validation_error_count(state: Dict[str, Any]) -> bool:
    errs = state["outputs"].get("validation_errors") or []
    ok = len(errs) <= MAX_VALIDATION_ERRORS_FOR_OK
    state["guards"]["validation_error_count_ok"] = {
        "ok": ok,
        "count": len(errs),
        "max": MAX_VALIDATION_ERRORS_FOR_OK,
    }
    if not ok:
        state["errors"].append(
            f"Too many validation errors ({len(errs)}). Maximum allowed is {MAX_VALIDATION_ERRORS_FOR_OK}."
        )
    return ok

def _stage_ocr(state: Dict[str, Any], file_bytes: bytes) -> None:
    state["stage"] = "ocr"
    text = get_text_from_document(file_bytes, state["filename"])
    state["outputs"]["ocr_text"] = text or ""

def _stage_llm(state: Dict[str, Any]) -> None:
    state["stage"] = "llm"
    ocr_text = state["outputs"].get("ocr_text") or ""
    reduced = reduce_invoice_text(ocr_text)
    state["outputs"]["reduced_text"] = reduced
    # LLM consumes reduced text (deterministic selection of relevant regions).
    state["outputs"]["llm_data"] = structure_text_with_llm(reduced, state["filename"])

def _stage_validation(state: Dict[str, Any]) -> None:
    state["stage"] = "validation"
    llm_data = state["outputs"].get("llm_data") or {}
    _, _, _, _, _, validation_errors = validate_extraction(llm_data, state["filename"])
    state["outputs"]["validation_errors"] = validation_errors or []

def run_extraction_pipeline(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Deterministic, step-based extraction pipeline with explicit state transitions and guards.

    Stages:
    - OCR: get_text_from_document
    - LLM: structure_text_with_llm (consumes reduce_invoice_text output)
    - Validation: validate_extraction

    Guard conditions:
    - minimum OCR text length
    - non-empty LLM JSON output
    - critical fields present (date, amounts, vendor/customer)

    On guard failure:
    - mark state as failed/partial
    - stop execution early
    """
    state = _init_extraction_state(filename)
    try:
        # 1) OCR stage + guard
        _stage_ocr(state, file_bytes)
        if not _guard_min_ocr_text(state):
            state["status"] = "failed"
            return state

        # 2) LLM stage + guard
        _stage_llm(state)
        if not _guard_nonempty_llm_output(state):
            state["status"] = "failed"
            return state

        # 3) Validation stage + guards
        _stage_validation(state)
        if not _guard_critical_fields(state):
            state["status"] = "partial"
            return state
        if not _guard_validation_error_count(state):
            state["status"] = "partial"
            return state

        # Attach debug metadata to llm_data for downstream mapping (if executed)
        llm_data = state["outputs"]["llm_data"]
        ocr_text = state["outputs"]["ocr_text"] or ""
        llm_data["_invoice_text"] = ocr_text[:MAX_RAW_TEXT_STORE_CHARS]
        llm_data["_validation_errors"] = state["outputs"]["validation_errors"] or []

        state["status"] = "ok"
        return state

    except Exception as e:
        state["status"] = "failed"
        state["errors"].append(str(e))
        return state

# Backwards-compatible name used by app.py (now returns state, not raw LLM dict).
def robust_invoice_processor(pdf_bytes: bytes, filename: str) -> dict:
    return run_extraction_pipeline(pdf_bytes, filename)

# -------------------- Posting rules (data-driven) --------------------
# Optional: convert a register entry into a balanced journal entry.
# This is intentionally rule-driven (conditions + postings), so you can add mappings
# without changing Python logic everywhere.
DEFAULT_COA = {
    "AR": "1100",
    "AP": "2000",
    "SALES": "4000",
    "COGS": "5000",
    "FREIGHT": "5100",
    "SOFTWARE": "5200",
    "VAT_PAYABLE": "2100",
    "VAT_RECOVERABLE": "1400",
    "CASH": "1000",
}

DEFAULT_RULES = [
    {
        "condition": {"Type": "Sales"},
        "posting": {
            "dr": [{"account": "AR", "amount": "Gross Amount (EUR)"}],
            "cr": [
                {"account": "SALES", "amount": "Nett Amount (EUR)"},
                {"account": "VAT_PAYABLE", "amount": "VAT Amount (EUR)"},
            ],
        },
    },
    {
        "condition": {"Type": "Purchase", "VAT Category": "Zero Rated"},
        "posting": {
            "dr": [{"account": "FREIGHT", "amount": "Nett Amount (EUR)"}],
            "cr": [{"account": "AP", "amount": "Gross Amount (EUR)"}],
        },
    },
    {
        "condition": {"Type": "Unclassified", "Vendor Name_regex": ".*Google Cloud.*"},
        "posting": {
            "dr": [{"account": "SOFTWARE", "amount": "Gross Amount (EUR)"}],
            "cr": [{"account": "AP", "amount": "Gross Amount (EUR)"}],
        },
    },
]

def _match_rule(entry: dict, rules: list) -> Optional[dict]:
    import re as _re
    for r in rules:
        cond = r.get("condition", {})
        ok = True
        for k, v in cond.items():
            if k.endswith("_regex"):
                field = k[:-6]
                if not _re.match(str(v), str(entry.get(field, "") or ""), flags=_re.I):
                    ok = False; break
            else:
                if str(entry.get(k, "")).strip() != str(v).strip():
                    ok = False; break
        if ok:
            return r
    return None

def _amount_field_to_value(entry: dict, field: str) -> float:
    if "(EUR)" in field:
        return float(entry.get(field) or 0.0)
    mapping = {
        "Nett Amount": "Nett Amount (EUR)",
        "VAT Amount": "VAT Amount (EUR)",
        "Gross Amount": "Gross Amount (EUR)",
    }
    f = mapping.get(field, field)
    return float(entry.get(f) if entry.get(f) is not None else entry.get(field, 0.0) or 0.0)

def build_journal_from_entry(entry: dict, coa: dict = None, rules: list = None) -> dict:
    coa = {**DEFAULT_COA, **(coa or {})}
    rules = rules or DEFAULT_RULES

    if entry.get("FX Error"):
        return {"status": "blocked", "reason": "Needs FX", "entry": entry}
    if not entry.get("Date"):
        return {"status": "blocked", "reason": "Missing Date", "entry": entry}

    rule = _match_rule(entry, rules)
    if not rule:
        return {"status": "blocked", "reason": "Unmapped", "entry": entry}

    lines = []
    for side in ("dr", "cr"):
        for post in rule["posting"].get(side, []):
            acct_key = post["account"]
            acct = coa.get(acct_key, acct_key)
            amt = _amount_field_to_value(entry, post["amount"])
            lines.append({
                "account_code": acct,
                "debit": round(amt, 2) if side == "dr" else 0.0,
                "credit": round(amt, 2) if side == "cr" else 0.0,
            })

    d = round(sum(x["debit"] for x in lines), 2)
    c = round(sum(x["credit"] for x in lines), 2)
    if d != c:
        return {"status": "blocked", "reason": f"Imbalance {d} != {c}", "entry": entry, "lines": lines}

    return {
        "status": "posted",
        "journal": {
            "entry_date": entry["Date"],
            "memo": entry.get("Description") or f"{entry.get('Vendor Name')} / {entry.get('Invoice Number')}",
            "currency": entry.get("Currency"),
            "fx_rate": entry.get("FX Rate (ccy->EUR)"),
            "client_id": entry.get("client_id"),
            "lines": lines,
        }
    }

__all__ = [
    "robust_invoice_processor",
    "run_extraction_pipeline",
    "build_journal_from_entry",
    "DEFAULT_COA",
    "DEFAULT_RULES",
]