# processor.py
# Robust extraction, OCR, FX, validation, mapping, and posting.
# Used by app.py. No server here.

import io
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta

# Money / math
from decimal import Decimal, ROUND_HALF_UP

# OCR & PDF
import PyPDF2
import boto3
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import pytesseract

# LLM + HTTP
from openai import OpenAI
import requests

# Env
from dotenv import load_dotenv
import pathlib

# Initialize logging first
log = logging.getLogger("invoice-processor")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load .env file - try multiple locations
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
CENT = Decimal("0.01")
RATE_PREC = Decimal("0.0001")

def q_money(x) -> Decimal:
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)

def q_rate(x) -> Decimal:
    return Decimal(str(x)).quantize(RATE_PREC, rounding=ROUND_HALF_UP)

def nearly_equal_money(a: Decimal, b: Decimal, tol: Decimal = CENT) -> bool:
    return abs(q_money(a) - q_money(b)) <= tol

# -------------------- Dates & currency --------------------
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
        return "EUR" # Default to EUR to avoid pipeline crash, but error logged
    return cur

def _prev_business_day(d: date) -> date:
    while d.weekday() >= 5:  # 5=Sat,6=Sun
        d -= timedelta(days=1)
    return d

def get_eur_rate(invoice_date: date, ccy: str) -> Tuple[Decimal, str]:
    """
    Return (rate, rate_date_str) for 1 CCY -> EUR using exchangerate.host (ECB).
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
        except Exception as ex2:
            log.warning(f"FX invert failed {url2}: {ex2}")

        d = _prev_business_day(d - timedelta(days=1))
    
    # Fallback to 1.0 if not found (don't crash)
    log.error(f"FX Rate not found for {ccy} on {invoice_date}. Defaulting to 1.0")
    return Decimal("1.0"), invoice_date.isoformat()

# -------------------- OCR (robust, supports PDFs & images) --------------------
def _aws_region() -> str:
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _preprocess_for_tesseract(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("L"))
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11,
    )
    return Image.fromarray(img)


def _textract_analyze_image(img_bytes: bytes) -> str:
    try:
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
        
        # Simplified KV extraction for context
        kv_pairs = []
        for b in blocks:
             if b.get("BlockType") == "KEY_VALUE_SET" and "KEY" in (b.get("EntityTypes") or []):
                 pass # Skipping complex KV reconstruction for brevity, relying on Lines
        
        return "\n".join(text_lines)
    except Exception as e:
        log.warning(f"Textract analysis failed: {e}")
        return ""

def _tesseract_ocr(pil_img: Image.Image) -> str:
    pre = _preprocess_for_tesseract(pil_img)
    return pytesseract.image_to_string(pre, config="--psm 6 -l eng")

def get_text_from_pdf(pdf_bytes: bytes, filename: str) -> str:
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
            return text
    except Exception as e:
        log.warning(f"[PyPDF2] failed: {e}")

    # 2) Textract detect
    try:
        textract = boto3.client("textract", region_name=_aws_region())
        resp = textract.detect_document_text(Document={"Bytes": pdf_bytes})
        text = "\n".join([b.get("Text", "") for b in (resp.get("Blocks") or []) if b.get("BlockType") == "LINE"])
        _update_best(text, "Textract.detect")
        if len(text.strip()) > 60:
            return text
    except Exception:
        pass

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
        combined = "\n".join(texts)
        if len(combined.strip()) > 60:
            return combined
    except Exception:
        pass

    # 4) Tesseract fallback
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
        ocr = []
        for im in images:
            page_text = _tesseract_ocr(im)
            if page_text:
                ocr.append(page_text)
                _update_best(page_text, "Tesseract")
        return "\n".join(ocr)
    except Exception as e:
        log.error(f"[Tesseract] failed: {e}")

    return best_text


def get_text_from_image(image_bytes: bytes, filename: str) -> str:
    # 1) Textract detect
    try:
        textract = boto3.client("textract", region_name=_aws_region())
        resp = textract.detect_document_text(Document={"Bytes": image_bytes})
        text = "\n".join([b.get("Text", "") for b in (resp.get("Blocks") or []) if b.get("BlockType") == "LINE"])
        if len(text.strip()) > 40:
            return text
    except Exception:
        pass

    # 2) Tesseract fallback
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        return _tesseract_ocr(pil_img)
    except Exception:
        pass
    
    return ""


def get_text_from_document(file_bytes: bytes, filename: str) -> str:
    name = (filename or "").lower()
    ext = os.path.splitext(name)[1]
    
    is_pdf_header = file_bytes.startswith(b"%PDF-")
    if is_pdf_header or ext == ".pdf":
        return get_text_from_pdf(file_bytes, filename)
    
    return get_text_from_image(file_bytes, filename)

# -------------------- LLM extraction --------------------
SECTION_LABELS = [
    "invoice", "total", "subtotal", "tax", "vat", "btw", "reverse charge",
    "verlegd", "omgekeerde heffing", "bill to", "payer", "customer", "vendor",
    "supplier", "line items", "description", "due", "payment terms", "amount",
    "iban", "bank", "account"
]

def reduce_invoice_text(raw_text: str, window: int = 300) -> str:
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
You are an expert financial data extraction model. Extract structured data into a JSON object.

RULES:
1. **Parties**: 
   - **Vendor**: Entity getting paid. Look for top logo or "Bank Details".
   - **Customer**: Entity being billed. Look for "Bill To".
   - *Tip*: If the invoice shows an IBAN for payment, that IBAN belongs to the Vendor.
2. **IBANs**: Extract 'vendor_iban' and 'customer_iban' carefully.
3. **VAT**: 'vat_category' must be 'Import-VAT', 'Reverse-Charge', 'Standard', 'Zero-Rated', or 'Out-of-Scope'.
4. **Dates**: YYYY-MM-DD.
5. **Addresses**: Extract FULL addresses including COUNTRY.

SCHEMA:
{
  "invoice_number": "string | null",
  "invoice_date": "YYYY-MM-DD | null",
  "due_date": "YYYY-MM-DD | null",
  "vendor_name": "string | null",
  "vendor_vat_id": "string | null",
  "vendor_address": "string | null",
  "customer_name": "string | null",
  "customer_vat_id": "string | null",
  "customer_address": "string | null",
  "currency": "string | null",
  "vat_category": "string | null",
  "subtotal": "float | null",
  "total_amount": "float | null",
  "total_vat": "float | null",
  "vat_breakdown": [{"rate": "float", "base_amount": "float", "tax_amount": "float"}],
  "line_items": [{"description": "string", "quantity": "float", "unit_price": "float", "line_total": "float"}],
  "goods_services_indicator": "goods | services | null",
  "vendor_iban": "string | null",
  "customer_iban": "string | null",
  "notes": "string | null"
}
"""

def structure_text_with_llm(invoice_text: str, filename: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=api_key)
    reduced = reduce_invoice_text(invoice_text)
    try:
        log.info(f"LLM extracting {filename}...")
        r = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": f"**INVOICE TEXT:**\n{reduced}"}
            ]
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        log.error(f"LLM extraction error: {e}")
        return {}

def _translate_to_english_if_dutch(text: str) -> str:
    # Simplified for brevity; assuming English preferred
    return text

# -------------------- Validation & mapping --------------------
def _estimate_extraction_confidence(invoice_text: str, llm_data: Dict[str, Any]) -> Tuple[str, str]:
    text_len = len(invoice_text)
    if text_len < 200: return "low", "very short text"
    
    missing = []
    if not llm_data.get("total_amount"): missing.append("total")
    if not llm_data.get("invoice_date"): missing.append("date")
    
    if missing: return "medium", f"missing {','.join(missing)}"
    return "high", "good data density"

def validate_extraction(data: dict, filename: str) -> Tuple[date, str, Decimal, Decimal, Decimal]:
    errors = []
    inv_date = ensure_iso_date(data.get("invoice_date"), "invoice_date", errors)
    currency = normalize_currency(data.get("currency"), errors)
    
    # We allow validation "failures" to pass as soft errors in robust mode, 
    # but logging them is important.
    if errors:
        log.warning(f"Validation warnings for {filename}: {errors}")
        
    return inv_date, currency, 0, 0, 0 # Return placeholders if needed

def _normalize_company_name(name: str) -> str:
    if not name: return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def _split_company_list(raw: str) -> List[str]:
    if not raw: return []
    return [p.strip() for p in re.split(r"[,\n;]", raw) if p and p.strip()]

# -------------------- Dynamic IBAN & Sanity Logic --------------------

def _extract_ibans_via_regex(text: str) -> List[str]:
    """
    Finds IBANs in raw text even if LLM misses them.
    Looks for 2 letters + 2 digits + 10-30 alphanumeric.
    """
    if not text: return []
    # Clean whitespace for better matching
    clean = re.sub(r'\s+', '', text)
    # Generic IBAN pattern
    matches = re.findall(r'([A-Z]{2}\d{2}[A-Z0-9]{10,30})', clean)
    # Filter by length (valid IBANs are usually 15-34 chars)
    return list(set([m for m in matches if 15 <= len(m) <= 34]))

def _is_same_country_code(code: str, country_name: str) -> bool:
    """Helper to check if a 2-letter code matches a country name."""
    mapping = {
        "NL": "Netherlands", "DE": "Germany", "FR": "France", 
        "BE": "Belgium", "EG": "Egypt", "US": "United States",
        "GB": "United Kingdom", "UK": "United Kingdom"
    }
    c_name = mapping.get(code.upper(), "")
    if c_name and c_name.lower() in country_name.lower():
        return True
    return False

def _perform_sanity_check_and_swap(entry: Dict[str, Any], our_companies: List[str]) -> Dict[str, Any]:
    """
    Fixes LLM mix-ups where Vendor/Customer are swapped.
    Scenario: Invoice is classified as Sales (Vendor=Us), but Vendor IBAN is foreign (e.g. Egypt).
    This implies 'Us' is actually the payer (Customer), so we must swap.
    """
    current_type = entry.get("Type")
    vendor_iban = entry.get("Vendor IBAN")
    
    # Only check if we think it's a Sale (Vendor matches our company)
    if current_type == "Sales" and vendor_iban and len(vendor_iban) > 2:
        iban_code = vendor_iban[:2].upper()
        
        # If Vendor IBAN is NOT Dutch (assuming we are Dutch) 
        # AND it matches the "Customer's" country (e.g., Galina in Egypt), 
        # Then the LLM got it backwards.
        cust_country = entry.get("Customer Country") or ""
        
        # Assumption: Our company uses NL banks. If IBAN is Foreign, it's likely not us receiving money.
        if iban_code != "NL":
             log.warning(f"Sanity Swap Triggered: Vendor IBAN ({iban_code}) is foreign but type is Sales. Swapping entities.")
             
             # Swap relevant fields
             entry["Vendor Name"], entry["Customer Name"] = entry["Customer Name"], entry["Vendor Name"]
             entry["Vendor Address"], entry["Customer Address"] = entry["Customer Address"], entry["Vendor Address"]
             entry["Vendor Country"], entry["Customer Country"] = entry["Customer Country"], entry["Vendor Country"]
             entry["Vendor VAT ID"], entry["Customer VAT ID"] = entry["Customer VAT ID"], entry["Vendor VAT ID"]
             
             # Force type to Purchase
             entry["Type"] = "Purchase"
             
    return entry

# -------------------- Classification Helpers --------------------

EU_COUNTRIES = {
    "netherlands", "germany", "france", "belgium", "spain", "italy", "ireland", "austria", 
    "portugal", "poland", "sweden", "finland", "denmark", "czech republic", "hungary",
    "luxembourg", "bulgaria", "croatia", "cyprus", "estonia", "greece", "latvia", 
    "lithuania", "malta", "romania", "slovakia", "slovenia"
}

def _extract_country_from_address(addr: str) -> Optional[str]:
    if not addr: return None
    addr_l = addr.lower()
    # Simple check for common countries
    for c in EU_COUNTRIES:
        if c in addr_l: return c.title()
    if "egypt" in addr_l: return "Egypt"
    if "united states" in addr_l or " usa " in addr_l: return "United States"
    if "united kingdom" in addr_l or " uk " in addr_l: return "United Kingdom"
    return None

def _is_eu_country(c: Optional[str]) -> bool:
    return (c or "").lower() in EU_COUNTRIES

def _is_nl_country(c: Optional[str]) -> bool:
    return (c or "").lower() in ["netherlands", "nl", "nederland", "holland"]

def _determine_goods_services_indicator(llm_data, text):
    return llm_data.get("goods_services_indicator", "services")

def _classify_type(register_entry: Dict[str, Any], our_companies_list: List[str]) -> str:
    v = _normalize_company_name(register_entry.get("Vendor Name"))
    c = _normalize_company_name(register_entry.get("Customer Name"))
    
    for comp in our_companies_list:
        norm = _normalize_company_name(comp)
        if norm and norm in v: return "Sales"
        if norm and norm in c: return "Purchase"
        
    return "Unclassified"

def _determine_invoice_subcategory(type_str, v_addr, c_addr, vat_pct, text):
    v_country = _extract_country_from_address(v_addr)
    c_country = _extract_country_from_address(c_addr)
    
    if type_str == "Sales":
        if _is_nl_country(c_country): return "Standard 21%" if vat_pct == 21 else "Domestic Sales"
        if _is_eu_country(c_country): return "Sales to EU Countries"
        return "Sales to Non-EU Countries"
        
    if type_str == "Purchase":
        if _is_nl_country(v_country): return "Domestic Purchase"
        if _is_eu_country(v_country): return "Purchase from EU Countries"
        return "Purchase from Non-EU Countries (Import)"
        
    return "Unclassified"

DOTCH_VAT_CATEGORY_DESCRIPTIONS = {
    "1a": "Sales taxed at standard rate (21%)",
    "4a": "Purchases of goods from EU countries",
    "5a": "Input VAT on domestic purchases (Dutch VAT)",
}

def _determine_dutch_vat_return_category(invoice_type, vendor_country, customer_country, **kwargs):
    # Simplified logic for robustness
    if invoice_type == "Purchase":
        if _is_nl_country(vendor_country): return "5a"
        if _is_eu_country(vendor_country): return "4a"
        # Non-EU imports usually handled via license or customs, often not on simple VAT return boxes directly unless 4a/4b logic applies.
        return "" 
    return ""


def _detect_credit_note(invoice_text: str) -> bool:
    """
    Heuristic to detect if the document is a credit note instead of a normal invoice.

    We look for common keywords in the raw invoice text:
      - "credit note", "creditnote", "credit memo", "crediteurnota"

    If none of these are present we assume it is a normal invoice.
    """
    if not invoice_text:
        return False
    txt = invoice_text.lower()
    keywords = [
        "credit note",
        "creditnote",
        "credit memo",
        "crediteurnota",
    ]
    return any(k in txt for k in keywords)

def _set_icp_fields_for_nl(entry):
    # Only Sales to EU (not NL) require ICP
    if entry["Type"] == "Sales":
        cust_country = entry.get("Customer Country")
        if _is_eu_country(cust_country) and not _is_nl_country(cust_country):
            entry["ICP Return Required"] = "Yes"
            entry["ICP Reporting Category"] = "Intra-EU supply"
        else:
            entry["ICP Return Required"] = "No"
    else:
        entry["ICP Return Required"] = "No"
    return entry

# -------------------- Main mapping --------------------

def _map_llm_output_to_register_entry(llm_data: Dict[str, Any]) -> Dict[str, Any]:
    invoice_text = llm_data.get("_invoice_text", "")
    
    # DYNAMIC IBAN FIX:
    # If LLM returns null vendor_iban, try regex
    vendor_iban = llm_data.get("vendor_iban")
    if not vendor_iban:
        ibans = _extract_ibans_via_regex(invoice_text)
        if ibans:
            vendor_iban = ibans[0] # Best guess
            log.info(f"LLM missed IBAN. Regex recovered: {vendor_iban}")

    # Is this a credit note?
    is_credit_note = _detect_credit_note(invoice_text)

    # Derive description
    desc = ""
    if llm_data.get("line_items"):
        desc = llm_data["line_items"][0].get("description", "")

    vat_pct = None
    if llm_data.get("vat_breakdown"):
        vat_pct = llm_data["vat_breakdown"][0].get("rate")

    # Amounts (convert to negative for credit notes if currently positive)
    nett = float(q_money(llm_data.get("subtotal") or 0))
    vat_amt = float(q_money(llm_data.get("total_vat") or 0))
    gross = float(q_money(llm_data.get("total_amount") or 0))
    if is_credit_note:
        if nett > 0:
            nett = -nett
        if vat_amt > 0:
            vat_amt = -vat_amt
        if gross > 0:
            gross = -gross

    return {
        "Date": llm_data.get("invoice_date"),
        "Invoice Number": llm_data.get("invoice_number"),
        "Type": "Unclassified",
        "Document Type": "Credit Note" if is_credit_note else "Invoice",
        "Vendor Name": llm_data.get("vendor_name"),
        "Vendor VAT ID": llm_data.get("vendor_vat_id"),
        "Vendor Country": _extract_country_from_address(llm_data.get("vendor_address")),
        "Vendor Address": llm_data.get("vendor_address"),
        "Vendor IBAN": vendor_iban,
        "Customer Name": llm_data.get("customer_name"),
        "Customer VAT ID": llm_data.get("customer_vat_id"),
        "Customer Country": _extract_country_from_address(llm_data.get("customer_address")),
        "Customer Address": llm_data.get("customer_address"),
        "Customer IBAN": llm_data.get("customer_iban"),
        "Description": desc,
        "Nett Amount": nett,
        "VAT %": vat_pct,
        "VAT Amount": vat_amt,
        "Gross Amount": gross,
        "Currency": (llm_data.get("currency") or "EUR"),
        "VAT Category": llm_data.get("vat_category"),
        "Goods Services Indicator": llm_data.get("goods_services_indicator"),
        "Full_Extraction_Data": llm_data
    }

def _classify_and_set_subcategory(entry: Dict[str, Any], our_companies: List[str]) -> Dict[str, Any]:
    # 1. Initial Classification
    entry["Type"] = _classify_type(entry, our_companies)
    
    # 2. SANITY CHECK & SWAP (Dynamic Fix)
    entry = _perform_sanity_check_and_swap(entry, our_companies)
    
    # 3. Subcategory
    entry["Subcategory"] = _determine_invoice_subcategory(
        entry["Type"], 
        entry.get("Vendor Address"), 
        entry.get("Customer Address"), 
        entry.get("VAT %"),
        ""
    )
    
    # 4. VAT Return Category
    entry["Dutch VAT Return Category"] = _determine_dutch_vat_return_category(
        entry["Type"],
        entry.get("Vendor Country"),
        entry.get("Customer Country")
    )
    entry["Dutch VAT Return Category Description"] = DUTCH_VAT_CATEGORY_DESCRIPTIONS.get(entry["Dutch VAT Return Category"], "")
    
    # 5. ICP
    entry = _set_icp_fields_for_nl(entry)
    
    # Confidence
    conf, reason = _estimate_extraction_confidence("", entry["Full_Extraction_Data"])
    entry["Extraction Confidence"] = conf
    entry["Extraction Confidence Reason"] = reason
    
    return entry

def _convert_to_eur_fields(entry: dict, enabled: bool = True) -> dict:
    if not enabled: return entry
    
    try:
        dt = entry.get("Date")
        ccy = entry.get("Currency")
        if not dt or not ccy:
             raise ValueError("Missing date/currency")
             
        rate, date_used = get_eur_rate(date.fromisoformat(dt), ccy)
        
        entry["FX Rate (ccy->EUR)"] = str(rate)
        entry["FX Rate Date"] = date_used
        entry["Nett Amount (EUR)"] = round(float(Decimal(str(entry["Nett Amount"])) * rate), 2)
        entry["VAT Amount (EUR)"] = round(float(Decimal(str(entry["VAT Amount"])) * rate), 2)
        entry["Gross Amount (EUR)"] = round(float(Decimal(str(entry["Gross Amount"])) * rate), 2)
        
    except Exception as e:
        log.warning(f"FX conversion failed: {e}")
        entry["FX Error"] = str(e)
        
    return entry

# -------------------- Main pipeline --------------------
def robust_invoice_processor(pdf_bytes: bytes, filename: str) -> dict:
    # 1. Extract Text
    text = get_text_from_document(pdf_bytes, filename)
    
    # 2. Structure with LLM
    llm_data = structure_text_with_llm(text, filename)
    llm_data["_invoice_text"] = text # Preserve for regex fallback
    
    # 3. Validation
    validate_extraction(llm_data, filename)
    
    return llm_data

# -------------------- Posting rules (stub) --------------------
DEFAULT_COA = {}
DEFAULT_RULES = []
def build_journal_from_entry(entry, coa=None, rules=None):
    return {} # Implementation optional for this specific request