import logging
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import pathlib

# Configure logging FIRST before using it
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("invoice-api")

# Load environment variables from .env file - try multiple locations
env_paths = [
    pathlib.Path(__file__).parent / '.env',  # Same directory as app.py
    pathlib.Path.cwd() / '.env',  # Current working directory
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

# Verify OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    log.warning("WARNING: OPENAI_API_KEY not found in environment variables!")
    log.warning("Please ensure your .env file contains: OPENAI_API_KEY=your_key_here")
else:
    log.info("OpenAI API key loaded successfully")

# Import from processor
from processor import (
    robust_invoice_processor,
    _map_llm_output_to_register_entry,
    _classify_and_set_subcategory,
    _convert_to_eur_fields,
    _split_company_list,
)

# Configuration
conversion_enabled: bool = True  # Toggle currency conversion on/off

# Create FastAPI app
app = FastAPI(
    title="Invoice Transaction Register Extractor",
    description=(
        "Upload invoice PDFs and extract structured transaction register data. "
        "Automatically classifies as Purchase or Sales with detailed subcategories."
    ),
    version="5.0.0 (Subcategory Classification)",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Invoice Transaction Register Extractor API with Subcategory Classification",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload - Upload a single invoice PDF",
            "upload_multiple": "POST /upload-multiple - Upload multiple invoice PDFs",
        },
        "supported_formats": ["PDF"],
        "features": [
            "Automatic Purchase/Sales classification",
            "EU vs Non-EU country detection",
            "VAT subcategory classification (Standard 21%, Reduced 9%, etc.)",
            "Import VAT detection",
            "Currency conversion to EUR"
        ]
    }

@app.post("/upload")
async def upload_and_extract(
    file: UploadFile = File(...),
    our_companies: str = Form(
        ...,
        description=(
            "Company name(s): single (e.g., 'PAE NL BV') or comma-/newline-separated "
            "(e.g., 'PAE NL BV, Biofount (Netherlands) B.V.')"
        ),
    ),
):
    """
    Upload a single invoice PDF and extract structured data with subcategory classification.
    """
    try:
        # Validate file type
        if not (file.content_type or "").startswith("application/pdf"):
            return JSONResponse(status_code=400, content={
                "filename": file.filename,
                "status": "error",
                "error": f"Unsupported content-type: {file.content_type}. Only application/pdf is accepted."
            })

        # Parse company list
        our_companies_list = _split_company_list(our_companies)
        if not our_companies_list:
            return JSONResponse(status_code=400, content={
                "filename": file.filename,
                "status": "error",
                "error": "No company name(s) provided. Provide a single name or a comma/newline-separated list."
            })

        # Read file bytes
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={
                "filename": file.filename,
                "status": "error",
                "error": "File is empty."
            })

        log.info(f"Processing {file.filename}...")

        # Step 1: Extract and process invoice
        llm_data = robust_invoice_processor(file_bytes, file.filename)

        # Step 2: Map to register entry format
        register_entry = _map_llm_output_to_register_entry(llm_data)

        # Step 3: Classify type and set subcategory
        register_entry = _classify_and_set_subcategory(register_entry, our_companies_list)

        # Step 4: Apply currency conversion
        register_entry = _convert_to_eur_fields(register_entry, conversion_enabled)

        # Step 5: Add conversion info to Full_Extraction_Data for audit trace
        if "Full_Extraction_Data" in register_entry:
            register_entry["Full_Extraction_Data"]["fx_conversion"] = {
                "enabled": conversion_enabled,
                "rate": register_entry.get("FX Rate (ccy->EUR)"),
                "rate_date": register_entry.get("FX Rate Date"),
                "error": register_entry.get("FX Error")
            }

        return {
            "filename": file.filename,
            "status": "success",
            "register_entry": register_entry,
        }

    except Exception as e:
        log.exception(f"Error processing {file.filename}")
        return JSONResponse(status_code=500, content={
            "filename": file.filename,
            "status": "error",
            "error": str(e),
            "register_entry": None,
        })

@app.post("/upload-multiple")
async def upload_multiple_and_extract(
    files: List[UploadFile] = File(...),
    our_companies: str = Form(
        ...,
        description=(
            "Company name(s): single or comma-/newline-separated"
        )
    ),
):
    """
    Upload multiple invoice PDFs and extract structured data with subcategory classification.
    """
    if not files:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "error": "No files provided",
            "results": [],
        })

    try:
        # Parse company list
        our_companies_list = _split_company_list(our_companies)
        if not our_companies_list:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "error": "No company name(s) provided. Provide a single name or a comma/newline-separated list.",
                "results": [],
            })

        results = []

        for file in files:
            if not file.filename or not (file.content_type or "").startswith("application/pdf"):
                log.warning(f"Skipping invalid file: {file.filename}")
                results.append({
                    "file_name": file.filename or "unknown",
                    "status": "error",
                    "error": "Invalid file or content type (not PDF).",
                    "register_entry": None,
                })
                continue

            log.info(f"Processing file: {file.filename}")
            try:
                file_bytes = await file.read()
                if not file_bytes:
                    raise ValueError("File is empty.")

                # Step 1: Extract and process invoice
                llm_data = robust_invoice_processor(file_bytes, file.filename)

                # Step 2: Map to register entry format
                register_entry = _map_llm_output_to_register_entry(llm_data)

                # Step 3: Classify type and set subcategory
                register_entry = _classify_and_set_subcategory(register_entry, our_companies_list)

                # Step 4: Apply currency conversion
                register_entry = _convert_to_eur_fields(register_entry, conversion_enabled)

                # Step 5: Add conversion info to Full_Extraction_Data for audit trace
                if "Full_Extraction_Data" in register_entry:
                    register_entry["Full_Extraction_Data"]["fx_conversion"] = {
                        "enabled": conversion_enabled,
                        "rate": register_entry.get("FX Rate (ccy->EUR)"),
                        "rate_date": register_entry.get("FX Rate Date"),
                        "error": register_entry.get("FX Error")
                    }

                results.append({
                    "file_name": file.filename,
                    "status": "success",
                    "register_entry": register_entry,
                })

            except Exception as e:
                log.exception(f"Failed to process {file.filename}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "error": str(e),
                    "register_entry": None,
                })

        # Calculate summary statistics
        from decimal import Decimal
        from processor import q_money

        total_files = len(files)
        successful_files = sum(1 for r in results if r.get("status") == "success")
        failed_files = sum(1 for r in results if r.get("status") == "error")

        total_nett = Decimal("0.00")
        total_vat = Decimal("0.00")
        total_gross = Decimal("0.00")

        total_nett_eur = Decimal("0.00")
        total_vat_eur = Decimal("0.00")
        total_gross_eur = Decimal("0.00")

        for r in results:
            if r.get("status") == "success" and r.get("register_entry"):
                e = r["register_entry"]
                # Native currency totals
                total_nett += q_money(e.get("Nett Amount", 0.0) or 0.0)
                total_vat += q_money(e.get("VAT Amount", 0.0) or 0.0)
                total_gross += q_money(e.get("Gross Amount", 0.0) or 0.0)

                # EUR converted totals (if conversion was successful)
                nett_eur = e.get("Nett Amount (EUR)")
                vat_eur = e.get("VAT Amount (EUR)")
                gross_eur = e.get("Gross Amount (EUR)")

                if nett_eur is not None:
                    total_nett_eur += q_money(nett_eur)
                if vat_eur is not None:
                    total_vat_eur += q_money(vat_eur)
                if gross_eur is not None:
                    total_gross_eur += q_money(gross_eur)

        response = {
            "status": "success",
            "summary": {
                "total_files": total_files,
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_nett_amount": round(float(total_nett), 2),
                "total_vat_amount": round(float(total_vat), 2),
                "total_gross_amount": round(float(total_gross), 2),
                "note": "Native currency totals (summed regardless of currency)."
            },
            "results": results,
        }

        # Add EUR converted summary if conversion is enabled
        if conversion_enabled:
            response["eur_converted_summary"] = {
                "total_nett_amount_eur": round(float(total_nett_eur), 2),
                "total_vat_amount_eur": round(float(total_vat_eur), 2),
                "total_gross_amount_eur": round(float(total_gross_eur), 2),
                "note": "All amounts converted to EUR using historical exchange rates (ECB reference via exchangerate.host)."
            }

        return response

    except Exception as e:
        log.exception("Error processing multiple files")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": str(e),
            "results": [],
        })

# Local dev entrypoint
if __name__ == "__main__":
    import uvicorn
    log.info("Starting uvicorn server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

