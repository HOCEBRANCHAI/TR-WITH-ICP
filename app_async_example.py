
from task_queue import task_queue
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
@app.on_event("startup")
async def startup_event():
    """Start background workers when app starts"""
    await task_queue.start_workers()
    log.info("Task queue workers started")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background workers when app shuts down"""
    await task_queue.stop_workers()
    log.info("Task queue workers stopped")

# ============================================================================
# MODIFY /upload ENDPOINT
# ============================================================================
@app.post("/upload")
async def upload_and_extract(
    file: UploadFile = File(...),
    our_companies: str = Form(...),
):
    """
    Upload invoice - returns immediately with job ID, processes in background.
    """
    try:
        # Validate file type
        content_type = (file.content_type or "").lower()
        if not (
            content_type.startswith("application/pdf")
            or content_type.startswith("image/")
        ):
            return JSONResponse(status_code=400, content={
                "filename": file.filename,
                "status": "error",
                "error": f"Unsupported content-type: {file.content_type}."
            })

        # Parse company list
        our_companies_list = _split_company_list(our_companies)
        if not our_companies_list:
            return JSONResponse(status_code=400, content={
                "filename": file.filename,
                "status": "error",
                "error": "No company name(s) provided."
            })

        # Read file bytes
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={
                "filename": file.filename,
                "status": "error",
                "error": "File is empty."
            })

        # Enqueue processing task (returns immediately)
        job_id = await task_queue.enqueue(
            process_invoice_sync,
            file_bytes=file_bytes,
            filename=file.filename,
            our_companies_list=our_companies_list
        )

        # Return immediately with job ID
        return {
            "job_id": job_id,
            "status": "accepted",
            "filename": file.filename,
            "message": f"Invoice processing started. Check status at /job/{job_id}"
        }

    except Exception as e:
        log.exception(f"Error accepting invoice {file.filename}")
        return JSONResponse(status_code=500, content={
            "filename": file.filename,
            "status": "error",
            "error": str(e),
        })
                                                                              
# ===========================================================================
# ADD PROCESSING FUNCTION (runs in background)
# ============================================================================
def process_invoice_sync(
    file_bytes: bytes,
    filename: str,
    our_companies_list: list
) -> dict:
    """
    Synchronous invoice processing function.
    This runs in a background worker thread.
    """
    try:
        log.info(f"Processing {filename}...")

        # Step 1: Extract and process invoice
        llm_data = robust_invoice_processor(file_bytes, filename)

        # Step 2: Map to register entry format
        register_entry = _map_llm_output_to_register_entry(llm_data, filename)

        # Step 3: Classify type and set subcategory
        register_entry = _classify_and_set_subcategory(register_entry, our_companies_list)

        # Step 4: Apply currency conversion
        register_entry = _convert_to_eur_fields(register_entry, conversion_enabled)

        # Step 5: Add conversion info to Full_Extraction_Data
        if "Full_Extraction_Data" in register_entry:
            register_entry["Full_Extraction_Data"]["fx_conversion"] = {
                "enabled": conversion_enabled,
                "rate": register_entry.get("FX Rate (ccy->EUR)"),
                "rate_date": register_entry.get("FX Rate Date"),
                "error": register_entry.get("FX Error")
            }
            register_entry["Full_Extraction_Data"].pop("_invoice_text", None)

        # Step 6: Format for frontend
        formatted_entry = _format_register_entry_for_frontend(register_entry)

        return {
            "filename": filename,
            "status": "success",
            "register_entry": formatted_entry,
        }

    except Exception as e:
        log.exception(f"Error processing {filename}")
        raise Exception(f"Processing failed: {str(e)}")

# ============================================================================
# ADD JOB STATUS ENDPOINT
# ============================================================================
@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status and result of a background job.
    
    Returns:
        - status: "pending", "processing", "completed", or "failed"
        - result: Transaction data (if completed)
        - error: Error message (if failed)
    """
    job = task_queue.get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={
            "error": "Job not found"
        })
    
    return job

# ============================================================================
# OPTIONAL: ADD JOB LIST ENDPOINT (for debugging)
# ============================================================================
@app.get("/jobs")
async def get_all_jobs():
    """Get all jobs (for debugging/monitoring)"""
    return task_queue.get_all_jobs()

# ============================================================================
# OPTIONAL: ADD JOB CLEANUP ENDPOINT
# ============================================================================
@app.post("/jobs/cleanup")
async def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up old completed/failed jobs"""
    task_queue.cleanup_old_jobs(max_age_hours)
    return {"message": f"Cleaned up jobs older than {max_age_hours} hours"}










