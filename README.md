# Invoice Transaction Register Extractor API

A production-ready FastAPI application for extracting and classifying invoice data with automatic VAT categorization for Dutch VAT returns.

## Features

- **Automatic Invoice Classification**: Purchase vs Sales
- **VAT Subcategory Classification**: Standard 21%, Reduced 9%, EU/Non-EU countries
- **Dutch VAT Return Categories**: Automatic mapping to categories 1a, 1b, 1e, 2a, 3a, 3b, 4a, 4b, 5a
- **Multi-format PDF Support**: PyPDF2, AWS Textract, OCR fallback
- **Currency Conversion**: Automatic conversion to EUR using historical exchange rates
- **Robust OCR**: Multiple fallback methods for text extraction

## API Endpoints

- `POST /upload` - Upload a single invoice PDF
- `POST /upload-multiple` - Upload multiple invoice PDFs
- `GET /` - API documentation and health check
- `GET /docs` - Interactive API documentation (Swagger UI)

## Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key for LLM-based extraction

Optional:
- `AWS_ACCESS_KEY_ID` - For AWS Textract (if using)
- `AWS_SECRET_ACCESS_KEY` - For AWS Textract (if using)

## Deployment

This application is configured for deployment on Render.

### Render Deployment

1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy using the provided `Procfile` and `runtime.txt`

## Requirements

See `requirements.txt` for all dependencies.

## License

Proprietary

