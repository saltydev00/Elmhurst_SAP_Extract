# SAP Property Data Extractor

Professional web application for extracting property data from SAP energy calculation reports.

## Features

- 🏠 **Multi-Property Support** - Process documents with multiple properties
- 📊 **Clean Excel Output** - Organized data ready for analysis
- 🚀 **Fast Processing** - Automatic property detection and data extraction
- 🔒 **Secure** - Files processed locally, not stored
- 📱 **Professional Interface** - Clean, user-friendly web application

## Extracted Data

- Property identifiers (Block X, Plot Y)
- CO2 emissions (t/year)
- Space heating values (240)
- Water heating values (247)
- Shower heating values (247a)
- Total DHW calculations

## Usage

1. Upload your SAP PDF document
2. Click "Extract Property Data"
3. Download the organized Excel file
4. Use data for analysis and reporting

## Technology

- Built with Streamlit
- PDF processing with PyMuPDF
- Excel export with pandas/openpyxl
- Optimized for professional use

## Deployment

This application is designed for easy deployment to Streamlit Cloud with custom domain support.

## Requirements

See `requirements.txt` for Python dependencies.