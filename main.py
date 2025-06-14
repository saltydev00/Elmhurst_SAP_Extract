import os
import json
import base64
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
import io
import tempfile
import shutil
import re

# PDF handling imports
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PyMuPDF not installed. Install with: pip install pymupdf")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_SUPPORT = True
except ImportError:
    PDF2IMAGE_SUPPORT = False
    print("⚠️  pdf2image not installed. Install with: pip install pdf2image")

# Load environment variables (optional for Streamlit)
try:
    load_dotenv()
except:
    pass

class ModelProvider(Enum):
    # OpenAI Models
    GPT4O = ("openai", "gpt-4o", 0.005, 0.015)  # $5/$15 per 1M tokens
    GPT4O_MINI = ("openai", "gpt-4o-mini", 0.00015, 0.0006)  # $0.15/$0.60 per 1M tokens
    GPT4_TURBO = ("openai", "gpt-4-turbo", 0.01, 0.03)  # $10/$30 per 1M tokens
    GPT4_VISION = ("openai", "gpt-4-vision-preview", 0.01, 0.03)  # $10/$30 per 1M tokens - DEPRECATED
    
    # Claude Models (Updated to current versions)
    CLAUDE_3_5_SONNET = ("claude", "claude-3-5-sonnet-20241022", 0.003, 0.015)  # $3/$15 per 1M tokens
    CLAUDE_3_5_HAIKU = ("claude", "claude-3-5-haiku-20241022", 0.001, 0.005)  # $1/$5 per 1M tokens
    CLAUDE_3_OPUS = ("claude", "claude-3-opus-20240229", 0.015, 0.075)  # $15/$75 per 1M tokens
    CLAUDE_3_SONNET = ("claude", "claude-3-5-sonnet-20240620", 0.003, 0.015)  # $3/$15 per 1M tokens
    CLAUDE_3_HAIKU = ("claude", "claude-3-haiku-20240307", 0.00025, 0.00125)  # $0.25/$1.25 per 1M tokens
    
    # Gemini Models
    GEMINI_1_5_PRO = ("gemini", "gemini-1.5-pro", 0.00125, 0.005)  # $1.25/$5 per 1M tokens
    GEMINI_1_5_FLASH = ("gemini", "gemini-1.5-flash", 0.000075, 0.0003)  # $0.075/$0.30 per 1M tokens
    GEMINI_1_0_PRO = ("gemini", "gemini-1.0-pro", 0.0005, 0.0015)  # $0.50/$1.50 per 1M tokens
    
    # Grok Models (Note: Pricing is estimated as it's not publicly available)
    GROK_BETA = ("grok", "grok-beta", 0.01, 0.03)  # Estimated pricing

    def __init__(self, provider, model_name, input_cost, output_cost):
        self.provider = provider
        self.model_name = model_name
        self.input_cost_per_1m = input_cost
        self.output_cost_per_1m = output_cost

class SAPMultiModelExtractor:
    def __init__(self):
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "claude": os.getenv("CLAUDE_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "grok": os.getenv("GROK_API_KEY")
        }
        self._setup_clients()
        self.results = []
        
    def _setup_clients(self):
        """Initialize all API clients"""
        self.clients = {}
        
        # OpenAI
        if self.api_keys["openai"]:
            import openai
            self.clients["openai"] = openai.OpenAI(api_key=self.api_keys["openai"])
        
        # Claude
        if self.api_keys["claude"]:
            import anthropic
            self.clients["claude"] = anthropic.Anthropic(api_key=self.api_keys["claude"])
        
        # Gemini
        if self.api_keys["gemini"]:
            import google.generativeai as genai
            genai.configure(api_key=self.api_keys["gemini"])
            self.clients["gemini"] = genai
        
        # Grok - Note: This is placeholder as Grok API is not publicly available
        if self.api_keys["grok"]:
            print("Note: Grok API integration is not yet available publicly")
    
    def analyze_sap_structure(self, pdf_path: str) -> Dict[str, any]:
        """Analyze SAP document structure and create extraction templates"""
        if not PDF_SUPPORT:
            return {"summary_pages": [], "worksheet_pages": [], "extraction_template": None}
        
        try:
            doc = fitz.open(pdf_path)
            analysis = {
                "summary_pages": [],
                "worksheet_pages": [], 
                "target_section_pages": [],
                "extraction_template": self._create_extraction_template()
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Find summary pages (Property and CO2 info)
                if "Property" in text and ("CO2 Emissions" in text or "CO₂ Emissions" in text):
                    analysis["summary_pages"].append(page_num)
                
                # Find SAP worksheet pages
                if "SAP 10 WORKSHEET" in text:
                    analysis["worksheet_pages"].append(page_num)
                
                # Find pages with our specific target section
                if ("10a. Fuel costs - using BEDF prices (572)" in text and 
                    "12a. Carbon dioxide emissions - Individual heating systems" in text):
                    analysis["target_section_pages"].append(page_num)
                    
                    # Extract the specific section for analysis
                    section_text = self._extract_target_section(text)
                    analysis[f"section_text_page_{page_num}"] = section_text
            
            doc.close()
            return analysis
            
        except Exception as e:
            print(f"Error analyzing SAP structure: {e}")
            return {"summary_pages": [], "worksheet_pages": [], "extraction_template": None}
    
    def _create_extraction_template(self) -> Dict:
        """Create extraction template based on SAP document patterns"""
        return {
            "target_section": {
                "start_marker": "10a. Fuel costs - using BEDF prices (572)",
                "end_marker": "12a. Carbon dioxide emissions - Individual heating systems",
                "target_items": {
                    "240": {
                        "pattern": r'(\d+(?:\.\d+)?)\s*\(240\)',
                        "context": "EPC rating value (left of 240)"
                    },
                    "247": {
                        "pattern": r'(\d+(?:\.\d+)?)\s*\(247\)',
                        "context": "Energy cost component 1 (left of 247)"
                    },
                    "247a": {
                        "pattern": r'(\d+(?:\.\d+)?)\s*\(247a\)',
                        "context": "Energy cost component 2 (left of 247a)"
                    }
                }
            }
        }
    
    def _extract_target_section(self, full_text: str) -> str:
        """Extract the specific section between the two markers"""
        lines = full_text.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            if "10a. Fuel costs - using BEDF prices (572)" in line:
                in_section = True
                section_lines.append(line)
            elif "12a. Carbon dioxide emissions - Individual heating systems" in line:
                section_lines.append(line)
                break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines)
    
    def find_sap_pages(self, pdf_path: str) -> Dict[str, List[int]]:
        """Use PyMuPDF's text extraction to identify relevant SAP pages
        
        Returns a dict with:
        - 'summary_pages': Pages containing property summary (with "Property" and "CO2 Emissions")
        - 'worksheet_pages': Pages containing SAP 10 WORKSHEET with items (240), (247), (247a)
        """
        if not PDF_SUPPORT:
            raise Exception("PyMuPDF not installed. Cannot use OCR pre-processing.")
        
        relevant_pages = {
            'summary_pages': [],
            'worksheet_pages': []
        }
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                
                # Convert to lowercase for case-insensitive matching
                text_lower = text.lower()
                
                # Check for summary page indicators
                if ('property' in text_lower and 
                    ('co2 emissions' in text_lower or 'co₂ emissions' in text_lower)):
                    relevant_pages['summary_pages'].append(page_num)
                    print(f"  📋 Found summary page: Page {page_num + 1}")
                
                # Check for SAP 10 WORKSHEET with specific items
                if 'sap 10 worksheet' in text_lower:
                    # Look for our target items
                    if ('(240)' in text and ('(247)' in text or '(247a)' in text)):
                        relevant_pages['worksheet_pages'].append(page_num)
                        print(f"  📊 Found SAP worksheet page: Page {page_num + 1}")
                    # Sometimes the items might be on the next page
                    elif '(240)' in text:
                        relevant_pages['worksheet_pages'].append(page_num)
                        print(f"  📊 Found SAP worksheet page with item (240): Page {page_num + 1}")
                # Check if this page has our target items even without "SAP 10 WORKSHEET" header
                elif ('(240)' in text and '(247)' in text and '(247a)' in text):
                    relevant_pages['worksheet_pages'].append(page_num)
                    print(f"  📊 Found page with all target items: Page {page_num + 1}")
            
            pdf_document.close()
            
            # If no summary pages found, check first few pages
            if not relevant_pages['summary_pages'] and len(pdf_document) > 0:
                print("  ⚠️  No clear summary page found, will check first page")
                relevant_pages['summary_pages'].append(0)
            
            return relevant_pages
            
        except Exception as e:
            print(f"  ❌ Error analyzing PDF structure: {str(e)}")
            return {'summary_pages': [0], 'worksheet_pages': []}
    
    def extract_text_regions(self, pdf_path: str, page_num: int, patterns: List[str]) -> Dict[str, str]:
        """Extract text regions around specific patterns from a PDF page
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-indexed)
            patterns: List of patterns to search for (e.g., ["(240)", "(247)", "(247a)"])
        
        Returns:
            Dict mapping pattern to extracted value
        """
        if not PDF_SUPPORT:
            return {}
        
        extracted_values = {}
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            if page_num >= len(pdf_document):
                return extracted_values
            
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Split text into lines for easier processing
            lines = text.split('\n')
            
            for pattern in patterns:
                # Look for the pattern in the text
                for i, line in enumerate(lines):
                    if pattern in line:
                        # Extract the value from the same line or next line
                        # Try to find a number after the pattern
                        value_match = re.search(r'\b' + re.escape(pattern) + r'\s*[:\s]?\s*([\d,]+\.?\d*)', line)
                        if value_match:
                            extracted_values[pattern] = value_match.group(1).replace(',', '')
                        else:
                            # Check the next line for a value
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                # Look for a number at the start of the next line
                                number_match = re.match(r'^([\d,]+\.?\d*)', next_line)
                                if number_match:
                                    extracted_values[pattern] = number_match.group(1).replace(',', '')
                                else:
                                    # Try to find any number in the next line
                                    any_number = re.search(r'([\d,]+\.?\d+)', next_line)
                                    if any_number:
                                        extracted_values[pattern] = any_number.group(1).replace(',', '')
            
            # Special handling for Property and CO2 emissions
            if "Property" in patterns:
                # Look for property pattern like "Block X, Plot Y"
                property_match = re.search(r'Property[:\s]+([^\n]+)', text)
                if property_match:
                    extracted_values["Property"] = property_match.group(1).strip()
            
            if "CO2 Emissions" in patterns:
                # Look for CO2 emissions with various formats
                co2_patterns = [
                    r'CO[2₂]\s+Emissions\s*\(t/year\)[:\s]*([\d.]+)',
                    r'CO[2₂]\s+Emissions[:\s]*([\d.]+)\s*t/year',
                    r'CO[2₂]\s+Emissions[:\s]*([\d.]+)'
                ]
                for co2_pattern in co2_patterns:
                    co2_match = re.search(co2_pattern, text, re.IGNORECASE)
                    if co2_match:
                        extracted_values["CO2 Emissions"] = co2_match.group(1)
                        break
            
            pdf_document.close()
            
        except Exception as e:
            print(f"  ❌ Error extracting text regions: {str(e)}")
        
        return extracted_values
    
    def extract_from_image(self, image_path: str, model: ModelProvider) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract data using specified model and return results with metrics"""
        
        # Check if this is a PDF and we can use OCR pre-processing
        ocr_results = {}
        relevant_pages = None
        if image_path.lower().endswith('.pdf') and PDF_SUPPORT:
            print(f"  🔍 Using OCR pre-processing to identify relevant pages...")
            relevant_pages = self.find_sap_pages(image_path)
            
            # Try to extract values directly using OCR first
            if relevant_pages['summary_pages']:
                page_num = relevant_pages['summary_pages'][0]
                summary_values = self.extract_text_regions(
                    image_path, page_num, 
                    ["Property", "CO2 Emissions"]
                )
                ocr_results.update(summary_values)
            
            if relevant_pages['worksheet_pages']:
                for page_num in relevant_pages['worksheet_pages']:
                    worksheet_values = self.extract_text_regions(
                        image_path, page_num,
                        ["(240)", "(247)", "(247a)"]
                    )
                    # Only update if we found values
                    for key, value in worksheet_values.items():
                        if value and key not in ocr_results:
                            ocr_results[key] = value
            
            if ocr_results:
                print(f"  ✅ OCR extracted values: {ocr_results}")
        
        prompt = """You are analyzing a SAP energy calculation report. This may be a single property report or a combined document with multiple properties.

EXTRACT THE FOLLOWING VALUES:

1. **Property Identifier**: Look for "Property" field in any summary table (usually "Block X, Plot Y" format)

2. **CO2 Emissions**: Look for "CO₂ Emissions (t/year)" or "CO2 Emissions (t/year)" in summary tables

3. **EPC Rating (240)**: Find the section titled exactly:
   "SAP 10 WORKSHEET FOR New Build (As Designed) (Version 10.2, February 2022)
   CALCULATION OF EPC COSTS, EMISSIONS AND PRIMARY ENERGY"
   Within this section, locate item "(240)" and extract the numerical value next to it.

4. **Total Energy Cost (247 + 247a)**: In the same SAP 10 WORKSHEET section:
   - Find item "(247)" and extract its numerical value
   - Find item "(247a)" and extract its numerical value  
   - Calculate their sum

IMPORTANT INSTRUCTIONS:
- If this is a multi-property document, extract values from the FIRST complete property section found
- Look carefully through the entire document text for these specific numbered items
- For items (240), (247), (247a): these are usually in a structured list or table format
- Extract only numerical values (ignore currency symbols, units in parentheses)
- If any value is not found, use "NOT_FOUND"

Return ONLY a valid JSON object in this exact format:
{
  "property": "extracted_property_value",
  "co2_emissions": "extracted_co2_value", 
  "epc_rating_240": "extracted_240_value",
  "energy_cost_247": "extracted_247_value",
  "energy_cost_247a": "extracted_247a_value",
  "total_energy_cost": "sum_of_247_and_247a"
}

Do not include any explanation or additional text."""
        
        start_time = time.time()
        tokens_used = {"input": 0, "output": 0}
        temp_dir = None
        processed_image_path = image_path
        
        try:
            # Handle PDF conversion
            if image_path.lower().endswith('.pdf'):
                print(f"  📄 Converting PDF to image...")
                temp_dir = tempfile.mkdtemp(prefix="sap_pdf_extract_")
                try:
                    image_paths = self.convert_pdf_to_images(image_path, temp_dir)
                    if image_paths:
                        # Smart page selection - minimal processing
                        print(f"  ✅ Converted {len(image_paths)} pages")
                        
                        # Use OCR pre-processing to identify relevant pages
                        all_results = {}
                        total_tokens = {"input": 0, "output": 0}
                        pages_processed = 0
                        
                        # Check if we already have OCR results from pre-processing
                        if ocr_results:
                            # Map OCR results to expected field names
                            all_results["property"] = ocr_results.get("Property", "NOT_FOUND")
                            all_results["co2_emissions"] = ocr_results.get("CO2 Emissions", "NOT_FOUND")
                            all_results["epc_rating_240"] = ocr_results.get("(240)", "NOT_FOUND")
                            all_results["energy_cost_247"] = ocr_results.get("(247)", "NOT_FOUND")
                            all_results["energy_cost_247a"] = ocr_results.get("(247a)", "NOT_FOUND")
                            print(f"  📊 Using OCR pre-extracted values")
                        
                        # Determine which pages to process with AI
                        pages_to_process = []
                        
                        if relevant_pages:
                            # Add summary pages first (usually has property and CO2)
                            for page_idx in relevant_pages.get('summary_pages', []):
                                if page_idx < len(image_paths) and len(pages_to_process) < 2:
                                    pages_to_process.append(page_idx)
                            
                            # Add worksheet pages if we still need SAP data
                            if (all_results.get("epc_rating_240") == "NOT_FOUND" or 
                                all_results.get("energy_cost_247") == "NOT_FOUND" or
                                all_results.get("energy_cost_247a") == "NOT_FOUND"):
                                for page_idx in relevant_pages.get('worksheet_pages', []):
                                    if page_idx < len(image_paths) and len(pages_to_process) < 3:
                                        if page_idx not in pages_to_process:
                                            pages_to_process.append(page_idx)
                        
                        # If no relevant pages identified, fall back to first 3 pages
                        if not pages_to_process:
                            pages_to_process = list(range(min(3, len(image_paths))))
                        
                        print(f"  📊 Smart extraction: Processing {len(pages_to_process)} relevant pages: {[p+1 for p in pages_to_process]}")
                        
                        # Process only the identified relevant pages
                        for page_idx in pages_to_process:
                            try:
                                print(f"    Processing page {page_idx + 1}...")
                                page_result, page_tokens = self._extract_from_single_page(
                                    image_paths[page_idx], prompt, model
                                )
                                
                                # Update only missing values or NOT_FOUND values
                                for key, value in page_result.items():
                                    if key not in all_results or all_results[key] == "NOT_FOUND":
                                        all_results[key] = value
                                
                                total_tokens["input"] += page_tokens.get("input", 0)
                                total_tokens["output"] += page_tokens.get("output", 0)
                                pages_processed += 1
                                
                                # Check if we have all required data
                                has_all_data = (
                                    all_results.get("property") != "NOT_FOUND" and
                                    all_results.get("co2_emissions") != "NOT_FOUND" and
                                    all_results.get("epc_rating_240") != "NOT_FOUND" and
                                    all_results.get("energy_cost_247") != "NOT_FOUND" and
                                    all_results.get("energy_cost_247a") != "NOT_FOUND"
                                )
                                
                                if has_all_data:
                                    print(f"  ✅ Found all required data!")
                                    break
                                    
                            except Exception as e:
                                print(f"    ⚠️  Error processing page {page_idx + 1}: {str(e)}")
                        
                        # Ensure all expected fields exist
                        expected_fields = ["property", "co2_emissions", "epc_rating_240", 
                                         "energy_cost_247", "energy_cost_247a", "total_energy_cost"]
                        for field in expected_fields:
                            if field not in all_results:
                                all_results[field] = "NOT_FOUND"
                        
                        # Calculate total energy cost if both values are found
                        if (all_results.get("energy_cost_247") != "NOT_FOUND" and 
                            all_results.get("energy_cost_247a") != "NOT_FOUND"):
                            try:
                                cost_247 = float(all_results["energy_cost_247"])
                                cost_247a = float(all_results["energy_cost_247a"])
                                all_results["total_energy_cost"] = str(cost_247 + cost_247a)
                            except:
                                pass
                        
                        print(f"  📊 Processed {pages_processed} pages total (Pages: {[p+1 for p in pages_to_process[:pages_processed]]})")
                        
                        # Add page information to results for tracking
                        all_results["pages_processed"] = [p+1 for p in pages_to_process[:pages_processed]]
                        result = all_results
                        tokens_used = total_tokens
                        
                    else:
                        raise Exception("Failed to convert PDF to images")
                except Exception as pdf_error:
                    raise Exception(f"PDF conversion failed: {str(pdf_error)}")
            
            else:
                # Non-PDF single image processing
                # Extract using the appropriate model
                if model.provider == "openai":
                    result, tokens = self._extract_openai(processed_image_path, prompt, model.model_name)
                elif model.provider == "claude":
                    result, tokens = self._extract_claude(processed_image_path, prompt, model.model_name)
                elif model.provider == "gemini":
                    result, tokens = self._extract_gemini(processed_image_path, prompt, model.model_name)
                elif model.provider == "grok":
                    result, tokens = self._extract_grok(processed_image_path, prompt, model.model_name)
                else:
                    raise ValueError(f"Unknown provider: {model.provider}")
                
                tokens_used = tokens
            
        except Exception as e:
            print(f"Error with {model.name}: {str(e)}")
            result = {
                "property": "ERROR", 
                "co2_emissions": "ERROR", 
                "epc_rating_240": "ERROR",
                "energy_cost_247": "ERROR", 
                "energy_cost_247a": "ERROR",
                "total_energy_cost": "ERROR",
                "error": str(e)
            }
        
        finally:
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        end_time = time.time()
        
        # Calculate costs
        input_cost = (tokens_used["input"] / 1_000_000) * model.input_cost_per_1m
        output_cost = (tokens_used["output"] / 1_000_000) * model.output_cost_per_1m
        total_cost = input_cost + output_cost
        
        metrics = {
            "model": model.name,
            "provider": model.provider,
            "processing_time": end_time - start_time,
            "input_tokens": tokens_used["input"],
            "output_tokens": tokens_used["output"],
            "total_tokens": tokens_used["input"] + tokens_used["output"],
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
        
        return result, metrics
    
    def extract_all_properties_from_document(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract data from all properties in a combined SAP document"""
        if not PDF_SUPPORT:
            return []
        
        try:
            doc = fitz.open(pdf_path)
            all_properties = []
            
            print(f"📄 Processing combined document: {os.path.basename(pdf_path)}")
            print(f"📊 Total pages: {len(doc)}")
            
            # First pass: Identify all property sections
            property_sections = self._identify_property_sections(doc)
            print(f"🏠 Found {len(property_sections)} property sections")
            
            # Second pass: Extract data from each property
            for i, section in enumerate(property_sections, 1):
                print(f"\n🔍 Processing property {i}/{len(property_sections)}...")
                property_data = self._extract_single_property_data(doc, section)
                
                if property_data["property"] != "NOT_FOUND":
                    all_properties.append(property_data)
                    print(f"  ✅ {property_data['property']}: {property_data['co2_emissions']} t/year")
                else:
                    print(f"  ⚠️  Property {i}: Could not extract property name")
            
            doc.close()
            
            print(f"\n📋 Successfully extracted {len(all_properties)} properties")
            return all_properties
            
        except Exception as e:
            print(f"Error processing combined document: {e}")
            return []
    
    def _identify_property_sections(self, doc) -> List[Dict]:
        """Identify separate property sections in the document"""
        sections = []
        current_section = None
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Check if this page starts a new property (has summary table)
            if self._is_property_summary_page(text):
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "start_page": page_num,
                    "summary_page": page_num,
                    "worksheet_pages": [],
                    "target_section_pages": []
                }
            
            # Add to current section if it exists
            if current_section:
                # Check if this page has SAP worksheet
                if "SAP 10 WORKSHEET" in text:
                    current_section["worksheet_pages"].append(page_num)
                
                # Check if this page has our target section
                if ("10a. Fuel costs - using BEDF prices (572)" in text and 
                    "12a. Carbon dioxide emissions - Individual heating systems" in text):
                    current_section["target_section_pages"].append(page_num)
                
                # Update end page
                current_section["end_page"] = page_num
        
        # Don't forget the last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _is_property_summary_page(self, text: str) -> bool:
        """Check if a page is a property summary page"""
        indicators = [
            ("Property" in text and ("CO2 Emissions" in text or "CO₂ Emissions" in text)),
            ("Block" in text and "Plot" in text and "Emissions" in text),
            ("EPC" in text and "Rating" in text and "Property" in text)
        ]
        return any(indicators)
    
    def _extract_single_property_data(self, doc, section: Dict) -> Dict[str, str]:
        """Extract all data for a single property section"""
        property_data = self._empty_result_set()
        
        # Extract from summary page
        summary_page = doc[section["summary_page"]]
        summary_text = summary_page.get_text()
        summary_data = self._extract_summary_data_from_text(summary_text)
        property_data.update(summary_data)
        
        # Extract from target section pages
        for page_num in section["target_section_pages"]:
            page = doc[page_num]
            text = page.get_text()
            section_data = self._extract_section_values_from_text(text)
            
            # Update only missing values
            for key, value in section_data.items():
                if property_data.get(key) == "NOT_FOUND":
                    property_data[key] = value
        
        # Remove old heating data extraction - no longer needed
        
        # Calculate total energy cost
        if (property_data.get("energy_cost_247") != "NOT_FOUND" and 
            property_data.get("energy_cost_247a") != "NOT_FOUND"):
            try:
                cost_247 = float(property_data["energy_cost_247"])
                cost_247a = float(property_data["energy_cost_247a"])
                property_data["total_energy_cost"] = str(cost_247 + cost_247a)
            except:
                property_data["total_energy_cost"] = "CALCULATION_ERROR"
        
        return property_data
    
    def _extract_summary_data_from_text(self, text: str) -> Dict[str, str]:
        """Extract property and CO2 data from text"""
        results = {}
        
        # Extract property using improved patterns
        property_patterns = [
            r'(Block\s+\d+[,\s]*Plot\s+\d+)',
            r'Property[:\s]+([^,\n\r]+?)(?:,|\n|\r|$)',
        ]
        
        found_property = None
        for pattern in property_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate.lower() != "reference" and "block" in candidate.lower():
                    found_property = candidate
                    break
                elif "block" in candidate.lower() and "plot" in candidate.lower():
                    found_property = candidate
                    break
                elif not found_property:
                    found_property = candidate
        
        results["property"] = found_property if found_property else "NOT_FOUND"
        
        # Extract CO2 emissions
        co2_patterns = [
            r'CO₂?\s*Emissions[^0-9]*(\d+(?:\.\d+)?)',
            r'CO2\s*Emissions[^0-9]*(\d+(?:\.\d+)?)',
            r'\(t/year\)[^0-9]*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in co2_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results["co2_emissions"] = match.group(1)
                break
        else:
            results["co2_emissions"] = "NOT_FOUND"
        
        return results
    
    def _extract_section_values_from_text(self, text: str) -> Dict[str, str]:
        """Extract (240), (247), (247a) values from text"""
        # First try within the target section
        section_text = self._extract_target_section(text)
        
        # If section is empty or too short, use full text
        if len(section_text.strip()) < 100:
            section_text = text
            
        results = {}
        
        target_items = {
            "240": r'(\d+(?:\.\d+)?)\s*\(240\)',
            "247": r'(\d+(?:\.\d+)?)\s*\(247\)',
            "247a": r'(\d+(?:\.\d+)?)\s*\(247a\)'
        }
        
        for item_key, base_pattern in target_items.items():
            patterns = [
                # Standard patterns with various spacing
                base_pattern,
                rf'(\d+(?:\.\d+)?)\s+\({item_key}\)',
                rf'(\d+(?:\.\d+)?)\t+\({item_key}\)',
                rf'(\d+(?:\.\d+)?)\s*\({item_key}\)',
                # Try with line breaks
                rf'(\d+(?:\.\d+)?)\n\s*\({item_key}\)',
                # Try with more flexible spacing and punctuation
                rf'(\d+(?:\.\d+)?)[^\w\d]*\({item_key}\)',
                # Try patterns like "123.45 (247)" anywhere in text
                rf'(?:^|\s)(\d+(?:\.\d+)?)\s*\({item_key}\)',
            ]
            
            found_value = None
            search_text = section_text
            
            # Try patterns in section first, then full text if not found
            for attempt in [section_text, text]:
                if found_value:
                    break
                    
                for p in patterns:
                    match = re.search(p, attempt, re.MULTILINE)
                    if match:
                        found_value = match.group(1)
                        print(f"    Found ({item_key}): {found_value} using pattern: {p[:30]}...")
                        break
            
            # Store results
            if item_key == "240":
                results["epc_rating_240"] = found_value if found_value else "NOT_FOUND"
            elif item_key == "247":
                results["energy_cost_247"] = found_value if found_value else "NOT_FOUND"
            elif item_key == "247a":
                results["energy_cost_247a"] = found_value if found_value else "NOT_FOUND"
            
            # Debug output if not found
            if not found_value:
                print(f"    ❌ Could not find ({item_key}) in text")
                # Show context around the item number if it exists
                item_matches = re.findall(rf'\({item_key}\)', text)
                if item_matches:
                    print(f"       Found {len(item_matches)} instances of ({item_key}) but no adjacent numbers")
        
        return results
    
    
    def save_properties_to_excel(self, properties: List[Dict], output_filename: str = None) -> str:
        """Save extracted properties to Excel file"""
        if not properties:
            print("❌ No properties to save")
            return None
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sap_properties_{timestamp}.xlsx"
        
        try:
            # Create DataFrame with all properties
            df = pd.DataFrame(properties)
            
            # Reorder columns for better readability
            column_order = [
                "property",
                "co2_emissions", 
                "epc_rating_240",
                "energy_cost_247",
                "energy_cost_247a",
                "total_energy_cost"
            ]
            
            # Only include columns that exist
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # Rename columns for Excel
            column_names = {
                "property": "Property",
                "co2_emissions": "CO2 Emissions (t/year)",
                "epc_rating_240": "Space Heating",
                "energy_cost_247": "Water Heating", 
                "energy_cost_247a": "Shower",
                "total_energy_cost": "Total DHW"
            }
            
            df = df.rename(columns=column_names)
            
            # Save to Excel
            df.to_excel(output_filename, index=False)
            
            print(f"✅ Saved {len(properties)} properties to {output_filename}")
            return output_filename
            
        except Exception as e:
            print(f"❌ Error saving to Excel: {e}")
            return None
    
    def extract_values_with_precision(self, pdf_path: str) -> Dict[str, str]:
        """Extract SAP values using precise OCR analysis of the target section"""
        if not PDF_SUPPORT:
            return self._empty_result_set()
        
        try:
            # First, analyze the document structure
            analysis = self.analyze_sap_structure(pdf_path)
            results = self._empty_result_set()
            
            print(f"📊 Document analysis:")
            print(f"  Summary pages: {analysis['summary_pages']}")
            print(f"  Worksheet pages: {analysis['worksheet_pages']}")
            print(f"  Target section pages: {analysis['target_section_pages']}")
            
            # Extract from summary pages first
            if analysis['summary_pages']:
                summary_data = self._extract_summary_data(pdf_path, analysis['summary_pages'][0])
                results.update(summary_data)
            
            # Extract from target section pages
            if analysis['target_section_pages']:
                for page_num in analysis['target_section_pages']:
                    section_data = self._extract_section_values(pdf_path, page_num, analysis['extraction_template'])
                    results.update(section_data)
            
            # Calculate total if both components found
            if (results.get("energy_cost_247") != "NOT_FOUND" and 
                results.get("energy_cost_247a") != "NOT_FOUND"):
                try:
                    cost_247 = float(results["energy_cost_247"])
                    cost_247a = float(results["energy_cost_247a"])
                    results["total_energy_cost"] = str(cost_247 + cost_247a)
                except:
                    results["total_energy_cost"] = "CALCULATION_ERROR"
            
            return results
            
        except Exception as e:
            print(f"Error in precision extraction: {e}")
            return self._empty_result_set()
    
    def _empty_result_set(self) -> Dict[str, str]:
        """Return empty result set with all expected fields"""
        return {
            "property": "NOT_FOUND",
            "co2_emissions": "NOT_FOUND",
            "epc_rating_240": "NOT_FOUND",
            "energy_cost_247": "NOT_FOUND",
            "energy_cost_247a": "NOT_FOUND",
            "total_energy_cost": "NOT_FOUND"
        }
    
    def _extract_summary_data(self, pdf_path: str, page_num: int) -> Dict[str, str]:
        """Extract property and CO2 data from summary page"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            results = {}
            
            # Extract property using improved patterns - prioritize Block/Plot format
            property_patterns = [
                # First try to find Block X, Plot Y pattern directly
                r'(Block\s+\d+[,\s]*Plot\s+\d+)',
                # Then try Property field patterns, but filter out "Reference"
                r'Property[:\s]+([^,\n\r]+?)(?:,|\n|\r|$)',
                r'Property[:\s]*([^\n\r]+?)(?:\n|\r|$)',
            ]
            
            found_property = None
            for pattern in property_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Skip if it's just "Reference" - look for actual Block/Plot
                    if candidate.lower() != "reference" and "block" in candidate.lower():
                        found_property = candidate
                        break
                    elif "block" in candidate.lower() and "plot" in candidate.lower():
                        found_property = candidate
                        break
                    elif not found_property:  # Keep as fallback if no Block/Plot found
                        found_property = candidate
            
            results["property"] = found_property if found_property else "NOT_FOUND"
            
            # Extract CO2 emissions
            co2_patterns = [
                r'CO₂?\s*Emissions[^0-9]*(\d+(?:\.\d+)?)',
                r'CO2\s*Emissions[^0-9]*(\d+(?:\.\d+)?)',
                r'\(t/year\)[^0-9]*(\d+(?:\.\d+)?)',
            ]
            
            for pattern in co2_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    results["co2_emissions"] = match.group(1)
                    break
            else:
                results["co2_emissions"] = "NOT_FOUND"
            
            return results
            
        except Exception as e:
            print(f"Error extracting summary data: {e}")
            return {"property": "NOT_FOUND", "co2_emissions": "NOT_FOUND"}
    
    def _extract_section_values(self, pdf_path: str, page_num: int, template: Dict) -> Dict[str, str]:
        """Extract (240), (247), (247a) values from the specific section"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            # Extract the target section
            section_text = self._extract_target_section(text)
            print(f"🔍 Analyzing section on page {page_num + 1}...")
            
            results = {}
            target_items = template["target_section"]["target_items"]
            
            for item_key, item_config in target_items.items():
                # Try multiple pattern variations for better matching
                patterns = [
                    item_config["pattern"],  # Primary pattern
                    rf'(\d+(?:\.\d+)?)\s+\({item_key}\)',  # With space
                    rf'(\d+(?:\.\d+)?)\s*\({item_key}\)',  # With optional space
                    rf'(\d+(?:\.\d+)?)\t+\({item_key}\)',  # With tab
                ]
                
                found_value = None
                for pattern in patterns:
                    match = re.search(pattern, section_text)
                    if match:
                        found_value = match.group(1)
                        print(f"  ✅ Found ({item_key}): {found_value}")
                        break
                
                if found_value:
                    # Map to result keys
                    if item_key == "240":
                        results["epc_rating_240"] = found_value
                    elif item_key == "247":
                        results["energy_cost_247"] = found_value
                    elif item_key == "247a":
                        results["energy_cost_247a"] = found_value
                else:
                    print(f"  ❌ Not found: ({item_key})")
                    if item_key == "240":
                        results["epc_rating_240"] = "NOT_FOUND"
                    elif item_key == "247":
                        results["energy_cost_247"] = "NOT_FOUND"
                    elif item_key == "247a":
                        results["energy_cost_247a"] = "NOT_FOUND"
            
            return results
            
        except Exception as e:
            print(f"Error extracting section values: {e}")
            return {
                "epc_rating_240": "NOT_FOUND",
                "energy_cost_247": "NOT_FOUND", 
                "energy_cost_247a": "NOT_FOUND"
            }
    
    def _extract_from_single_page(self, image_path: str, prompt: str, model: ModelProvider) -> Tuple[Dict, Dict]:
        """Extract data from a single page using the appropriate model"""
        if model.provider == "openai":
            return self._extract_openai(image_path, prompt, model.model_name)
        elif model.provider == "claude":
            return self._extract_claude(image_path, prompt, model.model_name)
        elif model.provider == "gemini":
            return self._extract_gemini(image_path, prompt, model.model_name)
        elif model.provider == "grok":
            return self._extract_grok(image_path, prompt, model.model_name)
        else:
            raise ValueError(f"Unknown provider: {model.provider}")
    
    def _extract_openai(self, image_path: str, prompt: str, model_name: str) -> Tuple[Dict, Dict]:
        """Extract using OpenAI models with retry logic"""
        
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine the correct MIME type
        if image_path.lower().endswith('.png'):
            mime_type = "image/png"
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = "image/jpeg"
        elif image_path.lower().endswith('.webp'):
            mime_type = "image/webp"
        elif image_path.lower().endswith('.gif'):
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"  # Default fallback
        
        # Retry logic for connection issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.clients["openai"].chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300,
                    temperature=0,
                    timeout=60  # Longer timeout for large images
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:  # Not the last attempt
                    if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                        print(f"    🔄 Retry {attempt + 1}/{max_retries} due to connection issue...")
                        time.sleep(2)  # Wait 2 seconds before retry
                        continue
                
                # Final attempt or non-retryable error
                if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                    raise Exception(f"Network connection issue after {max_retries} attempts: {error_msg}")
                elif "authentication" in error_msg.lower() or "api" in error_msg.lower():
                    raise Exception(f"API authentication issue: {error_msg}")
                elif "rate limit" in error_msg.lower():
                    raise Exception(f"Rate limit exceeded: {error_msg}")
                else:
                    raise Exception(f"OpenAI API error: {error_msg}")
        
        # Parse response
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "property": "PARSE_ERROR", 
                    "co2_emissions": "PARSE_ERROR",
                    "epc_rating_240": "PARSE_ERROR",
                    "energy_cost_247": "PARSE_ERROR", 
                    "energy_cost_247a": "PARSE_ERROR",
                    "total_energy_cost": "PARSE_ERROR"
                }
        
        tokens = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
        
        return result, tokens
    
    def _extract_claude(self, image_path: str, prompt: str, model_name: str) -> Tuple[Dict, Dict]:
        """Extract using Claude models"""
        # Process and optimize image for Claude
        processed_image_path = self._process_image_for_claude(image_path)
        
        try:
            with open(processed_image_path, "rb") as image_file:
                image_data = image_file.read()
                
            # Check file size (Claude has a 5MB limit)
            if len(image_data) > 5 * 1024 * 1024:  # 5MB
                raise Exception("Image too large for Claude API (>5MB)")
                
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine media type
            if processed_image_path.lower().endswith('.png'):
                media_type = "image/png"
            elif processed_image_path.lower().endswith(('.jpg', '.jpeg')):
                media_type = "image/jpeg"
            elif processed_image_path.lower().endswith('.webp'):
                media_type = "image/webp"
            elif processed_image_path.lower().endswith('.gif'):
                media_type = "image/gif"
            else:
                # Default to JPEG
                media_type = "image/jpeg"
            
            message = self.clients["claude"].messages.create(
                model=model_name,
                max_tokens=300,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
        finally:
            # Clean up processed image if it's different from original
            if processed_image_path != image_path and os.path.exists(processed_image_path):
                os.unlink(processed_image_path)
        
        # Parse response
        content = message.content[0].text
        try:
            result = json.loads(content)
        except:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "property": "PARSE_ERROR", 
                    "co2_emissions": "PARSE_ERROR",
                    "epc_rating_240": "PARSE_ERROR",
                    "energy_cost_247": "PARSE_ERROR", 
                    "energy_cost_247a": "PARSE_ERROR",
                    "total_energy_cost": "PARSE_ERROR"
                }
        
        # Claude provides token usage in the message object
        tokens = {
            "input": message.usage.input_tokens,
            "output": message.usage.output_tokens
        }
        
        return result, tokens
    
    def _process_image_for_claude(self, image_path: str) -> str:
        """Process and optimize image for Claude API"""
        try:
            # Open and process the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (Claude works best with images under 1600px and under 5MB)
                max_size = 1600  # Allow larger images for better text clarity
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save as optimized PNG or JPEG based on content
                temp_path = tempfile.mktemp(suffix='.png')
                
                # Try PNG first for better text clarity
                img.save(temp_path, 'PNG', optimize=True)
                
                # Check file size and compress if needed
                if os.path.getsize(temp_path) > 4.5 * 1024 * 1024:  # 4.5MB threshold
                    # Too large, switch to JPEG with high quality
                    os.unlink(temp_path)
                    temp_path = tempfile.mktemp(suffix='.jpg')
                    img.save(temp_path, 'JPEG', quality=95, optimize=True)
                    
                    # If still too large, reduce quality
                    if os.path.getsize(temp_path) > 4.5 * 1024 * 1024:
                        os.unlink(temp_path)
                        temp_path = tempfile.mktemp(suffix='.jpg')
                        img.save(temp_path, 'JPEG', quality=85, optimize=True)
                
                return temp_path
                
        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}")
            return image_path  # Return original if processing fails
    
    def _extract_gemini(self, image_path: str, prompt: str, model_name: str) -> Tuple[Dict, Dict]:
        """Extract using Gemini models"""
        genai = self.clients["gemini"]
        model = genai.GenerativeModel(model_name)
        
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        
        # Parse response
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        
        try:
            result = json.loads(content)
        except:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {
                    "property": "PARSE_ERROR", 
                    "co2_emissions": "PARSE_ERROR",
                    "epc_rating_240": "PARSE_ERROR",
                    "energy_cost_247": "PARSE_ERROR", 
                    "energy_cost_247a": "PARSE_ERROR",
                    "total_energy_cost": "PARSE_ERROR"
                }
        
        # Estimate tokens (Gemini doesn't provide exact counts)
        tokens = {
            "input": len(prompt.split()) * 1.3 + 85,  # Rough estimate for image
            "output": len(content.split()) * 1.3
        }
        
        return result, tokens
    
    def _extract_grok(self, image_path: str, prompt: str, model_name: str) -> Tuple[Dict, Dict]:
        """Placeholder for Grok extraction"""
        # This is a placeholder as Grok API is not publicly available
        return {
            "property": "GROK_NOT_AVAILABLE",
            "co2_emissions": "GROK_NOT_AVAILABLE"
        }, {"input": 0, "output": 0}
    
    def test_all_models(self, image_path: str, ground_truth: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Test all available models and compare results"""
        results = []
        
        print(f"\nTesting all models on: {image_path}")
        print("=" * 80)
        
        for model in ModelProvider:
            # Skip if API key not available
            if not self.api_keys.get(model.provider):
                print(f"Skipping {model.name} - No API key found")
                continue
            
            # Skip deprecated models that are known to fail
            deprecated_models = ["claude-3-sonnet-20240229", "gpt-4-vision-preview"]
            if any(deprecated in model.model_name for deprecated in deprecated_models):
                print(f"Skipping {model.name} - Model deprecated and no longer available")
                continue
            
            print(f"\nTesting {model.name}...")
            
            try:
                extraction, metrics = self.extract_from_image(image_path, model)
                
                # Add extraction results to metrics
                result = {
                    **metrics,
                    "property_extracted": extraction.get("property", "ERROR"),
                    "co2_extracted": extraction.get("co2_emissions", "ERROR"),
                    "epc_rating_240": extraction.get("epc_rating_240", "ERROR"),
                    "energy_cost_247": extraction.get("energy_cost_247", "ERROR"),
                    "energy_cost_247a": extraction.get("energy_cost_247a", "ERROR"),
                    "total_energy_cost": extraction.get("total_energy_cost", "ERROR"),
                    "error": extraction.get("error", None)
                }
                
                # Add accuracy if ground truth provided
                if ground_truth:
                    result["property_correct"] = result["property_extracted"] == ground_truth.get("property")
                    result["co2_correct"] = result["co2_extracted"] == ground_truth.get("co2_emissions")
                    result["fully_correct"] = result["property_correct"] and result["co2_correct"]
                
                results.append(result)
                
                print(f"✅ Property: {result['property_extracted']}")
                print(f"✅ CO2: {result['co2_extracted']}")
                print(f"✅ EPC Rating (240): {result['epc_rating_240']}")
                print(f"✅ Energy Cost (247): {result['energy_cost_247']}")
                print(f"✅ Energy Cost (247a): {result['energy_cost_247a']}")
                print(f"✅ Total Energy Cost: {result['total_energy_cost']}")
                print(f"💵 Cost: ${result['total_cost']:.6f}")
                print(f"⏱️  Time: {result['processing_time']:.2f}s")
                
            except Exception as e:
                print(f"Error with {model.name}: {str(e)}")
                results.append({
                    "model": model.name,
                    "provider": model.provider,
                    "error": str(e),
                    "property_extracted": "ERROR",
                    "co2_extracted": "ERROR",
                    "epc_rating_240": "ERROR",
                    "energy_cost_247": "ERROR",
                    "energy_cost_247a": "ERROR",
                    "total_energy_cost": "ERROR",
                    "total_cost": 0,
                    "processing_time": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_cost": 0,
                    "output_cost": 0
                })
        
        return pd.DataFrame(results)
    
    def generate_cost_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate a cost analysis report"""
        report = {
            "summary": {
                "total_models_tested": len(results_df),
                "successful_extractions": len(results_df[results_df['error'].isna()]),
                "total_cost": results_df['total_cost'].sum(),
                "average_processing_time": results_df['processing_time'].mean()
            },
            "by_provider": {},
            "recommendations": {}
        }
        
        # Analysis by provider
        for provider in results_df['provider'].unique():
            provider_data = results_df[results_df['provider'] == provider]
            report["by_provider"][provider] = {
                "models_tested": len(provider_data),
                "average_cost": provider_data['total_cost'].mean(),
                "average_time": provider_data['processing_time'].mean(),
                "success_rate": len(provider_data[provider_data['error'].isna()]) / len(provider_data)
            }
        
        # Find best models
        successful_results = results_df[results_df['error'].isna()].copy()
        
        if not successful_results.empty:
            # Best by cost
            cheapest = successful_results.nsmallest(3, 'total_cost')
            report["recommendations"]["most_cost_effective"] = [
                {
                    "model": row['model'],
                    "cost_per_extraction": row['total_cost'],
                    "processing_time": row['processing_time']
                }
                for _, row in cheapest.iterrows()
            ]
            
            # Best by speed
            fastest = successful_results.nsmallest(3, 'processing_time')
            report["recommendations"]["fastest"] = [
                {
                    "model": row['model'],
                    "processing_time": row['processing_time'],
                    "cost_per_extraction": row['total_cost']
                }
                for _, row in fastest.iterrows()
            ]
            
            # Best accuracy (if ground truth provided)
            if 'fully_correct' in successful_results.columns:
                accurate_models = successful_results[successful_results['fully_correct'] == True]
                if not accurate_models.empty:
                    best_accurate = accurate_models.nsmallest(3, 'total_cost')
                    report["recommendations"]["best_accurate_and_cheap"] = [
                        {
                            "model": row['model'],
                            "cost_per_extraction": row['total_cost'],
                            "processing_time": row['processing_time']
                        }
                        for _, row in best_accurate.iterrows()
                    ]
        
        return report
    
    def convert_pdf_to_images(self, pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
        """Convert PDF pages to images"""
        if not PDF_SUPPORT and not PDF2IMAGE_SUPPORT:
            raise Exception("No PDF support available. Install pymupdf or pdf2image")
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sap_pdf_")
        
        image_paths = []
        
        # Try PyMuPDF first (usually faster)
        if PDF_SUPPORT:
            try:
                pdf_document = fitz.open(pdf_path)
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    # Use higher DPI for better text clarity - SAP reports need crisp text
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scale for better OCR
                    image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
                    pix.save(image_path)
                    image_paths.append(image_path)
                pdf_document.close()
                return image_paths
            except Exception as e:
                print(f"PyMuPDF failed: {e}, trying pdf2image...")
        
        # Fallback to pdf2image
        if PDF2IMAGE_SUPPORT:
            try:
                # Higher DPI for better text extraction from SAP reports
                images = convert_from_path(pdf_path, dpi=200)
                for i, image in enumerate(images):
                    image_path = os.path.join(output_dir, f"page_{i + 1}.png")
                    image.save(image_path, 'PNG', optimize=True)
                    image_paths.append(image_path)
                return image_paths
            except Exception as e:
                raise Exception(f"Failed to convert PDF: {e}")
        
        return image_paths
    
    def is_sap_summary_page(self, image_path: str) -> bool:
        """Quick check if an image is likely a SAP summary page"""
        # This is a simple heuristic - you could make it more sophisticated
        # by actually checking for the presence of "Property" and "CO2 Emissions" text
        return True  # For now, process all pages
    
    def process_pdf_file(self, pdf_path: str, model: ModelProvider) -> List[Dict]:
        """Process a PDF file and extract data from relevant pages"""
        print(f"\n📄 Processing PDF: {os.path.basename(pdf_path)}")
        
        # Convert PDF to images
        temp_dir = tempfile.mkdtemp(prefix="sap_pdf_")
        try:
            image_paths = self.convert_pdf_to_images(pdf_path, temp_dir)
            print(f"  📑 Converted to {len(image_paths)} pages")
            
            results = []
            for i, image_path in enumerate(image_paths):
                # Only process first page of each property (SAP summary pages)
                # You can add logic here to detect summary pages
                if i == 0 or (i > 0 and i % 15 == 0):  # Assuming 15 pages per property
                    print(f"  📊 Extracting from page {i + 1}...")
                    try:
                        extraction, metrics = self.extract_from_image(image_path, model)
                        if extraction.get('property') != 'ERROR' and extraction.get('property') != 'NOT_FOUND':
                            results.append({
                                'pdf_file': os.path.basename(pdf_path),
                                'page_number': i + 1,
                                'property': extraction.get('property'),
                                'co2_emissions': extraction.get('co2_emissions'),
                                'cost': metrics['total_cost'],
                                'processing_time': metrics['processing_time']
                            })
                            print(f"    ✅ Found: {extraction.get('property')} - {extraction.get('co2_emissions')} t/year")
                    except Exception as e:
                        print(f"    ❌ Error on page {i + 1}: {str(e)}")
            
            return results
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_all_files_in_directory(self, directory: str = "test", model: Optional[ModelProvider] = None) -> pd.DataFrame:
        """Test all PDF and image files in a directory"""
        if not os.path.exists(directory):
            print(f"❌ Directory '{directory}' not found!")
            return pd.DataFrame()
        
        # Use default model if not specified
        if model is None:
            model = ModelProvider.GEMINI_1_5_FLASH  # Cheapest option
        
        print(f"\n📁 Testing all files in '{directory}' directory with {model.name}")
        print("=" * 80)
        
        # Find all PDF and image files
        pdf_extensions = ('.pdf',)
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
        
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(pdf_extensions)]
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
        
        print(f"Found: {len(pdf_files)} PDFs, {len(image_files)} images")
        
        all_results = []
        total_cost = 0
        
        # Process PDFs
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            try:
                results = self.process_pdf_file(pdf_path, model)
                all_results.extend(results)
                pdf_cost = sum(r['cost'] for r in results)
                total_cost += pdf_cost
                print(f"  💰 Cost for {pdf_file}: ${pdf_cost:.6f}")
            except Exception as e:
                print(f"  ❌ Error processing {pdf_file}: {str(e)}")
        
        # Process standalone images
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)
            try:
                extraction, metrics = self.extract_from_image(img_path, model)
                if extraction.get('property') != 'ERROR':
                    all_results.append({
                        'pdf_file': 'N/A',
                        'page_number': 1,
                        'property': extraction.get('property'),
                        'co2_emissions': extraction.get('co2_emissions'),
                        'cost': metrics['total_cost'],
                        'processing_time': metrics['processing_time']
                    })
                    total_cost += metrics['total_cost']
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {str(e)}")
        
        # Create DataFrame and save results
        if all_results:
            df = pd.DataFrame(all_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"test_directory_results_{timestamp}.xlsx"
            df.to_excel(output_file, index=False)
            
            print(f"\n📊 SUMMARY:")
            print(f"Total properties found: {len(df)}")
            print(f"Total cost: ${total_cost:.4f}")
            print(f"Average cost per property: ${total_cost/len(df):.6f}")
            print(f"Results saved to: {output_file}")
            
            return df
        else:
            print("\n❌ No data extracted from any files")
            return pd.DataFrame()
    
    def batch_process_with_best_model(self, image_paths: List[str], model: ModelProvider) -> pd.DataFrame:
        """Process multiple images with the specified model"""
        results = []
        total_cost = 0
        
        print(f"\n📊 Processing {len(image_paths)} images with {model.name}")
        print("=" * 60)
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                extraction, metrics = self.extract_from_image(image_path, model)
                
                result = {
                    'file_path': image_path,
                    'filename': os.path.basename(image_path),
                    'property': extraction.get('property', 'ERROR'),
                    'co2_emissions': extraction.get('co2_emissions', 'ERROR'),
                    'epc_rating_240': extraction.get('epc_rating_240', 'ERROR'),
                    'energy_cost_247': extraction.get('energy_cost_247', 'ERROR'),
                    'energy_cost_247a': extraction.get('energy_cost_247a', 'ERROR'),
                    'total_energy_cost': extraction.get('total_energy_cost', 'ERROR'),
                    'cost': metrics['total_cost'],
                    'processing_time': metrics['processing_time'],
                    'input_tokens': metrics['input_tokens'],
                    'output_tokens': metrics['output_tokens'],
                    'error': extraction.get('error', None)
                }
                
                results.append(result)
                total_cost += metrics['total_cost']
                
                print(f"  ✅ Property: {result['property']}")
                print(f"  ✅ CO2: {result['co2_emissions']}")
                print(f"  ✅ EPC (240): {result['epc_rating_240']}")
                print(f"  ✅ Energy Cost: {result['total_energy_cost']}")
                print(f"  💵 Cost: ${result['cost']:.6f}")
                print(f"  ⏱️  Time: {result['processing_time']:.2f}s")
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ❌ Error: {error_msg}")
                
                results.append({
                    'file_path': image_path,
                    'filename': os.path.basename(image_path),
                    'property': 'ERROR',
                    'co2_emissions': 'ERROR',
                    'epc_rating_240': 'ERROR',
                    'energy_cost_247': 'ERROR',
                    'energy_cost_247a': 'ERROR',
                    'total_energy_cost': 'ERROR',
                    'cost': 0,
                    'processing_time': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'error': error_msg
                })
        
        print(f"\n📊 BATCH PROCESSING SUMMARY:")
        print(f"Total images processed: {len(results)}")
        successful = len([r for r in results if r['property'] != 'ERROR'])
        print(f"Successful extractions: {successful}")
        print(f"Total cost: ${total_cost:.4f}")
        if successful > 0:
            print(f"Average cost per successful extraction: ${total_cost/successful:.6f}")
        
        return pd.DataFrame(results)

# Interactive menu system
def print_menu():
    """Display the main menu"""
    print("\n" + "="*80)
    print("SAP DOCUMENT EXTRACTION - MODEL TESTING & COMPARISON")
    print("="*80)
    print("\n1. Test ALL models (comprehensive comparison)")
    print("2. Test specific provider (OpenAI, Claude, Gemini)")
    print("3. Test single model")
    print("4. Batch process with selected model")
    print("5. View previous test results")
    print("6. Quick test - Process all files in 'test' folder")
    print("7. 🔬 Precision OCR extraction test (test1.pdf, testhighlight.pdf)")
    print("8. 🏠 Extract ALL properties from combined document")
    print("9. Exit")
    print("\n" + "="*80)

def print_provider_menu():
    """Display provider selection menu"""
    print("\nSelect Provider:")
    print("1. OpenAI")
    print("2. Claude (Anthropic)")
    print("3. Google Gemini")
    print("4. Back to main menu")

def print_model_menu(provider: str):
    """Display model selection menu for a specific provider"""
    print(f"\nSelect {provider} model:")
    
    models = {
        "openai": [
            ("1", ModelProvider.GPT4O, "GPT-4o (Latest, balanced)"),
            ("2", ModelProvider.GPT4O_MINI, "GPT-4o Mini (Cheap, fast)"),
            ("3", ModelProvider.GPT4_TURBO, "GPT-4 Turbo"),
            ("4", ModelProvider.GPT4_VISION, "GPT-4 Vision Preview"),
        ],
        "claude": [
            ("1", ModelProvider.CLAUDE_3_5_SONNET, "Claude 3.5 Sonnet (Latest, balanced)"),
            ("2", ModelProvider.CLAUDE_3_5_HAIKU, "Claude 3.5 Haiku (Fast & cheap)"),
            ("3", ModelProvider.CLAUDE_3_OPUS, "Claude 3 Opus (Most capable)"),
        ],
        "gemini": [
            ("1", ModelProvider.GEMINI_1_5_PRO, "Gemini 1.5 Pro (Most capable)"),
            ("2", ModelProvider.GEMINI_1_5_FLASH, "Gemini 1.5 Flash (Fast & cheap)"),
            ("3", ModelProvider.GEMINI_1_0_PRO, "Gemini 1.0 Pro"),
        ]
    }
    
    if provider.lower() in models:
        for idx, model, desc in models[provider.lower()]:
            cost_info = f"(~${model.input_cost_per_1m:.3f} per 1M input tokens)"
            print(f"{idx}. {desc} {cost_info}")
        print(f"{len(models[provider.lower()]) + 1}. Back to provider menu")
        return models[provider.lower()]
    return []

def get_image_path():
    """Get image path from user with better validation and auto-discovery"""
    
    # First, try to auto-discover PDF files
    current_dir_pdfs = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if current_dir_pdfs:
        print("\n📁 Found PDF files in current directory:")
        for i, pdf in enumerate(current_dir_pdfs, 1):
            print(f"  {i}. {pdf}")
        print("\n📝 Options:")
        print("  - Enter a number to select a PDF")
        print("  - Enter 'custom' to specify a different path")
        print("  - Enter 'list' to see all supported files")
        
        choice = input("\nYour choice: ").strip()
        
        # Handle number selection
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(current_dir_pdfs):
                return current_dir_pdfs[idx]
            else:
                print(f"❌ Invalid selection! Please choose 1-{len(current_dir_pdfs)}")
        
        elif choice.lower() != 'custom' and choice.lower() != 'list':
            # Try to match partial filename
            matching_pdfs = [pdf for pdf in current_dir_pdfs if choice.lower() in pdf.lower()]
            if len(matching_pdfs) == 1:
                print(f"✅ Found match: {matching_pdfs[0]}")
                return matching_pdfs[0]
            elif len(matching_pdfs) > 1:
                print(f"\n🔍 Multiple matches found:")
                for i, pdf in enumerate(matching_pdfs, 1):
                    print(f"  {i}. {pdf}")
                selection = input("\nSelect number: ").strip()
                if selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(matching_pdfs):
                        return matching_pdfs[idx]
    
    # Fallback to manual path entry
    while True:
        print("\n📁 Please provide the path to your SAP report file")
        print("Supported formats: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP")
        print("Examples:")
        print("  - C:\\Automation\\sap_report.pdf")
        print("  - C:\\Users\\YourName\\Documents\\report.pdf")
        print("  - ./test/sap_page1.png")
        print("  - Type 'list' to see files in current directory")
        print("  - Type 'test' to use test folder")
        
        path = input("\nEnter file path: ").strip()
        
        if path.lower() == 'test':
            # Look for test folder
            if os.path.exists('test'):
                return 'test'
            else:
                print("⚠️  No 'test' folder found in current directory.")
                continue
                
        elif path.lower() == 'list':
            # List files in current directory
            print("\n📷 Files in current directory:")
            supported_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            files = [f for f in os.listdir('.') if f.lower().endswith(supported_extensions)]
            
            if files:
                for i, file in enumerate(files, 1):
                    print(f"  {i}. {file}")
                print("\nYou can copy-paste one of these filenames.")
            else:
                print("  No supported files found in current directory.")
            continue
            
        # Check if it's a directory
        if os.path.isdir(path):
            print(f"\n❌ Error: '{path}' is a directory, not a file!")
            print("📁 Looking for files in this directory...")
            
            supported_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            files = [f for f in os.listdir(path) if f.lower().endswith(supported_extensions)]
            
            if files:
                print(f"\nFound {len(files)} files:")
                for i, file in enumerate(files[:10], 1):  # Show first 10
                    print(f"  {i}. {file}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")
                    
                print("\nTry entering the full path, for example:")
                print(f"  {os.path.join(path, files[0])}")
            else:
                print("  No supported files found in this directory.")
            continue
            
        # Check if file exists
        elif os.path.exists(path):
            # Verify it's a supported file
            if not path.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                print(f"\n⚠️  Warning: '{path}' is not a supported file type.")
                proceed = input("Continue anyway? (y/n): ").lower()
                if proceed != 'y':
                    continue
            return path
            
        else:
            print(f"\n❌ File not found: {path}")
            
            # Try to help with common issues
            if '\\' in path and not path.startswith('C:\\'):
                print("💡 Tip: Try using the full path starting with C:\\ or another drive letter")
            
            # Check if file exists in current directory
            filename = os.path.basename(path)
            if os.path.exists(filename):
                print(f"\n💡 Found '{filename}' in current directory!")
                use_this = input(f"Use './{filename}' instead? (y/n): ").lower()
                if use_this == 'y':
                    return filename
            
            retry = input("\nTry again? (y/n): ").lower()
            if retry != 'y':
                return None

def get_ground_truth():
    """Optionally get ground truth from user"""
    print("\nDo you want to provide ground truth for accuracy testing? (y/n): ", end="")
    if input().lower() == 'y':
        property_val = input("Enter the correct Property value: ").strip()
        co2_val = input("Enter the correct CO2 emissions value: ").strip()
        return {"property": property_val, "co2_emissions": co2_val}
    return None

def get_batch_image_paths():
    """Get multiple image paths for batch processing"""
    print("\n📁 BATCH FILE SELECTION")
    print("-" * 40)
    print("Options:")
    print("1. Enter paths one by one")
    print("2. Process all images in a directory")
    print("3. Select from current directory")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    image_paths = []
    
    if choice == '1':
        print("\nEnter image paths (one per line, empty line to finish):")
        while True:
            path = input().strip()
            if not path:
                break
            if os.path.exists(path) and os.path.isfile(path):
                image_paths.append(path)
            else:
                print(f"⚠️  File not found or not a file: {path}")
                
    elif choice == '2':
        dir_path = input("\nEnter directory path: ").strip()
        if os.path.isdir(dir_path):
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            images = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                     if f.lower().endswith(image_extensions)]
            
            if images:
                print(f"\nFound {len(images)} images. Process all? (y/n): ", end="")
                if input().lower() == 'y':
                    image_paths = images
                else:
                    print("\nSelect images to process (comma-separated numbers):")
                    for i, img in enumerate(images, 1):
                        print(f"  {i}. {os.path.basename(img)}")
                    
                    indices = input("\nEnter numbers (e.g., 1,3,5): ").split(',')
                    for idx in indices:
                        try:
                            i = int(idx.strip()) - 1
                            if 0 <= i < len(images):
                                image_paths.append(images[i])
                        except:
                            pass
            else:
                print("No image files found in directory!")
                
    elif choice == '3':
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
        images = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
        
        if images:
            print("\nSelect images from current directory:")
            for i, img in enumerate(images, 1):
                print(f"  {i}. {img}")
            
            print("\nEnter numbers (e.g., 1,3,5 or 'all' for all): ")
            selection = input().strip()
            
            if selection.lower() == 'all':
                image_paths = images
            else:
                indices = selection.split(',')
                for idx in indices:
                    try:
                        i = int(idx.strip()) - 1
                        if 0 <= i < len(images):
                            image_paths.append(images[i])
                    except:
                        pass
        else:
            print("No image files found in current directory!")
    
    return image_paths

def main():
    """Main interactive loop"""
    print("\n🚀 Starting SAP Document Extraction Tool...")
    print("Loading API keys from .env file...")
    
    # Show current working directory
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Initialize extractor
    try:
        extractor = SAPMultiModelExtractor()
        print("✅ Successfully loaded API keys!")
        
        # Check which APIs are available
        available_apis = []
        for provider, key in extractor.api_keys.items():
            if key:
                available_apis.append(provider)
        
        print(f"\nAvailable APIs: {', '.join(available_apis)}")
        
        if not available_apis:
            print("\n❌ No API keys found! Please add your API keys to the .env file.")
            return
        
        # Test API connectivity
        print("\n🔍 Testing API connectivity...")
        for provider in available_apis:
            try:
                if provider == "openai" and extractor.clients.get("openai"):
                    # Simple test call to OpenAI
                    import openai
                    client = extractor.clients["openai"]
                    models = client.models.list()
                    print(f"  ✅ OpenAI: Connected ({len(models.data)} models available)")
                elif provider == "claude" and extractor.clients.get("claude"):
                    # Simple test for Claude (just check if client is initialized)
                    print(f"  ✅ Claude: API key configured")
                elif provider == "gemini" and extractor.clients.get("gemini"):
                    print(f"  ✅ Gemini: API key configured")
            except Exception as e:
                print(f"  ❌ {provider.upper()}: {str(e)}")
                # Remove from available APIs if connection fails
                if provider in available_apis:
                    available_apis.remove(provider)
            
    except Exception as e:
        print(f"\n❌ Error initializing: {e}")
        return
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            # Test all models
            print("\n📊 COMPREHENSIVE MODEL TESTING")
            print("-" * 40)
            
            image_path = get_image_path()
            if not image_path:
                continue
                
            ground_truth = get_ground_truth()
            
            print("\n⏳ Testing all models... This may take a few minutes and incur costs.")
            confirm = input("Continue? (y/n): ").lower()
            if confirm != 'y':
                continue
            
            results_df = extractor.test_all_models(image_path, ground_truth)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(f"model_comparison_{timestamp}.csv", index=False)
            results_df.to_excel(f"model_comparison_{timestamp}.xlsx", index=False)
            
            # Generate and display cost report
            cost_report = extractor.generate_cost_report(results_df)
            
            print("\n" + "="*80)
            print("📈 COST ANALYSIS REPORT")
            print("="*80)
            print(f"Total models tested: {cost_report['summary']['total_models_tested']}")
            print(f"Total cost: ${cost_report['summary']['total_cost']:.4f}")
            print(f"Average time: {cost_report['summary']['average_processing_time']:.2f}s")
            
            if 'most_cost_effective' in cost_report['recommendations']:
                print("\n💰 MOST COST-EFFECTIVE MODELS:")
                for i, model in enumerate(cost_report['recommendations']['most_cost_effective'], 1):
                    print(f"{i}. {model['model']}: ${model['cost_per_extraction']:.6f}/extraction")
            
            # Save report
            with open(f"cost_report_{timestamp}.json", "w") as f:
                json.dump(cost_report, f, indent=2)
            
            print(f"\n✅ Results saved to model_comparison_{timestamp}.csv/.xlsx")
            print(f"✅ Cost report saved to cost_report_{timestamp}.json")
            
        elif choice == '2':
            # Test specific provider
            print_provider_menu()
            provider_choice = input("\nEnter your choice (1-4): ").strip()
            
            provider_map = {'1': 'openai', '2': 'claude', '3': 'gemini'}
            if provider_choice in provider_map:
                provider = provider_map[provider_choice]
                
                if not extractor.api_keys.get(provider):
                    print(f"\n❌ No API key found for {provider}!")
                    continue
                
                image_path = get_image_path()
                if not image_path:
                    continue
                
                print(f"\n⏳ Testing all {provider.upper()} models...")
                
                results = []
                for model in ModelProvider:
                    if model.provider == provider:
                        print(f"\nTesting {model.name}...")
                        try:
                            extraction, metrics = extractor.extract_from_image(image_path, model)
                            print(f"✅ Property: {extraction.get('property', 'ERROR')}")
                            print(f"✅ CO2: {extraction.get('co2_emissions', 'ERROR')}")
                            print(f"💵 Cost: ${metrics['total_cost']:.6f}")
                            print(f"⏱️  Time: {metrics['processing_time']:.2f}s")
                            
                            results.append({**metrics, **extraction})
                        except Exception as e:
                            print(f"❌ Error: {str(e)}")
                
                if results:
                    df = pd.DataFrame(results)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    df.to_csv(f"{provider}_comparison_{timestamp}.csv", index=False)
                    print(f"\n✅ Results saved to {provider}_comparison_{timestamp}.csv")
                    
        elif choice == '3':
            # Test single model
            print_provider_menu()
            provider_choice = input("\nEnter your choice (1-4): ").strip()
            
            provider_map = {'1': 'openai', '2': 'claude', '3': 'gemini'}
            if provider_choice in provider_map:
                provider = provider_map[provider_choice]
                models = print_model_menu(provider)
                
                model_choice = input("\nEnter your choice: ").strip()
                
                if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
                    selected_model = models[int(model_choice)-1][1]
                    
                    image_path = get_image_path()
                    if not image_path:
                        continue
                    
                    # Now PDF conversion is handled internally by extract_from_image
                    print(f"\n⏳ Testing {selected_model.name}...")
                    
                    try:
                        extraction, metrics = extractor.extract_from_image(image_path, selected_model)
                        
                        print("\n" + "="*50)
                        print("📊 EXTRACTION RESULTS")
                        print("="*50)
                        print(f"Model: {selected_model.name}")
                        print(f"Property: {extraction.get('property', 'ERROR')}")
                        print(f"CO2 Emissions: {extraction.get('co2_emissions', 'ERROR')}")
                        print(f"EPC Rating (240): {extraction.get('epc_rating_240', 'ERROR')}")
                        print(f"Energy Cost (247): {extraction.get('energy_cost_247', 'ERROR')}")
                        print(f"Energy Cost (247a): {extraction.get('energy_cost_247a', 'ERROR')}")
                        print(f"Total Energy Cost: {extraction.get('total_energy_cost', 'ERROR')}")
                        print(f"\n💰 COST METRICS:")
                        print(f"Input tokens: {metrics['input_tokens']}")
                        print(f"Output tokens: {metrics['output_tokens']}")
                        print(f"Total cost: ${metrics['total_cost']:.6f}")
                        print(f"Processing time: {metrics['processing_time']:.2f}s")
                        
                    except Exception as e:
                        print(f"\n❌ Error: {str(e)}")
                        
        elif choice == '4':
            # Batch process
            print("\n📁 BATCH PROCESSING")
            print("-" * 40)
            
            # Select model
            print("\nFirst, select the model to use for batch processing:")
            print("\n1. Use recommended (Gemini 1.5 Flash - cheapest)")
            print("2. Select specific model")
            
            model_choice = input("\nEnter choice (1-2): ").strip()
            
            if model_choice == '1':
                selected_model = ModelProvider.GEMINI_1_5_FLASH
            else:
                # Go through model selection process
                print_provider_menu()
                provider_choice = input("\nEnter your choice (1-4): ").strip()
                provider_map = {'1': 'openai', '2': 'claude', '3': 'gemini'}
                
                if provider_choice not in provider_map:
                    continue
                    
                provider = provider_map[provider_choice]
                models = print_model_menu(provider)
                model_idx = input("\nEnter your choice: ").strip()
                
                if not model_idx.isdigit() or int(model_idx) > len(models):
                    continue
                    
                selected_model = models[int(model_idx)-1][1]
            
            # Get image files using the new helper function
            image_paths = get_batch_image_paths()
            
            if not image_paths:
                print("No valid images selected!")
                continue
            
            print(f"\n📊 Ready to process {len(image_paths)} images with {selected_model.name}")
            print("\nSelected files:")
            for i, path in enumerate(image_paths[:5], 1):
                print(f"  {i}. {os.path.basename(path)}")
            if len(image_paths) > 5:
                print(f"  ... and {len(image_paths) - 5} more")
            
            # Estimate cost
            estimated_cost = len(image_paths) * selected_model.input_cost_per_1m * 0.0001  # Rough estimate
            print(f"\n💰 Estimated cost: ~${estimated_cost:.4f}")
            
            confirm = input("\nProceed? (y/n): ").lower()
            if confirm != 'y':
                continue
            
            # Process batch
            results_df = extractor.batch_process_with_best_model(image_paths, selected_model)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"batch_extraction_{timestamp}.xlsx"
            results_df.to_excel(output_file, index=False)
            print(f"\n✅ Results saved to {output_file}")
            
            # Show summary
            print(f"\n📊 BATCH SUMMARY:")
            print(f"Total files processed: {len(results_df)}")
            print(f"Successful extractions: {len(results_df[results_df['property'] != 'ERROR'])}")
            print(f"Total cost: ${results_df['cost'].sum():.4f}")
            
        elif choice == '5':
            # View previous results
            print("\n📂 Previous test results in current directory:")
            files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.json')) and ('comparison' in f or 'report' in f or 'extraction' in f)]
            
            if not files:
                print("No previous results found!")
                continue
                
            for i, file in enumerate(sorted(files, reverse=True)[:10], 1):
                print(f"{i}. {file}")
            
            print("\nPress Enter to continue...")
            input()
            
        elif choice == '6':
            # Quick test
            print("\n🚀 QUICK TEST - Processing all files in current directory")
            print("-" * 60)
            
            # Auto-discover files
            pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
            image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'))]
            
            if not pdf_files and not image_files:
                print("❌ No PDF or image files found in current directory!")
                input("Press Enter to continue...")
                continue
            
            print(f"📊 Found: {len(pdf_files)} PDF files, {len(image_files)} image files")
            
            if pdf_files:
                print("\n📄 PDF files:")
                for i, pdf in enumerate(pdf_files, 1):
                    print(f"  {i}. {pdf}")
            
            if image_files:
                print("\n🖼️  Image files:")
                for i, img in enumerate(image_files, 1):
                    print(f"  {i}. {img}")
            
            # Select model for quick test
            print("\n🤖 Select model for quick test:")
            print("1. Claude 3.5 Sonnet (Recommended)")
            print("2. Claude 3.5 Haiku (Fast & cheap)")
            print("3. Gemini 1.5 Flash (Cheapest)")
            
            model_choice = input("\nEnter choice (1-3): ").strip()
            model_map = {
                '1': ModelProvider.CLAUDE_3_5_SONNET,
                '2': ModelProvider.CLAUDE_3_5_HAIKU,
                '3': ModelProvider.GEMINI_1_5_FLASH
            }
            
            selected_model = model_map.get(model_choice, ModelProvider.CLAUDE_3_5_SONNET)
            
            print(f"\n⏳ Processing with {selected_model.name}...")
            
            try:
                all_results = []
                total_cost = 0
                
                # Process PDF files
                for pdf_file in pdf_files:
                    print(f"\n📄 Processing {pdf_file}...")
                    try:
                        results = extractor.process_pdf_file(pdf_file, selected_model)
                        if results:
                            all_results.extend(results)
                            file_cost = sum(r['cost'] for r in results)
                            total_cost += file_cost
                            print(f"  ✅ Extracted {len(results)} properties, Cost: ${file_cost:.6f}")
                        else:
                            print(f"  ⚠️  No data extracted from {pdf_file}")
                    except Exception as e:
                        print(f"  ❌ Error processing {pdf_file}: {str(e)}")
                
                # Process image files
                for img_file in image_files:
                    print(f"\n🖼️  Processing {img_file}...")
                    try:
                        extraction, metrics = extractor.extract_from_image(img_file, selected_model)
                        if extraction.get('property') not in ['ERROR', 'NOT_FOUND']:
                            all_results.append({
                                'file': img_file,
                                'property': extraction.get('property'),
                                'co2_emissions': extraction.get('co2_emissions'),
                                'cost': metrics['total_cost'],
                                'processing_time': metrics['processing_time']
                            })
                            total_cost += metrics['total_cost']
                            print(f"  ✅ {extraction.get('property')} - {extraction.get('co2_emissions')}")
                        else:
                            print(f"  ⚠️  No data extracted from {img_file}")
                    except Exception as e:
                        print(f"  ❌ Error processing {img_file}: {str(e)}")
                
                # Show results
                if all_results:
                    print(f"\n🎉 QUICK TEST SUMMARY:")
                    print(f"Total properties extracted: {len(all_results)}")
                    print(f"Total cost: ${total_cost:.4f}")
                    print(f"Average cost per property: ${total_cost/len(all_results):.6f}")
                    
                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    df = pd.DataFrame(all_results)
                    output_file = f"quick_test_results_{timestamp}.xlsx"
                    df.to_excel(output_file, index=False)
                    print(f"📊 Results saved to: {output_file}")
                    
                    # Show first few results
                    print(f"\n📋 First few results:")
                    for i, result in enumerate(all_results[:5], 1):
                        print(f"  {i}. {result.get('property', 'N/A')} - {result.get('co2_emissions', 'N/A')} t/year")
                    
                    if len(all_results) > 5:
                        print(f"  ... and {len(all_results) - 5} more (see {output_file})")
                else:
                    print("\n❌ No data extracted from any files!")
                    print("💡 Tips:")
                    print("  - Ensure files contain SAP energy calculation reports")
                    print("  - Check that Property and CO2 emissions are clearly visible")
                    print("  - Try a different model or check your API keys")
                    
            except Exception as e:
                print(f"\n❌ Error during quick test: {str(e)}")
        
        elif choice == '7':
            # Hidden option for testing precision extraction
            print("\n🔬 PRECISION EXTRACTION TEST")
            print("-" * 50)
            
            # Test with the provided sample files
            test_files = ['test1.pdf', 'testhighlight.pdf']
            available_files = [f for f in test_files if os.path.exists(f)]
            
            if not available_files:
                print("❌ No test files found (test1.pdf, testhighlight.pdf)")
                input("Press Enter to continue...")
                continue
            
            print(f"Found test files: {available_files}")
            
            for file in available_files:
                print(f"\n📋 Testing precision extraction on {file}:")
                try:
                    results = extractor.extract_values_with_precision(file)
                    
                    print(f"\n📊 PRECISION EXTRACTION RESULTS for {file}:")
                    print("=" * 60)
                    for key, value in results.items():
                        print(f"{key}: {value}")
                    
                except Exception as e:
                    print(f"❌ Error testing {file}: {str(e)}")
            
            input("\nPress Enter to continue...")
            
        elif choice == '7.5':
            # Hidden debug option
            print("\n🔍 DEBUG EXTRACTION - Show what's found for (247) and (247a)")
            print("-" * 60)
            
            image_path = get_image_path()
            if not image_path or not image_path.lower().endswith('.pdf'):
                print("❌ Need a PDF file for debugging")
                continue
            
            try:
                # Show detailed extraction for first property
                doc = fitz.open(image_path)
                analysis = extractor.analyze_sap_structure(image_path)
                
                if analysis['target_section_pages']:
                    page_num = analysis['target_section_pages'][0]
                    page = doc[page_num]
                    text = page.get_text()
                    
                    print(f"\n📄 Analyzing page {page_num + 1} for (247) and (247a)...")
                    
                    # Show what (247) patterns exist
                    for pattern_name in ['247', '247a']:
                        print(f"\n🔍 Looking for ({pattern_name}):")
                        instances = re.findall(rf'\([^)]*{pattern_name}[^)]*\)', text)
                        if instances:
                            print(f"  Found instances: {instances}")
                            
                            # Show context around each instance
                            for instance in instances:
                                matches = list(re.finditer(re.escape(instance), text))
                                for match in matches:
                                    start = max(0, match.start() - 50)
                                    end = min(len(text), match.end() + 50)
                                    context = text[start:end].replace('\n', ' ')
                                    print(f"  Context: ...{context}...")
                        else:
                            print(f"  No ({pattern_name}) found in text")
                
                doc.close()
                
            except Exception as e:
                print(f"Error: {e}")
            
            input("\nPress Enter to continue...")
            
        elif choice == '8':
            # Process combined document
            print("\n🏠 EXTRACT ALL PROPERTIES FROM COMBINED DOCUMENT")
            print("-" * 60)
            
            # Get PDF file
            image_path = get_image_path()
            if not image_path:
                continue
            
            if not image_path.lower().endswith('.pdf'):
                print("❌ This feature only works with PDF files")
                input("Press Enter to continue...")
                continue
            
            print(f"\n⏳ Processing combined document: {os.path.basename(image_path)}")
            print("This may take several minutes for large documents...")
            
            confirm = input("\nProceed with full document extraction? (y/n): ").lower()
            if confirm != 'y':
                continue
            
            try:
                # Extract all properties
                all_properties = extractor.extract_all_properties_from_document(image_path)
                
                if all_properties:
                    print(f"\n📊 EXTRACTION SUMMARY:")
                    print("=" * 60)
                    
                    # Show summary of extracted properties
                    for i, prop in enumerate(all_properties, 1):
                        print(f"{i:2d}. {prop['property']:<20} | CO2: {prop['co2_emissions']:>6} | Space: {prop['epc_rating_240']:>8}")
                    
                    # Save to Excel
                    excel_file = extractor.save_properties_to_excel(all_properties)
                    
                    if excel_file:
                        print(f"\n✅ All properties saved to: {excel_file}")
                        print(f"📋 Total properties extracted: {len(all_properties)}")
                        
                        # Show column summary
                        print(f"\n📊 EXTRACTION STATISTICS:")
                        found_co2 = len([p for p in all_properties if p['co2_emissions'] != 'NOT_FOUND'])
                        found_space_heating = len([p for p in all_properties if p['epc_rating_240'] != 'NOT_FOUND'])
                        found_water_heating = len([p for p in all_properties if p['energy_cost_247'] != 'NOT_FOUND'])
                        found_shower = len([p for p in all_properties if p['energy_cost_247a'] != 'NOT_FOUND'])
                        found_total_dhw = len([p for p in all_properties if p['total_energy_cost'] != 'NOT_FOUND'])
                        
                        print(f"  CO2 Emissions found: {found_co2}/{len(all_properties)}")
                        print(f"  Space Heating (240) found: {found_space_heating}/{len(all_properties)}")
                        print(f"  Water Heating (247) found: {found_water_heating}/{len(all_properties)}")
                        print(f"  Shower (247a) found: {found_shower}/{len(all_properties)}")
                        print(f"  Total DHW calculated: {found_total_dhw}/{len(all_properties)}")
                    
                else:
                    print("\n❌ No properties could be extracted from the document")
                    print("💡 Tips:")
                    print("  - Ensure the PDF contains SAP energy reports")
                    print("  - Check that properties have clear 'Block X, Plot Y' identifiers")
                    
            except Exception as e:
                print(f"\n❌ Error processing document: {str(e)}")
            
            input("\nPress Enter to continue...")
            
        elif choice == '9':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid choice! Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()