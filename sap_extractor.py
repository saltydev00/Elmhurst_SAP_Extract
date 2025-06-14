import os
import re
import tempfile
import shutil
from typing import Dict, List
from datetime import datetime
import pandas as pd
from PIL import Image

# PDF handling imports
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_SUPPORT = True
except ImportError:
    PDF2IMAGE_SUPPORT = False


class SAPPropertyExtractor:
    """Streamlined SAP property data extractor for Streamlit deployment"""
    
    def __init__(self):
        pass
    
    def extract_all_properties_from_document(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract data from all properties in a combined SAP document"""
        if not PDF_SUPPORT:
            raise Exception("PyMuPDF not available. Please install with: pip install pymupdf")
        
        try:
            doc = fitz.open(pdf_path)
            all_properties = []
            
            # First pass: Identify all property sections
            property_sections = self._identify_property_sections(doc)
            
            # Second pass: Extract data from each property
            for i, section in enumerate(property_sections, 1):
                property_data = self._extract_single_property_data(doc, section)
                
                if property_data["property"] != "NOT_FOUND":
                    all_properties.append(property_data)
            
            doc.close()
            return all_properties
            
        except Exception as e:
            raise Exception(f"Error processing document: {e}")
    
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
                base_pattern,
                rf'(\d+(?:\.\d+)?)\s+\({item_key}\)',
                rf'(\d+(?:\.\d+)?)\t+\({item_key}\)',
                rf'(\d+(?:\.\d+)?)\n\s*\({item_key}\)',
                rf'(\d+(?:\.\d+)?)[^\w\d]*\({item_key}\)',
                rf'(?:^|\s)(\d+(?:\.\d+)?)\s*\({item_key}\)',
            ]
            
            found_value = None
            
            # Try patterns in section first, then full text if not found
            for attempt in [section_text, text]:
                if found_value:
                    break
                    
                for p in patterns:
                    match = re.search(p, attempt, re.MULTILINE)
                    if match:
                        found_value = match.group(1)
                        break
            
            # Store results
            if item_key == "240":
                results["epc_rating_240"] = found_value if found_value else "NOT_FOUND"
            elif item_key == "247":
                results["energy_cost_247"] = found_value if found_value else "NOT_FOUND"
            elif item_key == "247a":
                results["energy_cost_247a"] = found_value if found_value else "NOT_FOUND"
        
        return results
    
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
    
    def save_properties_to_excel(self, properties: List[Dict], output_filename: str = None) -> str:
        """Save extracted properties to Excel file"""
        if not properties:
            raise Exception("No properties to save")
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sap_properties_{timestamp}.xlsx"
        
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
        
        return output_filename