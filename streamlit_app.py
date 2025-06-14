import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime
import base64
from io import BytesIO

# Import the streamlined extraction class
from sap_extractor import SAPPropertyExtractor

# Page configuration
st.set_page_config(
    page_title="SAP Property Data Extractor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f4e79 0%, #2e7bb4 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2e7bb4;
        margin: 1rem 0;
    }
    .stats-container {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .upload-section {
        border: 2px dashed #2e7bb4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_download_link(df, filename):
    """Create a download link for the Excel file"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='SAP Properties')
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Download Excel File</a>'

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏠 SAP Property Data Extractor</h1>
        <p>Professional SAP document processing for property developers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🚀 Fast Processing</h3>
            <p>Extract data from multiple properties in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Clean Output</h3>
            <p>Get organized Excel files ready for analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🔒 Secure</h3>
            <p>Your documents are processed locally and securely</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("## 📄 Upload Your SAP Document")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your SAP PDF file here",
        type=['pdf'],
        help="Upload a PDF containing SAP energy calculation reports. Supports multi-property documents."
    )
    
    if uploaded_file is not None:
        # Show file details
        st.markdown(f"""
        <div class="success-box">
            <strong>📎 File uploaded:</strong> {uploaded_file.name}<br>
            <strong>📏 Size:</strong> {uploaded_file.size / 1024 / 1024:.2f} MB
        </div>
        """, unsafe_allow_html=True)
        
        # Processing button
        if st.button("🔄 Extract Property Data", type="primary", use_container_width=True):
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize extractor
                status_text.text("🔧 Initializing extraction system...")
                progress_bar.progress(10)
                
                extractor = SAPPropertyExtractor()
                
                # Save uploaded file temporarily
                status_text.text("💾 Processing uploaded file...")
                progress_bar.progress(20)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract properties
                status_text.text("🔍 Scanning document for properties...")
                progress_bar.progress(40)
                
                all_properties = extractor.extract_all_properties_from_document(tmp_file_path)
                
                progress_bar.progress(80)
                
                if all_properties:
                    # Create DataFrame
                    df = pd.DataFrame(all_properties)
                    
                    # Reorder and rename columns
                    column_order = [
                        "property", "co2_emissions", "epc_rating_240",
                        "energy_cost_247", "energy_cost_247a", "total_energy_cost"
                    ]
                    
                    existing_columns = [col for col in column_order if col in df.columns]
                    df = df[existing_columns]
                    
                    # Rename columns
                    column_names = {
                        "property": "Property",
                        "co2_emissions": "CO2 Emissions (t/year)",
                        "epc_rating_240": "Space Heating",
                        "energy_cost_247": "Water Heating",
                        "energy_cost_247a": "Shower",
                        "total_energy_cost": "Total DHW"
                    }
                    df = df.rename(columns=column_names)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Extraction completed successfully!")
                    
                    # Show results
                    st.markdown("## 📊 Extraction Results")
                    
                    # Statistics
                    total_properties = len(all_properties)
                    found_co2 = len([p for p in all_properties if p['co2_emissions'] != 'NOT_FOUND'])
                    found_space = len([p for p in all_properties if p['epc_rating_240'] != 'NOT_FOUND'])
                    found_water = len([p for p in all_properties if p['energy_cost_247'] != 'NOT_FOUND'])
                    found_shower = len([p for p in all_properties if p['energy_cost_247a'] != 'NOT_FOUND'])
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Properties Found", total_properties)
                    with col2:
                        st.metric("CO2 Data", f"{found_co2}/{total_properties}")
                    with col3:
                        st.metric("Space Heating", f"{found_space}/{total_properties}")
                    with col4:
                        st.metric("Water Heating", f"{found_water}/{total_properties}")
                    with col5:
                        st.metric("Shower Data", f"{found_shower}/{total_properties}")
                    
                    # Show data table
                    st.markdown("### 📋 Property Data")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"sap_properties_{timestamp}.xlsx"
                    
                    # Create download link
                    download_link = create_download_link(df, filename)
                    
                    st.markdown("### 📥 Download Results")
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
                        <h4>Your Excel file is ready!</h4>
                        <p style="margin: 1rem 0;">Click below to download your extracted property data</p>
                        <div style="font-size: 1.2rem;">
                            {download_link}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Summary info
                    st.markdown(f"""
                    <div class="stats-container">
                        <strong>📈 Summary:</strong><br>
                        • Successfully extracted {total_properties} properties<br>
                        • File: {filename}<br>
                        • Processed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown("""
                    <div class="error-box">
                        <strong>❌ No properties found</strong><br>
                        The uploaded file doesn't appear to contain recognizable SAP property data.
                        Please ensure the PDF contains SAP energy calculation reports with property identifiers.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                progress_bar.progress(100)
                st.markdown(f"""
                <div class="error-box">
                    <strong>❌ Processing Error</strong><br>
                    {str(e)}<br><br>
                    Please ensure your PDF file is valid and contains SAP energy reports.
                </div>
                """, unsafe_allow_html=True)
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>How it works:</h4>
        <p>
            1. Upload your SAP PDF document (single or multi-property)<br>
            2. Our system automatically identifies property sections<br>
            3. Extracts key data: Property ID, CO2 emissions, Space heating, Water heating, Shower data<br>
            4. Download clean Excel file ready for analysis
        </p>
        <p><em>All processing is done securely - your files are not stored.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()