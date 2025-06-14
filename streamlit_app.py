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
    page_title="Elmhurst SAP Report Data Extractor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal, clean styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp {
        background: #ffffff;
    }
    
    .main-header {
        text-align: left;
        padding: 3rem 0 2rem 0;
        margin: -3rem 0 3rem 0;
    }
    
    .main-header h1 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.5rem;
        font-weight: 300;
        color: #1a1a1a;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .info-box {
        background: #fafafa;
        padding: 2rem;
        border-radius: 8px;
        margin: 2rem 0;
        font-size: 0.95rem;
        color: #4a4a4a;
        border: 1px solid #f0f0f0;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .info-box strong {
        font-weight: 500;
        color: #1a1a1a;
    }
    
    .success-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }
    
    .error-box {
        background: #fef5f5;
        border: 1px solid #f0d0d0;
        color: #d32f2f;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }
    
    .stats-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 2rem;
        border-radius: 8px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.2s ease;
        letter-spacing: -0.01em;
    }
    
    .stButton > button:hover {
        background: #000000;
        transform: translateY(-1px);
    }
    
    .stMetric {
        background: #fafafa;
        border-radius: 8px;
        padding: 1.25rem;
        border: 1px solid #f0f0f0;
    }
    
    .stMetric label {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        color: #6a6a6a;
    }
    
    .stMetric [data-testid="metric-value"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1a1a1a;
    }
    
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #1a1a1a;
        font-size: 1.25rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #1a1a1a;
        font-size: 1.1rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    .footer-text {
        text-align: center;
        color: #8a8a8a;
        padding: 3rem 0 2rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 400;
    }
    
    .footer-text p {
        margin: 0;
    }
    
    /* Clean file uploader */
    .stFileUploader {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 2rem;
        background: #fafafa;
    }
    
    .stFileUploader > div {
        background: #fafafa;
    }
    
    /* Remove unnecessary Streamlit branding */
    .css-1y4p8pa {
        padding-top: 2rem;
    }
    
    /* Clean progress bar */
    .stProgress > div > div {
        background-color: #e0e0e0;
    }
    
    .stProgress > div > div > div {
        background-color: #1a1a1a;
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
    # Minimal header
    st.markdown("""
    <div class="main-header">
        <h1>Elmhurst SAP Report Data Extractor</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Minimal info box
    st.markdown("""
    <div class="info-box">
        <strong>Extracts the following data:</strong><br><br>
        • Property identifiers (Block X, Plot Y)<br>
        • CO2 emissions (t/year)<br> 
        • Space heating costs<br>
        • Water heating and shower costs<br>
        • Total DHW calculations
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### Upload Document")
    
    uploaded_file = st.file_uploader(
        "Drop your PDF file here or click to browse",
        type=['pdf'],
        help="Supports single or multi-property SAP reports"
    )
    
    if uploaded_file is not None:
        # Show file details
        st.markdown(f"""
        <div class="success-box">
            <strong>File uploaded:</strong> {uploaded_file.name}<br>
            <strong>Size:</strong> {uploaded_file.size / 1024 / 1024:.2f} MB
        </div>
        """, unsafe_allow_html=True)
        
        # Processing button
        if st.button("Extract Data", type="primary", use_container_width=True):
            
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
                    st.markdown("### Results")
                    
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
                    st.markdown("#### Extracted Data")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download section
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"elmhurst_sap_data_{timestamp}.xlsx"
                    
                    # Create download link
                    download_link = create_download_link(df, filename)
                    
                    st.markdown("#### Download")
                    st.markdown(f"""
                    <div class="stats-box" style="text-align: center;">
                        <div style="font-size: 1.1rem; margin-bottom: 1rem;">
                            {download_link}
                        </div>
                        <small>File: {filename}</small>
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
    
    # Minimal footer
    st.markdown("""
    <div class="footer-text">
        <p>Files are processed locally and not stored.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()