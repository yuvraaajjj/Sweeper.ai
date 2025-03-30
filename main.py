from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import os
import hashlib
import re
from dotenv import load_dotenv
from huggingface_hub import login
from flask import Flask, request, jsonify, send_file, render_template, url_for, redirect, flash, session
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

# Flask App
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")  # Add a secret key for session

# File Upload Config
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limit uploads to 16MB

# Define State
class DataState(BaseModel):
    file_path: str = ""
    raw_data: pd.DataFrame = None  
    cleaned_data: pd.DataFrame = None
    vector_db: any = None
    analysis_report: str = ""
    domain: str = ""
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Step 1: Read File
def read_file(state: DataState) -> dict:
    """Reads CSV file and updates state."""
    try:
        logger.info(f"Reading file from {state.file_path}")
        df = pd.read_csv(state.file_path)
        logger.info(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        return {"raw_data": df}
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return {"raw_data": None}

# Step 2: Clean Data
def clean_data(state: DataState) -> dict:
    """Cleans dataset as per LLM recommendations."""
    if state.raw_data is None:
        logger.warning("No raw data to clean")
        return {"cleaned_data": None, "file_path": state.file_path}
    
    logger.info("Starting data cleaning process")
    df = state.raw_data.copy()

    # Drop empty columns and duplicate columns
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert 'Date' column to YYYY-MM-DD
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

    # Remove commas from numerical columns and convert
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '', regex=True)
            except Exception as e:
                logger.warning(f"Could not clean column {col}: {str(e)}")
    
    # Fix the deprecated 'errors=ignore' warning
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            logger.debug(f"Column {col} cannot be converted to numeric: {str(e)}")

    # Save cleaned file
    cleaned_file_path = state.file_path.replace(".csv", "_cleaned.csv")
    df.to_csv(cleaned_file_path, index=False)
    logger.info(f"Saved cleaned file to {cleaned_file_path}")

    # Return dictionary with updated values
    return {
        "cleaned_data": df,
        "file_path": cleaned_file_path
    }

# Step 3: Convert to Vectors
def convert_to_vectors(state: DataState) -> dict:
    """Converts cleaned data into FAISS vector storage."""
    if state.cleaned_data is None:
        logger.warning("No cleaned data to vectorize")
        return {"vector_db": None}

    logger.info("Starting vectorization process")
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    file_hash = hashlib.md5(state.file_path.encode()).hexdigest()
    faiss_index_path = f"faiss_indexes/{file_hash}"
    os.makedirs("faiss_indexes", exist_ok=True)

    vector_db = None
    batch_size = 5000
    
    if os.path.exists(faiss_index_path):
        logger.info(f"Loading existing FAISS index from {faiss_index_path}")
        vector_db = FAISS.load_local(faiss_index_path, model, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new FAISS index")
        total_rows = len(state.cleaned_data)
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = state.cleaned_data.iloc[batch_start:batch_end]
            documents = [Document(page_content=" | ".join(map(str, row)), metadata={"index": index}) for index, row in batch_df.iterrows()]
            
            if vector_db is None:
                vector_db = FAISS.from_documents(documents, model)
            else:
                vector_db.add_documents(documents)
            
            logger.info(f"Processed batch {batch_start}-{batch_end} of {total_rows} rows")

        if vector_db is not None:
            vector_db.save_local(faiss_index_path)
            logger.info(f"Saved FAISS index to {faiss_index_path}")

    return {"vector_db": vector_db}

# Step 4: Create LLM
def create_llm():
    """Creates LLM using Google Gemini API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        raise ValueError("GOOGLE_API_KEY is required for analysis")
    
    os.environ["GOOGLE_API_KEY"] = api_key
    logger.info("Initializing Gemini LLM")
    return GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Step 5: Analyze Data
def analyze_data(state: DataState) -> dict:
    """Uses LLM to analyze dataset and classify domain."""
    if state.vector_db is None:
        logger.warning("No vector database available for analysis")
        return {
            "analysis_report": "No data available for analysis.",
            "domain": "Unknown"
        }
    
    logger.info("Starting data analysis with LLM")
    try:
        llm = create_llm()

        retrieved_data = state.vector_db.similarity_search("Analyze this dataset", k=5)
        sample_data = "\n".join([doc.page_content for doc in retrieved_data])
        logger.info(f"Retrieved {len(retrieved_data)} sample documents for analysis")

        prompt_template = PromptTemplate.from_template("""
            You are a data analyst. Analyze the dataset and provide:

            - **Domain Classification:** <One-word domain>  
            - **Key Insights:** <List insights>  
            - **Preprocessing Steps:** <List steps>  
            - **Issues:** <List issues>  
            - **Feature Engineering Suggestions:** <List of relevant features that can be used for model training>  
            - **Potential Data Enhancements:** <Suggestions to improve dataset quality, e.g., collecting more data, handling missing values differently, feature transformations, etc.>  
            - **Recommended Machine Learning Models:** <List of ML models suitable for this type of data>  

            Sample Data:  
            {sample_data}
                    
            """)

        chain = (
            {"sample_data": lambda _: sample_data}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        analysis_report = chain.invoke({})
        logger.info("Successfully generated analysis report")

        # Extract domain
        domain = "Unknown"
        domain_match = re.search(r"Domain Classification:\s*(.+?)(?:\n|$)", analysis_report, re.IGNORECASE)
        if domain_match:
            domain = domain_match.group(1).strip()
            logger.info(f"Detected domain: {domain}")
        
        # Return a dictionary with the analysis results
        logger.info(f"Analysis complete. Domain: {domain}")
        return {
            "analysis_report": analysis_report,
            "domain": domain
        }
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return {
            "analysis_report": f"Error analyzing data: {str(e)}",
            "domain": "Error"
        }

# Workflow Graph
workflow = StateGraph(DataState)
workflow.add_node("read_file", read_file)
workflow.add_node("clean_data", clean_data)
workflow.add_node("convert_to_vectors", convert_to_vectors)
workflow.add_node("analyze_data", analyze_data)

workflow.add_edge("read_file", "clean_data")
workflow.add_edge("clean_data", "convert_to_vectors")
workflow.add_edge("convert_to_vectors", "analyze_data")

workflow.set_entry_point("read_file")
workflow.set_finish_point("analyze_data")

app_workflow = workflow.compile()

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logger.info("Received POST request for file upload")
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.warning("No file part in request")
            flash("No file part")
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            logger.warning("No file selected")
            flash("No file selected")
            return redirect(request.url)
        
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            try:
                # Secure the filename to prevent security issues
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                logger.info(f"Saving uploaded file to {file_path}")
                # Save the uploaded file
                file.save(file_path)
                
                # Process the file through the workflow
                logger.info("Starting workflow processing")
                state = DataState(file_path=file_path)
                
                # Get the initial state before workflow
                original_file_path = state.file_path
                
                # Run workflow
                final_state_dict = app_workflow.invoke(state)
                
                # Important: Extract analysis results directly from the returned dict
                analysis_report = final_state_dict.get("analysis_report", "No analysis available")
                domain = final_state_dict.get("domain", "Unknown")
                
                # Store in session
                session['analysis_report'] = analysis_report
                session['domain'] = domain
                
                logger.info(f"Analysis report: {analysis_report[:100]}...")  # Log first 100 chars
                logger.info(f"Domain: {domain}")
                
                # Determine cleaned file path
                if original_file_path.endswith('.csv'):
                    cleaned_file_path = original_file_path.replace(".csv", "_cleaned.csv")
                else:
                    cleaned_file_path = original_file_path.replace(".xlsx", "_cleaned.csv")
                
                # Store the file path in the session for download
                session['cleaned_file_path'] = cleaned_file_path
                download_filename = os.path.basename(cleaned_file_path)
                logger.info(f"Processing complete. Cleaned file: {download_filename}")
                
                # Set a flash message for successful upload
                flash("File processed successfully!")
                
                # Redirect to the result page with the download link
                return redirect(url_for('download_result', filename=download_filename))
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                flash(f"Error processing file: {str(e)}")
                return render_template('index.html', error=f"Error processing file: {str(e)}")
        else:
            logger.warning(f"Invalid file format: {file.filename}")
            flash("Invalid file format. Please upload a CSV or Excel file.")
            return render_template('index.html', error="Invalid file format. Please upload a CSV or Excel file.")
    
    # If GET request, just render the upload form
    logger.info("Rendering index page")
    return render_template('index.html')

# Update the download_result route to include analysis data
@app.route('/result/<filename>')
def download_result(filename):
    logger.info(f"Rendering result page for file: {filename}")
    
    # Generate file download URL
    file_url = url_for('download_file', filename=filename)
    
    # Get analysis data from session
    analysis_report = session.get('analysis_report', 'No analysis available')
    domain = session.get('domain', 'Unknown')
    
    logger.info(f"Rendering result page with domain: {domain}")
    logger.info(f"Analysis report excerpt: {analysis_report[:100] if analysis_report else 'None'}...")
    
    return render_template('result.html', 
                          file_url=file_url, 
                          analysis_report=analysis_report,
                          domain=domain)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logger.info(f"Download request for file: {file_path}")
    if os.path.exists(file_path):
        logger.info(f"Sending file: {file_path}")
        return send_file(file_path, as_attachment=True)
    else:
        logger.error(f"File not found: {file_path}")
        flash("File not found")
        return "File not found", 404

# Create static folder structure if it doesn't exist
os.makedirs('static/image', exist_ok=True)

if __name__ == '__main__':
    logger.info("Starting Flask server")
    # Set use_reloader=False to prevent automatic reloading
    app.run(debug=True, use_reloader=False)