"""
Refactored Flask Application for DTE Admission Chatbot
Streamlined CSV-based admission analysis with advanced analytics.
"""

import os
import logging
import socket
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import google.generativeai as genai
from data_processor import AdmissionDataProcessor
from analytics_engine import AdvancedAnalyticsEngine
from pandas_query_processor import PandasQueryProcessor
try:
    from enhanced_query_processor import EnhancedQueryProcessor
    USE_ENHANCED_PROCESSOR = True
except ImportError:
    USE_ENHANCED_PROCESSOR = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
DEFAULT_PORTS = [5000, 5001, 5002, 5003, 8000, 8080]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables
data_processor = None
analytics_engine = None
pandas_processor = None
gemini_model = None

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_gemini():
    """Initialize Gemini AI model."""
    global gemini_model
    try:
        # Try to get API key from environment or hardcoded value
        api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini model initialized successfully")
            return True
        else:
            logger.warning("GEMINI_API_KEY not found. Advanced AI features will be limited.")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing Gemini: {e}")
        return False

def load_default_data():
    """Load default data file if available."""
    global data_processor, analytics_engine, pandas_processor
    
    # Check for existing data files
    possible_files = [
        'admission_data.csv',
        'EnggAdmissions2024.csv',
        'data.csv'
    ]
    
    for filename in possible_files:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            try:
                api_key = os.getenv('GEMINI_API_KEY')
                data_processor = AdmissionDataProcessor(filepath, api_key)
                if data_processor.process_data():
                    analytics_engine = AdvancedAnalyticsEngine(data_processor)
                    if USE_ENHANCED_PROCESSOR:
                        pandas_processor = EnhancedQueryProcessor(data_processor, api_key)
                    else:
                        pandas_processor = PandasQueryProcessor(data_processor, api_key)
                    logger.info(f"Successfully loaded default data from {filename}")
                    return True
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue
    
    logger.info("No default data file found. Upload a CSV file to get started.")
    return False

def find_available_port(start_port=5000):
    """Find an available port starting from the given port."""
    ports_to_try = [start_port] + [p for p in DEFAULT_PORTS if p != start_port]
    
    for port in ports_to_try:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result != 0:  # Port is available
                return port
        except Exception:
            continue
    
    # If no predefined port is available, find any available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process data."""
    global data_processor, analytics_engine, pandas_processor
    
    logger.info("=== UPLOAD ENDPOINT CALLED ===")
    
    try:
        logger.info("Step 1: Checking for file in request")
        if 'file' not in request.files:
            logger.error("No file found in request.files")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        logger.info(f"Step 2: File object received: {file}")
        logger.info(f"Step 2a: File filename: {file.filename}")
        
        if file.filename == '':
            logger.error("Empty filename provided")
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Step 3: Checking file type for: {file.filename}")
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed. Please upload CSV, XLSX, or XLS files.'}), 400
        
        # Save uploaded file
        logger.info("Step 4: Preparing to save file")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Step 4a: Secure filename: {filename}")
        logger.info(f"Step 4b: Full filepath: {filepath}")
        
        # Ensure upload directory exists
        logger.info("Step 5: Creating upload directory if needed")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        logger.info("Step 6: Saving file to disk")
        file.save(filepath)
        logger.info(f"Step 6a: File saved successfully: {filename}")
        
        # Process the data with dynamic analysis
        logger.info("Step 7: Initializing data processor")
        api_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"Step 7a: API key present: {bool(api_key)}")
        
        logger.info("Step 8: Creating AdmissionDataProcessor")
        try:
            data_processor = AdmissionDataProcessor(filepath, api_key)
            logger.info("Step 8a: AdmissionDataProcessor created successfully")
        except Exception as e:
            logger.error(f"Step 8 FAILED: Error creating AdmissionDataProcessor: {e}")
            import traceback
            logger.error(f"Step 8 TRACEBACK: {traceback.format_exc()}")
            raise
        
        logger.info("Step 9: Processing data")
        try:
            process_result = data_processor.process_data()
            logger.info(f"Step 9a: Process data result: {process_result}")
            if not process_result:
                logger.error("Step 9 FAILED: process_data() returned False")
                return jsonify({'error': 'Failed to process uploaded data. Please check the file format.'}), 400
        except Exception as e:
            logger.error(f"Step 9 FAILED: Error in process_data(): {e}")
            import traceback
            logger.error(f"Step 9 TRACEBACK: {traceback.format_exc()}")
            raise
        
        logger.info("Step 10: Creating analytics engine")
        try:
            analytics_engine = AdvancedAnalyticsEngine(data_processor)
            logger.info("Step 10a: AdvancedAnalyticsEngine created successfully")
        except Exception as e:
            logger.error(f"Step 10 FAILED: Error creating AdvancedAnalyticsEngine: {e}")
            import traceback
            logger.error(f"Step 10 TRACEBACK: {traceback.format_exc()}")
            raise
        
        logger.info("Step 10b: Creating pandas query processor")
        try:
            if USE_ENHANCED_PROCESSOR:
                pandas_processor = EnhancedQueryProcessor(data_processor, api_key)
                logger.info("Step 10c: EnhancedQueryProcessor created successfully")
            else:
                pandas_processor = PandasQueryProcessor(data_processor, api_key)
                logger.info("Step 10c: PandasQueryProcessor created successfully")
        except Exception as e:
            logger.error(f"Step 10b FAILED: Error creating PandasQueryProcessor: {e}")
            import traceback
            logger.error(f"Step 10b TRACEBACK: {traceback.format_exc()}")
            raise
        
        # Get data summary
        logger.info("Step 11: Getting data summary")
        try:
            data_info = data_processor.get_data_info()
            logger.info(f"Step 11a: Data info retrieved: {data_info}")
        except Exception as e:
            logger.error(f"Step 11 FAILED: Error getting data info: {e}")
            import traceback
            logger.error(f"Step 11 TRACEBACK: {traceback.format_exc()}")
            raise
        
        logger.info("Step 12: Preparing response")
        response_data = {
            'message': 'File uploaded and processed successfully!',
            'data_info': data_info
        }
        logger.info(f"Step 12a: Response prepared: {response_data}")
        
        logger.info("=== UPLOAD SUCCESSFUL ===")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"=== UPLOAD FAILED ===")
        logger.error(f"MAIN ERROR: {e}")
        import traceback
        logger.error(f"MAIN TRACEBACK: {traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while processing the file.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries with advanced analytics."""
    try:
        if not data_processor or not analytics_engine:
            return jsonify({
                'error': 'No data loaded. Please upload a CSV file first.',
                'suggestions': [
                    'Upload a CSV file with columns: College Name, College Type, Regional Office, Course Name, Intake, Admissions'
                ]
            }), 400
        
        # Support both 'query' and 'question' for compatibility
        user_query = request.json.get('query', '') or request.json.get('question', '')
        user_query = user_query.strip() if user_query else ''
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        logger.info(f"Processing query: {user_query}")
        
        # Try pandas processor first for natural language queries
        if pandas_processor:
            logger.info("Using pandas processor for natural language query")
            pandas_result = pandas_processor.process_query(user_query)
            
            if pandas_result.get('success'):
                response = {
                    'answer': pandas_result.get('answer', ''),
                    'data': pandas_result.get('data'),
                    'insights': [pandas_result.get('explanation', '')],
                    'pandas_code': pandas_result.get('pandas_code', ''),
                    'suggestions': pandas_processor.get_sample_queries()[:5]
                }
                return jsonify(response)
        
        # Fallback to analytics engine for structured analysis
        logger.info("Falling back to analytics engine")
        result = analytics_engine.process_query(user_query)
        
        # Convert QueryResult to JSON response
        response = {
            'answer': result.answer,
            'data': result.data,
            'insights': result.insights or []
        }
        
        # Add chart information if available
        if result.chart_type:
            response['chart_type'] = result.chart_type
        
        # Add query suggestions
        suggestions = analytics_engine.get_suggestions(user_query)
        response['suggestions'] = suggestions
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'An error occurred while processing your query.',
            'error_details': str(e),
            'suggestions': ['Please try rephrasing your question or upload a new data file.']
        }), 500

@app.route('/fallback_chat', methods=['POST'])
def fallback_chat():
    """Fallback chat using Gemini AI for unstructured queries."""
    try:
        if not gemini_model:
            return jsonify({'error': 'AI model not available'}), 500
        
        if not data_processor:
            return jsonify({'error': 'No data loaded. Please upload a CSV file first.'}), 400
        
        # Support both 'query' and 'question' for compatibility
        user_query = request.json.get('query', '') or request.json.get('question', '')
        user_query = user_query.strip() if user_query else ''
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get data info and dynamic analysis for context
        data_info = data_processor.get_data_info()
        column_info = data_processor.get_column_info_for_llm()
        
        # Create enhanced prompt for Gemini with dynamic column understanding
        prompt = f"""
        You are analyzing admission data for educational institutions. 
        
        {column_info}
        
        Data Summary:
        - Total colleges: {data_info.get('total_colleges', 0)}
        - Total intake: {data_info.get('total_intake', 0)}
        - Total admissions: {data_info.get('total_admissions', 0)}
        - Total vacancies: {data_info.get('total_vacancies', 0)}
        - Overall vacancy percentage: {data_info.get('overall_vacancy_percentage', 0)}%
        
        Available columns in the data:
        - college_name, college_type, regional_office, course_name
        - intake, admissions, vacancies, percentage_vacancies
        
        User Query: "{user_query}"
        
        Provide a helpful response based on the data context. If specific analysis is needed, 
        generate Python pandas code that works with the available columns.
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            return jsonify({
                'answer': response.text.strip(),
                'source': 'AI Assistant',
                'suggestions': analytics_engine.get_suggestions(user_query) if analytics_engine else []
            })
            
        except Exception as ai_error:
            logger.error(f"Gemini API error: {ai_error}")
            return jsonify({
                'error': 'AI service temporarily unavailable. Please try the structured queries.',
                'suggestions': analytics_engine.get_suggestions(user_query) if analytics_engine else []
            }), 500
        
    except Exception as e:
        logger.error(f"Error in fallback chat: {e}")
        return jsonify({'error': 'An error occurred while processing your query.'}), 500

@app.route('/data_info', methods=['GET'])
def get_data_info():
    """Get information about loaded data."""
    if not data_processor:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        data_info = data_processor.get_data_info()
        return jsonify(data_info)
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return jsonify({'error': 'Error retrieving data information'}), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get query suggestions based on dynamic data analysis."""
    if not data_processor or not analytics_engine:
        return jsonify({
            'suggestions': [
                'Upload a CSV file to get started',
                'Any CSV format is supported - the system will automatically understand your data structure'
            ]
        })
    
    try:
        partial_query = request.args.get('q', '')
        
        # Get dynamic suggestions first
        dynamic_suggestions = data_processor.get_suggested_queries()
        
        # Get analytics engine suggestions
        analytics_suggestions = analytics_engine.get_suggestions(partial_query)
        
        # Combine and deduplicate
        all_suggestions = list(dict.fromkeys(dynamic_suggestions + analytics_suggestions))
        
        return jsonify({'suggestions': all_suggestions[:8]})  # Limit to 8 suggestions
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return jsonify({'suggestions': []})

@app.route('/analysis_info', methods=['GET'])
def get_analysis_info():
    """Get dynamic CSV analysis information."""
    if not data_processor:
        return jsonify({'error': 'No data loaded'}), 400
    
    try:
        analysis_info = data_processor.get_dynamic_analysis_info()
        column_info = data_processor.get_column_info_for_llm()
        
        return jsonify({
            'analysis_info': analysis_info,
            'column_description': column_info,
            'success': True
        })
    except Exception as e:
        logger.error(f"Error getting analysis info: {e}")
        return jsonify({'error': 'Error retrieving analysis information'}), 500

@app.route('/test_gemini', methods=['GET'])
def test_gemini():
    """Test Gemini API connectivity."""
    try:
        if not gemini_model:
            return jsonify({
                'status': 'error',
                'message': 'Gemini API not configured',
                'details': 'No API key found in environment'
            }), 400
        
        # Simple test query
        test_prompt = "Respond with exactly: 'API working correctly'"
        response = gemini_model.generate_content(test_prompt)
        
        return jsonify({
            'status': 'success',
            'message': 'Gemini API is working',
            'test_response': response.text.strip(),
            'api_configured': True
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Gemini API test failed',
            'error': str(e),
            'api_configured': bool(gemini_model)
        }), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear analytics cache."""
    if analytics_engine:
        analytics_engine.clear_cache()
        return jsonify({'message': 'Cache cleared successfully'})
    return jsonify({'message': 'No cache to clear'})

# Initialize the application
def initialize_app():
    """Initialize the application with default settings."""
    # Create upload directory
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize Gemini
    initialize_gemini()
    
    # Try to load default data
    load_default_data()
    
    logger.info("Application initialized successfully")

if __name__ == '__main__':
    initialize_app()
    
    # Check for required dependencies
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        print("Please install required packages: pip install pandas numpy openpyxl")
        exit(1)
    
    # Get port from environment or find available port
    port = int(os.environ.get('PORT', 0))
    if port == 0:
        port = find_available_port()
    
    # Start the application
    logger.info(f"Starting Flask application on port {port}...")
    print(f"\n🚀 Server starting on http://localhost:{port}")
    print(f"📊 DTE Admission Chatbot is ready!")
    print(f"💡 If port {port} doesn't work, try setting PORT environment variable")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=port)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is still in use. Trying to find another port...")
            backup_port = find_available_port(port + 1)
            logger.info(f"Starting on backup port {backup_port}")
            print(f"\n🔄 Retrying on http://localhost:{backup_port}")
            app.run(debug=True, host='0.0.0.0', port=backup_port)
        else:
            raise e