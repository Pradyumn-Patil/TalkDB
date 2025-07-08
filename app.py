# app.py
import os
import logging
from flask import Flask, request, jsonify, render_template
from query_handler import QueryHandler

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:
    logger.warning("python-dotenv not installed. Will try to use environment variables directly.")
    load_dotenv = lambda: None

# ==============================================================================
# --- Configuration Section ---
# ==============================================================================
EXCEL_FILE = '/Users/pradyumn/Desktop/DTE_admission_chatbot/EnggAdmissions2024.xlsx'

app = Flask(__name__)

def initialize_query_handler():
    """Initialize the query handler with data."""
    try:
        if not os.path.exists(EXCEL_FILE):
            logger.error(f"Excel file not found: {EXCEL_FILE}")
            raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")
            
        handler = QueryHandler(EXCEL_FILE)
        logger.info("Query handler initialized successfully")
        return handler
        
    except Exception as e:
        logger.error(f"Error initializing query handler: {str(e)}")
        return None

@app.route('/')
def index():
    """Renders the chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and generate responses."""
    try:
        if query_handler is None:
            logger.error("Query handler not initialized")
            return jsonify({'error': 'System not properly initialized. Please check the logs.'}), 500
            
        user_query = request.json.get('query')
        if not user_query:
            logger.error("No query provided")
            return jsonify({'error': 'No query provided'}), 400

        logger.debug(f"Processing query: {user_query}")
        
        # Process the query using PandasAI
        result = query_handler.process_query(user_query)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
            
        logger.info("Successfully processed chat request")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# Load environment variables
load_dotenv()

# Initialize query handler
logger.info("Initializing query handler...")
query_handler = initialize_query_handler()

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    if query_handler is not None:
        logger.info("Starting Flask server...")
        app.run(debug=True)
    else:
        logger.error("Application not starting due to initialization errors")
        print("Application will not start due to initialization errors. Please check the logs above.")