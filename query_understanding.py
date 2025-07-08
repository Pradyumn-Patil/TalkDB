import google.generativeai as genai
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class QueryUnderstanding:
    """Converts natural language queries to structured analysis parameters using Gemini."""
    
    def __init__(self, model):
        """Initialize with a Gemini model instance."""
        self.model = model
        # Define available columns and their meanings
        self.available_columns = {
            'ProgramID': 'Program or course identifier',
            'InstituteCode': 'Unique code for each institute',
            'InstituteName': 'Name of the educational institute',
            'DistrictName': 'District where institute is located',
            'RegionName': 'Regional office or area',
            'Status': 'Type of institute (e.g., Government, Private)',
            'Autonomu': 'Autonomy status of the institute',
            'Minority': 'Minority status of the institute'
        }
        
        # Define unsupported but commonly requested metrics
        self.unsupported_metrics = {
            'intake': 'student intake capacity',
            'admissions': 'number of admissions',
            'vacancies': 'vacant seats',
            'cutoff': 'cutoff marks',
            'fees': 'fee structure'
        }

    def _clean_json_response(self, text: str) -> str:
        """Clean the response text to get valid JSON."""
        # Remove markdown code block markers if present
        text = text.replace('```json', '').replace('```', '')
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def parse_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Parse a natural language query into structured parameters."""
        try:
            # Create a prompt that explains available and unavailable data
            prompt = f"""
            You are analyzing engineering college admission data. Here are the ONLY available columns:
            {json.dumps(self.available_columns, indent=2)}

            The following metrics are NOT available in the dataset:
            {json.dumps(self.unsupported_metrics, indent=2)}

            For the query: "{query}"

            1. First, determine if the query can be answered with the available columns.
            2. If NOT answerable, respond with a JSON object:
               {{
                 "can_answer": false,
                 "explanation": "Clear explanation of why the query cannot be answered and what data is missing"
               }}
            
            3. If answerable, respond with a JSON object containing:
               {{
                 "can_answer": true,
                 "query_type": "One of: DISTRIBUTION, COUNT_BY, GROUP_SUMMARY",
                 "parameters": {{
                   "group_by": ["column names to group by"],
                   "metrics": ["columns to analyze"],
                   "filters": {{"column": "value"}} or {{}},
                   "limit": number or null
                 }},
                 "explanation": "How you'll answer with available data"
               }}

            Example unsupported query: "Which college has highest intake?"
            Response: {{"can_answer": false, "explanation": "Cannot determine highest intake as intake capacity data is not available in the dataset. Available institute data only includes name, location, and type."}}

            Example supported query: "How many government colleges are there in each region?"
            Response: {{"can_answer": true, "query_type": "DISTRIBUTION", "parameters": {{"group_by": ["RegionName"], "metrics": ["Status"], "filters": {{"Status": "Government"}}, "limit": null}}, "explanation": "Will show count of government colleges by region using Status and RegionName columns"}}

            IMPORTANT: Return ONLY the JSON object, no markdown formatting or other text.
            """

            # Get model response
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0},
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            )

            # Clean and parse the response
            try:
                cleaned_text = self._clean_json_response(response.text)
                parsed = json.loads(cleaned_text)
                logger.info(f"Successfully parsed query: {query}")
                logger.debug(f"Parsed result: {parsed}")
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {response.text}")
                logger.error(f"JSON error: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error in parse_query: {str(e)}")
            return None

    def _validate_query_structure(self, query_dict: Dict[str, Any]) -> None:
        """Validate the structure of the parsed query."""
        required_fields = {'query_type', 'parameters', 'explanation'}
        missing_fields = required_fields - set(query_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        if not isinstance(query_dict['parameters'], dict):
            raise ValueError("Parameters must be a dictionary")

        if not isinstance(query_dict['explanation'], str):
            raise ValueError("Explanation must be a string")
            
        # Add default values for optional parameters if needed
        if query_dict['query_type'] in ['HIGHEST', 'MINIMUM']:
            if 'limit' not in query_dict['parameters']:
                query_dict['parameters']['limit'] = 1 