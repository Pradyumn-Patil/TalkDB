"""
Pandas-based Query Processor for Natural Language to Pandas Code
Uses LLM to understand user queries and generate appropriate pandas code.
"""

import pandas as pd
import numpy as np
import logging
import json
import re
from typing import Dict, Any, Optional, Tuple
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

class PandasQueryProcessor:
    """Processes natural language queries by generating and executing pandas code."""
    
    def __init__(self, data_processor, api_key: Optional[str] = None):
        """Initialize with data processor and optional API key."""
        self.data_processor = data_processor
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            logger.warning("No Gemini API key provided. Limited to basic queries.")
            self.model = None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and return results."""
        logger.info(f"Processing pandas query: {query}")
        
        # Debug info collection
        debug_info = {
            'original_query': query,
            'processing_steps': []
        }
        
        if self.data_processor.processed_data is None:
            return {
                'success': False,
                'error': 'No data loaded',
                'answer': 'Please upload a CSV file first.'
            }
        
        # Get data context
        data_context = self._get_data_context()
        debug_info['data_context'] = {
            'columns': list(self.data_processor.processed_data.columns),
            'shape': self.data_processor.processed_data.shape,
            'sample_row': self.data_processor.processed_data.iloc[0].to_dict() if len(self.data_processor.processed_data) > 0 else {}
        }
        debug_info['processing_steps'].append("1. Data context extracted")
        
        # Generate pandas code using LLM
        if self.model:
            debug_info['processing_steps'].append("2. Using LLM to generate pandas code")
            pandas_code, explanation = self._generate_pandas_code(query, data_context)
            debug_info['llm_response'] = {
                'generated_code': pandas_code,
                'explanation': explanation
            }
        else:
            debug_info['processing_steps'].append("2. Using fallback processing (no LLM)")
            pandas_code, explanation = self._fallback_query_processing(query)
            debug_info['fallback_used'] = True
        
        if not pandas_code:
            debug_info['processing_steps'].append("3. No code generated, using fallback response")
            response = self._use_fallback_response(query, data_context)
            response['debug_info'] = debug_info
            return response
        
        debug_info['processing_steps'].append("3. Executing pandas code")
        # Execute pandas code safely
        result = self._execute_pandas_code(pandas_code)
        debug_info['execution_result'] = {
            'result_type': type(result).__name__,
            'result_preview': str(result)[:200] if result is not None else 'None'
        }
        
        debug_info['processing_steps'].append("4. Formatting response")
        # Format response
        response = self._format_response(result, explanation, pandas_code)
        response['debug_info'] = debug_info
        
        # Log debug info
        logger.debug(f"Query processing debug info: {json.dumps(debug_info, indent=2)}")
        
        return response
    
    def _get_data_context(self) -> str:
        """Get context about the loaded data for LLM."""
        df = self.data_processor.processed_data
        
        context = f"""
Available DataFrame columns: {list(df.columns)}
Data shape: {df.shape[0]} rows, {df.shape[1]} columns
Column types: {dict(df.dtypes.astype(str))}

Column mapping (original -> standardized):
{json.dumps(self.data_processor.analysis_result.get('standardized_mapping', {}), indent=2)}

Sample data (first 3 rows):
{df.head(3).to_string()}

Available for queries:
- Filter by any column value
- Group by operations
- Statistical summaries
- Count and aggregate functions
- Search and pattern matching
"""
        return context
    
    def _generate_pandas_code(self, query: str, data_context: str) -> Tuple[str, str]:
        """Generate pandas code using LLM."""
        prompt = f"""
You are a pandas expert. Generate Python pandas code to answer the user's question about their data.

Data Context:
{data_context}

User Query: "{query}"

IMPORTANT CONTEXT:
- This dataset contains institution data WITHOUT intake/admissions numbers (those columns are all 0)
- Focus on analyzing the available data: institution names, types, locations, and courses
- DO NOT calculate vacancy rates or admission statistics since the data doesn't contain real numbers

Instructions:
1. Generate ONLY the pandas code needed to answer the query
2. Use 'df' as the dataframe variable name
3. The final result should be assigned to a variable called 'result'
4. Include comments explaining each step
5. Handle edge cases (empty results, missing values)
6. For text searches, use case-insensitive matching with str.contains()
7. Return results in a user-friendly format (not just indexes)
8. IMPORTANT: Do NOT use df.append() as it's deprecated. Use pd.concat() instead
9. For questions about "what is this dataset about":
   - Provide a comprehensive overview
   - List unique values for key categorical columns
   - Show sample records
   - Explain what type of data this is

Common Query Patterns:
- "What is this dataset about?" → Provide full overview with samples
- "Show colleges" → Display college names with their details
- "Count by type/region" → Group and count appropriately
- "Find specific college" → Use str.contains for flexible matching

Respond in this exact format:
PANDAS_CODE:
```python
# Your pandas code here
result = ...
```

EXPLANATION:
Brief explanation of what the code does and what the results mean

Example for "what is this dataset about":
PANDAS_CODE:
```python
# Create comprehensive dataset overview
overview_parts = []

# Basic info
overview_parts.append(f"Dataset Overview:")
overview_parts.append(f"- Total Records: {len(df)}")
overview_parts.append(f"- Columns: {', '.join(df.columns)}")

# Get unique counts for key columns
unique_colleges = df['college_name'].nunique()
unique_courses = df['course_name'].nunique()
unique_regions = df['regional_office'].nunique()
overview_parts.append(f"\\nUnique Counts:")
overview_parts.append(f"- {unique_colleges} different colleges/institutions")
overview_parts.append(f"- {unique_courses} different courses/programs")
overview_parts.append(f"- {unique_regions} different regions/districts")

# College types breakdown
college_types = df['college_type'].value_counts()
overview_parts.append(f"\\nCollege Types:")
for ctype, count in college_types.items():
    overview_parts.append(f"- {ctype}: {count} institutions")

# Sample data
overview_parts.append(f"\\nSample Records:")
sample_df = df[['college_name', 'course_name', 'regional_office', 'college_type']].head(3)
for idx, row in sample_df.iterrows():
    overview_parts.append(f"- {row['college_name']} ({row['college_type']}) in {row['regional_office']} offers {row['course_name']}")

result = '\\n'.join(overview_parts)
```

EXPLANATION:
This provides a comprehensive overview of the educational institutions dataset, showing the total records, unique counts, college type distribution, and sample records.
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Extract code and explanation
            code_match = re.search(r'PANDAS_CODE:\s*```python\s*(.*?)\s*```', text, re.DOTALL)
            explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?:$|\n\n)', text, re.DOTALL)
            
            if code_match:
                code = code_match.group(1).strip()
                explanation = explanation_match.group(1).strip() if explanation_match else "Query processed successfully"
                return code, explanation
            else:
                logger.error("LLM response did not contain valid pandas code")
                return None, None
                
        except Exception as e:
            logger.error(f"Error generating pandas code: {e}")
            return None, None
    
    def _fallback_query_processing(self, query: str) -> Tuple[str, str]:
        """Simple fallback processing without LLM."""
        query_lower = query.lower()
        df_var = "df"
        
        # Basic patterns
        if "how many" in query_lower or "count" in query_lower:
            if "college" in query_lower:
                return f"result = {df_var}['college_name'].nunique()", "Counting unique colleges"
            else:
                return f"result = len({df_var})", "Counting total records"
        
        elif "show" in query_lower or "list" in query_lower:
            if "first" in query_lower:
                n = 5  # default
                numbers = re.findall(r'\d+', query)
                if numbers:
                    n = int(numbers[0])
                return f"result = {df_var}.head({n})", f"Showing first {n} records"
            elif "all" in query_lower:
                return f"result = {df_var}", "Showing all data"
        
        elif "column" in query_lower:
            return f"result = list({df_var}.columns)", "Listing all column names"
        
        elif "about" in query_lower or "summary" in query_lower or "describe" in query_lower:
            code = f"""# Get comprehensive data summary
df = {df_var}  # Reference to dataframe

# Basic info
total_records = len(df)
columns = list(df.columns)
data_types = df.dtypes.to_dict()

# Get unique counts for key columns  
unique_counts = {{}}
for col in ['college_name', 'college_type', 'regional_office', 'course_name']:
    if col in df.columns:
        unique_counts[col] = df[col].nunique()

# College type breakdown
college_types = df['college_type'].value_counts() if 'college_type' in df.columns else None

# Sample data
sample_df = df[['college_name', 'course_name', 'regional_office', 'college_type']].head(3)
sample_records = []
for idx, row in sample_df.iterrows():
    sample_records.append(f"- {{row['college_name']}} ({{row['college_type']}}) in {{row['regional_office']}} offers {{row['course_name']}}")

# Build overview
overview_parts = []
overview_parts.append(f"Dataset Overview:")
overview_parts.append(f"- Total Records: {{total_records}}")
overview_parts.append(f"- Columns: {{', '.join(columns)}}")

overview_parts.append(f"\\nUnique Counts:")
for col, count in unique_counts.items():
    col_display = col.replace('_', ' ').title()
    overview_parts.append(f"- {{count}} different {{col_display}}")

if college_types is not None:
    overview_parts.append(f"\\nCollege Types:")
    for ctype, count in college_types.items():
        overview_parts.append(f"- {{ctype}}: {{count}} institutions")

overview_parts.append(f"\\nSample Records:")
overview_parts.extend(sample_records)

overview_parts.append(f"\\nThis dataset appears to be a list of educational institutions with their courses, types, and locations.")

result = '\\n'.join(overview_parts)"""
            return code, "Providing comprehensive dataset overview"
        
        return None, None
    
    def _execute_pandas_code(self, code: str) -> Any:
        """Safely execute pandas code."""
        # Create safe namespace
        namespace = {
            'df': self.data_processor.processed_data.copy(),
            'pd': pd,
            'np': np,
            'result': None
        }
        
        try:
            # Execute code
            exec(code, namespace)
            return namespace.get('result')
        except Exception as e:
            logger.error(f"Error executing pandas code: {e}")
            logger.error(f"Code was: {code}")
            return f"Error: {str(e)}"
    
    def _format_response(self, result: Any, explanation: str, code: str) -> Dict[str, Any]:
        """Format the query result for response."""
        response = {
            'success': True,
            'explanation': explanation,
            'pandas_code': code
        }
        
        if isinstance(result, pd.DataFrame):
            if len(result) > 0:
                response['answer'] = f"{explanation}\n\nFound {len(result)} results."
                response['data'] = {
                    'type': 'dataframe',
                    'headers': list(result.columns),
                    'rows': result.head(50).to_dict('records'),
                    'total_rows': len(result)
                }
            else:
                response['answer'] = f"{explanation}\n\nNo matching records found."
                response['data'] = None
        
        elif isinstance(result, pd.Series):
            response['answer'] = f"{explanation}\n\n{result.to_string()}"
            response['data'] = {
                'type': 'series',
                'data': result.to_dict()
            }
        
        elif isinstance(result, (int, float, str)):
            response['answer'] = f"{explanation}\n\nResult: {result}"
            response['data'] = result
        
        elif isinstance(result, list):
            response['answer'] = f"{explanation}\n\n{', '.join(map(str, result[:20]))}"
            if len(result) > 20:
                response['answer'] += f"... and {len(result) - 20} more"
            response['data'] = result
        
        elif isinstance(result, str) and result.startswith("Error:"):
            response['success'] = False
            response['answer'] = "Sorry, I encountered an error processing your query."
            response['error'] = result
        
        else:
            response['answer'] = f"{explanation}\n\n{str(result)}"
            response['data'] = str(result)
        
        return response
    
    def _use_fallback_response(self, query: str, data_context: str) -> Dict[str, Any]:
        """Provide a helpful fallback response when unable to process query."""
        return {
            'success': True,
            'answer': f"""I understand you're asking about "{query}". 
            
Based on your data, here's what I can tell you:
- You have {self.data_processor.processed_data.shape[0]} records
- The data includes: {', '.join(self.data_processor.processed_data.columns)}

Try asking specific questions like:
- "Show me all government colleges"
- "How many colleges are in Pune?"
- "List colleges by type"
- "Show first 10 records"
""",
            'data': None,
            'explanation': "Providing general data overview"
        }
    
    def get_sample_queries(self) -> list:
        """Get sample queries based on available data."""
        df = self.data_processor.processed_data
        queries = []
        
        if 'college_name' in df.columns:
            queries.extend([
                "What is this dataset about?",
                "Show me all colleges",
                "How many colleges are there?",
                "List first 10 colleges"
            ])
        
        if 'college_type' in df.columns:
            types = df['college_type'].unique()[:3]
            queries.extend([
                f"Show all {types[0]} colleges" if len(types) > 0 else "Show colleges by type",
                "Count colleges by type"
            ])
        
        if 'regional_office' in df.columns:
            regions = df['regional_office'].unique()[:3]
            if len(regions) > 0:
                queries.append(f"Which colleges are in {regions[0]}?")
        
        queries.extend([
            "Summarize this data",
            "What columns are available?",
            "Show data statistics"
        ])
        
        return queries