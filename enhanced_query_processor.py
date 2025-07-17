"""
Enhanced Query Processor with Better Understanding of User Intent
Analyzes queries deeply and generates accurate pandas code
"""

import pandas as pd
import numpy as np
import logging
import json
import re
from typing import Dict, Any, Optional, Tuple, List
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

class EnhancedQueryProcessor:
    """Enhanced processor that deeply analyzes queries and available data."""
    
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
            
        # Analyze available data on initialization
        self._analyze_available_data()
    
    def _analyze_available_data(self):
        """Analyze what data is actually available in the dataset."""
        if self.data_processor.processed_data is None:
            self.data_summary = None
            return
            
        df = self.data_processor.processed_data
        
        # Get all column information
        self.data_summary = {
            'columns': list(df.columns),
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'unique_values': {},
            'sample_values': {},
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': [],
            'categorical_columns': [],
            'available_filters': []
        }
        
        # Analyze each column
        for col in df.columns:
            # Get unique values for categorical columns
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                if len(unique_vals) < 50:  # Only store if reasonable number
                    self.data_summary['unique_values'][col] = list(unique_vals)
                self.data_summary['categorical_columns'].append(col)
            else:
                self.data_summary['numeric_columns'].append(col)
            
            # Get sample values
            self.data_summary['sample_values'][col] = df[col].dropna().head(5).tolist()
        
        # Identify available filters based on actual data
        if 'college_type' in df.columns:
            self.data_summary['available_filters'].append('college type (Government, Private, etc.)')
        if 'regional_office' in df.columns:
            self.data_summary['available_filters'].append('region/district')
        if 'course_name' in df.columns:
            self.data_summary['available_filters'].append('course/program')
        
        # Check for additional columns from raw data
        if hasattr(self.data_processor, 'raw_data'):
            raw_cols = list(self.data_processor.raw_data.columns)
            self.data_summary['raw_columns'] = raw_cols
            
            # Check for autonomous/minority status
            if any('autonom' in col.lower() for col in raw_cols):
                self.data_summary['available_filters'].append('autonomous status')
            if any('minority' in col.lower() for col in raw_cols):
                self.data_summary['available_filters'].append('minority status')
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with enhanced understanding."""
        logger.info(f"Processing enhanced query: {query}")
        
        if self.data_processor.processed_data is None:
            return {
                'success': False,
                'error': 'No data loaded',
                'answer': 'Please upload a CSV file first.'
            }
        
        # Enhanced query analysis
        query_analysis = self._analyze_query_intent(query)
        
        # Generate pandas code based on deep analysis
        if self.model:
            pandas_code, explanation = self._generate_enhanced_pandas_code(
                query, query_analysis, self._get_enhanced_data_context()
            )
        else:
            pandas_code, explanation = self._enhanced_fallback_processing(query, query_analysis)
        
        if not pandas_code:
            return self._provide_helpful_response(query, query_analysis)
        
        # Execute and format
        result = self._execute_pandas_code(pandas_code)
        return self._format_enhanced_response(result, explanation, pandas_code, query_analysis)
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Deeply analyze what the user is asking for."""
        query_lower = query.lower()
        
        analysis = {
            'query_type': None,
            'filters': [],
            'aggregation': None,
            'sorting': None,
            'limit': None,
            'keywords': [],
            'entities': []
        }
        
        # Identify query type
        if any(word in query_lower for word in ['how many', 'count', 'number of', 'total']):
            analysis['query_type'] = 'count'
        elif any(word in query_lower for word in ['list', 'show', 'display', 'which', 'what are']):
            analysis['query_type'] = 'list'
        elif any(word in query_lower for word in ['highest', 'maximum', 'most', 'top']):
            analysis['query_type'] = 'maximum'
            analysis['sorting'] = 'descending'
        elif any(word in query_lower for word in ['lowest', 'minimum', 'least', 'bottom']):
            analysis['query_type'] = 'minimum'
            analysis['sorting'] = 'ascending'
        elif any(word in query_lower for word in ['average', 'mean']):
            analysis['query_type'] = 'average'
        elif any(word in query_lower for word in ['group', 'by type', 'by region', 'breakdown']):
            analysis['query_type'] = 'groupby'
        
        # Extract filters
        if 'government' in query_lower:
            analysis['filters'].append(('college_type', 'Government'))
        if 'private' in query_lower:
            analysis['filters'].append(('college_type', 'Un-Aided'))
        if 'autonomous' in query_lower:
            analysis['filters'].append(('autonomous', 'Autonomous'))
        if 'non-autonomous' in query_lower:
            analysis['filters'].append(('autonomous', 'Non-Autonomous'))
        if 'minority' in query_lower:
            analysis['filters'].append(('minority', 'Minority'))
        if 'non-minority' in query_lower:
            analysis['filters'].append(('minority', 'Non-Minority'))
        
        # Extract location filters
        location_keywords = ['pune', 'mumbai', 'nagpur', 'nashik', 'amravati', 'kolhapur']
        for location in location_keywords:
            if location in query_lower:
                analysis['filters'].append(('regional_office', location.title()))
        
        # Extract limits
        limit_match = re.search(r'(top|first|last)\s+(\d+)', query_lower)
        if limit_match:
            analysis['limit'] = int(limit_match.group(2))
        
        # Extract specific column references
        if 'intake' in query_lower:
            analysis['keywords'].append('intake')
        if 'admission' in query_lower:
            analysis['keywords'].append('admissions')
        if 'vacanc' in query_lower:
            analysis['keywords'].append('vacancies')
        
        return analysis
    
    def _get_enhanced_data_context(self) -> str:
        """Get comprehensive context about available data."""
        df = self.data_processor.processed_data
        raw_df = getattr(self.data_processor, 'raw_data', None)
        
        context = f"""
Dataset Information:
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Processed Columns: {list(df.columns)}
"""
        
        if raw_df is not None:
            context += f"- Raw Data Columns: {list(raw_df.columns)}\n"
        
        context += f"""
Column Details:
"""
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            context += f"- {col} ({dtype}): {null_count} null values"
            
            if col in self.data_summary['unique_values']:
                unique_vals = self.data_summary['unique_values'][col][:10]
                context += f", Values: {unique_vals}"
            context += "\n"
        
        # Add information about actual data availability
        context += f"""
Available Filters: {', '.join(self.data_summary['available_filters'])}

IMPORTANT NOTES:
1. If the data contains 'Autonomu' column, it likely means 'Autonomous' status
2. Check both processed data (df) and raw data (self.data_processor.raw_data) if needed
3. The Status column contains college types like 'Government', 'Un-Aided', etc.
4. Some columns might have slight spelling variations in the raw data
"""
        
        return context
    
    def _generate_enhanced_pandas_code(self, query: str, query_analysis: Dict, data_context: str) -> Tuple[str, str]:
        """Generate pandas code with enhanced understanding."""
        prompt = f"""
You are an expert data analyst. Generate pandas code to answer the user's query accurately.

Query: "{query}"

Query Analysis:
{json.dumps(query_analysis, indent=2)}

Available Data Context:
{data_context}

Instructions:
1. CRITICAL: Check if the requested information exists in the data
2. If filtering by autonomous/minority status, check raw_data columns too
3. Generate pandas code that precisely answers the query
4. Use 'df' for processed data, 'raw_data' for raw data (NOT self.data_processor.raw_data)
5. Handle edge cases and missing values properly
6. For counts, ensure you're counting the right thing (rows vs unique values)
7. For filters, use appropriate string matching (exact vs contains)
8. ALWAYS assign final result to 'result' variable
9. Available variables: df, raw_data, pd, np

Special Handling:
- If querying autonomous/minority status, you MUST use raw data
- Government colleges: Status == 'Government' or contains 'Government'
- Private colleges: Status == 'Un-Aided' 
- Autonomous status: In raw data, 'Autonomu' column has values 'Autonomous' or 'Non-Autonomous' (NOT 'Yes'/'No')
- Minority status: In raw data, 'Minority' column has values 'Minority' or 'Non-Minority' (NOT 'Yes'/'No')
- For merging: Use 'InstituteName' from raw data to match with 'college_name' in processed data

Generate code in this format:
PANDAS_CODE:
```python
# Your code here
result = ...
```

EXPLANATION:
What the code does and any assumptions made
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Extract code and explanation
            code_match = re.search(r'PANDAS_CODE:\s*```python\s*(.*?)\s*```', text, re.DOTALL)
            explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?:$|\n\n)', text, re.DOTALL)
            
            if code_match:
                code = code_match.group(1).strip()
                explanation = explanation_match.group(1).strip() if explanation_match else "Query processed"
                return code, explanation
            
        except Exception as e:
            logger.error(f"Error generating enhanced pandas code: {e}")
        
        return None, None
    
    def _enhanced_fallback_processing(self, query: str, query_analysis: Dict) -> Tuple[str, str]:
        """Enhanced fallback without LLM."""
        query_type = query_analysis['query_type']
        filters = query_analysis['filters']
        
        # Build filter conditions
        filter_code = ""
        if filters:
            conditions = []
            for col, val in filters:
                if col in ['college_type', 'regional_office', 'course_name']:
                    conditions.append(f"(df['{col}'].str.contains('{val}', case=False, na=False))")
            if conditions:
                filter_code = f"mask = {' & '.join(conditions)}\nfiltered_df = df[mask]\n"
            else:
                filter_code = "filtered_df = df\n"
        else:
            filter_code = "filtered_df = df\n"
        
        # Generate code based on query type
        if query_type == 'count':
            code = filter_code + "result = len(filtered_df)"
            explanation = "Counting matching records"
        elif query_type == 'list':
            limit = query_analysis.get('limit', 20)
            code = filter_code + f"result = filtered_df.head({limit})"
            explanation = f"Listing first {limit} matching records"
        elif query_type == 'groupby':
            code = "result = df.groupby('college_type').size().sort_values(descending=True)"
            explanation = "Grouping and counting by college type"
        else:
            # Default to showing summary
            code = "result = df.describe(include='all')"
            explanation = "Showing data summary"
        
        return code, explanation
    
    def _execute_pandas_code(self, code: str) -> Any:
        """Execute pandas code with access to all necessary data."""
        # Create namespace with all needed references
        namespace = {
            'df': self.data_processor.processed_data.copy(),
            'raw_data': self.data_processor.raw_data.copy() if hasattr(self.data_processor, 'raw_data') else None,
            'pd': pd,
            'np': np,
            'self': self,  # Allow access to self
            'data_processor': self.data_processor,  # Direct access to data processor
            'result': None,
            # Built-in functions
            'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'print': print, 'chr': chr, 'ord': ord,
            'min': min, 'max': max, 'sum': sum, 'abs': abs,
            'round': round, 'sorted': sorted, 'reversed': reversed,
            'enumerate': enumerate, 'zip': zip, 'range': range,
            'any': any, 'all': all
        }
        
        try:
            logger.debug(f"Executing pandas code:\n{code}")
            exec(code, namespace)
            result = namespace.get('result')
            logger.debug(f"Execution successful. Result type: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"Error executing pandas code: {e}")
            logger.error(f"Code was:\n{code}")
            return f"Error: {str(e)}"
    
    def _format_enhanced_response(self, result: Any, explanation: str, code: str, query_analysis: Dict) -> Dict[str, Any]:
        """Format response with enhanced information."""
        response = {
            'success': True,
            'explanation': explanation,
            'pandas_code': code,
            'query_analysis': query_analysis
        }
        
        # Format result based on type
        if isinstance(result, pd.DataFrame):
            if len(result) > 0:
                # Provide natural language summary
                if query_analysis['query_type'] == 'count':
                    response['answer'] = f"Found {len(result)} records matching your criteria."
                else:
                    response['answer'] = f"{explanation}\n\nShowing {min(len(result), 50)} of {len(result)} results."
                
                response['data'] = {
                    'type': 'dataframe',
                    'headers': list(result.columns),
                    'rows': result.head(50).to_dict('records'),
                    'total_rows': len(result)
                }
            else:
                response['answer'] = "No records found matching your criteria."
                response['data'] = None
        
        elif isinstance(result, pd.Series):
            response['answer'] = f"{explanation}\n\n{result.to_string()}"
            response['data'] = {
                'type': 'series',
                'data': result.to_dict()
            }
        
        elif isinstance(result, (int, float)):
            # Provide contextual answer
            if query_analysis['query_type'] == 'count':
                entity = 'records'
                if any(f in query_analysis['filters'] for f in [('college_type', 'Government')]):
                    entity = 'government colleges'
                response['answer'] = f"There are {result} {entity} in the dataset."
            else:
                response['answer'] = f"{explanation}\n\nResult: {result}"
            response['data'] = result
        
        elif isinstance(result, str):
            if result.startswith("Error:"):
                response['success'] = False
                response['answer'] = "I encountered an error processing your query. Please try rephrasing."
                response['error'] = result
            else:
                response['answer'] = result
                response['data'] = result
        
        else:
            response['answer'] = f"{explanation}\n\n{str(result)}"
            response['data'] = str(result)
        
        return response
    
    def _provide_helpful_response(self, query: str, query_analysis: Dict) -> Dict[str, Any]:
        """Provide helpful response when unable to generate code."""
        available_info = []
        if self.data_summary:
            if self.data_summary['categorical_columns']:
                available_info.append(f"categorical data: {', '.join(self.data_summary['categorical_columns'][:5])}")
            if self.data_summary['numeric_columns']:
                available_info.append(f"numeric data: {', '.join(self.data_summary['numeric_columns'][:5])}")
        
        response = {
            'success': True,
            'answer': f"""I understand you're asking: "{query}"

Based on the available data, I can help you with:
- Counting colleges by type, region, or other criteria
- Listing colleges with specific characteristics
- Analyzing distribution across regions
- Filtering by {', '.join(self.data_summary['available_filters'])} 

The dataset contains {', '.join(available_info)}.

Try asking specific questions like:
- "How many government colleges are there?"
- "List autonomous colleges in Pune"
- "Show minority institutions by region"
- "Count colleges by type"
""",
            'data': None,
            'query_analysis': query_analysis
        }
        
        return response
    
    def get_sample_queries(self) -> List[str]:
        """Get sample queries based on available data."""
        queries = []
        
        if self.data_summary:
            # Basic queries
            queries.extend([
                "What is this dataset about?",
                "How many colleges are there?",
                "Show me the first 10 colleges"
            ])
            
            # Type-based queries
            if 'college_type' in self.data_summary['unique_values']:
                types = self.data_summary['unique_values']['college_type']
                if 'Government' in types:
                    queries.append("How many government colleges are there?")
                if 'Un-Aided' in types:
                    queries.append("List private colleges")
            
            # Location-based queries
            if 'regional_office' in self.data_summary['unique_values']:
                regions = self.data_summary['unique_values']['regional_office'][:3]
                if regions:
                    queries.append(f"Show colleges in {regions[0]}")
            
            # Complex queries based on available columns
            if 'autonomous' in [col.lower() for col in self.data_summary.get('raw_columns', [])]:
                queries.extend([
                    "Show autonomous government colleges",
                    "How many non-autonomous colleges are there?"
                ])
            
            if 'minority' in [col.lower() for col in self.data_summary.get('raw_columns', [])]:
                queries.extend([
                    "List minority institutions",
                    "Count non-minority colleges by type"
                ])
            
            # Advanced queries
            queries.extend([
                "Group colleges by type and region",
                "Which region has the most colleges?",
                "Show government colleges in Pune"
            ])
        
        return queries[:10]  # Return top 10 suggestions