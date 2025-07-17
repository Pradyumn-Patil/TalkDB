"""
Dynamic CSV Analyzer using LLM for column understanding and mapping.
Analyzes any CSV file structure and maps columns to standardized format.
"""

import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import os
import google.generativeai as genai

logger = logging.getLogger(__name__)

class DynamicCSVAnalyzer:
    """Analyzes CSV files dynamically using LLM to understand column meanings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Gemini API key."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            logger.warning("No Gemini API key provided. LLM analysis will be disabled.")
            self.model = None
    
    def analyze_csv_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV file structure and understand column meanings."""
        logger.info("=== STARTING CSV STRUCTURE ANALYSIS ===")
        try:
            logger.info(f"Analyzing file: {file_path}")
            
            # Load CSV with different encodings
            logger.info("Step 1: Loading CSV file")
            df = self._load_csv(file_path)
            if df is None:
                logger.error("Step 1 FAILED: Failed to load CSV file")
                return {"error": "Failed to load CSV file"}
            logger.info(f"Step 1 SUCCESS: Loaded CSV with {len(df)} rows, {len(df.columns)} columns")
            
            # Get basic info
            logger.info("Step 2: Getting basic info")
            try:
                basic_info = self._get_basic_info(df)
                logger.info(f"Step 2 SUCCESS: Basic info retrieved")
            except Exception as e:
                logger.error(f"Step 2 FAILED: Error getting basic info: {e}")
                raise
            
            # Get sample data for analysis
            logger.info("Step 3: Getting sample data")
            try:
                sample_data = self._get_sample_data(df)
                logger.info(f"Step 3 SUCCESS: Sample data for {len(sample_data)} columns")
            except Exception as e:
                logger.error(f"Step 3 FAILED: Error getting sample data: {e}")
                raise
            
            # Use LLM to understand columns if available
            logger.info("Step 4: Analyzing columns with LLM")
            try:
                column_analysis = self._analyze_columns_with_llm(df.columns.tolist(), sample_data)
                logger.info(f"Step 4 SUCCESS: Column analysis completed for {len(column_analysis)} columns")
            except Exception as e:
                logger.error(f"Step 4 FAILED: Error in column analysis: {e}")
                raise
            
            # Generate standardized mapping
            logger.info("Step 5: Generating standardized mapping")
            try:
                standardized_mapping = self._generate_standardized_mapping(column_analysis)
                logger.info(f"Step 5 SUCCESS: Generated mapping: {standardized_mapping}")
            except Exception as e:
                logger.error(f"Step 5 FAILED: Error generating mapping: {e}")
                raise
            
            result = {
                "success": True,
                "basic_info": basic_info,
                "sample_data": sample_data,
                "column_analysis": column_analysis,
                "standardized_mapping": standardized_mapping,
                "raw_columns": df.columns.tolist()
            }
            
            logger.info("=== CSV STRUCTURE ANALYSIS SUCCESSFUL ===")
            return result
            
        except Exception as e:
            logger.error(f"=== CSV STRUCTURE ANALYSIS FAILED ===")
            logger.error(f"Error analyzing CSV structure: {e}")
            import traceback
            logger.error(f"Analysis traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV with multiple encoding attempts."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully loaded CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading CSV with {encoding}: {e}")
                continue
        
        logger.error("Failed to load CSV with any encoding")
        return None
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the CSV."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    
    def _get_sample_data(self, df: pd.DataFrame, n_samples: int = 5) -> Dict[str, List]:
        """Get sample data for each column."""
        sample_data = {}
        
        for column in df.columns:
            # Get non-null samples
            non_null_values = df[column].dropna()
            if len(non_null_values) > 0:
                sample_size = min(n_samples, len(non_null_values))
                samples = non_null_values.head(sample_size).tolist()
                # Convert to string for JSON serialization
                samples = [str(val) for val in samples]
                sample_data[column] = samples
            else:
                sample_data[column] = []
        
        return sample_data
    
    def _analyze_columns_with_llm(self, columns: List[str], sample_data: Dict[str, List]) -> Dict[str, Dict]:
        """Use LLM to analyze column meanings and suggest mappings."""
        if not self.model:
            return self._fallback_column_analysis(columns, sample_data)
        
        try:
            # Prepare prompt for LLM
            prompt = self._create_analysis_prompt(columns, sample_data)
            
            response = self.model.generate_content(prompt)
            result = self._parse_llm_response(response.text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._fallback_column_analysis(columns, sample_data)
    
    def _create_analysis_prompt(self, columns: List[str], sample_data: Dict[str, List]) -> str:
        """Create prompt for LLM to analyze columns."""
        prompt = """
Analyze the following CSV columns and their sample data. For each column, determine:
1. What type of data it contains (e.g., institution name, course name, numeric count, etc.)
2. What it likely represents in an educational admission context
3. Suggest a standardized category it maps to

Standard categories to map to:
- college_name: Name of educational institution
- college_type: Type of institution (Government/Private/Autonomous)
- regional_office: Administrative region or location
- course_name: Academic program or course name
- intake: Total seat capacity or intake
- admissions: Number of students admitted
- other: Any other type of data

Columns and sample data:
"""
        
        for column, samples in sample_data.items():
            prompt += f"\nColumn: '{column}'\n"
            prompt += f"Sample values: {samples}\n"
        
        prompt += """
Respond with a JSON object where each key is the original column name and the value is an object with:
{
  "data_type": "description of data type",
  "likely_meaning": "what this column represents",
  "standardized_category": "one of the standard categories above",
  "confidence": "high/medium/low"
}

Example response format:
{
  "Institute Name": {
    "data_type": "text/string",
    "likely_meaning": "Name of educational institution",
    "standardized_category": "college_name",
    "confidence": "high"
  }
}
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Dict]:
        """Parse LLM response into structured format."""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.error("No valid JSON found in LLM response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            return {}
    
    def _fallback_column_analysis(self, columns: List[str], sample_data: Dict[str, List]) -> Dict[str, Dict]:
        """Fallback analysis using keyword matching when LLM is not available."""
        analysis = {}
        
        # Keywords for different categories
        keywords_map = {
            'college_name': ['college', 'institute', 'university', 'school', 'name'],
            'college_type': ['type', 'status', 'category', 'government', 'private'],
            'regional_office': ['region', 'office', 'location', 'area', 'zone'],
            'course_name': ['course', 'program', 'branch', 'specialization', 'degree'],
            'intake': ['intake', 'capacity', 'seats', 'total'],
            'admissions': ['admitted', 'admission', 'enrolled', 'selected']
        }
        
        for column in columns:
            column_lower = column.lower()
            best_match = 'other'
            confidence = 'low'
            
            # Check for keyword matches
            for category, keywords in keywords_map.items():
                if any(keyword in column_lower for keyword in keywords):
                    best_match = category
                    confidence = 'medium'
                    break
            
            # Check sample data for numeric patterns
            samples = sample_data.get(column, [])
            if samples and all(str(val).replace('.', '').replace(',', '').isdigit() for val in samples if val):
                if 'intake' in column_lower or 'capacity' in column_lower:
                    best_match = 'intake'
                    confidence = 'high'
                elif 'admit' in column_lower or 'enroll' in column_lower:
                    best_match = 'admissions'
                    confidence = 'high'
            
            analysis[column] = {
                "data_type": "numeric" if best_match in ['intake', 'admissions'] else "text",
                "likely_meaning": f"Likely {best_match.replace('_', ' ')} field",
                "standardized_category": best_match,
                "confidence": confidence
            }
        
        return analysis
    
    def _generate_standardized_mapping(self, column_analysis: Dict[str, Dict]) -> Dict[str, str]:
        """Generate mapping from original columns to standardized names."""
        mapping = {}
        
        # Priority mapping - prefer high confidence matches
        for original_col, analysis in column_analysis.items():
            category = analysis.get('standardized_category', 'other')
            confidence = analysis.get('confidence', 'low')
            
            if category != 'other':
                # If we already have a mapping for this category, prefer higher confidence
                if category in mapping.values():
                    existing_col = next(k for k, v in mapping.items() if v == category)
                    existing_confidence = column_analysis[existing_col].get('confidence', 'low')
                    
                    # Replace if current has higher confidence
                    confidence_order = {'low': 0, 'medium': 1, 'high': 2}
                    if confidence_order.get(confidence, 0) > confidence_order.get(existing_confidence, 0):
                        del mapping[existing_col]
                        mapping[original_col] = category
                else:
                    mapping[original_col] = category
        
        return mapping
    
    def get_suggested_queries(self, column_analysis: Dict[str, Dict], sample_data: Dict[str, List]) -> List[str]:
        """Generate suggested queries based on analyzed columns."""
        queries = []
        
        # Check what types of data we have
        has_college = any(analysis.get('standardized_category') == 'college_name' 
                         for analysis in column_analysis.values())
        has_intake = any(analysis.get('standardized_category') == 'intake' 
                        for analysis in column_analysis.values())
        has_admissions = any(analysis.get('standardized_category') == 'admissions' 
                            for analysis in column_analysis.values())
        has_type = any(analysis.get('standardized_category') == 'college_type' 
                      for analysis in column_analysis.values())
        has_course = any(analysis.get('standardized_category') == 'course_name' 
                        for analysis in column_analysis.values())
        
        # Generate relevant queries based on available data
        if has_college and has_intake:
            queries.append("Which college has the highest intake?")
        
        if has_college and has_admissions:
            queries.append("Which colleges have the most admissions?")
        
        if has_intake and has_admissions:
            queries.append("Show colleges with minimum vacancies")
            queries.append("Calculate vacancy percentages")
        
        if has_college and not has_intake and not has_admissions:
            # For institution lists without numeric data
            queries.append("List all colleges and their details")
            queries.append("Show colleges by region")
            
        if has_type:
            queries.append("Compare Government vs Private colleges")
            queries.append("How many colleges of each type?")
        
        if has_course:
            queries.append("Which course has the highest demand?")
            queries.append("Show course-wise analysis")
        
        # Add queries based on what data is actually available
        if has_college:
            queries.append("Show all colleges")
            queries.append("Group colleges by type")
            
        # Add generic queries
        queries.extend([
            "Show top 10 records",
            "Provide data summary",
            "Show all available data"
        ])
        
        return queries[:8]  # Limit to 8 suggestions