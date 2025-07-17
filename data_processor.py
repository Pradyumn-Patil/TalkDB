"""
Standardized CSV Data Processor for Admission Chatbot
Handles loading, validation, and enhancement of admission data.
Now supports dynamic CSV analysis using LLM for any file format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging
from csv_analyzer import DynamicCSVAnalyzer

logger = logging.getLogger(__name__)

class AdmissionDataProcessor:
    """Processes admission data with standardized column mapping and derived metrics."""
    
    # Standardized column mapping
    REQUIRED_COLUMNS = {
        'college_name': ['College Name', 'InstituteName', 'Institute Name', 'college_name'],
        'college_type': ['College Type', 'Status', 'Type', 'college_type'],
        'regional_office': ['Regional Office', 'RegionName', 'Region', 'regional_office'],
        'course_name': ['Course Name', 'CourseName', 'Program', 'course_name'],
        'intake': ['Intake', 'TotalIntake', 'Capacity', 'intake'],
        'admissions': ['Admissions', 'TotalAdmitted', 'Admitted', 'admissions']
    }
    
    def __init__(self, file_path: str, api_key: Optional[str] = None):
        """Initialize with CSV file path and optional API key for dynamic analysis."""
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.college_summary = None
        self.course_summary = None
        self.regional_summary = None
        self.csv_analyzer = DynamicCSVAnalyzer(api_key)
        self.dynamic_mapping = None
        self.analysis_result = None
        
    def load_data(self) -> bool:
        """Load and validate CSV data."""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                return False
                
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.raw_data = pd.read_csv(self.file_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error("Failed to load data with any encoding")
                return False
                
            logger.info(f"Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def analyze_csv_dynamically(self) -> bool:
        """Use dynamic CSV analyzer to understand the file structure."""
        logger.info("=== STARTING DYNAMIC CSV ANALYSIS ===")
        try:
            logger.info(f"Analyzing file: {self.file_path}")
            logger.info("Calling csv_analyzer.analyze_csv_structure")
            
            self.analysis_result = self.csv_analyzer.analyze_csv_structure(self.file_path)
            logger.info(f"Analysis result keys: {list(self.analysis_result.keys()) if self.analysis_result else 'None'}")
            
            if not self.analysis_result.get("success", False):
                error_msg = self.analysis_result.get('error', 'Unknown error')
                logger.error(f"Dynamic analysis failed: {error_msg}")
                logger.error(f"Full analysis result: {self.analysis_result}")
                return False
            
            self.dynamic_mapping = self.analysis_result.get("standardized_mapping", {})
            logger.info(f"Dynamic analysis completed. Found mapping: {self.dynamic_mapping}")
            logger.info("=== DYNAMIC CSV ANALYSIS SUCCESSFUL ===")
            return True
            
        except Exception as e:
            logger.error(f"=== DYNAMIC CSV ANALYSIS FAILED ===")
            logger.error(f"Error in dynamic CSV analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def validate_and_map_columns(self) -> Dict[str, str]:
        """Validate and map columns to standardized names using dynamic analysis first."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Try dynamic analysis first
        if self.analyze_csv_dynamically() and self.dynamic_mapping:
            # Use dynamic mapping, but invert it (original_col -> standard_name)
            column_mapping = {}
            for original_col, standard_category in self.dynamic_mapping.items():
                if standard_category in ['college_name', 'college_type', 'regional_office', 'course_name', 'intake', 'admissions']:
                    column_mapping[standard_category] = original_col
            
            logger.info(f"Using dynamic mapping: {column_mapping}")
            
            # Check if we have minimum required columns
            required_for_basic_analysis = ['college_name']
            missing_required = [col for col in required_for_basic_analysis if col not in column_mapping]
            
            if not missing_required:
                return column_mapping
            else:
                logger.warning(f"Dynamic mapping missing required columns: {missing_required}. Falling back to static mapping.")
        
        # Fallback to original static mapping
        column_mapping = {}
        available_columns = [col.strip() for col in self.raw_data.columns]
        
        for standard_name, possible_names in self.REQUIRED_COLUMNS.items():
            mapped_column = None
            for possible_name in possible_names:
                if possible_name in available_columns:
                    mapped_column = possible_name
                    break
            
            if mapped_column:
                column_mapping[standard_name] = mapped_column
            else:
                logger.warning(f"Could not find column for '{standard_name}'. Available: {available_columns}")
        
        # For dynamic analysis, be more flexible with requirements
        if self.analysis_result:
            # If we have any college/institution identifier, allow processing
            has_institution = any('college' in col.lower() or 'institute' in col.lower() or 'name' in col.lower() 
                                for col in available_columns)
            if has_institution and not column_mapping.get('college_name'):
                # Find the most likely college name column
                for col in available_columns:
                    if any(keyword in col.lower() for keyword in ['college', 'institute', 'university', 'school', 'name']):
                        column_mapping['college_name'] = col
                        break
        
        # Check minimum requirements
        if not column_mapping.get('college_name'):
            # For completely unknown formats, use the first text column as college name
            text_columns = [col for col in available_columns 
                          if self.raw_data[col].dtype == 'object']
            if text_columns:
                column_mapping['college_name'] = text_columns[0]
                logger.info(f"Using first text column '{text_columns[0]}' as college name")
        
        if not column_mapping.get('college_name'):
            raise ValueError("No suitable column found for college/institution names")
        
        return column_mapping
    
    def calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics: vacancies and percentage vacancies."""
        df = df.copy()
        
        # Handle cases where intake/admissions columns might be missing
        if 'intake' in df.columns:
            df['intake'] = pd.to_numeric(df['intake'], errors='coerce').fillna(0)
        else:
            df['intake'] = 0
            logger.info("No intake column found, setting to 0")
        
        if 'admissions' in df.columns:
            df['admissions'] = pd.to_numeric(df['admissions'], errors='coerce').fillna(0)
        else:
            df['admissions'] = 0
            logger.info("No admissions column found, setting to 0")
        
        # Calculate derived metrics only if we have meaningful data
        df['vacancies'] = df['intake'] - df['admissions']
        df['percentage_vacancies'] = np.where(
            df['intake'] > 0,
            (df['vacancies'] / df['intake']) * 100,
            0
        ).round(2)
        
        # Additional useful metrics
        df['admission_rate'] = np.where(
            df['intake'] > 0,
            (df['admissions'] / df['intake']) * 100,
            0
        ).round(2)
        
        return df
    
    def process_data(self) -> bool:
        """Process raw data into standardized format with derived metrics."""
        logger.info("=== STARTING DATA PROCESSING ===")
        try:
            logger.info("Step 1: Checking if raw data is loaded")
            if self.raw_data is None:
                logger.info("Raw data not loaded, attempting to load")
                if not self.load_data():
                    logger.error("Failed to load data")
                    return False
            else:
                logger.info(f"Raw data already loaded: {len(self.raw_data)} rows, {len(self.raw_data.columns)} columns")
            
            # Map columns
            logger.info("Step 2: Validating and mapping columns")
            try:
                column_mapping = self.validate_and_map_columns()
                logger.info(f"Column mapping successful: {column_mapping}")
            except Exception as e:
                logger.error(f"Column mapping failed: {e}")
                import traceback
                logger.error(f"Column mapping traceback: {traceback.format_exc()}")
                raise
            
            # Create processed dataframe with standardized column names
            logger.info("Step 3: Creating processed dataframe")
            try:
                processed_df = pd.DataFrame()
                for standard_name, original_name in column_mapping.items():
                    logger.info(f"Mapping {original_name} -> {standard_name}")
                    processed_df[standard_name] = self.raw_data[original_name]
                logger.info(f"Processed dataframe created with columns: {list(processed_df.columns)}")
            except Exception as e:
                logger.error(f"Failed to create processed dataframe: {e}")
                import traceback
                logger.error(f"Processed dataframe traceback: {traceback.format_exc()}")
                raise
            
            # Fill missing optional columns
            logger.info("Step 4: Filling missing optional columns")
            for col in ['college_type', 'regional_office', 'course_name']:
                if col not in processed_df.columns:
                    processed_df[col] = 'Unknown'
                    logger.info(f"Added missing column '{col}' with 'Unknown' values")
            
            # Calculate derived metrics
            logger.info("Step 5: Calculating derived metrics")
            try:
                self.processed_data = self.calculate_derived_metrics(processed_df)
                logger.info(f"Derived metrics calculated. Final columns: {list(self.processed_data.columns)}")
            except Exception as e:
                logger.error(f"Failed to calculate derived metrics: {e}")
                import traceback
                logger.error(f"Derived metrics traceback: {traceback.format_exc()}")
                raise
            
            # Generate summary tables
            logger.info("Step 6: Generating summary tables")
            try:
                self._generate_summaries()
                logger.info("Summary tables generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate summaries: {e}")
                import traceback
                logger.error(f"Summaries traceback: {traceback.format_exc()}")
                raise
            
            logger.info(f"=== DATA PROCESSING SUCCESSFUL: {len(self.processed_data)} records ===")
            return True
            
        except Exception as e:
            logger.error(f"=== DATA PROCESSING FAILED ===")
            logger.error(f"Error processing data: {e}")
            import traceback
            logger.error(f"Main processing traceback: {traceback.format_exc()}")
            return False
    
    def _generate_summaries(self):
        """Generate various summary tables for quick analysis."""
        if self.processed_data is None:
            return
        
        # College-wise summary
        college_cols = ['college_name']
        if 'college_type' in self.processed_data.columns:
            college_cols.append('college_type')
        if 'regional_office' in self.processed_data.columns:
            college_cols.append('regional_office')
            
        self.college_summary = self.processed_data.groupby(college_cols).agg({
            'intake': 'sum',
            'admissions': 'sum',
            'vacancies': 'sum',
            'course_name': 'count'
        }).rename(columns={'course_name': 'total_courses'}).reset_index()
        
        # Recalculate percentages for college summary
        self.college_summary['percentage_vacancies'] = np.where(
            self.college_summary['intake'] > 0,
            (self.college_summary['vacancies'] / self.college_summary['intake']) * 100,
            0
        ).round(2)
        
        # Course-wise summary
        if 'course_name' in self.processed_data.columns:
            self.course_summary = self.processed_data.groupby('course_name').agg({
                'intake': 'sum',
                'admissions': 'sum',
                'vacancies': 'sum',
                'college_name': 'count'
            }).rename(columns={'college_name': 'total_colleges'}).reset_index()
            
            self.course_summary['percentage_vacancies'] = np.where(
                self.course_summary['intake'] > 0,
                (self.course_summary['vacancies'] / self.course_summary['intake']) * 100,
                0
            ).round(2)
        
        # Regional summary
        if 'regional_office' in self.processed_data.columns:
            self.regional_summary = self.processed_data.groupby('regional_office').agg({
                'intake': 'sum',
                'admissions': 'sum',
                'vacancies': 'sum',
                'college_name': 'nunique',
                'course_name': 'count'
            }).rename(columns={
                'college_name': 'total_colleges',
                'course_name': 'total_programs'
            }).reset_index()
            
            self.regional_summary['percentage_vacancies'] = np.where(
                self.regional_summary['intake'] > 0,
                (self.regional_summary['vacancies'] / self.regional_summary['intake']) * 100,
                0
            ).round(2)
    
    def get_data_info(self) -> Dict:
        """Get comprehensive data information."""
        if self.processed_data is None:
            return {}
        
        info = {
            'total_records': int(len(self.processed_data)),
            'total_colleges': int(self.processed_data['college_name'].nunique()),
            'total_intake': int(self.processed_data['intake'].sum()),
            'total_admissions': int(self.processed_data['admissions'].sum()),
            'total_vacancies': int(self.processed_data['vacancies'].sum()),
            'overall_vacancy_percentage': round(
                (self.processed_data['vacancies'].sum() / self.processed_data['intake'].sum()) * 100, 2
            ) if self.processed_data['intake'].sum() > 0 else 0
        }
        
        if 'college_type' in self.processed_data.columns:
            info['college_types'] = {k: int(v) for k, v in self.processed_data['college_type'].value_counts().to_dict().items()}
        
        if 'regional_office' in self.processed_data.columns:
            info['regional_offices'] = {k: int(v) for k, v in self.processed_data['regional_office'].value_counts().to_dict().items()}
        
        if 'course_name' in self.processed_data.columns:
            info['total_courses'] = int(self.processed_data['course_name'].nunique())
        
        return info
    
    def search_colleges(self, filters: Dict) -> pd.DataFrame:
        """Search colleges with various filters."""
        if self.processed_data is None:
            return pd.DataFrame()
        
        result = self.processed_data.copy()
        
        for key, value in filters.items():
            if key in result.columns and value:
                if isinstance(value, str):
                    result = result[result[key].str.contains(value, case=False, na=False)]
                else:
                    result = result[result[key] == value]
        
        return result
    
    def get_top_colleges(self, metric: str = 'intake', n: int = 10, ascending: bool = False) -> pd.DataFrame:
        """Get top N colleges by specified metric."""
        if self.college_summary is None:
            return pd.DataFrame()
        
        if metric not in self.college_summary.columns:
            return pd.DataFrame()
        
        return self.college_summary.nlargest(n, metric) if not ascending else self.college_summary.nsmallest(n, metric)
    
    def get_regional_analysis(self) -> pd.DataFrame:
        """Get regional analysis summary."""
        return self.regional_summary if self.regional_summary is not None else pd.DataFrame()
    
    def get_course_analysis(self) -> pd.DataFrame:
        """Get course-wise analysis summary."""
        return self.course_summary if self.course_summary is not None else pd.DataFrame()
    
    def get_dynamic_analysis_info(self) -> Dict:
        """Get information from dynamic CSV analysis."""
        if not self.analysis_result:
            return {}
        
        info = {
            "success": self.analysis_result.get("success", False),
            "basic_info": self.analysis_result.get("basic_info", {}),
            "column_analysis": self.analysis_result.get("column_analysis", {}),
            "standardized_mapping": self.analysis_result.get("standardized_mapping", {}),
            "raw_columns": self.analysis_result.get("raw_columns", [])
        }
        
        return info
    
    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries based on dynamic analysis."""
        if not self.analysis_result:
            return ["Show all data", "Provide data summary"]
        
        column_analysis = self.analysis_result.get("column_analysis", {})
        sample_data = self.analysis_result.get("sample_data", {})
        
        return self.csv_analyzer.get_suggested_queries(column_analysis, sample_data)
    
    def get_column_info_for_llm(self) -> str:
        """Get column information formatted for LLM context."""
        if not self.analysis_result:
            return "No column analysis available."
        
        info = "CSV Column Analysis:\n"
        
        # Basic info
        basic_info = self.analysis_result.get("basic_info", {})
        info += f"Total rows: {basic_info.get('total_rows', 'Unknown')}\n"
        info += f"Total columns: {basic_info.get('total_columns', 'Unknown')}\n\n"
        
        # Column details
        column_analysis = self.analysis_result.get("column_analysis", {})
        sample_data = self.analysis_result.get("sample_data", {})
        
        info += "Column Details:\n"
        for col, analysis in column_analysis.items():
            info += f"- '{col}': {analysis.get('likely_meaning', 'Unknown meaning')}\n"
            info += f"  Type: {analysis.get('data_type', 'Unknown')}\n"
            info += f"  Category: {analysis.get('standardized_category', 'other')}\n"
            samples = sample_data.get(col, [])
            if samples:
                info += f"  Sample values: {samples[:3]}\n"
            info += "\n"
        
        # Available queries
        mapping = self.analysis_result.get("standardized_mapping", {})
        info += "Available data types for queries:\n"
        categories = set(mapping.values())
        for category in categories:
            if category != 'other':
                info += f"- {category.replace('_', ' ').title()}\n"
        
        return info