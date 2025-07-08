import pandas as pd
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, data: pd.DataFrame):
        """Initialize with a pandas DataFrame containing the admission data."""
        self.data = data
        self.available_columns = data.columns.tolist()

    def process_query(self, parsed_query: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], str]:
        """Process a parsed query and return results with explanation."""
        try:
            # Check if query can be answered
            if not parsed_query.get('can_answer', True):
                return None, parsed_query.get('explanation', 'Query cannot be answered with available data')

            query_type = parsed_query.get('query_type')
            params = parsed_query.get('parameters', {})
            
            # Apply any filters first
            df = self.data.copy()
            filters = params.get('filters', {})
            for col, value in filters.items():
                if col in df.columns:
                    df = df[df[col] == value]

            # Process based on query type
            if query_type == 'DISTRIBUTION':
                return self._process_distribution(df, params)
            elif query_type == 'COUNT_BY':
                return self._process_count_by(df, params)
            elif query_type == 'GROUP_SUMMARY':
                return self._process_group_summary(df, params)
            else:
                return None, f"Unsupported query type: {query_type}"

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return None, f"Error processing query: {str(e)}"

    def _process_distribution(self, df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, str]:
        """Process a distribution analysis query."""
        try:
            group_by = params.get('group_by', [])
            if isinstance(group_by, str):
                group_by = [group_by]

            # Ensure all group_by columns exist
            if not all(col in df.columns for col in group_by):
                missing = [col for col in group_by if col not in df.columns]
                return None, f"Missing columns: {missing}"

            # Calculate distribution
            result = df.groupby(group_by).size().reset_index(name='Count')
            
            # Add percentage
            total = result['Count'].sum()
            result['Percentage'] = (result['Count'] / total * 100).round(2)
            
            # Apply limit if specified
            if params.get('limit'):
                result = result.nlargest(params['limit'], 'Count')
            
            explanation = f"Analyzed distribution by {', '.join(group_by)} (showing counts and percentages)"
            return result, explanation

        except Exception as e:
            logger.error(f"Error in distribution analysis: {str(e)}")
            return None, f"Error in distribution analysis: {str(e)}"

    def _process_count_by(self, df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, str]:
        """Process a count analysis query."""
        try:
            group_by = params.get('group_by', [])
            if isinstance(group_by, str):
                group_by = [group_by]

            # Ensure all group_by columns exist
            if not all(col in df.columns for col in group_by):
                missing = [col for col in group_by if col not in df.columns]
                return None, f"Missing columns: {missing}"

            result = df.groupby(group_by).size().reset_index(name='Count')
            
            # Apply limit if specified
            if params.get('limit'):
                result = result.nlargest(params['limit'], 'Count')
            
            explanation = f"Counted entries by {', '.join(group_by)}"
            return result, explanation

        except Exception as e:
            logger.error(f"Error in count analysis: {str(e)}")
            return None, f"Error in count analysis: {str(e)}"

    def _process_group_summary(self, df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, str]:
        """Process a group summary query."""
        try:
            group_by = params.get('group_by', [])
            metrics = params.get('metrics', [])
            
            if isinstance(group_by, str):
                group_by = [group_by]

            # Ensure all columns exist
            all_cols = group_by + metrics
            if not all(col in df.columns for col in all_cols):
                missing = [col for col in all_cols if col not in df.columns]
                return None, f"Missing columns: {missing}"

            # Group by and calculate metrics
            result = df.groupby(group_by).agg({
                col: ['count', 'nunique'] for col in metrics
            }).reset_index()
            
            # Flatten column names
            result.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                            for col in result.columns]
            
            # Apply limit if specified
            if params.get('limit'):
                sort_col = f"{metrics[0]}_count" if metrics else group_by[0]
                result = result.nlargest(params['limit'], sort_col)
            
            explanation = f"Analyzed {', '.join(metrics)} grouped by {', '.join(group_by)}"
            return result, explanation

        except Exception as e:
            logger.error(f"Error in group summary: {str(e)}")
            return None, f"Error in group summary: {str(e)}" 