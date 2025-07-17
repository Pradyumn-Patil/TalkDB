"""
Advanced Analytics Engine for Admission Data
Handles complex queries and provides sophisticated analysis capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Structured result for analytics queries."""
    answer: str
    data: Optional[Dict[str, Any]] = None
    chart_type: Optional[str] = None
    insights: Optional[List[str]] = None

class AdvancedAnalyticsEngine:
    """Advanced analytics engine for processing complex admission queries."""
    
    def __init__(self, data_processor):
        """Initialize with a data processor instance."""
        self.data_processor = data_processor
        self.cache = {}
        
        # Query patterns for different types of analyses
        self.query_patterns = {
            'highest_intake': [
                r'(?:which|what).+(?:highest|maximum|most).+intake',
                r'(?:top|best).+intake',
                r'(?:college|institution).+(?:highest|maximum|most).+intake'
            ],
            'highest_admissions': [
                r'(?:which|what).+(?:highest|maximum|most).+admission',
                r'(?:top|best).+admission',
                r'(?:college|institution).+(?:highest|maximum|most).+admission'
            ],
            'minimum_vacancies': [
                r'(?:minimum|lowest|least).+(?:vacanc|percentage vacanc)',
                r'(?:college|institution).+(?:minimum|lowest|least).+(?:vacanc|percentage vacanc)',
                r'(?:government|private).+(?:minimum|lowest|least).+(?:vacanc|percentage vacanc)'
            ],
            'top_n': [
                r'(?:top|first)\s+(\d+)',
                r'(\d+)\s+(?:top|best|highest)',
                r'list\s+(?:top\s+)?(\d+)'
            ],
            'regional_analysis': [
                r'region.+(?:wise|analysis|comparison)',
                r'(?:regional office|region).+(?:intake|admission|capacity)',
                r'(?:which|what).+regional office.+(?:above|below).+average'
            ],
            'course_analysis': [
                r'(?:which|what).+course.+(?:highest|maximum|most)',
                r'(?:top|best).+course.+(?:demand|popular)',
                r'course.+(?:wise|analysis|comparison)'
            ],
            'college_type_filter': [
                r'government.+college',
                r'private.+(?:college|institution)',
                r'(?:autonomous|minority).+(?:college|institution)'
            ]
        }
    
    def process_query(self, query: str) -> QueryResult:
        """Process a natural language query and return structured results."""
        query_lower = query.lower().strip()
        
        try:
            # Check cache first
            if query_lower in self.cache:
                return self.cache[query_lower]
            
            # Identify query type and extract parameters
            query_type, params = self._analyze_query(query_lower)
            
            # Route to appropriate handler
            result = self._route_query(query_type, params, query_lower)
            
            # Cache result
            self.cache[query_lower] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResult(
                answer=f"I encountered an error processing your query: {str(e)}",
                insights=["Please try rephrasing your question or check if the data contains the required information."]
            )
    
    def _analyze_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Analyze query to determine type and extract parameters."""
        params = {}
        
        # Extract number for top N queries
        for pattern in self.query_patterns['top_n']:
            match = re.search(pattern, query)
            if match:
                params['n'] = int(match.group(1))
                break
        else:
            params['n'] = 10  # default
        
        # Extract college type filter
        if any(re.search(pattern, query) for pattern in self.query_patterns['college_type_filter']):
            if 'government' in query:
                params['college_type'] = 'Government'
            elif 'private' in query:
                params['college_type'] = 'Private'
            elif 'autonomous' in query:
                params['autonomous'] = True
            elif 'minority' in query:
                params['minority'] = True
        
        # Determine primary query type
        for query_type, patterns in self.query_patterns.items():
            if query_type == 'top_n' or query_type == 'college_type_filter':
                continue
            if any(re.search(pattern, query) for pattern in patterns):
                return query_type, params
        
        # Default to general analysis if no specific pattern matches
        return 'general_analysis', params
    
    def _route_query(self, query_type: str, params: Dict[str, Any], original_query: str) -> QueryResult:
        """Route query to appropriate handler based on type."""
        handlers = {
            'highest_intake': self._handle_highest_intake,
            'highest_admissions': self._handle_highest_admissions,
            'minimum_vacancies': self._handle_minimum_vacancies,
            'regional_analysis': self._handle_regional_analysis,
            'course_analysis': self._handle_course_analysis,
            'general_analysis': self._handle_general_analysis
        }
        
        handler = handlers.get(query_type, self._handle_general_analysis)
        return handler(params, original_query)
    
    def _handle_highest_intake(self, params: Dict[str, Any], query: str) -> QueryResult:
        """Handle queries about highest intake."""
        data = self.data_processor.college_summary.copy()
        
        if data.empty:
            return QueryResult(answer="No data available for analysis.")
        
        # Apply filters
        data = self._apply_filters(data, params)
        
        if data.empty:
            return QueryResult(answer="No colleges found matching your criteria.")
        
        # Get top colleges by intake
        top_colleges = data.nlargest(params.get('n', 10), 'intake')
        
        if len(top_colleges) == 1:
            college = top_colleges.iloc[0]
            answer = f"The college with the highest intake is {college['college_name']} with {college['intake']:,} seats."
        else:
            answer = f"Top {len(top_colleges)} colleges with highest intake:"
        
        insights = [
            f"Total intake across all colleges: {data['intake'].sum():,}",
            f"Average intake per college: {data['intake'].mean():.0f}",
            f"Highest intake: {top_colleges.iloc[0]['intake']:,}",
            f"Lowest intake in top {len(top_colleges)}: {top_colleges.iloc[-1]['intake']:,}"
        ]
        
        return QueryResult(
            answer=answer,
            data={
                'headers': ['College Name', 'College Type', 'Intake', 'Admissions', 'Vacancies', 'Vacancy %'],
                'rows': top_colleges[['college_name', 'college_type', 'intake', 'admissions', 'vacancies', 'percentage_vacancies']].to_dict('records')
            },
            chart_type='bar',
            insights=insights
        )
    
    def _handle_highest_admissions(self, params: Dict[str, Any], query: str) -> QueryResult:
        """Handle queries about highest admissions."""
        data = self.data_processor.college_summary.copy()
        
        if data.empty:
            return QueryResult(answer="No data available for analysis.")
        
        # Apply filters
        data = self._apply_filters(data, params)
        
        if data.empty:
            return QueryResult(answer="No colleges found matching your criteria.")
        
        # Get top colleges by admissions
        top_colleges = data.nlargest(params.get('n', 10), 'admissions')
        
        if len(top_colleges) == 1:
            college = top_colleges.iloc[0]
            answer = f"The college with the highest admissions is {college['college_name']} with {college['admissions']:,} students admitted."
        else:
            answer = f"Top {len(top_colleges)} colleges with highest admissions:"
        
        insights = [
            f"Total admissions across all colleges: {data['admissions'].sum():,}",
            f"Average admissions per college: {data['admissions'].mean():.0f}",
            f"Highest admissions: {top_colleges.iloc[0]['admissions']:,}",
            f"Overall admission rate: {(data['admissions'].sum() / data['intake'].sum() * 100):.1f}%"
        ]
        
        return QueryResult(
            answer=answer,
            data={
                'headers': ['College Name', 'College Type', 'Intake', 'Admissions', 'Admission Rate %'],
                'rows': top_colleges[['college_name', 'college_type', 'intake', 'admissions']].assign(
                    admission_rate=lambda x: (x['admissions'] / x['intake'] * 100).round(1)
                )[['college_name', 'college_type', 'intake', 'admissions', 'admission_rate']].to_dict('records')
            },
            chart_type='bar',
            insights=insights
        )
    
    def _handle_minimum_vacancies(self, params: Dict[str, Any], query: str) -> QueryResult:
        """Handle queries about minimum vacancies."""
        data = self.data_processor.college_summary.copy()
        
        if data.empty:
            return QueryResult(answer="No data available for analysis.")
        
        # Apply filters
        data = self._apply_filters(data, params)
        
        if data.empty:
            return QueryResult(answer="No colleges found matching your criteria.")
        
        # Filter out colleges with zero intake to avoid division issues
        data = data[data['intake'] > 0]
        
        # Get colleges with minimum vacancy percentage
        top_colleges = data.nsmallest(params.get('n', 10), 'percentage_vacancies')
        
        if len(top_colleges) == 1:
            college = top_colleges.iloc[0]
            answer = f"The college with minimum vacancy percentage is {college['college_name']} with {college['percentage_vacancies']:.1f}% vacancies."
        else:
            filter_text = " Government" if params.get('college_type') == 'Government' else ""
            answer = f"Top {len(top_colleges)}{filter_text} colleges with minimum vacancy percentage:"
        
        insights = [
            f"Average vacancy percentage: {data['percentage_vacancies'].mean():.1f}%",
            f"Lowest vacancy percentage: {top_colleges.iloc[0]['percentage_vacancies']:.1f}%",
            f"Highest vacancy percentage: {data['percentage_vacancies'].max():.1f}%",
            f"Total colleges analyzed: {len(data)}"
        ]
        
        return QueryResult(
            answer=answer,
            data={
                'headers': ['College Name', 'College Type', 'Intake', 'Admissions', 'Vacancies', 'Vacancy %'],
                'rows': top_colleges[['college_name', 'college_type', 'intake', 'admissions', 'vacancies', 'percentage_vacancies']].to_dict('records')
            },
            chart_type='bar',
            insights=insights
        )
    
    def _handle_regional_analysis(self, params: Dict[str, Any], query: str) -> QueryResult:
        """Handle regional analysis queries."""
        regional_data = self.data_processor.get_regional_analysis()
        
        if regional_data.empty:
            return QueryResult(answer="No regional data available for analysis.")
        
        # Check for above/below average queries
        if 'above average' in query or 'below average' in query:
            avg_admissions = regional_data['admissions'].mean()
            
            if 'above average' in query:
                filtered_data = regional_data[regional_data['admissions'] > avg_admissions]
                comparison = "above"
            else:
                filtered_data = regional_data[regional_data['admissions'] < avg_admissions]
                comparison = "below"
            
            answer = f"Regional offices with {comparison} average admissions (average: {avg_admissions:.0f}):"
            
            insights = [
                f"Average admissions per region: {avg_admissions:.0f}",
                f"Regions {comparison} average: {len(filtered_data)}",
                f"Total regions: {len(regional_data)}"
            ]
            
            return QueryResult(
                answer=answer,
                data={
                    'headers': ['Regional Office', 'Total Colleges', 'Total Intake', 'Total Admissions', 'Vacancy %'],
                    'rows': filtered_data[['regional_office', 'total_colleges', 'intake', 'admissions', 'percentage_vacancies']].to_dict('records')
                },
                chart_type='bar',
                insights=insights
            )
        else:
            # General regional analysis
            answer = "Region-wise intake capacity and admissions analysis:"
            
            insights = [
                f"Total regions: {len(regional_data)}",
                f"Region with highest intake: {regional_data.loc[regional_data['intake'].idxmax(), 'regional_office']}",
                f"Region with highest admissions: {regional_data.loc[regional_data['admissions'].idxmax(), 'regional_office']}",
                f"Overall vacancy percentage: {regional_data['percentage_vacancies'].mean():.1f}%"
            ]
            
            return QueryResult(
                answer=answer,
                data={
                    'headers': ['Regional Office', 'Total Colleges', 'Total Intake', 'Total Admissions', 'Total Vacancies', 'Vacancy %'],
                    'rows': regional_data.to_dict('records')
                },
                chart_type='bar',
                insights=insights
            )
    
    def _handle_course_analysis(self, params: Dict[str, Any], query: str) -> QueryResult:
        """Handle course analysis queries."""
        course_data = self.data_processor.get_course_analysis()
        
        if course_data.empty:
            return QueryResult(answer="No course data available for analysis.")
        
        # Get top courses by admissions (demand)
        top_courses = course_data.nlargest(params.get('n', 10), 'admissions')
        
        answer = f"Top {len(top_courses)} courses with highest admissions (most in demand):"
        
        insights = [
            f"Total courses available: {len(course_data)}",
            f"Most popular course: {top_courses.iloc[0]['course_name']} ({top_courses.iloc[0]['admissions']:,} admissions)",
            f"Total admissions across all courses: {course_data['admissions'].sum():,}",
            f"Average admissions per course: {course_data['admissions'].mean():.0f}"
        ]
        
        return QueryResult(
            answer=answer,
            data={
                'headers': ['Course Name', 'Total Colleges', 'Total Intake', 'Total Admissions', 'Vacancy %'],
                'rows': top_courses[['course_name', 'total_colleges', 'intake', 'admissions', 'percentage_vacancies']].to_dict('records')
            },
            chart_type='bar',
            insights=insights
        )
    
    def _handle_general_analysis(self, params: Dict[str, Any], query: str) -> QueryResult:
        """Handle general analysis queries."""
        data_info = self.data_processor.get_data_info()
        
        answer = "Here's a general analysis of the admission data:"
        
        insights = [
            f"Total colleges: {data_info.get('total_colleges', 0):,}",
            f"Total intake capacity: {data_info.get('total_intake', 0):,}",
            f"Total admissions: {data_info.get('total_admissions', 0):,}",
            f"Total vacancies: {data_info.get('total_vacancies', 0):,}",
            f"Overall vacancy percentage: {data_info.get('overall_vacancy_percentage', 0):.1f}%"
        ]
        
        # Add college type breakdown if available
        if 'college_types' in data_info:
            insights.append("College types breakdown:")
            for ctype, count in data_info['college_types'].items():
                insights.append(f"  {ctype}: {count:,}")
        
        return QueryResult(
            answer=answer,
            insights=insights
        )
    
    def _apply_filters(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to data based on parameters."""
        filtered_data = data.copy()
        
        if 'college_type' in params and 'college_type' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['college_type'] == params['college_type']]
        
        return filtered_data
    
    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
        logger.info("Query cache cleared")
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on partial input."""
        suggestions = [
            "Which college has the highest intake?",
            "Which colleges have maximum admissions?",
            "Which Government colleges have the minimum percentage of vacancies?",
            "What are the top 5 colleges with the minimum percentage of vacancies?",
            "Which regional offices have above-average admissions?",
            "Give me region-wise intake capacity and admissions.",
            "Which course has the highest admissions? List the top 10 courses in demand.",
            "Show me all Government colleges.",
            "What is the overall admission statistics?",
            "Compare Government vs Private colleges."
        ]
        
        if partial_query:
            # Simple matching based on keywords
            partial_lower = partial_query.lower()
            relevant_suggestions = [
                s for s in suggestions 
                if any(word in s.lower() for word in partial_lower.split())
            ]
            return relevant_suggestions[:5]
        
        return suggestions[:5]