from pandasai import SmartDataframe
from pandasai_google.google import GoogleGenerativeAI
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class QueryHandler:
    def __init__(self, excel_file: str):
        """Initialize the query handler with data source."""
        try:
            # Load the Excel file
            self.raw_data = pd.read_excel(excel_file)
            logger.info(f"Loaded data with shape: {self.raw_data.shape}")
            
            # Initialize Google Gemini LLM
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not found")
            
            llm = GoogleGenerativeAI(api_key=api_key)
            
            # Create SmartDataframe with enhanced prompts
            self.df = SmartDataframe(
                self.raw_data,
                config={
                    "llm": llm,
                    "verbose": True,
                    "enable_cache": True,
                    "custom_prompts": {
                        "system": """You are analyzing engineering college admission data for the DTE (Directorate of Technical Education).
                        
                        Available Data Columns:
                        - InstituteCode: Unique identifier for each institute
                        - InstituteName: Full name of the educational institute
                        - DistrictName: District where the institute is located
                        - RegionName: Regional office or administrative area
                        - Status: Type of institute (Government/Private)
                        - Autonomu: Autonomy status (Yes/No)
                        - Minority: Minority institution status (Yes/No)
                        - ProgramID: Identifier for specific programs/courses
                        
                        Important Notes:
                        1. Data NOT Available (explain this to users when asked):
                           - Intake capacity
                           - Admission numbers
                           - Vacancy information
                           - Fee structures
                           - Cutoff marks
                        
                        2. For Comparison Queries:
                           - Use visualizations for distribution analysis
                           - Show both counts and percentages
                           - Sort results for better readability
                        
                        3. For Regional Analysis:
                           - Group by RegionName for geographical insights
                           - Consider both district and region levels
                           - Use appropriate visualizations (bar charts, pie charts)
                        
                        4. For Institution Type Analysis:
                           - Consider Status, Autonomu, and Minority fields together
                           - Show relationships between these attributes
                           - Use cross-tabulation where appropriate
                        
                        Response Guidelines:
                        1. Always explain what analysis was performed
                        2. Suggest alternative analyses when requested data isn't available
                        3. Use visualizations for:
                           - Distribution comparisons
                           - Regional analysis
                           - Time series (if applicable)
                           - Category relationships
                        4. Format numbers for readability
                        5. Sort results meaningfully
                        """,
                        "generate_python_code": """
                        Instructions for code generation:
                        1. Always include proper error handling
                        2. Use seaborn for visualizations when possible
                        3. Set appropriate figure sizes
                        4. Use readable color palettes
                        5. Add proper labels and titles
                        6. Format numbers in output
                        7. Sort results meaningfully
                        8. Use plt.close() to clean up
                        
                        Example visualization code:
                        ```python
                        plt.figure(figsize=(10, 6))
                        sns.set_style("whitegrid")
                        sns.barplot(data=df, x="Status", y="count", palette="deep")
                        plt.title("Distribution of Colleges by Status")
                        plt.xlabel("Institution Status")
                        plt.ylabel("Number of Colleges")
                        plt.xticks(rotation=45)
                        ```
                        """
                    }
                }
            )
            logger.info("Successfully initialized SmartDataframe with custom configuration")
            
        except Exception as e:
            logger.error(f"Error initializing QueryHandler: {str(e)}")
            raise

    def process_query(self, query: str) -> dict:
        """Process a natural language query and return results."""
        try:
            # Execute query using PandasAI
            result = self.df.chat(query)
            
            # Format the response
            response = {
                "answer": str(result),  # Convert result to string for JSON serialization
                "data": None,  # Will be populated if result contains tabular data
                "visualization": None  # Will be populated if result contains a plot
            }
            
            # If result is a DataFrame, include it in the response
            if isinstance(result, pd.DataFrame):
                # Format the DataFrame for display
                result = result.round(2)  # Round numeric columns
                response["data"] = {
                    "headers": result.columns.tolist(),
                    "rows": result.to_dict('records')
                }
                
                # Generate appropriate visualization based on data
                if len(result) > 0:
                    if len(result.columns) >= 2:
                        plt.figure(figsize=(10, 6))
                        sns.set_style("whitegrid")
                        
                        # Choose visualization based on data types
                        numeric_cols = result.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_cols) > 0:
                            # Bar plot for numeric data
                            x_col = result.columns[0]
                            y_col = numeric_cols[0]
                            sns.barplot(data=result, x=x_col, y=y_col, palette="deep")
                            plt.xticks(rotation=45, ha='right')
                        else:
                            # Count plot for categorical data
                            sns.countplot(data=result, x=result.columns[0], palette="deep")
                            plt.xticks(rotation=45, ha='right')
                        
                        plt.tight_layout()
                        # Save plot to a temporary file or convert to base64
                        plt.savefig('temp_plot.png')
                        plt.close()
                        
                        # Add visualization path to response
                        response["visualization"] = 'temp_plot.png'
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "error": f"Failed to process query: {str(e)}"
            } 