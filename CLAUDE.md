# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a refactored Flask-based AI chatbot for analyzing DTE (Directorate of Technical Education) engineering college admission data. The system now supports standardized CSV uploads with advanced analytics and natural language processing.

## Common Development Commands

### Running the Application

**Refactored Version (Recommended):**
```bash
python app_refactored.py
```

**Original Version:**
```bash
python app.py
```

The application runs on http://localhost:5000

### Installing Dependencies
```bash
# For refactored version
pip install -r requirements_refactored.txt

# For original version
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test_api.py
```

### Linting
```bash
# Run flake8 linter
python -m flake8 .
```

## Architecture Overview (Refactored)

### Core Components

1. **app_refactored.py**: Main Flask application with:
   - CSV file upload functionality (`/upload` endpoint)
   - Advanced analytics via `/chat` endpoint
   - Fallback AI processing via `/fallback_chat` endpoint
   - Data info and suggestions endpoints

2. **data_processor.py**: Standardized data processor that:
   - Handles CSV files with flexible column mapping
   - Calculates derived metrics (vacancies, percentages)
   - Generates college, course, and regional summaries
   - Supports multiple encodings and file formats

3. **analytics_engine.py**: Advanced analytics engine with:
   - Pattern-based query analysis
   - Structured result formatting
   - Caching for performance
   - Support for complex queries like "top N", "minimum vacancies", etc.

4. **templates/index_enhanced.html**: Enhanced frontend with:
   - Drag-and-drop file upload
   - Real-time data statistics
   - Advanced table visualization
   - Responsive design with quick stats sidebar

### Data Format Requirements

The system expects CSV files with these standardized columns:
- **College Name**: Name of the educational institution
- **College Type**: Government/Private/Autonomous
- **Regional Office**: Administrative region
- **Course Name**: Program/course offered
- **Intake**: Total seat capacity
- **Admissions**: Number of students admitted

### Automatic Derived Metrics
The system automatically calculates:
- **Vacancies**: Intake - Admissions
- **Percentage Vacancies**: (Vacancies / Intake) × 100
- **Admission Rate**: (Admissions / Intake) × 100

### API Endpoints

- `GET /`: Main chat interface
- `POST /upload`: File upload and processing
- `POST /chat`: Advanced analytics queries
- `POST /fallback_chat`: AI-powered fallback for complex queries
- `GET /data_info`: Get loaded data statistics
- `GET /suggestions`: Get query suggestions
- `POST /clear_cache`: Clear analytics cache

### Supported Query Types

1. **Highest Intake**: "Which college has the highest intake?"
2. **Highest Admissions**: "Which colleges have maximum admissions?"
3. **Minimum Vacancies**: "Which Government colleges have minimum vacancy percentage?"
4. **Top N Queries**: "Top 5 colleges with lowest vacancies"
5. **Regional Analysis**: "Region-wise intake and admissions"
6. **Course Analysis**: "Which course has highest demand?"
7. **Filtered Queries**: "Government colleges", "Private institutions"

### API Configuration
Set the Gemini API key either:
1. In `.env` file: `GEMINI_API_KEY=your_key`
2. As environment variable: `export GEMINI_API_KEY=your_key`

### File Structure
```
├── app_refactored.py           # Main refactored application
├── data_processor.py           # Standardized data processing
├── analytics_engine.py         # Advanced analytics engine
├── templates/
│   ├── index.html             # Original template
│   └── index_enhanced.html    # Enhanced template with upload
├── uploads/                    # File upload directory
├── sample_data.csv            # Sample CSV format
├── requirements_refactored.txt # Dependencies for refactored version
└── CLAUDE.md                  # This file
```

### Important Patterns

1. **Flexible Column Mapping**: The system automatically maps various column name variations to standardized names
2. **Data Validation**: Comprehensive validation ensures data integrity and prevents errors
3. **Caching**: Query results are cached for improved performance
4. **Structured Results**: All analytics return structured QueryResult objects with answer, data, and insights
5. **Error Handling**: Graceful error handling with user-friendly messages

### Testing Approach
- Use `sample_data.csv` for testing the refactored system
- Upload CSV files via the web interface to test processing
- Test various query patterns to verify analytics engine
- Use `test_api.py` for API connectivity verification

## Development Tips

1. **Column Mapping**: When adding new data sources, update the `REQUIRED_COLUMNS` mapping in `data_processor.py`
2. **Query Patterns**: Add new query patterns to `analytics_engine.py` for better natural language understanding
3. **Frontend**: Use `index_enhanced.html` template for the full-featured interface
4. **Performance**: Clear cache periodically or implement cache expiration for production use
5. **Data Validation**: Always validate uploaded data format before processing
6. **Error Messages**: Provide specific, actionable error messages for better user experience

## Migration from Original Version

To migrate from the original Excel-based version:
1. Convert Excel files to CSV format
2. Ensure column names match the expected format (see sample_data.csv)
3. Use the refactored application (`app_refactored.py`)
4. Upload CSV files via the web interface