# DTE Admission Chatbot

An AI-powered chatbot that helps analyze engineering college admission data for DTE (Directorate of Technical Education).

## Features

- Interactive chat interface for querying admission data
- Real-time analysis of engineering college statistics
- Support for various queries about colleges, programs, and admission trends
- Beautiful web interface with responsive design

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd DTE_admission_chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root and add:
```
GEMINI_API_KEY=your_api_key_here
```

4. Add your data:
Place your `EnggAdmissions2024.xlsx` file in the project root directory.

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open the web interface in your browser
2. Type your question about engineering admissions
3. Get instant analysis and insights about:
   - College rankings
   - Program statistics
   - Regional distribution
   - Institution types
   - And more!

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not in repo)
- `EnggAdmissions2024.xlsx`: Data file (not in repo)

## Roadmap & Future Improvements

The current version is a basic implementation. Here's what's planned for the future:

### ✅ Enhanced Data Processing

#### Input Requirements
- Support for standardized XLSX format with columns:
  - College Name
  - College Type
  - Regional Office
  - Course Name
  - Intake
  - Admissions

#### Advanced Analytics
- Automated calculation of derived metrics:
  - Vacancies (Intake - Admissions)
  - Percentage Vacancies ((Vacancies / Intake) * 100)
- Sophisticated aggregation capabilities:
  - Total intake calculations
  - Average admissions
  - Region-wise summaries
- Advanced sorting and ranking
- Multi-level filtering

### ✅ Natural Language Processing Improvements

Support for complex queries like:
- "Which college has the highest intake?"
- "Which colleges have maximum admissions?"
- "Which Government colleges have the minimum percentage of vacancies?"
- "What are the top 5 colleges with the minimum percentage of vacancies?"
- "Which regional offices have above-average admissions?"
- "Give me region-wise intake capacity and admissions."
- "Which course has the highest admissions? List the top 10 courses in demand."

### ✅ Enhanced Answer Quality

- Non-hallucinated, verifiable responses
- Clear handling of unanswerable questions
- Adaptive response formatting:
  - Single value/name for simple queries
  - Ranked lists for "Top N" queries
  - Formatted tables for grouped data
  - Visual representations when appropriate

### ✅ Performance Optimization

- Token usage optimization
- Response time improvements
- Caching frequently requested analyses
- Efficient data structure usage

### ✅ Technical Improvements

- Python 3.9+ compatibility
- Enhanced Pandas integration for data manipulation
- Optimized Gemini API usage
- Improved error handling and validation
- Better data verification mechanisms

### ✅ UI/UX Enhancements

- Streamlined file upload process
- Better query input interface
- Enhanced result visualization
- Mobile-responsive design improvements
- Loading states and progress indicators

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 