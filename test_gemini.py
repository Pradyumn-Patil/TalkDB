import google.generativeai as genai
import pandas as pd
import os

# Your API key
GEMINI_API_KEY = ""

try:
    print("Configuring Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)
    
    print("\nListing available models:")
    for m in genai.list_models():
        print(f"- {m.name}")
    
    print("\nCreating model...")
    model = genai.GenerativeModel('models/gemini-pro')
    
    print("Testing with simple prompt...")
    response = model.generate_content("Say hello!")
    
    print("\nResponse from Gemini:")
    print(response.text)
    print("\nAPI key is working correctly!")
    
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("Please check if your API key is valid and has the necessary permissions.")

# Column Names
INSTITUTE_CODE_COLUMN = 'InstituteCode'
INSTITUTE_NAME_COLUMN = 'InstituteName'
PROGRAM_ID_COLUMN = 'ProgramID'

# Read the Excel file
excel_path = '/Users/pradyumn/Desktop/DTE_admission_chatbot/EnggAdmissions2024.xlsx'
df = pd.read_excel(excel_path)

# 1. Preview Data
print("\n=== First 5 Rows ===")
print(df.head())

# 2. Column Information
print("\n=== Column Analysis ===")
for column in df.columns:
    print(f"\nColumn: {column}")
    print(f"Data Type: {df[column].dtype}")
    print(f"Unique Values: {df[column].nunique()}")
    if df[column].dtype == 'object':
        print(f"Sample Values: {df[column].unique()[:3]}")
    
# 3. Dataset Summary
print("\n=== Dataset Summary ===")
print(f"Total Records: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print("\nColumn Names:", df.columns.tolist())

# Institute Statistics
print("\n=== Institute Statistics ===")
print(f"Total Unique Institutes: {df[INSTITUTE_CODE_COLUMN].nunique()}")
print("\nInstitute Types:")
print(df['Status'].value_counts())

# Program Statistics
print("\n=== Program Statistics ===")
print(f"Total Programs: {df[PROGRAM_ID_COLUMN].nunique()}")

# Regional Distribution
print("\n=== Regional Distribution ===")
print(df['RegionName'].value_counts()) 