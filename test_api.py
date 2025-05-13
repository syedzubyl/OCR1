#!/usr/bin/env python3
"""
Test script for the Aadhaar OCR API
Usage: python test_api.py <pdf_file_path> [pdf_password]
"""

import sys
import requests
import json
import os

def test_extract_api(pdf_path, password=None):
    """Test the /extract API endpoint with a local PDF file"""
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    # Check if the file is a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: File {pdf_path} is not a PDF")
        return
    
    # Prepare the API URL (change this to your Render URL when deployed)
    api_url = "http://127.0.0.1:5000/extract"
    
    # Prepare the files and data for the request
    files = {'file': open(pdf_path, 'rb')}
    data = {}
    
    if password:
        data['password'] = password
    
    print(f"Sending request to {api_url}...")
    
    try:
        # Send the request
        response = requests.post(api_url, files=files, data=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            result = response.json()
            
            # Check if there was an error
            if 'error' in result:
                print(f"API Error: {result['error']}")
                return
            
            # Print the extracted information
            print("\n=== Extracted Aadhaar Card Details ===\n")
            
            if result.get('aadhaar_number'):
                print(f"Aadhaar Number: {result['aadhaar_number']}")
            
            if result.get('name_tamil'):
                print(f"Name (Tamil): {result['name_tamil']}")
            
            if result.get('name'):
                print(f"Name: {result['name']}")
            
            if result.get('guardian_name'):
                print(f"Guardian Name: {result['guardian_name']}")
            
            if result.get('dob'):
                print(f"Date of Birth: {result['dob']}")
            
            if result.get('gender'):
                print(f"Gender: {result['gender']}")
            
            if result.get('address'):
                print(f"Address: {result['address']}")
            
            if result.get('district'):
                print(f"District: {result['district']}")
            
            if result.get('state'):
                print(f"State: {result['state']}")
            
            if result.get('pincode'):
                print(f"Pincode: {result['pincode']}")
            
            if result.get('phone'):
                print(f"Phone: {result['phone']}")
            
            if result.get('vid'):
                print(f"VID: {result['vid']}")
            
            # Save the raw text to a file for debugging
            with open('raw_text_output.txt', 'w', encoding='utf-8') as f:
                f.write(result.get('raw_text', ''))
            
            print("\nRaw text saved to 'raw_text_output.txt'")
            
            # Save the full JSON response to a file
            with open('api_response.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print("Full JSON response saved to 'api_response.json'")
            
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Close the file
        files['file'].close()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <pdf_file_path> [pdf_password]")
        sys.exit(1)
    
    # Get the PDF file path
    pdf_path = sys.argv[1]
    
    # Get the password if provided
    password = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Test the API
    test_extract_api(pdf_path, password)
