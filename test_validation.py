#!/usr/bin/env python3
"""
Test script for the Aadhaar OCR Validation API
Usage: python test_validation.py <pdf_file_path> <name> <dob> <state> <aadhaar_number> [pdf_password]
"""

import sys
import requests
import json
import os

def test_validate_api(pdf_path, name, dob, state, aadhaar_number, password=None):
    """Test the /validate API endpoint with a local PDF file and user details"""
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        return
    
    # Check if the file is a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: File {pdf_path} is not a PDF")
        return
    
    # Prepare the API URL (change this to your Render URL when deployed)
    api_url = "http://127.0.0.1:5000/validate"
    
    # Prepare the files and data for the request
    files = {'file': open(pdf_path, 'rb')}
    data = {
        'name': name,
        'dob': dob,
        'state': state,
        'aadhaar_number': aadhaar_number
    }
    
    if password:
        data['password'] = password
    
    print(f"Sending validation request to {api_url}...")
    print(f"User details: Name='{name}', DOB='{dob}', State='{state}', Aadhaar='{aadhaar_number}'")
    
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
            
            # Print the validation results
            print("\n=== Aadhaar Validation Results ===\n")
            
            # Overall result
            is_valid = result.get('is_valid', False)
            confidence = result.get('confidence_score', 0)
            
            if is_valid:
                print(f"✅ VALIDATION PASSED (Confidence: {confidence}%)")
            else:
                print(f"❌ VALIDATION FAILED (Confidence: {confidence}%)")
            
            # Print match details
            print("\nField Matches:")
            matches = result.get('matches', {})
            
            for field, match in matches.items():
                status = "✅ Match" if match else "❌ No Match"
                print(f"  {field}: {status}")
            
            # Print comparison details
            print("\nComparison Details:")
            extracted = result.get('extracted_details', {})
            user = result.get('user_details', {})
            
            print(f"  Name: '{user.get('name', '')}' vs '{extracted.get('name', '')}'")
            print(f"  DOB: '{user.get('dob', '')}' vs '{extracted.get('dob', '')}'")
            print(f"  State: '{user.get('state', '')}' vs '{extracted.get('state', '')}'")
            print(f"  Aadhaar: '{user.get('aadhaar_number', '')}' vs '{extracted.get('aadhaar_number', '')}'")
            
            # Save the full JSON response to a file
            with open('validation_response.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print("\nFull JSON response saved to 'validation_response.json'")
            
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
    if len(sys.argv) < 6:
        print("Usage: python test_validation.py <pdf_file_path> <name> <dob> <state> <aadhaar_number> [pdf_password]")
        sys.exit(1)
    
    # Get the arguments
    pdf_path = sys.argv[1]
    name = sys.argv[2]
    dob = sys.argv[3]
    state = sys.argv[4]
    aadhaar_number = sys.argv[5]
    
    # Get the password if provided
    password = sys.argv[6] if len(sys.argv) > 6 else None
    
    # Test the API
    test_validate_api(pdf_path, name, dob, state, aadhaar_number, password)
