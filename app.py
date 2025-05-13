import os
import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv
import io
import fitz  # PyMuPDF
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for the entire app with all origins
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size
app.secret_key = os.getenv('SECRET_KEY', 'dev_key')  # Secret key for session

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Aadhaar data model
class AadhaarData(BaseModel):
    aadhaar_number: str = ""
    name_tamil: str = ""
    name: str = ""
    guardian_name: str = ""
    dob: str = ""
    gender: str = ""
    address: str = ""
    district: str = ""
    state: str = ""
    pincode: str = ""
    phone: str = ""
    vid: str = ""
    raw_text: str = ""

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract text from image using Tesseract OCR
def extract_text_from_image(image: Image.Image) -> str:
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process with OpenCV for better OCR results
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get better results
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR the image
    custom_config = r'--oem 3 --psm 6'
    try:
        # Try with Tamil language support
        return pytesseract.image_to_string(thresh, config=custom_config, lang='eng+tam')
    except:
        # Fallback to English only
        return pytesseract.image_to_string(thresh, config=custom_config, lang='eng')

# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_bytes: bytes, password: str = None) -> str:
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.needs_pass and password:
            success = doc.authenticate(password)
            if not success:
                return "ERROR: Invalid password for PDF"

        # Extract text from each page
        for page in doc:
            text += page.get_text("text")

        # If text extraction yields little content, try image-based extraction
        if len(text.strip()) < 100:
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

            # Process each image with OCR
            for img in images:
                text += extract_text_from_image(img)

        return text
    except Exception as e:
        return f"ERROR: {str(e)}"

# Extract name logic
def extract_name_from_text(lines):
    unwanted_phrases = [
        "Digitally signed by DS Unique",
        "Identification Authority of India",
        "Government of India",
        "Signature Not Verified",
        "Tamil Nadu",
        "Kerala",
        "Karnataka",
        "Andhra Pradesh",
        "Telangana",
        "Maharashtra",
        "Gujarat",
        "Rajasthan",
        "Madhya Pradesh",
        "Uttar Pradesh",
        "Bihar",
        "West Bengal",
        "Odisha",
        "Punjab",
        "Haryana",
        "Jharkhand",
        "Chhattisgarh",
        "Uttarakhand",
        "Himachal Pradesh",
        "Jammu and Kashmir",
        "Assam",
        "Manipur",
        "Meghalaya",
        "Nagaland",
        "Tripura",
        "Arunachal Pradesh",
        "Mizoram",
        "Sikkim",
        "Goa",
    ]

    # First, try to find name after "Name:" label
    name_pattern = r'Name[:\s]+([A-Za-z\s\'-]+)'
    for line in lines:
        name_match = re.search(name_pattern, line, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            # Check if the extracted name is not a state or other unwanted phrase
            if name and all(phrase.lower() not in name.lower() for phrase in unwanted_phrases):
                return name

    # If not found with label, look for guardian name and infer the person's name
    guardian_pattern = r'(?:S/O|C/O|W/O|D/O)[.:]?\s*([A-Za-z\s\'-]+)'
    for line in lines:
        guardian_match = re.search(guardian_pattern, line, re.IGNORECASE)
        if guardian_match:
            # If we found a guardian name, look for a name before it in the same line
            full_line = line.strip()
            parts = re.split(r'\s*(?:S/O|C/O|W/O|D/O)\s*', full_line, flags=re.IGNORECASE)
            if len(parts) > 1 and parts[0].strip():
                name_part = parts[0].strip()
                # Check if the extracted name is not a state or other unwanted phrase
                if all(phrase.lower() not in name_part.lower() for phrase in unwanted_phrases):
                    return name_part

    # If still not found, try to find a name pattern in the text
    for line in lines:
        clean_line = line.strip()
        # Check for a line that looks like a name (all alphabetic, more than one word)
        if (
            re.match(r'^[A-Za-z\s\'-]+$', clean_line)
            and len(clean_line.split()) >= 2
            and len(clean_line.split()) <= 5  # Most names have 2-5 words
            and all(phrase.lower() not in clean_line.lower() for phrase in unwanted_phrases)
            and not re.search(r'(?:address|gender|dob|date of birth|male|female|district|state|pincode)',
                             clean_line, re.IGNORECASE)
        ):
            # Make sure it's not just a state name or other common label
            name_part = re.split(r'\s*(?:S/O|C/O|W/O|D/O)\s*', clean_line, flags=re.IGNORECASE)[0]
            name_part = re.sub(r'\s+[CWSD]\s*$', '', name_part).strip()
            name_part = re.sub(r'\s+', ' ', name_part)

            # Additional check to avoid state names or other common fields
            if len(name_part) > 3 and all(phrase.lower() not in name_part.lower() for phrase in unwanted_phrases):
                return name_part

    # Look for name near the guardian name
    for i, line in enumerate(lines):
        if "S/O" in line or "D/O" in line or "W/O" in line or "C/O" in line:
            # Check the line before for a potential name
            if i > 0:
                prev_line = lines[i-1].strip()
                if (
                    re.match(r'^[A-Za-z\s\'-]+$', prev_line)
                    and all(phrase.lower() not in prev_line.lower() for phrase in unwanted_phrases)
                    and not re.search(r'(?:address|gender|dob|date of birth|male|female|district|state|pincode)',
                                     prev_line, re.IGNORECASE)
                ):
                    return prev_line

    # If we still can't find a name, look for the guardian name as a fallback
    for line in lines:
        guardian_match = re.search(r'(?:S/O|C/O|W/O|D/O)[.:]?\s*([A-Za-z\s\'-]+)', line, re.IGNORECASE)
        if guardian_match:
            return f"[Guardian: {guardian_match.group(1).strip()}]"

    return ""

# Parse Aadhaar details
def parse_aadhaar_details(text: str) -> AadhaarData:
    data = AadhaarData()
    data.raw_text = text

    # If there was an error in text extraction
    if text.startswith("ERROR:"):
        return data

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Aadhaar Number
    aadhaar_match = re.search(r'\b(\d{4}\s\d{4}\s\d{4})\b', text)
    if aadhaar_match:
        data.aadhaar_number = aadhaar_match.group(1)

    # VID
    vid_match = re.search(r'VID[:\s]*(\d{4}\s\d{4}\s\d{4}\s\d{4})', text)
    if vid_match:
        data.vid = vid_match.group(1)

    # Name (Tamil and English)
    tamil_name_match = re.search(r'([\u0B80-\u0BFF\s]+)\n([A-Za-z\s\'-]+)', text)
    if tamil_name_match:
        data.name_tamil = tamil_name_match.group(1).strip()
        potential_name = tamil_name_match.group(2).strip().replace("\n", " ")
        potential_name = re.split(r'\s*(?:S/O|C/O|W/O|D/O)\s*', potential_name, flags=re.IGNORECASE)[0].strip()
        potential_name = re.sub(r'\s+[CWSD]\s*$', '', potential_name).strip()
        potential_name = re.sub(r'\s+', ' ', potential_name)

        # Check if the potential name is not a state name
        unwanted_state_names = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh"]
        if potential_name and all(state.lower() not in potential_name.lower() for state in unwanted_state_names):
            data.name = potential_name

    # If name is still not found or is a state name, use the enhanced extraction
    if not data.name or any(state.lower() in data.name.lower() for state in ["Tamil Nadu", "Kerala", "Karnataka"]):
        extracted_name = extract_name_from_text(lines)
        if extracted_name:
            data.name = extracted_name

    # Look for name near guardian name as a last resort
    if not data.name and data.guardian_name:
        # Try to find a name pattern in lines before guardian name is mentioned
        for i, line in enumerate(lines):
            if data.guardian_name in line:
                # Check previous lines for potential name
                for j in range(max(0, i-3), i):
                    prev_line = lines[j].strip()
                    if (re.match(r'^[A-Za-z\s\'-]+$', prev_line) and
                        not any(state.lower() in prev_line.lower() for state in ["Tamil Nadu", "Kerala", "Karnataka"]) and
                        not re.search(r'(?:address|gender|dob|date of birth|male|female|district|state|pincode)',
                                     prev_line, re.IGNORECASE)):
                        data.name = prev_line
                        break

    # Guardian Name
    # First try with a more specific pattern that looks for "Guardian Name:" label
    guardian_label_match = re.search(r'Guardian\s+Name[:\s]+([A-Za-z\s\'-]+)', text, re.IGNORECASE)
    if guardian_label_match:
        data.guardian_name = guardian_label_match.group(1).strip()
    else:
        # Fall back to the S/O, D/O pattern
        guardian_match = re.search(r'(S/o|C/o|D/o|W/o)[.:]?\s*([A-Za-z\s\'-]+)', text, re.IGNORECASE)
        if guardian_match:
            data.guardian_name = guardian_match.group(2).strip()

    # DOB
    dob_match = re.search(r'(DOB|Date of Birth|D\\.O\\.B)[:\s]*?(\d{1,2}[-/]\d{1,2}[-/]\d{4})', text, re.IGNORECASE)
    if dob_match:
        data.dob = dob_match.group(2).replace('-', '/')

    # Gender
    gender_match = re.search(r'\b(Male|Female|Transgender|M|F|T)\b', text, re.IGNORECASE)
    if gender_match:
        data.gender = gender_match.group(1).capitalize()

    # Address - improved extraction to separate state
    address_match = re.search(r'(?i)address[:\s]*(.*?)(?=\nDistrict|\nState|\n\d{6}|\nVID|\nDigitally|$)', text, re.DOTALL)
    if address_match:
        address_text = re.sub(r'(S/o|C/o|D/o|W/o)[.:]?\s*[A-Za-z\s\'-]+', '', address_match.group(1).strip(), flags=re.IGNORECASE)
        address_text = re.sub(r'\b\d{4}\s\d{4}\s\d{4}\b', '', address_text)
        address_text = re.sub(r'PO:.*?,', '', address_text)

        # List of Indian states to identify in the address
        indian_states = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
            "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
            "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
            "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
            "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
            "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
        ]

        # Extract state from address if present
        for state in indian_states:
            state_pattern = re.compile(r'(' + re.escape(state) + r')', re.IGNORECASE)
            if state_pattern.search(address_text):
                # If state is found in address, save it separately and remove from address
                if not data.state:
                    data.state = state
                # Remove state from address
                address_text = state_pattern.sub('', address_text)
                break

        # Clean up the address
        address_text = re.sub(r'(?i)\b(dist|state)\b.*?[,.]', '', address_text)
        address_text = re.sub(r'\n+', ' ', address_text).strip()
        address_text = re.sub(r'\s+', ' ', address_text).strip()
        address_text = re.sub(r'\s*-\s*\d{6}', '', address_text)  # Remove pincode
        address_text = re.sub(r'\s*,\s*$', '', address_text)  # Remove trailing comma
        address_text = re.sub(r'\s+,', ',', address_text)  # Fix spaces before commas

        data.address = address_text

    # District
    district_match = re.search(r'District[:\s]*(.*)', text, re.IGNORECASE)
    if district_match:
        data.district = district_match.group(1).strip().replace(',', '')

    # State - improved extraction with multiple patterns
    # First try with explicit "State:" label
    state_match = re.search(r'State[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
    if state_match:
        data.state = state_match.group(1).strip().rstrip(',')

    # If not found, try to extract from address
    if not data.state and data.address:
        # List of Indian states to look for in the address
        indian_states = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
            "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
            "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
            "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
            "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
            "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
            "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
        ]

        # Look for state names in the address
        for state in indian_states:
            if state.lower() in data.address.lower():
                data.state = state
                break

    # Try to find state with pincode pattern
    if not data.state:
        state_pincode_pattern = r'([A-Za-z\s]+)[\s-]+(\d{6})'
        state_pincode_match = re.search(state_pincode_pattern, text)
        if state_pincode_match:
            potential_state = state_pincode_match.group(1).strip()
            # Verify it's a state name
            if any(state.lower() in potential_state.lower() for state in indian_states):
                data.state = next((state for state in indian_states if state.lower() in potential_state.lower()), potential_state)

    # Try using the specialized state extraction function
    if not data.state:
        extracted_state = extract_state_from_pattern(text)
        if extracted_state:
            data.state = extracted_state

    # Try to extract state from common patterns in the raw text
    if not data.state:
        # Look for "Tamil Nadu" or other state names in the text
        for state in indian_states:
            if re.search(r'\b' + re.escape(state) + r'\b', text, re.IGNORECASE):
                data.state = state
                break

    # Clean up state name if found
    if data.state:
        data.state = data.state.strip().rstrip(',').rstrip('-').strip()

    # Pincode
    pincode_match = re.search(r'\b(\d{6})\b', text)
    if pincode_match:
        data.pincode = pincode_match.group(1)

    # Phone Number
    phone_match = re.search(r'\b(\d{10})\b', text)
    if phone_match:
        data.phone = phone_match.group(1)

    return data

# Process Aadhaar card from file path
def process_aadhaar_card(file_path, pdf_password=None):
    try:
        file_ext = os.path.splitext(file_path.lower())[1]

        if file_ext == '.pdf':
            # Process PDF file
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            text = extract_text_from_pdf(pdf_bytes, pdf_password)
        else:
            # Process image file with enhanced preprocessing
            image = Image.open(file_path)
            text = process_image_for_ocr(image)

        # Parse the extracted text
        aadhaar_data = parse_aadhaar_details(text)

        # Convert to dictionary for session storage
        return aadhaar_data.model_dump()

    except Exception as e:
        return {'error': str(e), 'raw_text': ''}

# Legacy helper functions for backward compatibility
def extract_aadhar_number(text):
    # Aadhar number pattern: XXXX XXXX XXXX or XXXX-XXXX-XXXX
    pattern = r'\d{4}[\s-]?\d{4}[\s-]?\d{4}'
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

def extract_name(text):
    # Name usually appears after "Name:" or similar text
    name_patterns = [
        r'Name[:\s]+([A-Za-z\s]+)',
        r'([A-Z][a-z]+\s[A-Z][a-z]+)'  # Simple pattern for first and last name
    ]

    for pattern in name_patterns:
        matches = re.search(pattern, text)
        if matches:
            return matches.group(1).strip()
    return None

def extract_dob(text):
    # DOB patterns: DD/MM/YYYY or DD-MM-YYYY
    dob_patterns = [
        r'DOB[:\s]+(\d{2}/\d{2}/\d{4})',
        r'Date of Birth[:\s]+(\d{2}/\d{2}/\d{4})',
        r'(\d{2}[/-]\d{2}[/-]\d{4})'
    ]

    for pattern in dob_patterns:
        matches = re.search(pattern, text)
        if matches:
            return matches.group(1)
    return None

def extract_gender(text):
    # Gender is usually a single word: MALE or FEMALE
    gender_pattern = r'(?:Gender|Sex)[:\s]+(MALE|FEMALE|Male|Female|M|F)'
    matches = re.search(gender_pattern, text)
    return matches.group(1) if matches else None

def extract_address(text):
    # Address is usually a multi-line text after "Address:" or similar
    address_pattern = r'Address[:\s]+(.+?)(?:\n\n|\Z)'
    matches = re.search(address_pattern, text, re.DOTALL)
    if matches:
        return matches.group(1).strip()
    return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "service": "aadhaar-ocr-api"
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']
    pdf_password = request.form.get('pdf_password', '')

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded PDF with password if provided
        results = process_aadhaar_card(filepath, pdf_password)

        # Store results in session for display
        session['ocr_results'] = results

        return redirect(url_for('results'))

    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'ocr_results' not in session:
        return redirect(url_for('index'))

    results = session['ocr_results']
    return render_template('results.html', results=results)

# API endpoint for extraction
@app.route('/extract', methods=['POST'])
def extract_aadhaar():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Please upload a PDF file."})

    file = request.files['file']
    password = request.form.get('password', None)

    if file.filename == '':
        return jsonify({"error": "No selected file. Please upload a PDF file."})

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a PDF file."})

    # Process the PDF file
    contents = file.read()
    text = extract_text_from_pdf(contents, password)

    # Parse the extracted text
    aadhaar_data = parse_aadhaar_details(text)

    # Add raw text to the response for debugging
    result = aadhaar_data.model_dump()
    result['raw_text'] = text

    # Print formatted JSON to terminal for debugging
    print("\nExtracted Aadhaar Details:")
    print(aadhaar_data.model_dump_json(indent=4))

    return jsonify(result)

# API endpoint for validation
@app.route('/validate', methods=['POST'])
def validate_aadhaar():
    """
    Validate user-entered details against extracted Aadhaar card details

    Expected JSON payload:
    {
        "user_details": {
            "name": "User entered name",
            "dob": "DD/MM/YYYY",
            "state": "User entered state",
            "aadhaar_number": "XXXX XXXX XXXX"
        }
    }

    Plus a PDF file upload with optional password
    """
    # Check if file is provided
    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Please upload a PDF file."})

    # Get the file and password
    file = request.files['file']
    password = request.form.get('password', None)

    # Check if file is valid
    if file.filename == '':
        return jsonify({"error": "No selected file. Please upload a PDF file."})

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a PDF file."})

    # Get user details from form data
    user_name = request.form.get('name', '').strip()
    user_dob = request.form.get('dob', '').strip()
    user_state = request.form.get('state', '').strip()
    user_aadhaar = request.form.get('aadhaar_number', '').strip()

    # Validate required fields
    if not user_name:
        return jsonify({"error": "Name is required"})

    if not user_dob:
        return jsonify({"error": "Date of Birth is required"})

    if not user_aadhaar:
        return jsonify({"error": "Aadhaar number is required"})

    # Process the PDF file
    contents = file.read()
    text = extract_text_from_pdf(contents, password)

    # Parse the extracted text
    aadhaar_data = parse_aadhaar_details(text)

    # Initialize validation results
    validation_results = {
        "is_valid": False,
        "matches": {
            "name": False,
            "dob": False,
            "state": False,
            "aadhaar_number": False
        },
        "extracted_details": aadhaar_data.model_dump(),
        "user_details": {
            "name": user_name,
            "dob": user_dob,
            "state": user_state,
            "aadhaar_number": user_aadhaar
        }
    }

    # Compare name (case-insensitive and fuzzy matching)
    extracted_name = aadhaar_data.name.lower() if aadhaar_data.name else ""
    user_name_lower = user_name.lower()

    # Check if user name is contained in extracted name or vice versa
    name_match = (user_name_lower in extracted_name or extracted_name in user_name_lower)

    # If not direct match, check for similarity (allowing for OCR errors)
    if not name_match and extracted_name and user_name_lower:
        # Simple word-based comparison (checking if most words match)
        extracted_words = set(extracted_name.split())
        user_words = set(user_name_lower.split())
        common_words = extracted_words.intersection(user_words)

        if len(common_words) >= min(len(extracted_words), len(user_words)) * 0.7:
            name_match = True

    validation_results["matches"]["name"] = name_match

    # Compare DOB (allowing for different formats)
    extracted_dob = aadhaar_data.dob
    if extracted_dob and user_dob:
        # Normalize date formats (handle DD/MM/YYYY, DD-MM-YYYY, etc.)
        extracted_dob_normalized = extracted_dob.replace('-', '/').replace('.', '/')
        user_dob_normalized = user_dob.replace('-', '/').replace('.', '/')

        # Check if dates match
        validation_results["matches"]["dob"] = (extracted_dob_normalized == user_dob_normalized)

    # Compare state (case-insensitive and fuzzy matching)
    extracted_state = aadhaar_data.state.lower() if aadhaar_data.state else ""
    user_state_lower = user_state.lower()

    # Direct match
    state_match = (extracted_state == user_state_lower)

    # If not direct match, check if one contains the other
    if not state_match and extracted_state and user_state_lower:
        state_match = (extracted_state in user_state_lower or user_state_lower in extracted_state)

        # If still no match, check for common abbreviations
        if not state_match:
            # Common state abbreviations
            state_abbr = {
                "tn": "tamil nadu",
                "ap": "andhra pradesh",
                "ts": "telangana",
                "ka": "karnataka",
                "mh": "maharashtra",
                "up": "uttar pradesh",
                "wb": "west bengal",
                "dl": "delhi",
                "hr": "haryana",
                "pb": "punjab",
                "rj": "rajasthan",
                "mp": "madhya pradesh",
                "gj": "gujarat",
                "or": "odisha",
                "jk": "jammu and kashmir"
            }

            # Check if user entered an abbreviation
            if user_state_lower in state_abbr and state_abbr[user_state_lower] == extracted_state:
                state_match = True
            # Check if user entered full name for an abbreviation
            elif extracted_state in state_abbr and state_abbr[extracted_state] == user_state_lower:
                state_match = True

    validation_results["matches"]["state"] = state_match

    # Compare Aadhaar number (ignoring spaces and formatting)
    extracted_aadhaar = ''.join(aadhaar_data.aadhaar_number.split()) if aadhaar_data.aadhaar_number else ""
    user_aadhaar_normalized = ''.join(user_aadhaar.split())

    validation_results["matches"]["aadhaar_number"] = (extracted_aadhaar == user_aadhaar_normalized)

    # Overall validation result - all critical fields must match
    validation_results["is_valid"] = (
        validation_results["matches"]["name"] and
        validation_results["matches"]["dob"] and
        validation_results["matches"]["aadhaar_number"]
    )

    # Add confidence score
    match_count = sum(1 for match in validation_results["matches"].values() if match)
    total_fields = len(validation_results["matches"])
    validation_results["confidence_score"] = round((match_count / total_fields) * 100, 2)

    return jsonify(validation_results)

# Helper function to extract state from address pattern
def extract_state_from_pattern(text):
    """
    Extract state from common address patterns in Aadhaar cards
    Example: "..., District Name, State Name - 123456"
    """
    # List of Indian states
    indian_states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
        "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
        "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
    ]

    # Common patterns for state in address
    patterns = [
        r',\s*([A-Za-z\s]+)\s*-\s*\d{6}',  # ", State Name - 123456"
        r',\s*([A-Za-z\s]+),\s*\d{6}',     # ", State Name, 123456"
        r',\s*([A-Za-z\s]+)$',             # ", State Name" at end of text
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            potential_state = match.strip()
            # Check if the extracted text matches or contains a state name
            for state in indian_states:
                if state.lower() == potential_state.lower() or state.lower() in potential_state.lower():
                    return state

    return None

# Helper function to process images for OCR
def process_image_for_ocr(image):
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Resize if needed
    height, width = img_cv.shape[:2]
    if max(height, width) > 3000:
        scale_factor = 3000 / max(height, width)
        img_cv = cv2.resize(img_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    elif min(height, width) < 800:
        scale_factor = 800 / min(height, width)
        img_cv = cv2.resize(img_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Apply multiple preprocessing techniques and combine results
    text = ""

    # Method 1: Basic grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    text += pytesseract.image_to_string(gray, config='--oem 3 --psm 6', lang='eng+tam')

    # Method 2: Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text += " " + pytesseract.image_to_string(thresh, config='--oem 3 --psm 6', lang='eng+tam')

    # Method 3: Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    text += " " + pytesseract.image_to_string(adaptive_thresh, config='--oem 3 --psm 6', lang='eng+tam')

    # Method 4: Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    text += " " + pytesseract.image_to_string(denoised, config='--oem 3 --psm 6', lang='eng+tam')

    return text

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
