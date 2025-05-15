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
import difflib  # For fuzzy string matching

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'tiff'}

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

# Extract text from image using Tesseract OCR with enhanced preprocessing
def extract_text_from_image(image: Image.Image) -> str:
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process with OpenCV for better OCR results
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply multiple preprocessing techniques and combine results for better accuracy

    # Method 1: Basic thresholding
    _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 2: Adaptive thresholding
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Method 3: Bilateral filtering for noise reduction while preserving edges
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh3 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 4: Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, thresh4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Method 5: Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, thresh5 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try different PSM modes for better text detection
    psm_modes = [6, 4, 3]  # 6=Block of text, 4=Single column, 3=Auto

    all_text = ""

    # Try with Tamil language support first
    try:
        for psm in psm_modes:
            custom_config = f'--oem 3 --psm {psm}'

            # Process with all preprocessing methods
            for thresh in [thresh1, thresh2, thresh3, thresh4, thresh5]:
                text = pytesseract.image_to_string(thresh, config=custom_config, lang='eng+tam')
                all_text += " " + text

        return all_text
    except:
        # Fallback to English only
        all_text = ""
        for psm in psm_modes:
            custom_config = f'--oem 3 --psm {psm}'

            # Process with all preprocessing methods
            for thresh in [thresh1, thresh2, thresh3, thresh4, thresh5]:
                text = pytesseract.image_to_string(thresh, config=custom_config, lang='eng')
                all_text += " " + text

        return all_text

# Extract text from PDF using PyMuPDF with enhanced processing
def extract_text_from_pdf(pdf_bytes: bytes, password: str = None) -> str:
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.needs_pass and password:
            success = doc.authenticate(password)
            if not success:
                return "ERROR: Invalid password for PDF"

        # Extract text from each page using multiple methods
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Method 1: Direct text extraction
            page_text = page.get_text("text")
            text += page_text + "\n"

            # Method 2: Extract text with different parameters
            page_text_blocks = page.get_text("blocks")
            for block in page_text_blocks:
                if isinstance(block, tuple) and len(block) > 4:
                    text += block[4] + "\n"

            # Method 3: Always perform image-based extraction for better accuracy
            # This helps with scanned documents or when text is embedded in images
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Apply OCR to the image
            img_text = extract_text_from_image(img)
            text += img_text + "\n"

            # Method 4: Extract text from specific regions if needed
            # This can be useful for targeting specific areas where names typically appear
            # For Aadhaar cards, names are usually in the top portion
            top_region = page.rect_for_bbox((0, 0, page.rect.width, page.rect.height * 0.3))
            pix_top = page.get_pixmap(clip=top_region, matrix=fitz.Matrix(2, 2))
            img_top = Image.frombytes("RGB", [pix_top.width, pix_top.height], pix_top.samples)
            top_text = extract_text_from_image(img_top)
            text += top_text + "\n"

        return text
    except Exception as e:
        return f"ERROR: {str(e)}"

# Define common Indian names for better name recognition
COMMON_INDIAN_FIRST_NAMES = [
    "Aarav", "Aditya", "Akshay", "Amit", "Ananya", "Anil", "Arjun", "Aryan",
    "Deepak", "Divya", "Gaurav", "Ishaan", "Karan", "Kavya", "Krishna", "Kunal",
    "Manish", "Meera", "Mohan", "Neha", "Nikhil", "Nisha", "Pooja", "Priya",
    "Rahul", "Raj", "Rajesh", "Ravi", "Rohit", "Sanjay", "Sarika", "Shivani",
    "Sneha", "Sonia", "Sunil", "Suresh", "Tanvi", "Varun", "Vijay", "Vikram"
]

# Enhanced name extraction with scoring system
def extract_name_from_text(lines, full_text=""):
    # Combine lines and full text for comprehensive analysis
    if not full_text:
        full_text = " ".join(lines)

    # List of unwanted phrases that should not be identified as names
    unwanted_phrases = [
        "Digitally signed by", "DS Unique", "Identification Authority", "Authority of India",
        "Government of India", "Signature Not Verified", "Aadhaar", "UIDAI", "Unique Identification",
        "Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana", "Maharashtra",
        "Gujarat", "Rajasthan", "Madhya Pradesh", "Uttar Pradesh", "Bihar", "West Bengal",
        "Odisha", "Punjab", "Haryana", "Jharkhand", "Chhattisgarh", "Uttarakhand",
        "Himachal Pradesh", "Jammu and Kashmir", "Assam", "Manipur", "Meghalaya",
        "Nagaland", "Tripura", "Arunachal Pradesh", "Mizoram", "Sikkim", "Goa",
        "Address", "Gender", "DOB", "Date of Birth", "Male", "Female", "District", "State",
        "Pincode", "Phone", "VID", "Year of Birth", "Enrollment", "Enrolment", "Registered",
        "Resident", "Citizen", "India", "Document", "Certificate", "Card", "Number"
    ]

    # Common words that should not be part of a name
    common_words = [
        "the", "and", "for", "this", "that", "with", "from", "your", "have", "has",
        "not", "yes", "no", "address", "gender", "dob", "birth", "male", "female",
        "district", "state", "pincode", "phone", "number", "card", "aadhaar"
    ]

    # Initialize candidates with scores
    name_candidates = []

    # Method 1: Look for explicit name labels
    name_patterns = [
        r'Name[:\s]+([A-Za-z\s\'-]+)',
        r'(?:^|\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})(?=\s*\n)',  # Capitalized name at start of line
        r'(?i)name of person[:\s]+([A-Za-z\s\'-]+)',
        r'(?i)holder[\'s]? name[:\s]+([A-Za-z\s\'-]+)',
        r'(?i)applicant[\'s]? name[:\s]+([A-Za-z\s\'-]+)'
    ]

    for pattern in name_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.group(1).strip()
            # Clean up the name
            name = re.sub(r'\s+', ' ', name)
            name = name.strip()

            # Skip if it contains unwanted phrases
            if any(phrase.lower() in name.lower() for phrase in unwanted_phrases):
                continue

            # Skip if it's too short or too long
            if len(name) < 3 or len(name) > 40:
                continue

            # Skip if it contains common words that shouldn't be in names
            if any(f" {word} " in f" {name.lower()} " for word in common_words):
                continue

            # Score: Higher for explicit name labels (10 points)
            score = 10

            # Bonus if name contains common Indian first names
            if any(first_name.lower() in name.lower() for first_name in COMMON_INDIAN_FIRST_NAMES):
                score += 3

            # Bonus for capitalized words (proper names)
            if re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)+$', name):
                score += 2

            name_candidates.append((name, score, "explicit_label"))

    # Method 2: Look for name before guardian indicators
    guardian_patterns = [
        r'([A-Za-z\s\'-]+)\s+(?:S/O|C/O|W/O|D/O|Son of|Daughter of|Wife of|Care of)[.:]?\s+([A-Za-z\s\'-]+)',
        r'([A-Za-z\s\'-]+)\s+(?:S/O|C/O|W/O|D/O)[.:]?\s+([A-Za-z\s\'-]+)'
    ]

    for pattern in guardian_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            name = match.group(1).strip()
            guardian = match.group(2).strip()

            # Clean up the name
            name = re.sub(r'\s+', ' ', name)
            name = name.strip()

            # Skip if it contains unwanted phrases
            if any(phrase.lower() in name.lower() for phrase in unwanted_phrases):
                continue

            # Skip if it's too short or too long
            if len(name) < 3 or len(name) > 40:
                continue

            # Skip if it contains common words that shouldn't be in names
            if any(f" {word} " in f" {name.lower()} " for word in common_words):
                continue

            # Score: Good for guardian-based names (8 points)
            score = 8

            # Bonus if name contains common Indian first names
            if any(first_name.lower() in name.lower() for first_name in COMMON_INDIAN_FIRST_NAMES):
                score += 3

            # Bonus for capitalized words (proper names)
            if re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)+$', name):
                score += 2

            name_candidates.append((name, score, "before_guardian"))

            # Also add guardian name as a candidate with lower score
            if guardian and len(guardian) > 3:
                guardian = re.sub(r'\s+', ' ', guardian)
                score_guardian = 5  # Lower score for guardian names

                # Bonus if guardian name contains common Indian first names
                if any(first_name.lower() in guardian.lower() for first_name in COMMON_INDIAN_FIRST_NAMES):
                    score_guardian += 2

                name_candidates.append((guardian, score_guardian, "guardian"))

    # Method 3: Look for standalone names (capitalized words at beginning of lines)
    for line in lines:
        line = line.strip()

        # Skip short lines or lines with unwanted content
        if len(line) < 3 or any(phrase.lower() in line.lower() for phrase in unwanted_phrases):
            continue

        # Look for capitalized words that might be names
        if re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}$', line) and not re.search(r'\d', line):
            name = line

            # Skip if it contains common words that shouldn't be in names
            if any(f" {word} " in f" {name.lower()} " for word in common_words):
                continue

            # Score: Medium for standalone capitalized names (7 points)
            score = 7

            # Bonus if name contains common Indian first names
            if any(first_name.lower() in name.lower() for first_name in COMMON_INDIAN_FIRST_NAMES):
                score += 3

            # Bonus for multiple words (more likely to be a full name)
            words = name.split()
            if len(words) >= 2:
                score += len(words) - 1

            name_candidates.append((name, score, "capitalized_line"))

    # Method 4: Look for names in the first few lines (often contain the name)
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        line = line.strip()

        # Skip short lines or lines with unwanted content
        if len(line) < 3 or any(phrase.lower() in line.lower() for phrase in unwanted_phrases):
            continue

        # Skip lines with common non-name indicators
        if re.search(r'(?:address|gender|dob|date of birth|male|female|district|state|pincode)', line, re.IGNORECASE):
            continue

        # Look for potential name patterns
        if re.match(r'^[A-Za-z\s\'-]+$', line) and len(line.split()) >= 2 and len(line.split()) <= 5:
            name = line

            # Skip if it contains common words that shouldn't be in names
            if any(f" {word} " in f" {name.lower()} " for word in common_words):
                continue

            # Score: Lower for names from first lines (6 points)
            score = 6

            # Bonus if name contains common Indian first names
            if any(first_name.lower() in name.lower() for first_name in COMMON_INDIAN_FIRST_NAMES):
                score += 3

            # Bonus for position (earlier lines more likely to have name)
            score += max(0, 5 - i)  # Bonus decreases with line number

            name_candidates.append((name, score, "first_lines"))

    # Method 5: Look for Tamil name followed by English name pattern
    tamil_name_match = re.search(r'([\u0B80-\u0BFF\s]+)\n([A-Za-z\s\'-]+)', full_text)
    if tamil_name_match:
        name = tamil_name_match.group(2).strip()

        # Clean up the name
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()

        # Skip if it contains unwanted phrases
        if not any(phrase.lower() in name.lower() for phrase in unwanted_phrases):
            # Score: High for Tamil-English pattern (9 points)
            score = 9

            # Bonus if name contains common Indian first names
            if any(first_name.lower() in name.lower() for first_name in COMMON_INDIAN_FIRST_NAMES):
                score += 3

            name_candidates.append((name, score, "tamil_english_pattern"))

    # If we have candidates, sort by score and return the highest
    if name_candidates:
        # Sort by score (descending)
        name_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return the highest scoring candidate
        return name_candidates[0][0]

    # If no candidates found, return empty string
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

    # Tamil Name
    tamil_name_match = re.search(r'([\u0B80-\u0BFF\s]+)', text)
    if tamil_name_match:
        data.name_tamil = tamil_name_match.group(1).strip()

    # Extract name using our enhanced algorithm
    # Pass both lines and full text for comprehensive analysis
    extracted_name = extract_name_from_text(lines, text)
    if extracted_name:
        data.name = extracted_name

    # If name is still not found, try additional methods
    if not data.name:
        # Try fuzzy matching with common patterns
        name_patterns = [
            r'(?:^|\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})(?=\s*\n)',
            r'Name[:\s]+([A-Za-z\s\'-]+)',
            r'([A-Za-z\s\'-]+)\s+(?:S/O|C/O|W/O|D/O)'
        ]

        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                potential_name = match.group(1).strip()
                # Clean up the name
                potential_name = re.sub(r'\s+', ' ', potential_name)

                # Check if it's a valid name (not too short, not a state name)
                if len(potential_name) >= 3 and not any(state.lower() in potential_name.lower()
                                                      for state in ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh"]):
                    data.name = potential_name
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
    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))

        file = request.files['file']
        pdf_password = request.form.get('pdf_password', '')

        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))

        # Print debugging information
        print(f"Received file: {file.filename}, Content type: {file.content_type}")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print(f"File saved to: {filepath}")

            # Process the uploaded PDF with password if provided
            results = process_aadhaar_card(filepath, pdf_password)

            # Print the results for debugging
            print(f"Extraction results: {results}")

            # Store results in session for display
            session['ocr_results'] = results

            return redirect(url_for('results'))
        else:
            flash('Invalid file type. Please upload a PDF file.')
            print(f"Invalid file type: {file.filename}")
            return redirect(url_for('index'))
    except Exception as e:
        # Log any exceptions that occur
        error_message = f"Error in upload_file: {str(e)}"
        print(error_message)
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('index'))

    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    try:
        # Check if results are in session
        if 'ocr_results' not in session:
            print("Warning: No OCR results found in session")
            return render_template('results.html', results={"error": "No OCR results found. Please upload a file first."})

        # Get results from session
        results = session['ocr_results']

        # Print debugging information
        print(f"Results route - Session data: {results}")

        # Check if results is a dictionary
        if not isinstance(results, dict):
            print(f"Warning: Results is not a dictionary, it's a {type(results)}")
            results = {"error": f"Invalid results format: {type(results)}"}

        # Ensure raw_text is available for debugging
        if 'raw_text' not in results:
            results['raw_text'] = "No raw text available"

        return render_template('results.html', results=results)
    except Exception as e:
        error_message = f"Error in results route: {str(e)}"
        print(error_message)
        return render_template('results.html', results={"error": error_message})

# API endpoint for extraction
@app.route('/extract', methods=['POST'])
def extract_aadhaar():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided. Please upload a PDF or image file."})

        file = request.files['file']
        password = request.form.get('password', None)

        if file.filename == '':
            return jsonify({"error": "No selected file. Please upload a PDF or image file."})

        # Print debugging information
        print(f"API: Received file: {file.filename}, Content type: {file.content_type}")

        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type: {file.filename}. Please upload a PDF or image file (jpg, jpeg, png, tiff)."})

        # Save the file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        file.save(temp_file.name)
        temp_file.close()

        print(f"API: File saved temporarily to: {temp_file.name}")

        # Process the file based on its type
        file_ext = os.path.splitext(file.filename.lower())[1]

        if file_ext == '.pdf':
            # Process PDF file
            with open(temp_file.name, 'rb') as f:
                pdf_bytes = f.read()
            text = extract_text_from_pdf(pdf_bytes, password)
        else:
            # Process image file
            image = Image.open(temp_file.name)
            text = process_image_for_ocr(image)

        # Clean up the temporary file
        os.unlink(temp_file.name)

        # Parse the extracted text
        aadhaar_data = parse_aadhaar_details(text)

        # Add raw text to the response for debugging
        result = aadhaar_data.model_dump()
        result['raw_text'] = text

        # Print formatted JSON to terminal for debugging
        print("\nExtracted Aadhaar Details:")
        print(aadhaar_data.model_dump_json(indent=4))

        return jsonify(result)
    except Exception as e:
        error_message = f"Error in extract_aadhaar: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message, "raw_text": "Error occurred during processing"})

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

    # Compare name using enhanced fuzzy matching
    extracted_name = aadhaar_data.name if aadhaar_data.name else ""

    # Use our fuzzy name matching function with a threshold of 0.7 (70% similarity)
    name_match_score = fuzzy_name_match(extracted_name, user_name, threshold=0.7)

    # Consider it a match if the score is above 0.7 (70% similarity)
    name_match = name_match_score >= 0.7

    # Store the match result
    validation_results["matches"]["name"] = name_match

    # Also store the match score for reference
    validation_results["match_scores"] = {
        "name": round(name_match_score * 100, 2)  # Convert to percentage
    }

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

# Helper function for fuzzy name matching
def fuzzy_name_match(name1, name2, threshold=0.7):
    """
    Compare two names using fuzzy matching to account for OCR errors and variations
    Returns a score between 0 and 1, where 1 is a perfect match
    """
    if not name1 or not name2:
        return 0.0

    # Convert to lowercase for comparison
    name1 = name1.lower()
    name2 = name2.lower()

    # Direct match check
    if name1 == name2:
        return 1.0

    # Check if one is contained in the other
    if name1 in name2 or name2 in name1:
        return 0.9

    # Split into words and check for word-level matches
    words1 = set(name1.split())
    words2 = set(name2.split())

    # Check common words
    common_words = words1.intersection(words2)
    if common_words:
        # Calculate word overlap ratio
        overlap_ratio = len(common_words) / max(len(words1), len(words2))
        if overlap_ratio >= threshold:
            return overlap_ratio

    # Use difflib for sequence matching
    sequence_ratio = difflib.SequenceMatcher(None, name1, name2).ratio()

    # Check individual words for high similarity
    max_word_ratio = 0
    for word1 in words1:
        for word2 in words2:
            if len(word1) > 2 and len(word2) > 2:  # Only compare meaningful words
                word_ratio = difflib.SequenceMatcher(None, word1, word2).ratio()
                max_word_ratio = max(max_word_ratio, word_ratio)

    # Return the best match score
    return max(sequence_ratio, max_word_ratio)

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
