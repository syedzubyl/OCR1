# Aadhaar Card OCR API

This API extracts information from Aadhaar cards in PDF format using OCR (Optical Character Recognition). It's designed to be used with React Native applications.

## Features

- PDF file upload with password support
- OCR processing to extract text from Aadhaar cards
- Tamil language support
- Extraction of specific details:
  - Aadhaar number
  - Name in Tamil
  - Name in English
  - Guardian name
  - Date of Birth
  - Gender
  - Address
  - District
  - State
  - Pincode
  - Phone number
  - VID (Virtual ID)
- RESTful API for integration with mobile apps

## API Documentation

### Extract Aadhaar Card Details

**Endpoint:** `/extract`

**Method:** POST

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type   | Required | Description                      |
| --------- | ------ | -------- | -------------------------------- |
| file      | File   | Yes      | PDF file containing Aadhaar card |
| password  | String | No       | Password for protected PDF files |

**Response:**

```json
{
  "aadhaar_number": "XXXX XXXX XXXX",
  "name_tamil": "தமிழ் பெயர்",
  "name": "English Name",
  "guardian_name": "Guardian Name",
  "dob": "DD/MM/YYYY",
  "gender": "Male/Female",
  "address": "Full address",
  "district": "District name",
  "state": "State name",
  "pincode": "XXXXXX",
  "phone": "XXXXXXXXXX",
  "vid": "XXXX XXXX XXXX XXXX",
  "raw_text": "Full extracted text"
}
```

**Error Response:**

```json
{
  "error": "Error message"
}
```

### Validate User Details Against Aadhaar Card

**Endpoint:** `/validate`

**Method:** POST

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter      | Type   | Required | Description                             |
| -------------- | ------ | -------- | --------------------------------------- |
| file           | File   | Yes      | PDF file containing Aadhaar card        |
| password       | String | No       | Password for protected PDF files        |
| name           | String | Yes      | User-entered name to validate           |
| dob            | String | Yes      | User-entered date of birth (DD/MM/YYYY) |
| state          | String | No       | User-entered state                      |
| aadhaar_number | String | Yes      | User-entered Aadhaar number             |

**Response:**

```json
{
  "is_valid": true,
  "confidence_score": 100.0,
  "matches": {
    "name": true,
    "dob": true,
    "state": true,
    "aadhaar_number": true
  },
  "extracted_details": {
    "aadhaar_number": "XXXX XXXX XXXX",
    "name_tamil": "தமிழ் பெயர்",
    "name": "English Name",
    "guardian_name": "Guardian Name",
    "dob": "DD/MM/YYYY",
    "gender": "Male/Female",
    "address": "Full address",
    "district": "District name",
    "state": "State name",
    "pincode": "XXXXXX",
    "phone": "XXXXXXXXXX",
    "vid": "XXXX XXXX XXXX XXXX"
  },
  "user_details": {
    "name": "User entered name",
    "dob": "DD/MM/YYYY",
    "state": "User entered state",
    "aadhaar_number": "XXXX XXXX XXXX"
  }
}
```

**Error Response:**

```json
{
  "error": "Error message"
}
```

## React Native Integration

Here are sample code snippets for integrating with React Native:

### Basic Extraction

```javascript
import React, { useState } from "react";
import { Button, Text, View, ActivityIndicator } from "react-native";
import DocumentPicker from "react-native-document-picker";

const AadhaarScanner = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const pickAndUploadPDF = async () => {
    try {
      setLoading(true);
      setError(null);

      // Pick a PDF file
      const res = await DocumentPicker.pick({
        type: [DocumentPicker.types.pdf],
      });

      // Create form data
      const formData = new FormData();
      formData.append("file", {
        uri: res[0].uri,
        type: res[0].type,
        name: res[0].name,
      });

      // Add password if needed
      // formData.append('password', 'your-pdf-password');

      // Upload to API
      const response = await fetch(
        "https://your-render-api-url.onrender.com/extract",
        {
          method: "POST",
          body: formData,
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={{ padding: 20 }}>
      <Button title="Scan Aadhaar Card" onPress={pickAndUploadPDF} />

      {loading && <ActivityIndicator size="large" color="#0000ff" />}

      {error && <Text style={{ color: "red" }}>{error}</Text>}

      {result && (
        <View style={{ marginTop: 20 }}>
          <Text>Name: {result.name}</Text>
          <Text>Aadhaar: {result.aadhaar_number}</Text>
          <Text>DOB: {result.dob}</Text>
          <Text>Address: {result.address}</Text>
          {/* Display other fields as needed */}
        </View>
      )}
    </View>
  );
};

export default AadhaarScanner;
```

### Validation with User Details

```javascript
import React, { useState } from "react";
import {
  Button,
  Text,
  View,
  ActivityIndicator,
  TextInput,
  StyleSheet,
  ScrollView,
} from "react-native";
import DocumentPicker from "react-native-document-picker";

const AadhaarValidator = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // User input fields
  const [name, setName] = useState("");
  const [dob, setDob] = useState("");
  const [state, setState] = useState("");
  const [aadhaarNumber, setAadhaarNumber] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);

  const pickPDF = async () => {
    try {
      const res = await DocumentPicker.pick({
        type: [DocumentPicker.types.pdf],
      });

      setSelectedFile({
        uri: res[0].uri,
        type: res[0].type,
        name: res[0].name,
      });
    } catch (err) {
      if (!DocumentPicker.isCancel(err)) {
        setError(err.message);
      }
    }
  };

  const validateDetails = async () => {
    // Validate required fields
    if (!name) {
      setError("Name is required");
      return;
    }

    if (!dob) {
      setError("Date of Birth is required");
      return;
    }

    if (!aadhaarNumber) {
      setError("Aadhaar Number is required");
      return;
    }

    if (!selectedFile) {
      setError("Please select an Aadhaar card PDF");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResult(null);

      // Create form data
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("name", name);
      formData.append("dob", dob);
      formData.append("state", state);
      formData.append("aadhaar_number", aadhaarNumber);

      // Add password if needed
      // formData.append('password', 'your-pdf-password');

      // Upload to API for validation
      const response = await fetch(
        "https://your-render-api-url.onrender.com/validate",
        {
          method: "POST",
          body: formData,
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView>
      <View style={styles.container}>
        <Text style={styles.title}>Aadhaar Verification</Text>

        <View style={styles.inputContainer}>
          <Text style={styles.label}>Full Name:</Text>
          <TextInput
            style={styles.input}
            value={name}
            onChangeText={setName}
            placeholder="Enter name as on Aadhaar"
          />
        </View>

        <View style={styles.inputContainer}>
          <Text style={styles.label}>Date of Birth (DD/MM/YYYY):</Text>
          <TextInput
            style={styles.input}
            value={dob}
            onChangeText={setDob}
            placeholder="DD/MM/YYYY"
            keyboardType="numbers-and-punctuation"
          />
        </View>

        <View style={styles.inputContainer}>
          <Text style={styles.label}>State:</Text>
          <TextInput
            style={styles.input}
            value={state}
            onChangeText={setState}
            placeholder="Enter state name"
          />
        </View>

        <View style={styles.inputContainer}>
          <Text style={styles.label}>Aadhaar Number:</Text>
          <TextInput
            style={styles.input}
            value={aadhaarNumber}
            onChangeText={setAadhaarNumber}
            placeholder="XXXX XXXX XXXX"
            keyboardType="number-pad"
          />
        </View>

        <View style={styles.fileSection}>
          <Button title="Select Aadhaar PDF" onPress={pickPDF} />
          {selectedFile && (
            <Text style={styles.fileInfo}>Selected: {selectedFile.name}</Text>
          )}
        </View>

        <Button
          title="Validate Details"
          onPress={validateDetails}
          disabled={loading}
        />

        {loading && (
          <ActivityIndicator
            size="large"
            color="#0000ff"
            style={styles.loader}
          />
        )}

        {error && <Text style={styles.error}>{error}</Text>}

        {result && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>
              Validation Result: {result.is_valid ? "✅ PASSED" : "❌ FAILED"}
            </Text>
            <Text style={styles.confidence}>
              Confidence Score: {result.confidence_score}%
            </Text>

            <Text style={styles.sectionTitle}>Field Matches:</Text>
            {Object.entries(result.matches).map(([field, match]) => (
              <Text key={field} style={styles.matchItem}>
                {field}: {match ? "✅ Match" : "❌ No Match"}
              </Text>
            ))}

            {!result.is_valid && (
              <Text style={styles.warning}>
                Please check your details and try again.
              </Text>
            )}
          </View>
        )}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  title: {
    fontSize: 22,
    fontWeight: "bold",
    marginBottom: 20,
    textAlign: "center",
  },
  inputContainer: {
    marginBottom: 15,
  },
  label: {
    fontSize: 16,
    marginBottom: 5,
    fontWeight: "500",
  },
  input: {
    borderWidth: 1,
    borderColor: "#ddd",
    padding: 10,
    borderRadius: 5,
    fontSize: 16,
  },
  fileSection: {
    marginVertical: 20,
  },
  fileInfo: {
    marginTop: 10,
    color: "green",
  },
  loader: {
    marginTop: 20,
  },
  error: {
    color: "red",
    marginTop: 15,
  },
  resultContainer: {
    marginTop: 20,
    padding: 15,
    backgroundColor: "#f5f5f5",
    borderRadius: 5,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 10,
  },
  confidence: {
    fontSize: 16,
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "500",
    marginTop: 10,
    marginBottom: 5,
  },
  matchItem: {
    fontSize: 14,
    marginLeft: 10,
    marginBottom: 5,
  },
  warning: {
    color: "orange",
    marginTop: 15,
    fontWeight: "500",
  },
});

export default AadhaarValidator;
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add environment variables:
   - `SECRET_KEY`: A random string for session encryption
5. Add the following build-time environment variables:
   - `PYTHON_VERSION`: 3.9.0
   - `TESSERACT_OCR_VERSION`: 4.1.1

## Prerequisites

- Python 3.9+
- Tesseract OCR 4.1.1+
- Poppler (for PDF processing)

## Local Development

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Start the development server:

   ```
   python app.py
   ```

3. Test the API at `http://127.0.0.1:5000/extract`

## Notes

- For best results, ensure the PDF is clear and readable
- The OCR accuracy depends on the quality of the input document
- This API is for demonstration purposes and should be used in accordance with privacy laws regarding Aadhaar data
# OCR
