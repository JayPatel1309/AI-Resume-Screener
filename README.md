# AI Resume Screener

AI Resume Screener is an intelligent web application designed to help recruiters and hiring managers efficiently filter and rank resumes based on job descriptions. It uses natural language processing (NLP) and machine learning to analyze applicant resumes and provide a match score.

## Project Structure

- `backend/`: FastAPI application that provides the API endpoints for resume text extraction, processing, and scoring.
- `frontend/`: Vanilla HTML, CSS, and JavaScript user interface for uploading resumes and job descriptions, and viewing the screening results.
- `data/`: Directory for storing datasets used for training or testing the machine learning models.
- `models/`: Directory containing pre-trained machine learning models and vectorizers.
- `notebooks/`: Jupyter notebooks used for data exploration, model training, and experimentation.

## Features

- **Resume Upload**: Upload PDF resumes directly to the system.
- **Job Description Matching**: Input a target job description for comparison.
- **AI-Powered Scoring**: Extracts text from PDFs and calculates relevance scores using cosine similarity and machine learning models (scikit-learn).
- **Interactive Dashboard**: View parsed resumes alongside their calculated match scores.

## Technologies Used

### Backend
- **Python**
- **FastAPI**: High-performance web framework for the API.
- **scikit-learn**: For feature extraction (CountVectorizer) and text similarity calculations.
- **PyPDF2**: For parsing and extracting text from PDF resumes.
- **pandas**: For data manipulation.
- **joblib**: For loading pre-trained models.

### Frontend
- **HTML5 / CSS3 / JavaScript**: Lightweight, responsive frontend interface.

## Setup and Installation

### Prerequisites
- Python 3.8+
- Modern Web Browser

### 1. Backend Setup
1. Navigate to the project root directory.
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required Python packages:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas PyPDF2 joblib python-multipart
   ```
4. Start the FastAPI development server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.

### 2. Frontend Setup
1. Open the `frontend/index.html` file in your preferred web browser. Alternatively, you can use a local development server (like the "Live Server" extension in VS Code) to serve the files.
2. Ensure the backend API is running so the frontend can successfully communicate with it.

## Usage
1. Launch both the backend API and the frontend application.
2. In the web interface, paste the desired job description.
3. Upload the applicant resumes (PDF format).
4. Click the screening button to process the resumes.
5. Review the results, which will display the candidates ranked by their matching score.