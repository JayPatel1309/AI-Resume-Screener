from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import joblib
import pandas as pd
import re
import io
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="AI Resume Screener API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# No Transcript Models
model_no_trans = joblib.load(r"P:\AI-Resume-Screener\models\ensemble_model.pkl")
scaler_no_trans = joblib.load(r"P:\AI-Resume-Screener\models\scaler.pkl")

# Transcript Models
model_trans = joblib.load(r"P:\AI-Resume-Screener\models\ensemble_model_transcript.pkl")
scaler_trans = joblib.load(r"P:\AI-Resume-Screener\models\scaler_transcript.pkl")
vectorizer_trans = joblib.load(r"P:\AI-Resume-Screener\models\word_vectorizer_transcript.pkl")

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

def clean_text(text):
    text = re.sub('http\\S+\\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\\S+', '', text)
    text = re.sub('@\\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\\x00-\\x7f]',r' ', text) 
    text = re.sub('\\s+', ' ', text)
    return text.lower()

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    union = set1.union(set2)
    return len(set1.intersection(set2)) / len(union) if len(union) != 0 else 0

# 2. THE SMART API ENDPOINT
@app.post("/predict")
async def predict_resume(
    resume: UploadFile = File(...), 
    jd: UploadFile = File(...),
    transcript: Optional[UploadFile] = File(None) # <-- THIS IS OPTIONAL NOW
):
    
    resume_text = clean_text(extract_text_from_pdf(await resume.read()))
    jd_text = clean_text(extract_text_from_pdf(await jd.read()))
    
    resume_len = len(resume_text.split())
    jd_len = len(jd_text.split())
    length_diff = abs(resume_len - jd_len)
    
    res_words = set(resume_text.split())
    jd_words = set(jd_text.split())
    overlap_res_jd = len(res_words.intersection(jd_words)) / len(jd_words) if len(jd_words) > 0 else 0
    jac_res_jd = jaccard_similarity(resume_text, jd_text)
    if transcript is not None:
        trans_text = clean_text(extract_text_from_pdf(await transcript.read()))
        trans_len = len(trans_text.split())
        
        # Vectorize using the transcript vectorizer
        res_vec = vectorizer_trans.transform([resume_text])
        jd_vec = vectorizer_trans.transform([jd_text])
        trans_vec = vectorizer_trans.transform([trans_text])
        
        cos_jd_res = cosine_similarity(jd_vec, res_vec)[0][0]
        cos_jd_trans = cosine_similarity(jd_vec, trans_vec)[0][0]
        cos_res_trans = cosine_similarity(res_vec, trans_vec)[0][0]
        
        jac_res_trans = jaccard_similarity(resume_text, trans_text)
        jac_jd_trans = jaccard_similarity(jd_text, trans_text)
        
        trans_words = set(trans_text.split())
        overlap_trans_jd = len(trans_words.intersection(jd_words)) / len(jd_words) if len(jd_words) > 0 else 0
        
        # Dynamically create DataFrame with all expected columns set to 0.0 (float)
        expected_cols = scaler_trans.feature_names_in_
        features = pd.DataFrame(0.0, index=[0], columns=expected_cols)
        
        # Fill in the features we calculated
        features.at[0, 'resume_length'] = resume_len
        features.at[0, 'jd_length'] = jd_len
        features.at[0, 'keyword_overlap_ratio_resume_jd'] = overlap_res_jd
        features.at[0, 'jd_resume_similarity'] = cos_jd_res
        features.at[0, 'jd_transcript_similarity'] = cos_jd_trans
        features.at[0, 'resume_transcript_similarity'] = cos_res_trans
        features.at[0, 'Transcript_length'] = trans_len
        features.at[0, 'keyword_overlap_ratio_transcript_jd'] = overlap_trans_jd
        features.at[0, 'jaccard_resume_jd'] = jac_res_jd
        features.at[0, 'jaccard_resume_transcript'] = jac_res_trans
        features.at[0, 'jaccard_jd_transcript'] = jac_jd_trans
        
        scaled_features = scaler_trans.transform(features)
        prediction = model_trans.predict(scaled_features)[0]
        model_used = "Transcript Model"
    else:
        vectorizer_no_trans = joblib.load(r"P:\AI-Resume-Screener\models\tfidf_vectorizer.pkl")
        
        res_vec = vectorizer_no_trans.transform([resume_text])
        jd_vec = vectorizer_no_trans.transform([jd_text])
        cos_jd_res = cosine_similarity(jd_vec, res_vec)[0][0]
    
        # Dynamically create DataFrame with all expected columns set to 0.0 (float)
        expected_cols = scaler_no_trans.feature_names_in_
        features = pd.DataFrame(0.0, index=[0], columns=expected_cols)
        
        # Fill in the features we calculated
        features.at[0, 'resume_length'] = resume_len
        features.at[0, 'jd_length'] = jd_len
        features.at[0, 'keyword_overlap_ratio_resume_jd'] = overlap_res_jd
        features.at[0, 'jd_resume_similarity'] = cos_jd_res
        features.at[0, 'jaccard_resume_jd'] = jac_res_jd
        
        scaled_features = scaler_no_trans.transform(features)
        prediction = model_no_trans.predict(scaled_features)[0]
        model_used = "No Transcript Model"
        
    decision = "select" if prediction == 1 else "reject"
    
    return {
        "decision": decision,
        "model_used": model_used
    }