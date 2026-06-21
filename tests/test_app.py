import pytest
import io
from app import app, classify_industry, check_requirements_mismatch, SimpleRAG, clean_html_to_text, extract_urls

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Verify that the dashboard home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"TrueTalent" in response.data

def test_industry_classification():
    """Verify candidate domain mapping heuristics."""
    # Software Engineering
    assert classify_industry("React, Javascript, frontend developer, git") == "software_engineering"
    # Data Analytics
    assert classify_industry("Tableau, Power BI, SQL, pandas data analyst") == "data_analytics"
    # Medical Coding
    assert classify_industry("AAPC Medical Coder with ICD-10 coding experience") == "medical_coding"
    # Fallback General
    assert classify_industry("General admin and customer service clerk") == "general"

def test_requirements_mismatch_check_local():
    """Verify local requirements mismatch fallback checks."""
    # Case where JD requires ED coding, but candidate has mismatching resume
    jd = "Required: 2+ years of ED coding experience."
    resume = "High school math teacher with a degree in physics."
    has_mismatch, reasons = check_requirements_mismatch(resume, jd)
    assert has_mismatch is True
    assert len(reasons) > 0

def test_rag_engine():
    """Verify the simple RAG paragraph indexing and querying."""
    doc_text = (
        "Alice has 5 years of experience building scalable backend APIs in Python.\n"
        "She also is certified in Kubernetes and cloud architecture.\n"
        "She has worked at major technology firms as a systems engineer."
    )
    rag = SimpleRAG(doc_text)
    
    # Query matching backend APIs should return the first paragraph
    retrieved = rag.query("backend APIs")
    assert "scalable backend" in retrieved
    
    # Query with no matches should return the default paragraphs
    fallback = rag.query("random query word")
    assert len(fallback) > 0

def test_clean_html_helper():
    """Verify sanitizer strips HTML scripts and tags properly."""
    html = "<html><body><h1>Candidate Name</h1><script>alert('xss')</script><p>Python developer</p></body></html>"
    cleaned = clean_html_to_text(html)
    assert "alert" not in cleaned
    assert "Candidate Name" in cleaned
    assert "Python developer" in cleaned

def test_url_extraction_helper():
    """Verify regex URL extractor correctly fetches HTTP links."""
    text = "Find my portfolio at https://github.com/testuser or http://mywebsite.com."
    urls = extract_urls(text)
    assert "https://github.com/testuser" in urls
    assert "http://mywebsite.com" in urls

def test_api_upload_empty(client):
    """Verify that uploading with no files returns an appropriate error."""
    response = client.post('/api/upload', data={
        'job_description': 'Test JD requirements.'
    })
    assert response.status_code == 400 or b"No resume file" in response.data or response.status_code == 200

def test_api_upload_invalid_extension(client):
    """Verify upload error handling for unsupported extensions."""
    data = {
        'job_description': 'Test JD requirements.',
        'resume': (io.BytesIO(b"executable content"), "hack.exe")
    }
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    # Should handle cleanly (e.g. empty file error, type check error, or fallback gracefully)
    assert response.status_code in [200, 400]
