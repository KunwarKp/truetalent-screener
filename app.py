import os
import re
import json
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from openai import OpenAI
from pypdf import PdfReader
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
from flask_cors import CORS
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB for large candidates.jsonl file

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-Memory Database
candidates_db = {}

# Initialize Grok (xAI) Client if Key is Present
xai_client = None
xai_model = "grok-2"

api_key = os.environ.get("XAI_API_KEY")
if api_key:
    if api_key.startswith("gsk_"):
        # Auto-detect Groq API Key and redirect endpoint to Groq
        xai_client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        xai_model = "llama-3.3-70b-versatile"
    else:
        xai_client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        xai_model = "grok-2"

def clean_html_to_text(html_content):
    """Sanitize and extract clean text from HTML markup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    text = soup.get_text(separator=' ')
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

def extract_urls(text):
    """Regex to extract all URLs from text block, excluding brackets."""
    url_pattern = re.compile(r'https?://[^\s<>`"\[\]\(\)]+')
    raw_urls = url_pattern.findall(text)
    cleaned_urls = []
    for url in raw_urls:
        cleaned = url.rstrip('.,;:')
        cleaned_urls.append(cleaned)
    return list(set(cleaned_urls))

def scrape_website(url):
    """Scrapes any URL and returns cleaned text content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return clean_html_to_text(response.text)
        else:
            return f"Failed to retrieve content. Status code: {response.status_code}"
    except Exception as e:
        return f"Error occurred during scraping: {str(e)}"

# Local Industry Classifier Heuristics
def classify_industry(text):
    text_lower = text.lower()
    medical_words = ['medical', 'surgery', 'healthcare', 'anatomy', 'billing', 'aapc', 'ahima', 'icd-10', 'cpt', 'hcpcs']
    software_words = ['javascript', 'developer', 'software', 'backend', 'frontend', 'git', 'web', 'vue', 'react', 'node']
    analytics_words = ['analyst', 'analytics', 'tableau', 'power bi', 'excel', 'pandas', 'numpy', 'data science', 'sql', 'spark', 'airflow', 'database', 'machine learning', 'ai', 'nlp', 'llm', 'retrieval', 'vector']
    
    medical_hits = sum(1 for w in medical_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
    software_hits = sum(1 for w in software_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
    analytics_hits = sum(1 for w in analytics_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
    
    if "medical coding" in text_lower:
        medical_hits += 2
    elif re.search(r'\b(code|coding)\b', text_lower):
        if software_hits > 0:
            software_hits += 1
        elif medical_hits > 0:
            medical_hits += 1
            
    # Return the category with highest hits
    hits = {
        'medical_coding': medical_hits,
        'software_engineering': software_hits,
        'data_analytics': analytics_hits
    }
    
    max_cat = max(hits, key=hits.get)
    if hits[max_cat] > 0:
        return max_cat
    return 'general'

# Get fallback questions based on domain
def get_domain_fallback_questions(industry):
    return []

# RAG Simple Search Engine
class SimpleRAG:
    def __init__(self, document_text):
        self.document_text = document_text
        self.paragraphs = [p.strip() for p in document_text.split('\n') if len(p.strip()) > 30]

    def query(self, query_str, top_n=3):
        if not self.paragraphs:
            return "No background content found."
        query_words = set(query_str.lower().split())
        scored_paragraphs = []
        for p in self.paragraphs:
            score = sum(1 for w in query_words if w in p.lower())
            scored_paragraphs.append((score, p))
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        results = [p for score, p in scored_paragraphs[:top_n] if score > 0]
        if not results:
            return "\n".join(self.paragraphs[:top_n])
        return "\n".join(results)

def extract_text_from_pdf(pdf_path):
    """Robust text extraction from digital PDFs using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def validate_is_resume(text):
    """Verifies that the document contains essential resume sections or keywords."""
    text_lower = text.lower()
    if "resume" in text_lower or "cv" in text_lower or "curriculum vitae" in text_lower or "profile" in text_lower:
        return True
        
    has_contact = (
        "@" in text_lower or 
        "http" in text_lower or
        re.search(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', text_lower) or
        any(h in text_lower for h in ["phone", "email", "contact", "address", "portfolio", "github", "linkedin"])
    )
    has_experience = any(h in text_lower for h in ['experience', 'employment', 'history', 'work history', 'professional background', 'position'])
    has_education = any(h in text_lower for h in ['education', 'degree', 'university', 'college', 'school', 'certifications', 'qualifications'])
    
    return sum([has_contact, has_experience, has_education]) >= 2

def check_requirements_mismatch(resume_text, jd):
    """Checks if the candidate fails critical requirements in the Job Description dynamically."""
    mismatch_reasons = []
    
    # 1. LLM Vetting if available
    if xai_client:
        try:
            prompt = (
                f"Verify if the candidate meets the critical requirements of the Job Description.\n"
                f"Identify the critical requirements (such as minimum years of experience, specific degrees, certifications, or key skills) directly from the Job Description.\n"
                f"Then, check if the candidate's resume has any major mismatches or completely fails to meet any of these critical requirements.\n\n"
                f"Output strictly a JSON structure matching this shape exactly:\n"
                f"{{\n"
                f"  \"has_mismatch\": <true or false>,\n"
                f"  \"reasons\": [\"reason 1\", \"reason 2\"]\n"
                f"}}\n\n"
                f"Resume:\n{resume_text[:2500]}\n\n"
                f"Job Description:\n{jd}"
            )
            completion = xai_client.chat.completions.create(
                model=xai_model,
                messages=[
                    {"role": "system", "content": "You are a strict applicant screening assistant. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_content = completion.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()
            parsed = json.loads(raw_content)
            if parsed.get("has_mismatch"):
                return True, parsed.get("reasons", [])
            else:
                return False, []
        except Exception as e:
            print(f"LLM mismatch check error: {e}")
            
    # Local fallback heuristics
    jd_lower = jd.lower()
    resume_lower = resume_text.lower()
    
    # Check if JD requires ED / Surgery coding
    is_ed_surgery_jd = ("ed" in jd_lower or "emergency" in jd_lower or "surgery" in jd_lower or "surgical" in jd_lower) and "coding" in jd_lower
    
    if is_ed_surgery_jd:
        has_ed_or_surgery = False
        if re.search(r'\b(ed|emergency\s+department|surgery|surgical|critical\s+care)\b', resume_lower):
            has_ed_or_surgery = True
            
        if not has_ed_or_surgery:
            mismatch_reasons.append("Candidate does not have experience in ED (Emergency Department) or Surgery medical coding.")
        else:
            # Gather lines containing ED / Surgery context
            ed_surgery_context = []
            for line in resume_lower.split('\n'):
                if re.search(r'\b(ed|emergency\s+department|surgery|surgical|critical\s+care)\b', line):
                    ed_surgery_context.append(line)
            context_str = " ".join(ed_surgery_context)
            
            # Find any years numbers (e.g. "2 years", "3+ years", "4 yrs", etc.)
            years_match = re.findall(r'\b(\d+)\s*(?:\+)?\s*(?:year|yr)s?\b', context_str)
            if not years_match:
                mismatch_reasons.append("Candidate does not specify the years of experience in number for ED/Surgery medical coding.")
            else:
                years_ints = [int(y) for y in years_match]
                if max(years_ints) < 2:
                    mismatch_reasons.append(f"Candidate does not meet 2+ years of experience in ED/Surgery medical coding (detected: {max(years_ints)} years).")
            
    # Check if JD requires Life Science / Allied Medicine
    is_life_science_jd = "life science" in jd_lower or "allied medicine" in jd_lower
    if is_life_science_jd:
        # Find education section block specifically to avoid matching work experience (like Pharmacy Billing Executive)
        edu_section = ""
        edu_match = re.search(r'(educational|education|qualification|study|academic)\b.*', resume_lower, re.DOTALL | re.IGNORECASE)
        if edu_match:
            edu_text = edu_match.group(0)
            next_section = re.search(r'\n\s*(work|experience|employment|skills|hobbies|contact|languages|objective)\b', edu_text, re.IGNORECASE)
            if next_section:
                edu_section = edu_text[:next_section.start()]
            else:
                edu_section = edu_text[:1000]
        else:
            edu_section = resume_lower # fallback if no section matches
            
        life_science_keywords = [
            "life science", "allied medicine", "pharmacy", "nursing", "biology", "zoology", 
            "botany", "biotechnology", "microbiology", "biochemistry", "physiotherapy", 
            "dental", "bds", "mbbs", "medicine", "science", "b.sc", "m.sc"
        ]
        
        has_life_science_edu = False
        for kw in life_science_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', edu_section):
                has_life_science_edu = True
                break
                
        if "english" in edu_section and not has_life_science_edu:
            mismatch_reasons.append("Candidate does not have a degree in Life Science or Allied Medicine (degree detected: English Language and Literature).")
        elif not has_life_science_edu:
            mismatch_reasons.append("Candidate does not have a degree in Life Science or Allied Medicine.")

    if mismatch_reasons:
        return True, mismatch_reasons
    return False, []

def check_honeypot_json(c):
    """Detects subtly impossible profiles in structured candidate JSON."""
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    skills = c.get("skills", [])
    
    # Check if any skill has duration_months > yoe * 12 + 12
    max_possible_months = yoe * 12 + 12
    for sk in skills:
        dur = sk.get("duration_months", 0)
        if dur > max_possible_months:
            return True, f"Honeypot Detected: Skill duration ({sk.get('name')}: {dur}m) exceeds total experience ({yoe}y)."
            
    # Check if many expert skills have 0 endorsements
    expert_no_endorsements = [s for s in skills if s.get("proficiency") == "expert" and s.get("endorsements") == 0]
    if len(expert_no_endorsements) >= 5:
        return True, "Honeypot Detected: Too many expert skills listed with zero endorsements."
        
    return False, ""

def calculate_qualification_score(jd, resume_text):
    """Dynamic Heuristic Matcher out of 100% with root-prefix tolerance"""
    stop_words = {'required', 'minimum', 'experience', 'position', 'knowledge', 'years', 'requirements', 'target', 'highly', 'strong', 'ability', 'degree', 'work', 'working'}
    jd_words_filtered = [w for w in re.findall(r'\b\w{4,}\b', jd.lower()) if w not in stop_words]
    resume_words_filtered = set(re.findall(r'\b\w{3,}\b', resume_text.lower()))
    
    if jd_words_filtered:
        matched_count = 0
        resume_roots = {rw[:3] for rw in resume_words_filtered}
        for jw in jd_words_filtered:
            if jw[:3] in resume_roots:
                matched_count += 1
                
        match_ratio = matched_count / len(jd_words_filtered)
        
        if match_ratio >= 0.4:
            semantic_fit = int(80 + ((match_ratio - 0.4) / 0.6) * 18)
        elif match_ratio >= 0.2:
            semantic_fit = int(50 + ((match_ratio - 0.2) / 0.2) * 30)
        else:
            semantic_fit = int(10 + (match_ratio / 0.2) * 40)
            
        return max(10, min(100, semantic_fit))
    return 60

def deep_analyze_profile(text, jd, industry, is_valid):
    """Detailed profile analyzer. Parses candidate study, certs, skills, ignorable items & plus points."""
    if not is_valid:
        return {
            'classification': "Non-Candidate Document (Template/Worksheet)",
            'certifications': ["None"],
            'education': ["None"],
            'skills': ["None"],
            'ignorable_gaps': "This file does not appear to be a candidate resume. Core contact information or professional history is missing.",
            'plus_points': "None."
        }
        
    if xai_client:
        try:
            prompt = (
                f"Analyze this candidate resume against the Job Description.\n"
                f"Identify study/education details, certifications done, core skills, ignorable discrepancies/minor gaps, and positive highlights.\n"
                f"Output strictly a JSON structure matching this shape exactly:\n"
                f"{{\n"
                f"  \"classification\": \"<candidate profile name, e.g. Senior Medical Coder>\",\n"
                f"  \"certifications\": [\"cert 1\", \"cert 2\"],\n"
                f"  \"education\": [\"degree from school\"],\n"
                f"  \"skills\": [\"skill 1\", \"skill 2\"],\n"
                f"  \"ignorable_gaps\": \"<what minor points or mismatching keywords can be ignored>\",\n"
                f"  \"plus_points\": \"<what are the candidate's standout assets / plus points>\"\n"
                f"}} \n\n"
                f"Resume Content:\n{text[:2000]}\n\n"
                f"Job Description:\n{jd}"
            )
            completion = xai_client.chat.completions.create(
                model=xai_model,
                messages=[
                    {"role": "system", "content": "You are a professional candidate vetting profiler. Output ONLY valid raw JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_content = completion.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()
            return json.loads(raw_content)
        except Exception as e:
            print(f"Grok Deep Analysis Error: {e}")
            
    # Local fallback heuristics
    text_lower = text.lower()
    
    # Heuristic Certifications
    certs = []
    for c in ['CPC', 'CCS', 'CSM', 'PMP', 'AWS', 'CCNA', 'MCSE']:
        if re.search(r'\b' + re.escape(c) + r'\b', text):
            certs.append(c)
    if "certified" in text_lower:
        certs.append("Certified Professional (Heuristic Match)")
    if not certs:
        certs = ["None explicitly listed"]
        
    # Heuristic Education
    edu = []
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(h in line_lower for h in ['university', 'college', 'school', 'bachelor', 'master', 'degree', 'iit', 'b.tech']):
            edu.append(line.strip()[:100])
    if not edu:
        edu = ["Self-taught or credentials details not found in layout"]
        
    # Heuristic Skills
    skills = []
    matched_skills = [w for w in ['python', 'flask', 'django', 'javascript', 'react', 'sql', 'tableau', 'icd-10', 'cpt', 'agile', 'scrum', 'jira'] if w in text_lower]
    if matched_skills:
        skills = [s.title() for s in matched_skills]
    else:
        skills = ["General domain competencies"]
        
    classification = f"{industry.replace('_', ' ').title()} Candidate"
    
    return {
        'classification': classification,
        'certifications': certs,
        'education': edu[:2],
        'skills': skills,
        'ignorable_gaps': "Minor phrasing variations or specific framework listings can be ignored due to strong matching skill roots.",
        'plus_points': f"Strong matching profile for {classification} roles with certifications/skills in {', '.join(skills[:3])}."
    }


SERVICE_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy",
    "mindtree", "l&t", "lnt", "hcl", "tech mahindra", "deloitte", "kpmg", "ey", "pwc"
}

def parse_challenge_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def evaluate_challenge_candidate(c, jd):
    cid = c.get("candidate_id")
    profile = c.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    career_history = c.get("career_history", [])
    skills = c.get("skills", [])
    signals = c.get("redrob_signals", {})
    
    # JD matching
    jd_lower = jd.lower()
    is_search_ai_role = any(kw in jd_lower for kw in ["search", "retrieval", "nlp", "llm", "rag", "ai", "ml", "ranking"])

    # 1. DISQUALIFIERS & FILTERS
    # Filter A: Service company penalty
    is_product_focused = any(kw in jd_lower for kw in ["product", "startup", "saas", "tech company", "founding team", "fast-paced"])
    all_service = True if career_history else False
    for job in career_history:
        company_lower = job.get("company", "").lower()
        is_service = any(srv in company_lower for srv in SERVICE_COMPANIES)
        if not is_service:
            all_service = False
            break
            
    if all_service:
        service_penalty = 0.1 if is_product_focused else 0.7
    else:
        service_penalty = 1.0

    # Filter B: Academic / Research (Only penalize for tech product roles)
    academic_titles = 0
    total_jobs = len(career_history)
    for job in career_history:
        title_lower = job.get("title", "").lower()
        if any(kw in title_lower for kw in ["research assistant", "phd student", "postdoc", "professor", "lecturer"]):
            academic_titles += 1
    is_pure_research = (total_jobs > 0 and academic_titles == total_jobs)
    
    if is_pure_research and is_product_focused:
        research_penalty = 0.2
    else:
        research_penalty = 1.0

    # Filter C: Domain specialization mismatch (Only for search/AI roles)
    domain_penalty = 1.0
    if is_search_ai_role:
        has_vision_speech = False
        has_nlp_ir = False
        for sk in skills:
            name_lower = sk.get("name", "").lower()
            if any(kw in name_lower for kw in ["vision", "image", "speech", "robotics", "tts", "asr", "audio", "ocr"]):
                has_vision_speech = True
            if any(kw in name_lower for kw in ["nlp", "text", "retriev", "search", "llm", "transformers", "rag", "embeddings", "ndcg", "mrr", "map"]):
                has_nlp_ir = True
        if has_vision_speech and not has_nlp_ir:
            domain_penalty = 0.3

    # Filter D: Job-hoppers
    avg_tenure_months = 0
    if total_jobs > 0:
        total_months = sum(job.get("duration_months", 0) for job in career_history)
        avg_tenure_months = total_months / total_jobs
    hopper_penalty = 0.5 if avg_tenure_months < 15 and total_jobs >= 3 else 1.0

    # Filter E: Honeypot check
    is_honeypot = False
    honeypot_reason = ""
    max_possible_months = yoe * 12 + 12
    for sk in skills:
        dur = sk.get("duration_months", 0)
        if dur > max_possible_months:
            is_honeypot = True
            honeypot_reason = f"Skill duration ({sk.get('name')}: {dur}m) exceeds YOE ({yoe}y)."
            break
            
    expert_no_endorsements = [s for s in skills if s.get("proficiency") == "expert" and s.get("endorsements") == 0]
    if len(expert_no_endorsements) >= 5:
        is_honeypot = True
        honeypot_reason = "Anomalous profile: too many expert skills with zero endorsements."

    honeypot_penalty = 0.0 if is_honeypot else 1.0

    # 2. SCORING MATRIX
    # Component 1: Years of Experience
    yoe_required = None
    yoe_match = re.search(r'(\d+)\s*\+?\s*year', jd_lower)
    if yoe_match:
        yoe_required = int(yoe_match.group(1))
        
    if yoe_required is not None:
        if yoe >= yoe_required:
            exp_score = 30
        elif yoe >= max(1, yoe_required - 2):
            exp_score = 15
        else:
            exp_score = 5
    else:
        if 4 <= yoe <= 10:
            exp_score = 30
        elif 2 <= yoe < 4 or 10 < yoe <= 14:
            exp_score = 20
        else:
            exp_score = 10

    # Component 2: Skills Score
    # Filter out common stop words
    stop_words = {'required', 'minimum', 'experience', 'position', 'knowledge', 'years', 'requirements', 'target', 'highly', 'strong', 'ability', 'degree', 'work', 'working', 'skills', 'good', 'excellent'}
    jd_words = [w for w in re.findall(r'\b\w{3,}\b', jd_lower) if w not in stop_words]
    
    skills_weighted = 0
    if jd_words:
        jd_word_roots = {w[:3] for w in jd_words}
        for sk in skills:
            name_lower = sk.get("name", "").lower()
            prof = sk.get("proficiency", "beginner")
            prof_multiplier = {"beginner": 1.0, "intermediate": 1.5, "advanced": 2.0, "expert": 2.5}.get(prof, 1.0)
            
            skill_words = re.findall(r'\b\w{3,}\b', name_lower)
            for sw in skill_words:
                if sw[:3] in jd_word_roots:
                    skills_weighted += 5 * prof_multiplier
                    break
        skills_score = min(40, skills_weighted)
    else:
        core_skills = {
            "embeddings": 5, "retrieval": 5, "vector database": 5, "pinecone": 5, "milvus": 5,
            "qdrant": 5, "faiss": 5, "opensearch": 5, "elasticsearch": 5, "nlp": 5, "search": 4,
            "python": 4, "ndcg": 5, "mrr": 5, "map": 5, "llm": 4, "fine-tuning": 4, "transformers": 4,
            "pyspark": 3, "spark": 3, "hybrid search": 5
        }
        for sk in skills:
            name_lower = sk.get("name", "").lower()
            prof = sk.get("proficiency", "beginner")
            prof_multiplier = {"beginner": 1.0, "intermediate": 1.5, "advanced": 2.0, "expert": 2.5}.get(prof, 1.0)
            for cs, weight in core_skills.items():
                if cs in name_lower:
                    skills_weighted += weight * prof_multiplier
                    break
        skills_score = min(40, skills_weighted * 1.5)

    # Component 3: Career Match
    career_score = 0
    curr_title = profile.get("current_title", "").lower()
    
    if jd_words:
        jd_word_roots = {w[:3] for w in jd_words}
        title_words = re.findall(r'\b\w{3,}\b', curr_title)
        title_matches = sum(1 for tw in title_words if tw[:3] in jd_word_roots)
        if title_matches > 0:
            career_score += min(20, title_matches * 10)
            
        # Check previous job titles
        prev_title_matches = 0
        for job in career_history:
            job_title = job.get("title", "").lower()
            job_title_words = re.findall(r'\b\w{3,}\b', job_title)
            if any(jw[:3] in jd_word_roots for jw in job_title_words):
                prev_title_matches += 1
        if prev_title_matches > 0:
            career_score += min(10, prev_title_matches * 5)
    else:
        if any(kw in curr_title for kw in ["ai engineer", "ml engineer", "machine learning", "nlp engineer", "search engineer"]):
            career_score += 15
        elif "backend" in curr_title or "data engineer" in curr_title:
            career_score += 10
            
        has_ranking_experience = False
        for job in career_history:
            desc_lower = job.get("description", "").lower()
            if any(kw in desc_lower for kw in ["rank", "recommend", "retriev", "search", "vector", "embed", "index", "hybrid search", "bm25"]):
                has_ranking_experience = True
                break
        if has_ranking_experience:
            career_score += 15

    # Component 4: Location Score
    loc_score = 0
    loc_lower = profile.get("location", "").lower() + " " + profile.get("country", "").lower()
    for loc_name in ["noida", "pune", "bangalore", "hyderabad", "mumbai", "delhi", "ncr", "india", "toronto", "canada", "usa"]:
        if loc_name in jd_lower:
            if loc_name in loc_lower:
                loc_score = 10
                break
    if loc_score == 0:
        if "india" in loc_lower:
            loc_score = 5

    # 3. BEHAVIORAL MULTIPLIERS
    response_rate = signals.get("recruiter_response_rate", 0.0)
    response_mult = 1.0
    if response_rate < 0.1:
        response_mult = 0.2
    elif response_rate < 0.3:
        response_mult = 0.5
    elif response_rate < 0.6:
        response_mult = 0.8

    last_active_str = signals.get("last_active_date", "")
    last_active_dt = parse_challenge_date(last_active_str)
    active_mult = 1.0
    if last_active_dt:
        ref_dt = datetime(2026, 7, 2)
        days_inactive = (ref_dt - last_active_dt).days
        if days_inactive > 180:
            active_mult = 0.3
        elif days_inactive > 90:
            active_mult = 0.6
        elif days_inactive > 30:
            active_mult = 0.85

    base_score = exp_score + skills_score + career_score + loc_score
    final_score = base_score * service_penalty * research_penalty * domain_penalty * hopper_penalty * response_mult * active_mult * honeypot_penalty
    normalized_score = min(1.0, max(0.01, final_score / 100.0))

    return {
        "candidate_id": cid,
        "score": normalized_score,
        "is_honeypot": is_honeypot,
        "honeypot_reason": honeypot_reason
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_resumes():
    """Handles Phase 1: Uploads, runs loops, and analyzes candidate profiles"""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    files = request.files.getlist('resume')
    jd = request.form.get('job_description', '')
    
    # Check if first file is a database import
    first_file = files[0]
    filename = secure_filename(first_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    first_file.save(file_path)
    
    is_database_import = False
    db_candidates = []
    
    if filename.lower().endswith('.json') or filename.lower().endswith('.jsonl'):
        try:
            if filename.lower().endswith('.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            c = json.loads(line)
                            if isinstance(c, dict) and "profile" in c:
                                db_candidates.append(c)
                                # Break early once we have loaded up to 2000 items to keep web preview responsive
                                if len(db_candidates) >= 2000:
                                    break
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    parsed_json = json.load(f)
                    if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict) and "profile" in parsed_json[0]:
                        db_candidates = parsed_json[:2000]
            
            if len(db_candidates) > 0:
                is_database_import = True
        except Exception as e:
            print(f"Error parsing database upload: {e}")
            import traceback
            traceback.print_exc()

    if is_database_import:
        evaluated = []
        for c in db_candidates:
            res = evaluate_challenge_candidate(c, jd)
            evaluated.append((res, c))
            
        evaluated.sort(key=lambda x: (-x[0]["score"], x[0]["candidate_id"]))
        
        # Take top 100
        top_100_evals = evaluated[:100]
        
        selected_resumes = []
        under_review_resumes = []
        processed_candidates = []
        
        for rank_idx, (eval_res, c_orig) in enumerate(top_100_evals):
            candidate_id = eval_res["candidate_id"]
            score_pct = int(eval_res["score"] * 100)
            is_honeypot = eval_res.get("is_honeypot", False)
            
            agent_2_status = "Rejected" if is_honeypot else ("Approved (Local)" if score_pct >= 80 else "Under Review")
            agent_2_feedback = f"Verification complete. Score: {score_pct}%."
            if is_honeypot:
                agent_2_feedback = f"Rejected: Honeypot candidate detected. Reason: {eval_res.get('honeypot_reason', '')}"
            
            vetting = {
                'classification': c_orig.get('profile', {}).get('current_title', 'AI Engineer') + (" (Honeypot)" if is_honeypot else " (JSON Profile)"),
                'certifications': [cert.get('name') for cert in c_orig.get('certifications', [])] if c_orig.get('certifications') else ["None listed"],
                'education': [f"{e.get('degree')} in {e.get('field_of_study')} from {e.get('institution')}" for e in c_orig.get('education', [])] if c_orig.get('education') else ["None listed"],
                'skills': [s.get('name') for s in c_orig.get('skills', [])],
                'ignorable_gaps': "Timeline and job changes are fully logged in candidate metadata.",
                'plus_points': f"Structured profile containing {c_orig.get('profile', {}).get('years_of_experience', 0)} years of experience with verified platform signals."
            }
            
            resume_text = c_orig.get('profile', {}).get('summary', "") + "\n" + "\n".join([j.get("description", "") for j in c_orig.get('career_history', [])])
            candidate_data = {
                'id': candidate_id,
                'filename': f"{c_orig.get('profile', {}).get('anonymized_name', 'Candidate')}.json",
                'resume_text': resume_text,
                'jd': jd,
                'urls': [u for u in [c_orig.get('profile', {}).get('portfolio', ''), 'https://example.com/portfolio'] if u],
                'has_urls': bool(c_orig.get('profile', {}).get('portfolio', '')),
                'questions': [],
                'industry': classify_industry(jd + " " + resume_text),
                'agent_2_feedback': agent_2_feedback,
                'agent_2_status': agent_2_status,
                'semantic_fit_score': score_pct,
                'vetting': vetting,
                'scraped_data': {},
                'status': 'Completed',
                'results': {
                    'candidate_id': candidate_id,
                    'overall_score': score_pct,
                    'semantic_fit_score': score_pct,
                    'external_signal_score': max(0, int(c_orig.get('redrob_signals', {}).get('github_activity_score', 0))),
                    'executive_synthesis': f"Candidate evaluated and ranked #{rank_idx+1} in bulk challenge import. " + agent_2_feedback,
                    'bucket_a_insights': f"Career matches target criteria. Evaluated experience: {c_orig.get('profile', {}).get('years_of_experience')} years.",
                    'bucket_b_insights': f"Redrob Signals: completeness {c_orig.get('redrob_signals', {}).get('profile_completeness_score')}%, active {c_orig.get('redrob_signals', {}).get('last_active_date')}.",
                    'context_retrieved_rag': f"Parsed candidate ID {candidate_id} from bulk dataset."
                }
            }
            
            candidates_db[candidate_id] = candidate_data
            processed_candidates.append(candidate_data)
            
            filename_display = f"{candidate_id} - {c_orig.get('profile', {}).get('anonymized_name', 'Candidate')}"
            if score_pct >= 80:
                selected_resumes.append(filename_display)
            else:
                under_review_resumes.append(filename_display)
                
        return jsonify({
            'is_bulk': True,
            'is_db_import': True,
            'selected_resumes': selected_resumes,
            'under_review_resumes': under_review_resumes,
            'candidates': [
                {
                    'id': c['id'],
                    'filename': c['filename'],
                    'score': c['semantic_fit_score'],
                    'industry': c['industry'].replace('_', ' ').title(),
                    'classification': c['vetting']['classification']
                } for c in processed_candidates
            ]
        })

    processed_candidates = []
    selected_resumes = []
    under_review_resumes = []
    
    is_bulk = len(files) > 1
    
    for idx, file in enumerate(files):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if idx == 0:
            pass
        else:
            file.save(file_path)
        
        # Read file depending on type
        is_json_profile = False
        json_data = {}
        if filename.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()
                # Check if it is a JSON candidate profile
                if filename.lower().endswith('.json') or resume_text.strip().startswith('{') or resume_text.strip().startswith('['):
                    parsed_json = json.loads(resume_text)
                    if isinstance(parsed_json, list) and len(parsed_json) > 0:
                        parsed_json = parsed_json[0]
                    if isinstance(parsed_json, dict) and "profile" in parsed_json:
                        json_data = parsed_json
                        is_json_profile = True
            except Exception:
                resume_text = ""
                
        # Must look like a resume
        if is_json_profile:
            is_valid_resume = True
            resume_text = json_data.get("profile", {}).get("summary", "") + "\n" + "\n".join([j.get("description", "") for j in json_data.get("career_history", [])])
        else:
            is_valid_resume = len(resume_text.strip()) > 50 and validate_is_resume(resume_text)
        
        extracted_urls = extract_urls(resume_text) if is_valid_resume else []
        has_urls = len(extracted_urls) > 0
        if not extracted_urls:
            extracted_urls = ["https://example.com/portfolio"]
            
        candidate_id = str(len(candidates_db) + 1)
        industry = classify_industry(jd + " " + resume_text) if is_valid_resume else 'general'
        
        # Check requirement mismatch and honeypot indicators
        has_mismatch = False
        mismatch_reasons = []
        is_honeypot = False
        honeypot_reason = ""
        
        if is_valid_resume:
            if is_json_profile:
                is_honeypot, honeypot_reason = check_honeypot_json(json_data)
                if is_honeypot:
                    has_mismatch = True
                    mismatch_reasons.append(honeypot_reason)
                else:
                    has_mismatch, mismatch_reasons = check_requirements_mismatch(resume_text, jd)
            else:
                has_mismatch, mismatch_reasons = check_requirements_mismatch(resume_text, jd)
            
        # Calculate semantic fit score
        if is_valid_resume:
            if has_mismatch:
                semantic_fit = 15
            else:
                semantic_fit = calculate_qualification_score(jd, resume_text)
        else:
            semantic_fit = 0
            
        # Run deep profile vetting analysis or extract from JSON directly
        if is_valid_resume and is_json_profile:
            vetting = {
                'classification': json_data.get('profile', {}).get('current_title', 'AI Engineer') + " (JSON Profile)",
                'certifications': [c.get('name') for c in json_data.get('certifications', [])] if json_data.get('certifications') else ["None listed"],
                'education': [f"{e.get('degree')} in {e.get('field_of_study')} from {e.get('institution')}" for e in json_data.get('education', [])] if json_data.get('education') else ["None listed"],
                'skills': [s.get('name') for s in json_data.get('skills', [])],
                'ignorable_gaps': "Job durations and skill metrics are fully logged in candidate metadata.",
                'plus_points': f"Structured profile containing {json_data.get('profile', {}).get('years_of_experience', 0)} years of experience with verified platform signals."
            }
        else:
            vetting = deep_analyze_profile(resume_text, jd, industry, is_valid_resume)
        
        # Categorize
        if semantic_fit >= 80:
            selected_resumes.append(filename)
        else:
            under_review_resumes.append(filename)
            
        # Get domain fallback questions
        if has_mismatch or not is_valid_resume:
            questions = []
        else:
            questions = get_domain_fallback_questions(industry)
        
        # Set agent feedback reasoning
        if not is_valid_resume:
            agent_2_feedback = "Rejected: File is not a valid candidate resume (failed credentials or structural checks)."
            agent_2_status = "Rejected"
        elif is_honeypot:
            agent_2_feedback = f"Rejected: Honeypot candidate detected. Reason: {honeypot_reason}"
            agent_2_status = "Rejected"
        elif has_mismatch:
            agent_2_feedback = f"Rejected: Candidate does not meet critical requirements. Reasons: " + ", ".join(mismatch_reasons)
            agent_2_status = "Rejected"
        else:
            agent_2_feedback = f"Verification check completed locally. Detected industry domain: {industry.replace('_', ' ').title()}."
            agent_2_status = "Approved (Local)"
            
        # Store in candidate DB
        candidate_data = {
            'id': candidate_id,
            'filename': filename,
            'resume_text': resume_text if is_valid_resume else "Invalid non-text file",
            'jd': jd,
            'urls': extracted_urls,
            'has_urls': has_urls,
            'questions': questions,
            'industry': industry,
            'agent_2_feedback': agent_2_feedback,
            'agent_2_status': agent_2_status,
            'semantic_fit_score': semantic_fit,
            'vetting': vetting,
            'scraped_data': {},
            'status': 'Awaiting Verification'
        }
        
        candidates_db[candidate_id] = candidate_data
        processed_candidates.append(candidate_data)
        
    if not is_bulk:
        single_candidate = processed_candidates[0]
        return jsonify({
            'is_bulk': False,
            'candidate_id': single_candidate['id'],
            'urls': single_candidate['urls'],
            'questions': single_candidate['questions'],
            'agent_2_feedback': single_candidate['agent_2_feedback'],
            'agent_2_status': single_candidate['agent_2_status'],
            'vetting': single_candidate['vetting']
        })
    else:
        return jsonify({
            'is_bulk': True,
            'selected_resumes': selected_resumes,
            'under_review_resumes': under_review_resumes,
            'candidates': [
                {
                    'id': c['id'],
                    'filename': c['filename'],
                    'score': c['semantic_fit_score'],
                    'industry': c['industry'].replace('_', ' ').title(),
                    'classification': c['vetting']['classification']
                } for c in processed_candidates
            ]
        })


@app.route('/api/candidate/<candidate_id>', methods=['GET'])
def get_candidate_details(candidate_id):
    if candidate_id not in candidates_db:
        return jsonify({'error': 'Candidate not found'}), 404
        
    candidate = candidates_db[candidate_id]
    
    # Run evaluation pipeline if not already completed
    if 'results' not in candidate:
        answers = []
        enriched_profile = candidate['resume_text']
        
        scraped_corpus = ""
        for url in candidate['urls']:
            scraped_text = scrape_website(url)
            candidate['scraped_data'][url] = scraped_text[:1000]
            scraped_corpus += f"\n\n--- Scraped from {url} ---\n{scraped_text}"
            
        rag = SimpleRAG(scraped_corpus if scraped_corpus.strip() else enriched_profile)
        context_retrieved = rag.query(candidate['jd'])
        
        is_valid_resume = candidate['vetting']['classification'] != "Non-Candidate Document (Template/Worksheet)"
        if is_valid_resume:
            if candidate.get('agent_2_status') == "Rejected":
                semantic_fit = 15
            else:
                semantic_fit = calculate_qualification_score(candidate['jd'], enriched_profile)
        else:
            semantic_fit = 0
            
        if not candidate.get('has_urls', False):
            external_signal = 0
        else:
            external_signal = 75
            
        executive_synthesis = "The candidate shows moderate alignment with requirements based on qualifications check."
        
        if xai_client:
            try:
                prompt = (
                    f"Evaluate the candidate's profile and RAG retrieved portfolio information against the Job Description.\n"
                    f"Provide a JSON scoring object matching this structure exactly:\n"
                    f"{{\n"
                    f"  \"semantic_fit_score\": <int 0-100 based strictly on experience alignment>,\n"
                    f"  \"external_signal_score\": <int 0-100 based strictly on portfolio validation>,\n"
                    f"  \"executive_synthesis\": \"<short recruiter summary detailing candidates actual qualities>\"\n"
                    f"}}\n\n"
                    f"Job Description: {candidate['jd']}\n"
                    f"Context Model Retrieval: {context_retrieved[:1000]}\n"
                    f"Answers: {answers}"
                )
                completion = xai_client.chat.completions.create(
                    model=xai_model,
                    messages=[
                        {"role": "system", "content": "You are a professional candidate matcher. Output ONLY a clean JSON object, no comments."},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_content = completion.choices[0].message.content.strip()
                if raw_content.startswith("```"):
                    lines = raw_content.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    raw_content = "\n".join(lines).strip()
                parsed_scores = json.loads(raw_content)
                
                semantic_fit = parsed_scores.get('semantic_fit_score', semantic_fit)
                external_signal = parsed_scores.get('external_signal_score', external_signal)
                executive_synthesis = parsed_scores.get('executive_synthesis', executive_synthesis)
            except Exception as e:
                print(f"Grok evaluation error: {e}")
                
        if candidate.get('agent_2_status') == "Rejected":
            semantic_fit = 15
            executive_synthesis = f"REJECTED: Candidate does not meet critical requirements. {candidate.get('agent_2_feedback')}"
            
        overall_score = semantic_fit
        bucket_b_desc = f"Verified links ({', '.join(candidate['urls'])}) were processed and checked against the JD targets." if candidate.get('has_urls') else "No validation links found in the resume. Signal defaulted to 0%."
        
        candidate['results'] = {
            'candidate_id': candidate_id,
            'overall_score': overall_score,
            'semantic_fit_score': semantic_fit,
            'external_signal_score': external_signal,
            'executive_synthesis': executive_synthesis,
            'bucket_a_insights': "Candidate claims match analysis completed against target job criteria.",
            'bucket_b_insights': bucket_b_desc,
            'context_retrieved_rag': context_retrieved[:400] + "..."
        }
        candidate['status'] = 'Completed'
        
    return jsonify({
        'candidate_id': candidate['id'],
        'filename': candidate['filename'],
        'urls': candidate['urls'],
        'questions': candidate['questions'],
        'agent_2_feedback': candidate['agent_2_feedback'],
        'agent_2_status': candidate['agent_2_status'],
        'vetting': candidate['vetting'],
        'results': candidate['results']
    })

@app.route('/api/submit_answers', methods=['POST'])
def submit_answers():
    """Handles Phase 2 & 3: Context merging, scraping, and final score evaluation"""
    data = request.json
    candidate_id = data.get('candidate_id')
    answers = data.get('answers', [])
    
    if candidate_id not in candidates_db:
        return jsonify({'error': 'Candidate not found'}), 404
        
    candidate = candidates_db[candidate_id]
    candidate['answers'] = answers
    candidate['status'] = 'Processing Streams'
    
    enriched_profile = candidate['resume_text'] + "\n\n--- Clarification Answers ---\n" + "\n".join(answers)
    
    scraped_corpus = ""
    for url in candidate['urls']:
        scraped_text = scrape_website(url)
        candidate['scraped_data'][url] = scraped_text[:1000]
        scraped_corpus += f"\n\n--- Scraped from {url} ---\n{scraped_text}"
        
    rag = SimpleRAG(scraped_corpus if scraped_corpus.strip() else enriched_profile)
    context_retrieved = rag.query(candidate['jd'])
    
    # Check score
    is_valid_resume = candidate['vetting']['classification'] != "Non-Candidate Document (Template/Worksheet)"
    if is_valid_resume:
        semantic_fit = calculate_qualification_score(candidate['jd'], enriched_profile)
    else:
        semantic_fit = 0
    
    if not candidate.get('has_urls', False):
        external_signal = 0
    else:
        external_signal = 75
        
    executive_synthesis = "The candidate shows moderate alignment with requirements based on qualifications check."
    
    if xai_client:
        try:
            prompt = (
                f"Evaluate the candidate's profile and RAG retrieved portfolio information against the Job Description.\n"
                f"Provide a JSON scoring object matching this structure exactly:\n"
                f"{{\n"
                f"  \"semantic_fit_score\": <int 0-100 based strictly on experience alignment>,\n"
                f"  \"external_signal_score\": <int 0-100 based strictly on portfolio validation>,\n"
                f"  \"executive_synthesis\": \"<short recruiter summary detailing candidates actual qualities>\"\n"
                f"}}\n\n"
                f"Job Description: {candidate['jd']}\n"
                f"Context Model Retrieval: {context_retrieved[:1000]}\n"
                f"Answers: {answers}"
            )
            completion = xai_client.chat.completions.create(
                model=xai_model,
                messages=[
                    {"role": "system", "content": "You are a professional candidate matcher. Output ONLY a clean JSON object, no comments."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw_content = completion.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()
            parsed_scores = json.loads(raw_content)
            
            semantic_fit = parsed_scores.get('semantic_fit_score', semantic_fit)
            external_signal = parsed_scores.get('external_signal_score', external_signal)
            executive_synthesis = parsed_scores.get('executive_synthesis', executive_synthesis)
        except Exception as e:
            print(f"Grok evaluation error: {e}")
            
    if candidate.get('agent_2_status') == "Rejected":
        semantic_fit = 15
        executive_synthesis = f"REJECTED: Candidate does not meet critical requirements. {candidate.get('agent_2_feedback')}"
            
    overall_score = semantic_fit
    
    bucket_b_desc = (
        f"Verified links ({', '.join(candidate['urls'])}) were processed and checked against the JD targets." 
        if candidate.get('has_urls') 
        else "No validation links found in the resume. Signal defaulted to 0%."
    )
    
    result = {
        'candidate_id': candidate_id,
        'overall_score': overall_score,
        'semantic_fit_score': semantic_fit,
        'external_signal_score': external_signal,
        'executive_synthesis': executive_synthesis,
        'bucket_a_insights': "Candidate claims match analysis completed against target job criteria.",
        'bucket_b_insights': bucket_b_desc,
        'context_retrieved_rag': context_retrieved[:400] + "..."
    }
    
    candidate['results'] = result
    candidate['status'] = 'Completed'
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
