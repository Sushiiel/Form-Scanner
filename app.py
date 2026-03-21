import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import google.generativeai as genai
import os
import chromedriver_autoinstaller
import json
from datetime import datetime
import time
import pandas as pd
import numpy as np
from io import StringIO
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import shutil

# Try to import sklearn, provide fallback if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

chromedriver_autoinstaller.install()

st.set_page_config(
    page_title="Form Intelligence System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 5px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== ML/AI FUNCTIONS ====================

def extract_keywords_tfidf(texts, top_n=10):
    """Extract important keywords using TF-IDF"""
    if not SKLEARN_AVAILABLE:
        all_words = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_words.extend([w for w in words if len(w) > 3])
        
        word_freq = Counter(all_words)
        return [(word, count) for word, count in word_freq.most_common(top_n)]
    
    try:
        if not texts or len(texts) == 0:
            return []
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        
        keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_n]
    except:
        return []

def calculate_question_similarity(questions):
    """Calculate semantic similarity between questions"""
    if not SKLEARN_AVAILABLE:
        return []
    
    try:
        if len(questions) < 2:
            return []
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(questions)
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        similar_pairs = []
        for i in range(len(questions)):
            for j in range(i+1, len(questions)):
                if similarity_matrix[i][j] > 0.3:
                    similar_pairs.append({
                        'q1': i+1,
                        'q2': j+1,
                        'similarity': round(similarity_matrix[i][j], 3),
                        'text1': questions[i][:50],
                        'text2': questions[j][:50]
                    })
        
        return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:5]
    except:
        return []

def analyze_question_patterns(questions_data):
    """Advanced analysis of question patterns"""
    analysis = {
        'total_questions': len(questions_data),
        'question_lengths': [],
        'word_frequencies': Counter(),
        'question_types_dist': Counter(),
        'required_ratio': 0,
        'avg_options_per_mcq': 0
    }
    
    all_text = []
    mcq_option_counts = []
    
    for q in questions_data:
        q_text = q.get('question', '')
        all_text.append(q_text)
        analysis['question_lengths'].append(len(q_text.split()))
        
        words = re.findall(r'\w+', q_text.lower())
        analysis['word_frequencies'].update(words)
        
        q_type = q.get('type', 'text')
        analysis['question_types_dist'][q_type] += 1
        
        if q_type in ['multiple_choice', 'checkboxes'] and q.get('options'):
            mcq_option_counts.append(len(q['options']))
    
    if analysis['total_questions'] > 0:
        analysis['required_ratio'] = sum(1 for q in questions_data if q.get('required')) / analysis['total_questions']
        analysis['avg_question_length'] = np.mean(analysis['question_lengths']) if analysis['question_lengths'] else 0
        analysis['std_question_length'] = np.std(analysis['question_lengths']) if len(analysis['question_lengths']) > 1 else 0
    
    if mcq_option_counts:
        analysis['avg_options_per_mcq'] = np.mean(mcq_option_counts)
    
    analysis['top_keywords'] = extract_keywords_tfidf(all_text, top_n=15)
    analysis['similar_questions'] = calculate_question_similarity([q.get('question', '') for q in questions_data])
    
    return analysis

def predict_form_difficulty(analysis):
    """Calculate form difficulty score"""
    weights = {
        'total_questions': 0.15,
        'required_ratio': 0.20,
        'avg_question_length': 0.15,
        'type_diversity': 0.20,
        'text_questions_ratio': 0.20
    }
    
    score = 0
    score += weights['total_questions'] * min(analysis['total_questions'] / 50, 1)
    score += weights['required_ratio'] * analysis['required_ratio']
    score += weights['avg_question_length'] * min(analysis['avg_question_length'] / 30, 1)
    
    type_count = len(analysis['question_types_dist'])
    score += weights['type_diversity'] * min(type_count / 5, 1)
    
    text_ratio = analysis['question_types_dist'].get('text', 0) / max(analysis['total_questions'], 1)
    score += weights['text_questions_ratio'] * text_ratio
    
    difficulty_score = score * 100
    
    if difficulty_score < 30:
        difficulty_level = "Easy"
    elif difficulty_score < 60:
        difficulty_level = "Medium"
    else:
        difficulty_level = "Hard"
    
    return {
        'score': round(difficulty_score, 2),
        'level': difficulty_level,
        'estimated_time_minutes': int(analysis['total_questions'] * 1.5 * (1 + score))
    }

# ==================== CORE FUNCTIONS ====================

def detect_form_type(url):
    """Detect if form is Google Forms or Microsoft Forms"""
    if "forms.gle" in url or "docs.google.com/forms" in url:
        return "google"
    elif "forms.office.com" in url or "forms.microsoft.com" in url or "forms.cloud.microsoft" in url:
        return "microsoft"
    else:
        return "unknown"

def extract_microsoft_form_questions(form_url, driver):
    """Extract questions from Microsoft Forms"""
    try:
        driver.get(form_url)
        time.sleep(3)  # Microsoft Forms needs more time to load
        
        try:
            form_title = driver.find_element(By.CLASS_NAME, "office-form-title").text
        except:
            try:
                form_title = driver.find_element(By.CSS_SELECTOR, "[role='heading']").text
            except:
                form_title = "Untitled Microsoft Form"
        
        try:
            form_description = driver.find_element(By.CLASS_NAME, "office-form-description").text
        except:
            form_description = ""
        
        questions = []
        
        # Microsoft Forms uses questionItem class
        question_elements = driver.find_elements(By.CSS_SELECTOR, "[data-automation-id='questionItem']")
        
        for idx, q_elem in enumerate(question_elements):
            try:
                # Get question text
                try:
                    question_text = q_elem.find_element(By.CSS_SELECTOR, "[data-automation-id='questionTitle']").text
                except:
                    question_text = q_elem.find_element(By.CLASS_NAME, "office-form-question-title").text
                
                if not question_text:
                    continue
                
                question_type = "text"
                options = []
                
                # Check for radio buttons (single choice)
                if q_elem.find_elements(By.CSS_SELECTOR, "input[type='radio']"):
                    question_type = "multiple_choice"
                    option_elements = q_elem.find_elements(By.CSS_SELECTOR, "[role='radio']")
                    for opt in option_elements:
                        try:
                            opt_text = opt.text
                            if opt_text:
                                options.append(opt_text)
                        except:
                            pass
                
                # Check for checkboxes (multiple choice)
                elif q_elem.find_elements(By.CSS_SELECTOR, "input[type='checkbox']"):
                    question_type = "checkboxes"
                    option_elements = q_elem.find_elements(By.CSS_SELECTOR, "[role='checkbox']")
                    for opt in option_elements:
                        try:
                            opt_text = opt.text
                            if opt_text:
                                options.append(opt_text)
                        except:
                            pass
                
                # Check for dropdown
                elif q_elem.find_elements(By.CSS_SELECTOR, "select") or q_elem.find_elements(By.CSS_SELECTOR, "[role='combobox']"):
                    question_type = "dropdown"
                    try:
                        select_elem = q_elem.find_element(By.CSS_SELECTOR, "select")
                        option_elements = select_elem.find_elements(By.TAG_NAME, "option")
                        for opt in option_elements:
                            opt_text = opt.text
                            if opt_text and opt_text != "Select":
                                options.append(opt_text)
                    except:
                        pass
                
                # Check for text input
                elif q_elem.find_elements(By.CSS_SELECTOR, "input[type='text']") or q_elem.find_elements(By.CSS_SELECTOR, "textarea"):
                    question_type = "text"
                
                # Check for date
                elif q_elem.find_elements(By.CSS_SELECTOR, "input[type='date']"):
                    question_type = "date"
                
                # Check for rating/scale
                elif q_elem.find_elements(By.CSS_SELECTOR, "[role='slider']") or "rating" in q_elem.get_attribute("class").lower():
                    question_type = "scale"
                
                # Check if required
                required = False
                try:
                    if q_elem.find_elements(By.CSS_SELECTOR, "[aria-required='true']"):
                        required = True
                    elif "*" in question_text or "required" in q_elem.text.lower():
                        required = True
                except:
                    pass
                
                questions.append({
                    "question": question_text,
                    "type": question_type,
                    "options": options,
                    "required": required,
                    "index": idx
                })
            except Exception as e:
                continue
        
        return {
            "title": form_title,
            "description": form_description,
            "questions": questions,
            "timestamp": datetime.now().isoformat(),
            "platform": "microsoft"
        }
    
    except Exception as e:
        return {"error": str(e)}

def extract_google_form_questions(form_url, driver):
    """Extract questions from Google Forms"""
    try:
        driver.get(form_url)
        time.sleep(2)
        
        try:
            form_title = driver.find_element(By.CLASS_NAME, "whsOEf").text
        except:
            form_title = "Untitled Form"
        
        try:
            form_description = driver.find_element(By.CLASS_NAME, "frebird-form-top-description").text
        except:
            form_description = ""
        
        questions = []
        question_elements = driver.find_elements(By.CLASS_NAME, "Qr7Oae")
        
        for idx, q_elem in enumerate(question_elements):
            try:
                question_text = q_elem.find_element(By.CLASS_NAME, "M7eMe").text
                
                question_type = "text"
                
                if q_elem.find_elements(By.CSS_SELECTOR, "[role='radio']"):
                    question_type = "multiple_choice"
                elif q_elem.find_elements(By.CSS_SELECTOR, "[role='checkbox']"):
                    question_type = "checkboxes"
                elif q_elem.find_elements(By.CSS_SELECTOR, "select"):
                    question_type = "dropdown"
                elif q_elem.find_elements(By.CSS_SELECTOR, "[role='slider']"):
                    question_type = "scale"
                elif q_elem.find_elements(By.CSS_SELECTOR, "input[type='date']"):
                    question_type = "date"
                elif q_elem.find_elements(By.CSS_SELECTOR, "input[type='time']"):
                    question_type = "time"
                
                options = []
                if question_type in ["multiple_choice", "checkboxes"]:
                    option_elements = q_elem.find_elements(By.CSS_SELECTOR, "[role='radio'], [role='checkbox']")
                    for opt in option_elements:
                        try:
                            opt_text = opt.find_element(By.TAG_NAME, "span").text
                            if opt_text:
                                options.append(opt_text)
                        except:
                            pass
                
                questions.append({
                    "question": question_text,
                    "type": question_type,
                    "options": options,
                    "required": "required" in q_elem.get_attribute("aria-required").lower() if q_elem.get_attribute("aria-required") else False,
                    "index": idx
                })
            except:
                continue
        
        return {
            "title": form_title,
            "description": form_description,
            "questions": questions,
            "timestamp": datetime.now().isoformat(),
            "platform": "google"
        }
    
    except Exception as e:
        return {"error": str(e)}

def extract_form_questions(form_url):
    """Extract all questions from a Google Form or Microsoft Form"""
    user_data_dir = None
    try:
        # Detect form type
        form_type = detect_form_type(form_url)
        
        if form_type == "unknown":
            return {"error": "Unsupported form type. Please use Google Forms or Microsoft Forms URLs."}
        
        if os.path.exists("/usr/bin/chromium"):
            chrome_path = "/usr/bin/chromium"
            chromedriver_path = "/usr/bin/chromedriver"
        else:
            chrome_path = None
            chromedriver_path = chromedriver_autoinstaller.install()
        
        chrome_options = Options()
        if chrome_path:
            chrome_options.binary_location = chrome_path
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
        else:
            chrome_options.add_argument("--start-maximized")
        
        # Fix: Add unique user data directory
        user_data_dir = tempfile.mkdtemp()
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--remote-debugging-port=0")  # Use random port
        
        if chromedriver_path:
            service = Service(chromedriver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            driver = webdriver.Chrome(options=chrome_options)
        
        # Extract based on form type
        if form_type == "google":
            result = extract_google_form_questions(form_url, driver)
        elif form_type == "microsoft":
            result = extract_microsoft_form_questions(form_url, driver)
        
        driver.quit()
        
        # Clean up temp directory
        try:
            shutil.rmtree(user_data_dir, ignore_errors=True)
        except:
            pass
        
        return result
    
    except Exception as e:
        # Clean up temp directory on error
        if user_data_dir:
            try:
                shutil.rmtree(user_data_dir, ignore_errors=True)
            except:
                pass
        return {"error": str(e)}

def generate_answers(questions_data, context=""):
    """Generate AI answers"""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""Generate realistic, professional answers for this form.

Context: {context if context else 'Professional context'}

Form Questions:
"""
        for i, q in enumerate(questions_data, 1):
            prompt += f"\n{i}. {q['question']}"
            prompt += f"\n   Type: {q['type']}"
            if q['options']:
                prompt += f"\n   Options: {', '.join(q['options'])}"
            prompt += "\n"
        
        prompt += "\n\nProvide answers in this exact JSON format:\n"
        prompt += "{\n"
        for i, q in enumerate(questions_data, 1):
            prompt += f'  "question_{i}": "answer_here",\n'
        prompt += "}"
        
        response = model.generate_content(prompt)
        
        try:
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            answers = json.loads(json_str)
            return answers
        except:
            return {"error": "Could not parse AI response"}
    
    except Exception as e:
        return {"error": str(e)}

# ==================== UI ====================

st.markdown("""
    <div class="main-header">
        <h1>Form Intelligence System</h1>
        <p>Automated form analysis and completion</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    context = st.text_area("Context", placeholder="Optional context for better answers")
    
    st.divider()

    st.caption("v3.1 - Multi-Platform Support")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Form Analyzer", "Smart Responder", "Batch Processing", "Data Export"])

with tab1:
    st.subheader("📝 Form Analysis")
    
    # Form type selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        form_url = st.text_input(
            "Form URL", 
            placeholder="Paste Google Forms or Microsoft Forms URL here...",
            help="Supports both Google Forms and Microsoft Forms"
        )
    
    with col2:
        st.write("")
        st.write("")
        if form_url:
            form_type = detect_form_type(form_url)
            if form_type == "google":
                st.success("🟢 Google")
            elif form_type == "microsoft":
                st.info("🔵 Microsoft")
            else:
                st.error("❌ Invalid")

    if st.button("Analyze Form", type="primary"):
        if form_url:
            with st.spinner("Extracting form data..."):
                form_data = extract_form_questions(form_url)
            
            if "error" in form_data:
                st.error(f"Error: {form_data['error']}")
            else:
                st.session_state['form_data'] = form_data
                st.session_state['form_url'] = form_url
                
                st.success("Form analyzed successfully")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Questions", len(form_data['questions']))
                with col2:
                    required = sum(1 for q in form_data['questions'] if q.get('required'))
                    st.metric("Required", required)
                with col3:
                    types = len(set(q['type'] for q in form_data['questions']))
                    st.metric("Question Types", types)
                
                st.write(f"**Form Title:** {form_data['title']}")
                if form_data.get('description'):
                    st.write(f"**Description:** {form_data['description']}")
                
                # Show platform
                platform = form_data.get('platform', 'Unknown')
                if platform == 'google':
                    st.info("🟢 **Platform:** Google Forms")
                elif platform == 'microsoft':
                    st.info("🔵 **Platform:** Microsoft Forms")
                
                # Analysis
                with st.spinner("Running analysis..."):
                    ml_analysis = analyze_question_patterns(form_data['questions'])
                    st.session_state['ml_analysis'] = ml_analysis
                
                # Display questions
                st.subheader("Questions")
                for i, q in enumerate(form_data['questions'], 1):
                    with st.expander(f"Question {i}: {q['question'][:70]}..."):
                        st.write(f"**Type:** {q['type']}")
                        st.write(f"**Required:** {'Yes' if q['required'] else 'No'}")
                        if q['options']:
                            st.write(f"**Options:** {', '.join(q['options'])}")
        else:
            st.warning("Please enter a form URL")
    
    # Generate and Fill
    if 'form_data' in st.session_state:
        st.divider()
        
        if st.button("🎯 Generate Answers", type="primary", use_container_width=True):
            with st.spinner("Generating answers..."):
                answers = generate_answers(st.session_state['form_data']['questions'], context)
                st.session_state['answers'] = answers
            
            if "error" not in answers:
                st.success("✅ Answers generated successfully!")
            else:
                st.error(f"❌ Error: {answers['error']}")
        
        # Display answers
        if 'answers' in st.session_state:
            st.subheader("Generated Answers")
            
            answers_list = []
            for i, q in enumerate(st.session_state['form_data']['questions'], 1):
                q_key = f"question_{i}"
                answer = st.session_state['answers'].get(q_key, "N/A")
                answers_list.append({
                    "Question": q['question'][:60],
                    "Type": q['type'],
                    "Answer": str(answer)[:100]
                })
            
            st.dataframe(answers_list, width='stretch')
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download JSON",
                    json.dumps(st.session_state['answers'], indent=2),
                    f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            with col2:
                csv = pd.DataFrame(answers_list).to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

with tab2:
    st.subheader("🤖 Smart Responder")
    st.write("Customize AI responses and create response profiles")
    
    if 'form_data' in st.session_state:
        # Response Profile Selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            response_profile = st.selectbox(
                "Response Profile",
                ["Professional", "Student", "Creative", "Technical", "Casual", "Custom"]
            )
        
        with col2:
            tone = st.selectbox("Tone", ["Neutral", "Enthusiastic", "Formal", "Friendly"])
        
        # Profile descriptions
        profile_configs = {
            "Professional": "Business-oriented, formal responses suitable for corporate forms",
            "Student": "Academic-focused responses for educational surveys and forms",
            "Creative": "Imaginative and unique responses for creative assessments",
            "Technical": "Detailed, technical responses with industry terminology",
            "Casual": "Relaxed, conversational responses for informal surveys",
            "Custom": "Define your own response style"
        }
        
        st.info(f"📋 {profile_configs[response_profile]}")
        
        # Custom instructions for the selected profile
        col1, col2 = st.columns(2)
        
        with col1:
            persona = st.text_input(
                "Persona/Role",
                placeholder="e.g., Software Engineer, Marketing Manager",
                help="Define who is filling this form"
            )
        
        with col2:
            response_length = st.select_slider(
                "Response Length",
                options=["Brief", "Moderate", "Detailed", "Comprehensive"],
                value="Moderate"
            )
        
        # Additional context
        st.subheader("Response Customization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.text_input("Location (optional)", placeholder="e.g., New York, USA")
            age_range = st.selectbox("Age Range (optional)", ["", "18-25", "26-35", "36-45", "46-55", "56+"])
        
        with col2:
            industry = st.text_input("Industry (optional)", placeholder="e.g., Technology, Healthcare")
            experience_level = st.selectbox("Experience Level", ["", "Beginner", "Intermediate", "Advanced", "Expert"])
        
        # Custom instructions
        custom_instructions = st.text_area(
            "Additional Instructions",
            placeholder="Any specific preferences or requirements for responses...",
            height=100
        )
        
        # Build context from all inputs
        full_context = f"Profile: {response_profile}\n"
        full_context += f"Tone: {tone}\n"
        full_context += f"Response Length: {response_length}\n"
        
        if persona:
            full_context += f"Persona: {persona}\n"
        if location:
            full_context += f"Location: {location}\n"
        if age_range:
            full_context += f"Age: {age_range}\n"
        if industry:
            full_context += f"Industry: {industry}\n"
        if experience_level:
            full_context += f"Experience: {experience_level}\n"
        if custom_instructions:
            full_context += f"\nAdditional Context: {custom_instructions}\n"
        
        st.divider()
        
        # Question-by-question customization
        st.subheader("Question-Specific Responses")
        
        enable_custom = st.checkbox("Enable question-by-question customization")
        
        if enable_custom:
            st.info("💡 Select specific questions to provide custom answers or leave blank for AI generation")
            
            if 'custom_answers' not in st.session_state:
                st.session_state['custom_answers'] = {}
            
            for i, q in enumerate(st.session_state['form_data']['questions'], 1):
                with st.expander(f"Q{i}: {q['question'][:80]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        custom_answer = st.text_input(
                            "Custom Answer",
                            key=f"custom_q_{i}",
                            placeholder="Leave blank for AI to generate",
                            value=st.session_state['custom_answers'].get(f"question_{i}", "")
                        )
                        
                        if custom_answer:
                            st.session_state['custom_answers'][f"question_{i}"] = custom_answer
                        elif f"question_{i}" in st.session_state['custom_answers']:
                            del st.session_state['custom_answers'][f"question_{i}"]
                    
                    with col2:
                        st.write(f"**Type:** {q['type']}")
                        if q['options']:
                            st.write(f"**Options:** {len(q['options'])}")
        
        st.divider()
        
        # Generate with smart context
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Generate Smart Answers", type="primary", use_container_width=True):
                with st.spinner("Generating customized answers..."):
                    # Generate AI answers for non-custom questions
                    questions_to_generate = []
                    question_indices = []
                    
                    for i, q in enumerate(st.session_state['form_data']['questions'], 1):
                        if f"question_{i}" not in st.session_state.get('custom_answers', {}):
                            questions_to_generate.append(q)
                            question_indices.append(i)
                    
                    if questions_to_generate:
                        ai_answers = generate_answers(questions_to_generate, full_context)
                        
                        if "error" not in ai_answers:
                            # Combine custom and AI answers
                            final_answers = st.session_state.get('custom_answers', {}).copy()
                            
                            for idx, q_num in enumerate(question_indices, 1):
                                ai_key = f"question_{idx}"
                                final_key = f"question_{q_num}"
                                if ai_key in ai_answers:
                                    final_answers[final_key] = ai_answers[ai_key]
                            
                            st.session_state['answers'] = final_answers
                            st.success(f"✅ Generated {len(questions_to_generate)} AI answers + {len(st.session_state.get('custom_answers', {}))} custom answers")
                        else:
                            st.error(f"Error: {ai_answers['error']}")
                    else:
                        st.session_state['answers'] = st.session_state.get('custom_answers', {})
                        st.success("✅ Using all custom answers")
        
        with col2:
            if st.button("🔄 Regenerate All", use_container_width=True):
                with st.spinner("Regenerating all answers..."):
                    st.session_state['custom_answers'] = {}
                    answers = generate_answers(st.session_state['form_data']['questions'], full_context)
                    st.session_state['answers'] = answers
                    
                    if "error" not in answers:
                        st.success("✅ All answers regenerated")
                    else:
                        st.error(f"Error: {answers['error']}")
        
        # Preview generated answers
        if 'answers' in st.session_state:
            st.divider()
            st.subheader("📄 Generated Responses Preview")
            
            preview_data = []
            for i, q in enumerate(st.session_state['form_data']['questions'], 1):
                q_key = f"question_{i}"
                answer = st.session_state['answers'].get(q_key, "N/A")
                
                source = "Custom" if q_key in st.session_state.get('custom_answers', {}) else "AI"
                
                preview_data.append({
                    "Q#": i,
                    "Question": q['question'][:50] + "..." if len(q['question']) > 50 else q['question'],
                    "Answer": str(answer)[:80] + "..." if len(str(answer)) > 80 else str(answer),
                    "Source": source
                })
            
            st.dataframe(preview_data, use_container_width=True, hide_index=True)
            
            # Save response profile
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                profile_name = st.text_input("Save Profile As", placeholder="e.g., My_Tech_Profile")
                
                if st.button("💾 Save Response Profile") and profile_name:
                    profile_data = {
                        "name": profile_name,
                        "profile": response_profile,
                        "tone": tone,
                        "persona": persona,
                        "response_length": response_length,
                        "location": location,
                        "age_range": age_range,
                        "industry": industry,
                        "experience_level": experience_level,
                        "custom_instructions": custom_instructions,
                        "full_context": full_context,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        "📥 Download Profile",
                        json.dumps(profile_data, indent=2),
                        f"profile_{profile_name}_{datetime.now().strftime('%Y%m%d')}.json",
                        "application/json"
                    )
                    st.success(f"Profile '{profile_name}' ready to download!")
            
            with col2:
                st.write("**Quick Stats**")
                total_q = len(st.session_state['form_data']['questions'])
                custom_q = len(st.session_state.get('custom_answers', {}))
                ai_q = total_q - custom_q
                
                st.metric("Total Questions", total_q)
                col_a, col_b = st.columns(2)
                col_a.metric("AI Generated", ai_q)
                col_b.metric("Custom", custom_q)
    
    else:
        st.info("📋 Please analyze a form first in the 'Form Analyzer' tab")
        
        st.subheader("Why Smart Responder?")
        
        features = [
            ("🎭", "**Response Profiles**", "Choose from pre-configured personas like Professional, Student, Technical, etc."),
            ("✍️", "**Custom Answers**", "Override AI for specific questions with your own responses"),
            ("🎨", "**Tone Control**", "Adjust response style: Formal, Friendly, Enthusiastic, or Neutral"),
            ("📊", "**Context Aware**", "Provide demographics and background for more relevant answers"),
            ("⚡", "**Hybrid Mode**", "Mix AI-generated and custom answers seamlessly"),
            ("💾", "**Save Profiles**", "Create reusable response profiles for future forms")
        ]
        
        for icon, title, desc in features:
            st.markdown(f"{icon} {title}")
            st.caption(desc)
            st.write("")

with tab3:
    st.subheader("Batch Processing")
    
    st.info("💡 Supports both Google Forms and Microsoft Forms in the same batch!")
    
    urls_input = st.text_area(
        "Form URLs (one per line)",
        height=150,
        placeholder="https://forms.gle/example1\nhttps://forms.office.com/example2\nhttps://forms.cloud.microsoft/example3"
    )
    
    delay_between = st.slider("Delay between forms (seconds)", 2, 10, 3)
    
    if st.button("Start Batch Processing", type="primary"):
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        if not urls:
            st.warning("Please enter at least one form URL")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for idx, url in enumerate(urls):
                status_text.text(f"Processing form {idx + 1}/{len(urls)}...")
                
                try:
                    form_data = extract_form_questions(url)
                    if "error" in form_data:
                        results.append({"URL": url[:50], "Status": "Failed", "Error": form_data["error"]})
                        progress_bar.progress((idx + 1) / len(urls))
                        continue
                    
                    start_time = time.time()
                    
                    # Detect platform
                    platform = form_data.get('platform', 'Unknown')
                    platform_icon = "🟢" if platform == 'google' else "🔵" if platform == 'microsoft' else "⚪"
                    
                    answers = generate_answers(form_data['questions'], context)
                    if "error" in answers:
                        results.append({"URL": url[:50], "Status": "Failed", "Error": answers["error"]})
                        progress_bar.progress((idx + 1) / len(urls))
                        continue
                    
                    elapsed = time.time() - start_time
                    
                    results.append({
                        "URL": url[:50],
                        "Platform": f"{platform_icon} {platform.title()}",
                        "Title": form_data.get("title", "Unknown")[:40],
                        "Status": "Success",
                        "Questions": len(form_data['questions']),
                        "Answers Generated": len(answers),
                        "Time": f"{elapsed:.1f}s"
                    })
                    
                    time.sleep(delay_between)
                    
                except Exception as e:
                    results.append({"URL": url[:50], "Status": "Failed", "Error": str(e)})
                
                progress_bar.progress((idx + 1) / len(urls))
            
            status_text.text("Batch processing complete")
            
            st.subheader("Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, width='stretch')
            
            success_count = sum(1 for r in results if r.get("Status") == "Success")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Forms", len(results))
            with col2:
                st.metric("Successful", success_count)
            with col3:
                st.metric("Failed", len(results) - success_count)
            
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv_results,
                f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

with tab4:
    st.subheader("Data Export")
    
    if 'form_data' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Form", st.session_state['form_data']['title'])
            st.metric("Questions", len(st.session_state['form_data']['questions']))
        
        with col2:
            st.metric("Answers Generated", "Yes" if 'answers' in st.session_state else "No")
            st.metric("Analysis Complete", "Yes" if 'ml_analysis' in st.session_state else "No")
        
        st.subheader("Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Form Structure"):
                export_data = {
                    'title': st.session_state['form_data']['title'],
                    'description': st.session_state['form_data']['description'],
                    'questions': st.session_state['form_data']['questions']
                }
                st.download_button(
                    "Download",
                    json.dumps(export_data, indent=2),
                    f"form_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        with col2:
            if 'ml_analysis' in st.session_state and st.button("Export Analysis"):
                ml_data = {
                    'form_title': st.session_state['form_data']['title'],
                    'timestamp': datetime.now().isoformat(),
                    'statistics': {
                        'total_questions': st.session_state['ml_analysis']['total_questions'],
                        'avg_length': st.session_state['ml_analysis']['avg_question_length'],
                        'required_ratio': st.session_state['ml_analysis']['required_ratio']
                    },
                    'keywords': st.session_state['ml_analysis']['top_keywords'],
                    'types': dict(st.session_state['ml_analysis']['question_types_dist'])
                }
                st.download_button(
                    "Download",
                    json.dumps(ml_data, indent=2),
                    f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        with col3:
            if 'answers' in st.session_state and st.button("Export Answers"):
                st.download_button(
                    "Download",
                    json.dumps(st.session_state['answers'], indent=2),
                    f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        st.divider()
        
        if st.button("Export Complete Package", type="primary"):
            package = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'url': st.session_state.get('form_url', 'N/A')
                },
                'form': st.session_state['form_data'],
                'analysis': st.session_state.get('ml_analysis', {}),
                'answers': st.session_state.get('answers', {})
            }
            
            st.download_button(
                "Download Package",
                json.dumps(package, indent=2),
                f"package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
        
        st.divider()
        
        if st.button("Clear Session"):
            for key in ['form_data', 'ml_analysis', 'answers', 'form_url']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Session cleared")
            st.rerun()
    
    else:
        st.info("No data available. Analyze a form first")

st.divider()
st.caption("Form Intelligence System v3.1 - Google Forms & Microsoft Forms Support")
