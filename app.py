import streamlit as st
import os
from dotenv import load_dotenv
import requests
from datetime import datetime
import json
from io import BytesIO
from PyPDF2 import PdfReader
import docx
import psycopg2
from psycopg2.extras import RealDictCursor
from ICSE_8th_Physicsstudy_agent import generate_study_material
from ICSE_8th_Physics_test_agent import generate_test, evaluate_answers

# Load environment variables
load_dotenv()

# Neon database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
NEON_API_URL = os.getenv("NEON_API_URL")

# Initialize database tables
def init_db():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Create file_uploads table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_uploads (
                id SERIAL PRIMARY KEY,
                board TEXT,
                class TEXT,
                subject TEXT,
                filename TEXT,
                file_size INTEGER,
                upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create generated_materials table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS generated_materials (
                id SERIAL PRIMARY KEY,
                board TEXT,
                class TEXT,
                subject TEXT,
                topic TEXT,
                params JSONB,
                files JSONB,
                outputs JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create tests table for test generation
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                id SERIAL PRIMARY KEY,
                board TEXT,
                class TEXT,
                subject TEXT,
                topic TEXT,
                test_params JSONB,
                test_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create test_submissions table for storing user answers and corrections
        cur.execute("""
            CREATE TABLE IF NOT EXISTS test_submissions (
                id SERIAL PRIMARY KEY,
                board TEXT,
                class TEXT,
                subject TEXT,
                topic TEXT,
                user_answers JSONB,
                corrections JSONB,
                submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.warning(f"Database initialization note: {e}")

# Save file record to database using direct PostgreSQL connection
def save_file_record(board, class_name, subject, filename, file_size):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO file_uploads (board, class, subject, filename, file_size, upload_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (board, class_name, subject, filename, file_size, datetime.now()))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Could not save to database: {e}")
        return False

def save_generated_material(board, class_name, subject, topic, params, files, outputs):
    """Save generated study material record via direct PostgreSQL"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO generated_materials (board, class, subject, topic, params, files, outputs, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (board, class_name, subject, topic, json.dumps(params), json.dumps(files), json.dumps(outputs), datetime.now()))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Error saving generated material: {e}")
        return False

def get_generated_materials(board, class_name, subject, topic):
    """Retrieve generated materials matching filters"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT * FROM generated_materials 
            WHERE board = %s AND class = %s AND subject = %s AND topic = %s
            ORDER BY created_at DESC
        """, (board, class_name, subject, topic))
        
        records = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in records]
    except Exception as e:
        st.warning(f"Error fetching history: {e}")
        return []


def save_test(board, class_name, subject, topic, test_params, test_data):
    """Save generated test via direct PostgreSQL"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO tests (board, class, subject, topic, test_params, test_data, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (board, class_name, subject, topic, json.dumps(test_params), json.dumps(test_data), datetime.now()))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Error saving test: {e}")
        return False


def get_tests(board, class_name, subject, topic):
    """Retrieve tests for a topic"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT * FROM tests 
            WHERE board = %s AND class = %s AND subject = %s AND topic = %s
            ORDER BY created_at DESC
        """, (board, class_name, subject, topic))
        
        records = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in records]
    except Exception as e:
        st.warning(f"Error fetching tests: {e}")
        return []


def save_test_submission(board, class_name, subject, topic, user_answers, corrections):
    """Save test submission and corrections"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO test_submissions (board, class, subject, topic, user_answers, corrections, submitted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (board, class_name, subject, topic, json.dumps(user_answers), json.dumps(corrections), datetime.now()))
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.warning(f"Error saving submission: {e}")
        return False


def extract_text_from_upload(uploaded_file):
    """Extract text content from supported uploads for prompting. Skip binary files."""
    name_lower = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    
    # Skip binary image files
    if name_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return None
    
    try:
        if name_lower.endswith('.pdf'):
            reader = PdfReader(BytesIO(data))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip() if text.strip() else None
        if name_lower.endswith('.docx'):
            document = docx.Document(BytesIO(data))
            text = "\n".join([p.text for p in document.paragraphs])
            return text.strip() if text.strip() else None
        # plain text (txt, csv, etc.)
        try:
            text = data.decode('utf-8').strip()
            return text if text else None
        except Exception:
            text = data.decode('latin-1', errors='ignore').strip()
            return text if text else None
    except Exception as e:
        return None

# Initialize database on app start
init_db()

# Set page configuration
st.set_page_config(
    page_title="euduAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("📚 euduAI")
st.markdown("---")

# Subject data with icons and chapter counts
ICSE_SUBJECTS_DATA = {
    "Mathematics": {"icon": "√", "chapters": 35, "color": "#87CEEB"},
    "Physics": {"icon": "⚙️", "chapters": 42, "color": "#FFD700"},
    "Chemistry": {"icon": "🧪", "chapters": 38, "color": "#DDA0DD"},
    "Biology": {"icon": "🔬", "chapters": 40, "color": "#90EE90"},
    "English Grammar": {"icon": "📝", "chapters": 25, "color": "#FFB6C1"},
    "English Literature": {"icon": "📖", "chapters": 28, "color": "#90EE90"},
    "History": {"icon": "📚", "chapters": 30, "color": "#DEB887"},
    "Geography": {"icon": "🌍", "chapters": 32, "color": "#20B2AA"}
}

JEE_SUBJECTS_DATA = {
    "Mathematics": {"icon": "√", "chapters": 35, "color": "#87CEEB"},
    "Physics": {"icon": "⚙️", "chapters": 42, "color": "#FFD700"},
    "Chemistry": {"icon": "🧪", "chapters": 38, "color": "#DDA0DD"},
    "Biology": {"icon": "🔬", "chapters": 40, "color": "#90EE90"}
}

# Topics/Themes data structure
ICSE_TOPICS = {
    "Class 8": {
        "Physics": {
            "Theme 1: Matter": "Focuses on the kinetic theory of matter and how particles behave in different states.\n• Kinetic Theory: Movement of particles in solids, liquids, and gases.\n• Energy Content: Comparing energy levels across the three states.\n• Changes of State: Melting (fusion), vaporization (boiling), evaporation, condensation, freezing, sublimation, and deposition.\n• State Diagrams: Visualizing phase changes.",
            "Theme 2: Physical Quantities and Measurement": "Develops skills in measuring density and understanding buoyancy.\n• Density Measurement: Using a Eureka can and measuring cylinder for irregular solids; basic concepts for fluids.\n• Floatation: Concept of sinking and floating related to density.\n• Relative Density: Definitions, units, and measurement techniques.",
            "Theme 3: Force and Pressure": "Covers the mechanics of force and its effects on objects and liquids.\n• Turning Effect: Moment of force (torque), definitions, and simple calculations.\n• Thrust and Pressure: Definitions, units, and factors affecting pressure.\n• Liquid & Gas Pressure: Qualitative understanding of pressure in liquids and atmospheric pressure.",
            "Theme 4: Energy": "Introduction to work, power, and different forms of mechanical energy.\n• Work: Definition, SI unit (Joule), and calculations in simple cases.\n• Energy Types: Kinetic and Potential energy (specifically Gravitational Potential Energy).\n• Energy Transformation: Common daily life examples and the difference between energy and power.",
            "Theme 5: Light Energy": "Explores the behavior of light as it interacts with different surfaces.\n• Refraction: Definition, examples, and behavior in different media.\n• Curved Mirrors: Concave and convex mirrors, focus, principal axis, and ray diagrams.\n• Images: Distinction between real and virtual images.\n• Dispersion: Splitting white light into constituent colors.",
            "Theme 6: Heat Transfer": "Details the movement of thermal energy and its effects.\n• Thermal Expansion: Linear, volume, and superficial expansion in solids, liquids, and gases.\n• Change of State: Deep dive into boiling vs. evaporation.",
            "Theme 7: Sound": "Examines the characteristics and propagation of sound waves.\n• Wave Properties: Pitch (frequency), loudness (amplitude), and monotone.\n• Units: Loudness measured in decibels (dB).\n• Instruments: How wind, membrane, and string instruments produce sound.",
            "Theme 8: Electricity": "Introduction to electrical circuits, safety, and static charges.\n• Household Consumption: Calculating energy in kilowatt-hours (kWh).\n• Wiring and Safety: Identifying live, neutral, and earth wires; use of fuses and circuit breakers.\n• Static Electricity: Conservation of charges, charging by conduction and induction.\n• Tools: Gold Leaf Electroscope and lightning conductors."
        }
    }
}

JEE_TOPICS = {
    "Class 9": {
        "Physics": {}
    },
    "Class 10": {
        "Physics": {}
    }
}

# Create tabs
tab1, tab2 = st.tabs(["ISCE", "JEE Foundation"])

# Helper function to display subjects
def display_subjects(tab_prefix, subjects_data):
    for subject, data in subjects_data.items():
        key = f"{tab_prefix}_{subject.lower().replace(' ', '_')}"
        is_selected = st.session_state.get(f'{tab_prefix}_selected_subject') == subject
        
        # Create custom HTML for subject display
        bg_color = "#FFE4B5" if is_selected else "transparent"
        
        subject_html = f"""
        <div style='
            background-color: {bg_color};
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 12px;
        '>
            <div style='
                background-color: {data['color']};
                width: 48px;
                height: 48px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                color: white;
                font-weight: bold;
                flex-shrink: 0;
            '>{data['icon']}</div>
            <div style='flex: 1;'>
                <div style='font-weight: bold; font-size: 16px; color: #333; margin-bottom: 4px;'>{subject}</div>
                <div style='font-size: 12px; color: #666;'>{data['chapters']} Chapters</div>
            </div>
        </div>
        """
        st.markdown(subject_html, unsafe_allow_html=True)
        
        # Invisible button for interaction
        if st.button("", key=key, use_container_width=True, help=subject):
            st.session_state[f'{tab_prefix}_selected_subject'] = subject
            st.rerun()

# ISCE Tab
with tab1:
    # Class selector dropdown
    classes = ["Select a Class", "Class 8", "Class 9", "Class 10"]
    selected_class = st.selectbox("Choose a Class:", classes, key="isce_class_select")
    
    if selected_class != "Select a Class":
        st.session_state['isce_selected_class'] = selected_class
    else:
        if 'isce_selected_class' in st.session_state:
            del st.session_state['isce_selected_class']
        if 'isce_selected_subject' in st.session_state:
            del st.session_state['isce_selected_subject']
    
    st.markdown("---")
    
    # Create two columns: left for navigation, right for content
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        if 'isce_selected_class' in st.session_state:
            st.subheader("Subjects")
            st.markdown("---")
            display_subjects("isce", ICSE_SUBJECTS_DATA)
    
    with right_col:
        # Display content based on selection
        if 'isce_selected_subject' in st.session_state:
            subject = st.session_state['isce_selected_subject']
            data = ICSE_SUBJECTS_DATA[subject]
            
            # Breadcrumb navigation
            st.markdown(f"**ICSE** > **{st.session_state['isce_selected_class']}** > **{subject}**")
            
            # Title
            st.title(f"{subject} Workspace")
            st.markdown("---")
            
            # Check if topics exist for this class and subject
            class_name = st.session_state['isce_selected_class']
            if class_name in ICSE_TOPICS and subject in ICSE_TOPICS[class_name]:
                topics_dict = ICSE_TOPICS[class_name][subject]
                if topics_dict:
                    # Topics dropdown
                    st.subheader("📚 Topics/Themes")
                    selected_topic = st.selectbox(
                        "Select a topic to view details:",
                        ["Select a topic"] + list(topics_dict.keys()),
                        key=f"isce_topic_{subject}"
                    )
                    
                    if selected_topic != "Select a topic":
                        st.markdown("---")
                        st.markdown(f"### {selected_topic}")
                        st.info(topics_dict[selected_topic])
                    st.markdown("---")
            
            # Tabs for different operations
            upload_tab, generate_tab, test_tab, history_tab = st.tabs(["📤 Upload", "✨ Generate", "🧪 Test", "📜 History"])
            
            with upload_tab:
                st.subheader("Upload Content")
                uploaded_files = st.file_uploader(
                    "Choose files",
                    type=["pdf", "txt", "jpg", "png", "docx"],
                    help="Supported: PDF, TXT, JPG, PNG, DOCX (Max 50MB)",
                    accept_multiple_files=True,
                    key="isce_uploader"
                )
                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
                    stored = []
                    for file in uploaded_files:
                        file_size_kb = file.size / 1024
                        st.write(f"📄 {file.name} ({file_size_kb:.2f} KB)")
                        text_content = extract_text_from_upload(file)
                        # Only store if we have valid text content (skip binary files)
                        if text_content:
                            if len(text_content) > 12000:
                                text_content = text_content[:12000]
                            stored.append({"name": file.name, "size": file.size, "text": text_content})
                            st.write(f"  ✓ Text extracted ({len(text_content)} chars)")
                        else:
                            st.write(f"  ⚠ Skipped (binary or empty)")
                        # Save basic upload record
                        save_file_record("ICSE", st.session_state['isce_selected_class'], subject, file.name, file.size)
                    # save into session for later use by generator
                    st.session_state['isce_uploaded_files'] = stored
                    st.info(f"✓ {len(stored)} text file(s) stored for generation")
            
            with generate_tab:
                st.subheader("🤖 AI Study Material Generator")
                st.info("Generate comprehensive study material using AI for this topic")
                
                # Only show for ICSE 8th Standard Physics
                if subject == "Physics" and st.session_state['isce_selected_class'] == "Class 8":
                    selected_topic_value = st.session_state.get(f"isce_topic_{subject}", "Select a topic")
                    if selected_topic_value != "Select a topic":
                        st.markdown("### Generation settings")
                        cols = st.columns(5)
                        with cols[0]:
                            mcq_count = st.number_input("MCQs", min_value=1, max_value=30, value=6, key="mcq_count")
                        with cols[1]:
                            fill_count = st.number_input("Fill Blanks", min_value=1, max_value=30, value=8, key="fill_count")
                        with cols[2]:
                            short_count = st.number_input("Short Q&A", min_value=1, max_value=30, value=6, key="short_count")
                        with cols[3]:
                            medium_count = st.number_input("Medium Q&A", min_value=1, max_value=30, value=4, key="medium_count")
                        with cols[4]:
                            long_count = st.number_input("Long Q&A", min_value=1, max_value=30, value=2, key="long_count")

                        # show uploaded files if any
                        user_docs = st.session_state.get('isce_uploaded_files', [])
                        if user_docs:
                            st.info(f"Using {len(user_docs)} uploaded document(s) as context for generation")

                        if st.button("Generate Study Material", key="isce_generate_btn", use_container_width=True):
                            counts = {"mcq": mcq_count, "fill": fill_count, "short": short_count, "medium": medium_count, "long": long_count}
                            with st.spinner("🔄 AI is generating comprehensive study material... This may take a minute..."):
                                try:
                                    result = generate_study_material(
                                        theme=selected_topic_value,
                                        class_name=st.session_state['isce_selected_class'],
                                        subject=subject,
                                        user_docs=user_docs,
                                        counts=counts
                                    )
                                    st.success("✅ Study material generated successfully!")

                                    outputs = {
                                        "study_content": result.get('study_content',''),
                                        "mcqs": result.get('mcqs',''),
                                        "true_false": result.get('true_false',''),
                                        "fill_blanks": result.get('fill_blanks',''),
                                        "short_qa": result.get('short_qa',''),
                                        "medium_qa": result.get('medium_qa',''),
                                        "long_qa": result.get('long_qa','')
                                    }

                                    # Save generated material to DB
                                    saved = save_generated_material(
                                        board="ICSE",
                                        class_name=st.session_state['isce_selected_class'],
                                        subject=subject,
                                        topic=selected_topic_value,
                                        params=counts,
                                        files=user_docs,
                                        outputs=outputs
                                    )
                                    if saved:
                                        st.info("Saved generated material to history")

                                    with st.expander("📖 Study Content & Key Points", expanded=True):
                                        st.markdown(outputs['study_content'] or 'No content generated')
                                    with st.expander("📝 MCQs (All Difficulty Levels)"):
                                        st.markdown(outputs['mcqs'] or 'No MCQs generated')
                                    with st.expander("✓/✗ True or False Questions"):
                                        st.markdown(outputs['true_false'] or 'No questions generated')
                                    with st.expander("_____ Fill in the Blanks"):
                                        st.markdown(outputs['fill_blanks'] or 'No questions generated')
                                    with st.expander("❓ Short Q&A (3-4 lines)"):
                                        st.markdown(outputs['short_qa'] or 'No questions generated')
                                    with st.expander("❓❓ Medium Q&A (6-7 lines)"):
                                        st.markdown(outputs['medium_qa'] or 'No questions generated')
                                    with st.expander("❓❓❓ Long Q&A (10-20 lines)"):
                                        st.markdown(outputs['long_qa'] or 'No questions generated')
                                except Exception as e:
                                    st.error(f"Error generating study material: {str(e)}")
                                    st.info("Please make sure your OpenAI API key is set correctly in the .env file")
                    else:
                        st.warning("Please select a topic from the Topics/Themes dropdown above")
                else:
                    st.warning("AI Study Material Generation is currently available only for ICSE 8th Standard Physics")
            
            with test_tab:
                st.subheader("🧪 AI Test Generator")
                st.info("Generate comprehensive tests based on the topic")
                
                if subject == "Physics" and st.session_state['isce_selected_class'] == "Class 8":
                    selected_topic_value = st.session_state.get(f"isce_topic_{subject}", "Select a topic")
                    if selected_topic_value != "Select a topic":
                        st.markdown("### Test Configuration")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            num_mcq = st.number_input("MCQs", min_value=2, max_value=20, value=5, key="test_mcq")
                        with col2:
                            num_tf = st.number_input("True/False", min_value=2, max_value=20, value=5, key="test_tf")
                        with col3:
                            num_fill = st.number_input("Fill Blanks", min_value=2, max_value=20, value=5, key="test_fill")
                        
                        col4, col5, col6 = st.columns(3)
                        with col4:
                            num_short = st.number_input("Short Q&A", min_value=1, max_value=15, value=3, key="test_short")
                        with col5:
                            num_medium = st.number_input("Medium Q&A", min_value=1, max_value=10, value=2, key="test_medium")
                        with col6:
                            num_long = st.number_input("Long Q&A", min_value=1, max_value=5, value=1, key="test_long")
                        
                        if st.button("Generate Test", key="generate_test_btn", use_container_width=True):
                            with st.spinner("🔄 Generating test questions... Please wait..."):
                                try:
                                    test_result = generate_test(
                                        theme=selected_topic_value,
                                        class_name=st.session_state['isce_selected_class'],
                                        subject=subject,
                                        num_mcq=num_mcq,
                                        num_true_false=num_tf,
                                        num_fill_blanks=num_fill,
                                        num_short_qa=num_short,
                                        num_medium_qa=num_medium,
                                        num_long_qa=num_long
                                    )
                                    
                                    st.session_state['test_data'] = test_result.get('questions', {})
                                    st.session_state['test_topic'] = selected_topic_value
                                    
                                    # Save test to DB
                                    test_params = {
                                        "num_mcq": num_mcq,
                                        "num_tf": num_tf,
                                        "num_fill": num_fill,
                                        "num_short": num_short,
                                        "num_medium": num_medium,
                                        "num_long": num_long
                                    }
                                    
                                    saved = save_test(
                                        board="ICSE",
                                        class_name=st.session_state['isce_selected_class'],
                                        subject=subject,
                                        topic=selected_topic_value,
                                        test_params=test_params,
                                        test_data=st.session_state['test_data']
                                    )
                                    
                                    if saved:
                                        st.success("✅ Test generated and saved!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating test: {str(e)}")
                        
                        # Display generated test if available
                        if 'test_data' in st.session_state and st.session_state.get('test_topic') == selected_topic_value:
                            st.markdown("---")
                            st.markdown("### 📋 Test Questions")
                            test_data = st.session_state['test_data']
                            
                            # Store user answers
                            user_answers = {}
                            
                            # MCQs
                            if 'mcqs' in test_data:
                                st.subheader("MCQs (Multiple Choice Questions)")
                                for difficulty, questions in test_data['mcqs'].items():
                                    st.markdown(f"**{difficulty}**")
                                    for idx, q in enumerate(questions):
                                        key = f"mcq_{difficulty}_{idx}"
                                        st.write(f"{idx + 1}. {q.get('question', '')}")
                                        answer = st.radio(
                                            "Select answer:",
                                            options=q.get('options', []),
                                            key=key
                                        )
                                        user_answers[key] = answer
                            
                            # True/False
                            if 'true_false' in test_data:
                                st.subheader("True or False")
                                for difficulty, questions in test_data['true_false'].items():
                                    st.markdown(f"**{difficulty}**")
                                    for idx, q in enumerate(questions):
                                        key = f"tf_{difficulty}_{idx}"
                                        st.write(f"{idx + 1}. {q.get('statement', '')}")
                                        answer = st.radio(
                                            "True or False:",
                                            options=["True", "False"],
                                            key=key
                                        )
                                        user_answers[key] = answer
                            
                            # Fill in the Blanks
                            if 'fill_blanks' in test_data:
                                st.subheader("Fill in the Blanks")
                                for difficulty, questions in test_data['fill_blanks'].items():
                                    st.markdown(f"**{difficulty}**")
                                    for idx, q in enumerate(questions):
                                        key = f"fill_{difficulty}_{idx}"
                                        st.write(f"{idx + 1}. {q.get('question', '')}")
                                        answer = st.text_input(
                                            "Your answer:",
                                            key=key
                                        )
                                        user_answers[key] = answer
                            
                            # Short Q&A
                            if 'short_qa' in test_data:
                                st.subheader("Short Q&A (3-4 lines)")
                                for difficulty, questions in test_data['short_qa'].items():
                                    st.markdown(f"**{difficulty}**")
                                    for idx, q in enumerate(questions):
                                        key = f"short_{difficulty}_{idx}"
                                        st.write(f"{idx + 1}. {q.get('question', '')}")
                                        answer = st.text_area(
                                            "Your answer (3-4 lines):",
                                            height=80,
                                            key=key
                                        )
                                        user_answers[key] = answer
                            
                            # Medium Q&A
                            if 'medium_qa' in test_data:
                                st.subheader("Medium Q&A (6-7 lines)")
                                for difficulty, questions in test_data['medium_qa'].items():
                                    st.markdown(f"**{difficulty}**")
                                    for idx, q in enumerate(questions):
                                        key = f"medium_{difficulty}_{idx}"
                                        st.write(f"{idx + 1}. {q.get('question', '')}")
                                        answer = st.text_area(
                                            "Your answer (6-7 lines):",
                                            height=120,
                                            key=key
                                        )
                                        user_answers[key] = answer
                            
                            # Long Q&A
                            if 'long_qa' in test_data:
                                st.subheader("Long Q&A (10-20 lines)")
                                for difficulty, questions in test_data['long_qa'].items():
                                    st.markdown(f"**{difficulty}**")
                                    for idx, q in enumerate(questions):
                                        key = f"long_{difficulty}_{idx}"
                                        st.write(f"{idx + 1}. {q.get('question', '')}")
                                        answer = st.text_area(
                                            "Your answer (10-20 lines):",
                                            height=150,
                                            key=key
                                        )
                                        user_answers[key] = answer
                            
                            st.markdown("---")
                            if st.button("Submit Test", use_container_width=True):
                                with st.spinner("🔄 Evaluating your answers..."):
                                    try:
                                        corrections = evaluate_answers(test_data, user_answers)
                                        
                                        # Save submission
                                        saved = save_test_submission(
                                            board="ICSE",
                                            class_name=st.session_state['isce_selected_class'],
                                            subject=subject,
                                            topic=selected_topic_value,
                                            user_answers=user_answers,
                                            corrections=corrections
                                        )
                                        
                                        st.session_state['test_corrections'] = corrections
                                        st.success("✅ Test submitted! View corrections below.")
                                    except Exception as e:
                                        st.error(f"Error evaluating test: {str(e)}")
                    else:
                        st.warning("Please select a topic from the Topics/Themes dropdown above")
                else:
                    st.warning("Test generation is currently available only for ICSE 8th Standard Physics")
                
                # Display corrections if available
                if 'test_corrections' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📝 Your Corrections")
                    corrections = st.session_state['test_corrections']
                    for q_id, feedback in corrections.items():
                        if q_id != "raw_feedback":
                            with st.expander(f"Question: {q_id}"):
                                st.write(f"**Status:** {'✅ Correct' if feedback.get('is_correct') else '❌ Incorrect'}")
                                st.write(f"**Score:** {feedback.get('score', 'N/A')}")
                                st.write(f"**Feedback:** {feedback.get('feedback', '')}")
                                st.write(f"**Correct Answer:** {feedback.get('correct_answer', '')}")
            
            with history_tab:
                st.subheader("History")
                st.info("View generated study materials for the selected topic")
                selected_topic_value = st.session_state.get(f"isce_topic_{subject}", "Select a topic")
                if selected_topic_value != "Select a topic":
                    if st.button("Load History", key="load_isce_history"):
                        records = get_generated_materials("ICSE", st.session_state['isce_selected_class'], subject, selected_topic_value)
                        if records:
                            for rec in records:
                                with st.expander(f"{rec.get('created_at','')} - {rec.get('topic','')} - {rec.get('subject','')}"):
                                    try:
                                        # Properly deserialize outputs from database
                                        outputs_str = rec.get('outputs', '{}')
                                        if isinstance(outputs_str, str):
                                            outputs = json.loads(outputs_str)
                                        else:
                                            outputs = outputs_str
                                        st.markdown(outputs.get('study_content',''))
                                        st.markdown("---")
                                        st.markdown(outputs.get('mcqs',''))
                                    except Exception as e:
                                        st.warning(f"Could not load outputs: {e}")
                        else:
                            st.info("No generated materials found for this topic")
                else:
                    st.info("Select a topic first to view history")
                
        elif 'isce_selected_class' in st.session_state:
            st.header(f"ICSE - {st.session_state['isce_selected_class']}")
            st.info("Please select a subject from the left panel.")
        else:
            st.header("ICSE")
            st.info("Please select a class from the dropdown above.")

# JEE Foundation Tab
with tab2:
    # Class selector dropdown
    classes = ["Select a Class", "Class 8", "Class 9", "Class 10"]
    selected_class = st.selectbox("Choose a Class:", classes, key="jee_class_select")
    
    if selected_class != "Select a Class":
        st.session_state['jee_selected_class'] = selected_class
    else:
        if 'jee_selected_class' in st.session_state:
            del st.session_state['jee_selected_class']
        if 'jee_selected_subject' in st.session_state:
            del st.session_state['jee_selected_subject']
    
    st.markdown("---")
    
    # Create two columns: left for navigation, right for content
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        if 'jee_selected_class' in st.session_state:
            st.subheader("Subjects")
            st.markdown("---")
            display_subjects("jee", JEE_SUBJECTS_DATA)
    
    with right_col:
        # Display content based on selection
        if 'jee_selected_subject' in st.session_state:
            subject = st.session_state['jee_selected_subject']
            data = JEE_SUBJECTS_DATA[subject]
            
            # Breadcrumb navigation
            st.markdown(f"**JEE Foundation** > **{st.session_state['jee_selected_class']}** > **{subject}**")
            
            # Title
            st.title(f"{subject} Workspace")
            st.markdown("---")
            
            # Tabs for different operations
            upload_tab, generate_tab, test_tab, history_tab = st.tabs(["📤 Upload", "✨ Generate", "🧪 Test", "📜 History"])
            
            with upload_tab:
                st.subheader("Upload Content")
                uploaded_files = st.file_uploader(
                    "Choose files",
                    type=["pdf", "jpg", "png", "docx"],
                    help="Supported: PDF, JPG, PNG, DOCX (Max 50MB)",
                    accept_multiple_files=True,
                    key="jee_uploader"
                )
                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
                    for file in uploaded_files:
                        file_size_kb = file.size / 1024
                        st.write(f"📄 {file.name} ({file_size_kb:.2f} KB)")
                        # Save to database
                        save_file_record("JEE Foundation", st.session_state['jee_selected_class'], subject, file.name, file.size)
                    st.info("✓ File records saved to database")
            
            with generate_tab:
                st.subheader("Generate Content")
                st.info("Generate learning materials and summaries for this subject")
            
            with test_tab:
                st.subheader("Test Yourself")
                st.info("Take quizzes and practice tests")
            
            with history_tab:
                st.subheader("History")
                st.info("View your recent activities and uploads")
                
        elif 'jee_selected_class' in st.session_state:
            st.header(f"JEE Foundation - {st.session_state['jee_selected_class']}")
            st.info("Please select a subject from the left panel.")
        else:
            st.header("JEE Foundation")
            st.info("Please select a class from the dropdown above.")
