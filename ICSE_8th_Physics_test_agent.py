import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import json

load_dotenv()

# State definition for test generation
class TestState(TypedDict):
    theme: str
    class_name: str
    subject: str
    num_mcq: int
    num_true_false: int
    num_fill_blanks: int
    num_short_qa: int
    num_medium_qa: int
    num_long_qa: int
    questions: dict
    user_answers: dict
    corrections: dict

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

def generate_test_prompt(theme: str, class_name: str, subject: str, num_mcq: int, num_true_false: int, 
                         num_fill_blanks: int, num_short_qa: int, num_medium_qa: int, num_long_qa: int) -> str:
    """Generate prompt for test creation"""
    return f"""You are an expert ICSE 8th Standard Physics teacher. Create a comprehensive test for the following topic:

Theme: {theme}
Class: {class_name}
Subject: {subject}
Board: ICSE

Generate EXACTLY the following number of questions for each section, with questions organized by difficulty levels (EASY, MEDIUM, HARD, HARDEST). Distribute questions roughly evenly across difficulty levels for each section.

REQUIREMENTS:
1. MCQs: Generate {num_mcq} multiple choice questions (4 options each, clearly mark correct answer)
2. TRUE OR FALSE: Generate {num_true_false} true/false statements
3. FILL IN THE BLANKS: Generate {num_fill_blanks} fill-in-the-blank questions
4. SHORT Q&A (3-4 lines): Generate {num_short_qa} short question & answer pairs
5. MEDIUM Q&A (6-7 lines): Generate {num_medium_qa} medium question & answer pairs
6. LONG Q&A (10-20 lines): Generate {num_long_qa} long question & answer pairs

Return the output as a valid JSON with this structure:
{{
    "mcqs": {{
        "EASY": [
            {{"question": "...", "options": ["A", "B", "C", "D"], "answer": "A", "explanation": "..."}}
        ],
        "MEDIUM": [...],
        "HARD": [...],
        "HARDEST": [...]
    }},
    "true_false": {{
        "EASY": [
            {{"statement": "...", "answer": true/false, "explanation": "..."}}
        ],
        ...
    }},
    "fill_blanks": {{
        "EASY": [
            {{"question": "... _____ ...", "answer": "word", "explanation": "..."}}
        ],
        ...
    }},
    "short_qa": {{
        "EASY": [
            {{"question": "...", "answer": "...", "points": 1}}
        ],
        ...
    }},
    "medium_qa": {{
        "EASY": [
            {{"question": "...", "answer": "...", "points": 3}}
        ],
        ...
    }},
    "long_qa": {{
        "EASY": [
            {{"question": "...", "answer": "...", "points": 5}}
        ],
        ...
    }}
}}

Ensure all content is appropriate for 8th Standard ICSE Physics syllabus. Make questions comprehensive and test understanding."""

def generate_test_questions(state: TestState) -> TestState:
    """Generate test questions using LLM"""
    prompt = generate_test_prompt(
        state['theme'], state['class_name'], state['subject'],
        state['num_mcq'], state['num_true_false'], state['num_fill_blanks'],
        state['num_short_qa'], state['num_medium_qa'], state['num_long_qa']
    )
    
    response = llm.invoke(prompt)
    response_text = response.content
    
    # Try to extract JSON from response
    try:
        # Look for JSON block in response
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            json_str = response_text[json_start:json_end].strip()
        elif '{' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
        else:
            json_str = response_text
        
        questions = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: return raw response if JSON parsing fails
        questions = {"raw_response": response_text}
    
    state['questions'] = questions
    return state

def build_test_graph():
    """Build the LangGraph workflow for test generation"""
    workflow = StateGraph(TestState)
    workflow.add_node("generate_test_questions", generate_test_questions)
    workflow.add_edge(START, "generate_test_questions")
    workflow.add_edge("generate_test_questions", END)
    return workflow.compile()

# Initialize the graph
test_graph = build_test_graph()

def generate_test(theme: str, class_name: str, subject: str, 
                 num_mcq: int = 5, num_true_false: int = 5, num_fill_blanks: int = 5,
                 num_short_qa: int = 3, num_medium_qa: int = 2, num_long_qa: int = 1):
    """Generate a test with specified number of questions per category"""
    initial_state: TestState = {
        "theme": theme,
        "class_name": class_name,
        "subject": subject,
        "num_mcq": num_mcq,
        "num_true_false": num_true_false,
        "num_fill_blanks": num_fill_blanks,
        "num_short_qa": num_short_qa,
        "num_medium_qa": num_medium_qa,
        "num_long_qa": num_long_qa,
        "questions": {},
        "user_answers": {},
        "corrections": {}
    }
    
    result = test_graph.invoke(initial_state)
    return result

def evaluate_answers(questions: dict, user_answers: dict) -> dict:
    """Evaluate user answers and provide corrections using LLM"""
    prompt = f"""You are an ICSE Physics teacher evaluating student answers.

QUESTIONS AND CORRECT ANSWERS:
{json.dumps(questions, indent=2)}

STUDENT ANSWERS:
{json.dumps(user_answers, indent=2)}

Evaluate each answer and provide:
1. Whether it's correct or incorrect
2. Score for this answer (0 for completely wrong, partial for partially correct, full for correct)
3. Detailed explanation of the correct answer
4. What the student missed or did incorrectly

Return as JSON with structure:
{{
    "question_id": {{
        "is_correct": true/false,
        "score": 0/partial/full,
        "feedback": "...",
        "correct_answer": "..."
    }}
}}"""

    response = llm.invoke(prompt)
    response_text = response.content
    
    try:
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            json_str = response_text[json_start:json_end].strip()
        elif '{' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
        else:
            json_str = response_text
        
        corrections = json.loads(json_str)
    except json.JSONDecodeError:
        corrections = {"raw_feedback": response_text}
    
    return corrections
