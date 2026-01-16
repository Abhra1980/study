import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator
import json

load_dotenv()

# State definition
class StudyMaterialState(TypedDict):
    theme: str
    class_name: str
    subject: str
    prompt: str
    study_content: str
    mcqs: str
    true_false: str
    fill_blanks: str
    short_qa: str
    medium_qa: str
    long_qa: str
    user_docs: list
    counts: dict

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

def generate_prompt(theme: str, class_name: str, subject: str, user_docs: list, counts: dict) -> str:
     """Generate comprehensive prompt for the agent"""
     docs_text = "\n".join([f"File: {d.get('name')}\n{d.get('text','(no extract)')}" for d in user_docs]) if user_docs else "No user documents provided."
     counts_text = json.dumps(counts)
     return f"""You are an expert ICSE teacher for 8th Standard Physics. Create comprehensive study material based on the following:

Theme: {theme}
Class: {class_name}
Subject: {subject}
Board: ICSE

User provided documents (if any):\n{docs_text}

Generation parameters (counts): {counts_text}

Create ONLY content for ICSE 8th Standard Physics. Use the user documents to inform and enrich examples and questions where relevant.

Produce the following sections. Respect the requested number of items for each question type (counts) and organize by difficulty levels (EASY, MEDIUM, HARD, HARDEST) where applicable.

1. STUDY CONTENT & KEY POINTS: overview, 5-7 key learning points with explanations, important definitions, practical applications.
2. MCQs: generate the specified total number of MCQs and split them across difficulty levels roughly evenly.
3. TRUE OR FALSE: generate the specified total number and split across difficulties.
4. FILL IN THE BLANKS: generate the specified total number and split across difficulties.
5. SHORT Q&A (3-4 lines): generate the specified total number split by difficulties.
6. MEDIUM Q&A (6-7 lines): generate the specified total number split by difficulties.
7. LONG Q&A (10-20 lines): generate the specified total number split by difficulties.

Format the output clearly with headings for each section and difficulty, and include answers for all questions."""

def generate_study_content(state: StudyMaterialState) -> StudyMaterialState:
    """Generate study content and key points"""
    prompt = generate_prompt(state['theme'], state['class_name'], state['subject'], state.get('user_docs', []), state.get('counts', {})) + "\n\nNow provide the STUDY CONTENT & KEY POINTS section only."
    response = llm.invoke(prompt)
    state['study_content'] = response.content
    return state

def generate_mcqs(state: StudyMaterialState) -> StudyMaterialState:
    """Generate MCQs for all difficulty levels"""
    prompt = generate_prompt(state['theme'], state['class_name'], state['subject'], state.get('user_docs', []), state.get('counts', {})) + "\n\nNow provide the MCQs section only, respecting the requested counts."
    response = llm.invoke(prompt)
    state['mcqs'] = response.content
    return state

def generate_true_false(state: StudyMaterialState) -> StudyMaterialState:
    """Generate True/False questions"""
    prompt = generate_prompt(state['theme'], state['class_name'], state['subject'], state.get('user_docs', []), state.get('counts', {})) + "\n\nNow provide the True/False section only, respecting the requested counts."
    response = llm.invoke(prompt)
    state['true_false'] = response.content
    return state

def generate_fill_blanks(state: StudyMaterialState) -> StudyMaterialState:
    """Generate Fill in the blanks"""
    prompt = generate_prompt(state['theme'], state['class_name'], state['subject'], state.get('user_docs', []), state.get('counts', {})) + "\n\nNow provide the Fill-in-the-Blanks section only, respecting the requested counts."
    response = llm.invoke(prompt)
    state['fill_blanks'] = response.content
    return state

def generate_short_qa(state: StudyMaterialState) -> StudyMaterialState:
    """Generate Short Question & Answers (3-4 lines)"""
    prompt = generate_prompt(state['theme'], state['class_name'], state['subject'], state.get('user_docs', []), state.get('counts', {})) + "\n\nNow provide the Short Q&A section only, respecting the requested counts and length."
    response = llm.invoke(prompt)
    state['short_qa'] = response.content
    return state

def generate_medium_qa(state: StudyMaterialState) -> StudyMaterialState:
    """Generate Medium Length Question & Answers (6-7 lines)"""
    prompt = generate_prompt(state['theme'], state['class_name'], state['subject'], state.get('user_docs', []), state.get('counts', {})) + "\n\nNow provide the Medium Q&A section only, respecting the requested counts and length."
    response = llm.invoke(prompt)
    state['medium_qa'] = response.content
    return state

def generate_long_qa(state: StudyMaterialState) -> StudyMaterialState:
    """Generate Long Question & Answers (10-20 lines)"""
    prompt = generate_prompt(state['theme'], state['class_name'], state['subject'], state.get('user_docs', []), state.get('counts', {})) + "\n\nNow provide the Long Q&A section only, respecting the requested counts and length."
    response = llm.invoke(prompt)
    state['long_qa'] = response.content
    return state

# Build the graph
def build_study_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(StudyMaterialState)
    
    # Add nodes
    workflow.add_node("generate_study_content", generate_study_content)
    workflow.add_node("generate_mcqs", generate_mcqs)
    workflow.add_node("generate_true_false", generate_true_false)
    workflow.add_node("generate_fill_blanks", generate_fill_blanks)
    workflow.add_node("generate_short_qa", generate_short_qa)
    workflow.add_node("generate_medium_qa", generate_medium_qa)
    workflow.add_node("generate_long_qa", generate_long_qa)
    
    # Add edges
    workflow.add_edge(START, "generate_study_content")
    workflow.add_edge("generate_study_content", "generate_mcqs")
    workflow.add_edge("generate_mcqs", "generate_true_false")
    workflow.add_edge("generate_true_false", "generate_fill_blanks")
    workflow.add_edge("generate_fill_blanks", "generate_short_qa")
    workflow.add_edge("generate_short_qa", "generate_medium_qa")
    workflow.add_edge("generate_medium_qa", "generate_long_qa")
    workflow.add_edge("generate_long_qa", END)
    
    return workflow.compile()

# Initialize the graph
study_graph = build_study_graph()

def generate_study_material(theme: str, class_name: str, subject: str, user_docs: list = None, counts: dict = None):
    """Main function to generate study material

    - user_docs: list of dicts {name: str, text: str}
    - counts: dict {mcq: int, fill: int, short: int, medium: int, long: int}
    """
    initial_state: StudyMaterialState = {
        "theme": theme,
        "class_name": class_name,
        "subject": subject,
        "prompt": "",
        "study_content": "",
        "mcqs": "",
        "true_false": "",
        "fill_blanks": "",
        "short_qa": "",
        "medium_qa": "",
        "long_qa": "",
        "user_docs": user_docs or [],
        "counts": counts or {}
    }

    # Run the graph
    result = study_graph.invoke(initial_state)
    return result
