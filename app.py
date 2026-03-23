import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import chromadb
import uuid
import os
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Career Assistant", layout="wide")
st.title("AI Resume & Job Matching Assistant")
st.caption("This demo app uses my personal resume and portfolio projects.")

BASE_DIR = Path(__file__).parent

job_link = st.text_input("Paste Job Description URL")
resume_path = BASE_DIR / "resume.pdf"

def read_resume(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

resume_text = read_resume(resume_path)
st.success("Resume loaded")

# ChromaDB for portfolio
client = chromadb.PersistentClient('vectorstore')
collection = client.get_or_create_collection(name="portfolio")

portfolio_path = BASE_DIR / "portfolio.csv"
portfolio_df = pd.read_csv(portfolio_path)

if not collection.count():
    for _, row in portfolio_df.iterrows():
        collection.add(
            documents=[row["all_text"]],
            metadatas={
                "link": row.get("link",""),
                "project_name": row.get("project_name",""),
                "tech_stack": row.get("tech_stack","")
            },
            ids=[str(uuid.uuid4())]
        )

st.info(f"ChromaDB loaded with {collection.count()} projects")


# LLM Setup
api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    temperature=0.3,
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)
parser = JsonOutputParser()


# Scrape Job Page
def scrape_job_page(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content if docs else ""

page_data = scrape_job_page(job_link) if job_link else None


# Job Extraction Prompt
prompt_extract = PromptTemplate.from_template("""
You are an expert information extraction system.

### CONTEXT
The following text was scraped from a company's career page and contains
information about a job posting. The text may include formatting noise,
navigation elements, or repeated content.

### TASK
Extract the job information from the text.

Return a JSON object with the following fields:

- role: The job title
- experience: Required experience level (years or level such as junior, mid, senior)
- skills: List of key technical or domain skills required for the role
- description: A concise 2–5 sentence summary of the job responsibilities

### RULES
- Extract information only from the provided text.
- Do NOT invent information.
- If a field is missing, return null.
- Skills must be returned as a list of strings.
- Return ONLY a valid JSON object.
- Do not include explanations, markdown, backticks, or extra text

### JSON FORMAT
{{
  "role": "Job title",
  "experience": "Experience requirement",
  "skills": ["skill1", "skill2", "skill3"],
  "description": "Short description of the role"
}}

### SCRAPED TEXT
{page_data}
""")

chain_extract = prompt_extract | llm | parser

job_data = None
if page_data:

    with st.spinner("Extracting job details..."):
        job_data = chain_extract.invoke({"page_data": page_data})

# Resume Extraction
prompt_resume = PromptTemplate.from_template("""
You are an expert resume parser.

### TASK
Extract structured information from the resume text.

Return a JSON object with the following fields:

- name: candidate name
- education: list of education entries
- skills: list of technical skills
- experience: list of professional experiences
- projects: list of projects mentioned

### RULES
- Extract information only from the text.
- Do NOT invent information.
- If something is missing return null.
- Skills must be a list of strings.
- Return ONLY valid JSON.

### JSON FORMAT
{{
  "name": "Candidate name",
  "education": ["degree1", "degree2"],
  "skills": ["skill1", "skill2"],
  "experience": ["experience1", "experience2"],
  "projects": ["project1", "project2"]
}}

### RESUME TEXT
{resume_text}
""")
chain_resume_extract = prompt_resume | llm | parser

resume_data = None
with st.spinner("Parsing resume..."):

    @st.cache_data
    def parse_resume(text):
        return chain_resume_extract.invoke({"resume_text": text})

    resume_data = parse_resume(resume_text)

    # resume_data = chain_resume_extract.invoke({"resume_text": resume_text})


# Semantic Matching
# HuggingFace embeddings for job description
if job_data:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Embed job
    description = job_data.get("description") or ""
    skills_list = job_data.get("skills") or []
    job_text = description + " " + " ".join(skills_list)

    job_embedding = embedding_model.embed_documents([job_text])[0]

    # Embed portfolio projects
    project_texts = portfolio_df['all_text'].tolist()
    project_embeddings = embedding_model.embed_documents(project_texts)

    # Compute cosine similarity
    from numpy import dot
    from numpy.linalg import norm

    similarities = [dot(job_embedding, p_emb) / (norm(job_embedding) * norm(p_emb)) for p_emb in project_embeddings]

    # Get top 5 projects
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
    top_semantic_projects = portfolio_df.iloc[top_indices]

    
# Skill gap analysis
import re

def normalize_skills(skills_list):
    normalized = set()
    
    if not skills_list:
        return normalized  # handle None or empty list
    
    for skill in skills_list:
        if not skill:
            continue
        skill = skill.lower().strip()

        # Extract text inside parentheses and split by comma
        paren_matches = re.findall(r'\((.*?)\)', skill)
        for match in paren_matches:
            for s in match.split(','):
                s = s.strip()
                if s:
                    normalized.add(s)

        # Remove parentheses part from main skill
        skill = re.sub(r'\(.*?\)', '', skill).strip()
        
        # Split remaining text by commas
        for s in skill.split(','):
            s = s.strip()
            if s:
                normalized.add(s)
    
    return normalized
    
def skill_gap(job_skills, resume_skills):
    job_set = set(job_skills)
    resume_set = set(resume_skills)
    matched = job_set.intersection(resume_set)
    missing = job_set.difference(resume_set)
    return list(matched), list(missing)

# Retrieving Portfolio Projects
def get_top_projects(job_text, n=7):
    results = collection.query(query_texts=[job_text], n_results=n)
    text = ""
    for meta in results['metadatas'][0]:
        text += f"""
Project: {meta.get("project_name")}
Technologies: {meta.get("tech_stack")}
Link: {meta.get("link")}
"""
    return text

top_projects_text = get_top_projects(job_text) if job_data else None

def extract_portfolio_skills(top_projects_text):
    """
    Extract a list of skills/technologies from the top portfolio projects.
    """
    skills = set()
    for line in top_projects_text.split("\n"):
        if line.startswith("Technologies:"):
            techs = line.replace("Technologies:", "").strip().split(",")
            techs = [t.strip() for t in techs if t.strip()]
            skills.update(techs)
    return list(skills)

if job_data:
    portfolio_skills = extract_portfolio_skills(top_projects_text)
    
    # Combine resume and portfolio skills
    combined_candidate_skills = resume_data.get("skills", []) + portfolio_skills
    
    # Normalize all skills
    job_skills_normalized = normalize_skills(job_data.get("skills" or []))
    job_skills_normalized = normalize_skills(job_data.get("skills") or [])
    candidate_skills_normalized = normalize_skills(combined_candidate_skills)
    
    # Compute matched / missing
    matched_skills_normalized = job_skills_normalized.intersection(candidate_skills_normalized)
    missing_skills_normalized = job_skills_normalized.difference(candidate_skills_normalized)

    # Display using original capitalization from resume/portfolio if desired
    matched_skills = [skill for skill in combined_candidate_skills if skill.lower() in matched_skills_normalized]
    missing_skills = [skill for skill in (job_data.get("skills") or []) if any(s.lower() in missing_skills_normalized for s in normalize_skills([skill]))]
    
# Resume & cover letter prompts
prompt_resume_tailor = PromptTemplate.from_template("""
You are an expert career assistant and resume writer.

Your task is to tailor a candidate's resume for a specific job role.

### JOB DESCRIPTION
{job_data}

### CANDIDATE RESUME
{resume_data}

### CANDIDATE PROJECTS
{portfolio_projects}

### INSTRUCTIONS
Rewrite the candidate's resume so that it better aligns with the job.

Focus on:
- highlighting relevant skills
- emphasizing relevant projects
- matching the language used in the job description
- keeping all information truthful

### OUTPUT FORMAT

Return a professional resume with the following sections:

NAME

PROFESSIONAL SUMMARY

SKILLS

EXPERIENCE

PROJECTS

EDUCATION

The resume should be concise, professional, and optimized for ATS systems.

Return only the resume text.
""")

prompt_cover_letter = PromptTemplate.from_template("""
You are an expert career assistant.

### CONTEXT
You have the following information:

- Job posting details: {job_data}
- Candidate resume content: {resume_data}
- Relevant portfolio projects: {portfolio_projects}

### TASK
Write a professional and persuasive **cover letter** tailored to this specific job. 
The cover letter should:

1. Address the hiring manager (use "Dear Hiring Manager" if name is unknown)
2. Highlight the candidate's most relevant skills and experience from the resume
3. Reference the most relevant portfolio projects
4. Match the tone of the job posting (formal, professional)
5. Be 3–5 paragraphs long
6. End with a polite call-to-action

### RULES
- Only use information provided in the context; do not invent details
- Keep it concise and impactful
- Output plain text (no JSON)
""")


# Sreamlit UI
if job_data:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Tailored Resume"):
            with st.spinner("Generating resume..."):
                chain_resume_tailor = prompt_resume_tailor | llm
                res = chain_resume_tailor.invoke({
                    "job_data": job_data,
                    "resume_data": resume_data,
                    "portfolio_projects": top_projects_text
                })
                tailored_resume = res.content
            st.subheader("Tailored Resume")
            st.text_area("", tailored_resume, height=400)
    
    with col2:
        if st.button("Generate Cover Letter"):
            with st.spinner("Writing cover letter..."):
                chain_cover = prompt_cover_letter | llm
                res = chain_cover.invoke({
                    "job_data": job_data,
                    "resume_data": resume_data,
                    "portfolio_projects": top_projects_text
                })
                cover_letter = res.content
            st.subheader("Cover Letter")
            st.text_area("", cover_letter, height=400)

if page_data:
    st.subheader("Extracted Job Data")
    st.json(job_data)

if job_data:
    st.subheader("Skill Gap Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Matched Skills**")
        st.write(matched_skills)
    
    with col2:
        st.markdown("**Missing Skills**")
        st.write(missing_skills)


if job_data and not top_semantic_projects.empty:
    st.subheader("Top Semantic Portfolio Matches")
    for i, row in top_semantic_projects.iterrows():
        st.markdown(f"**{i+1}. {row['project_name']}**")
        st.write(row['all_text'][:300] + "...")  # snippet
        st.markdown(f"[Project Link]({row['link']})")