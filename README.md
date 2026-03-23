# AI Career Assistant

### Resume Tailoring, Skill Gap Analysis & Cover Letter Generation using LLMs

An AI-powered career assistant that automates job application workflows by tailoring resumes, identifying skill gaps, retrieving relevant portfolio projects, and generating personalized cover letters.

The system combines **LLMs, semantic search, and vector databases** to align a candidate’s profile with job requirements.

* * *

# Project Overview

Job applications often require tailoring a resume and writing a custom cover letter for each role. This process is repetitive and time-consuming.

This project automates and enhances that process by:

1.  Scraping job descriptions from live URLs
2.  Extracting structured job requirements using an LLM
3.  Parsing and structuring resume content
4.  Retrieving relevant portfolio projects using semantic search
5.  Performing skill gap analysis
6.  Generating tailored resumes and writing personalized cover letters

All outputs are delivered through an interactive **Streamlit web application**.

* * *

# Technology Stack

| Component | Technology |
|--------|--------|
| Programming Language | Python |
| LLM | Groq API – Llama-3.3-70B-Versatile |
| Orchestration | LangChain |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Resume Processing | pdfplumber |
| Web Scraping | LangChain WebBaseLoader |
| Data Processing | Pandas, NumPy |
| UI | Streamlit |


* * *

# System Architecture


Job URL → Web Scraper → LLM (Job Extraction)  
                               ↓  
                        Structured Job Data  
                               ↓  
Resume PDF → Text Extraction → LLM (Resume Parsing)  
                               ↓  
                      Structured Resume Data  
                               ↓  
Portfolio CSV → ChromaDB (Embeddings)  
                               ↓  
         Semantic Retrieval of Relevant Projects  
                               ↓  
     Skill Gap Analysis (Job vs Resume + Portfolio)  
                               ↓  
     LLM → Tailored Resume + Cover Letter

* * *

# Key Features

## Job Description Scraping

The system retrieves job postings directly from URLs using **LangChain's WebBaseLoader**, which extracts visible text from the webpage.

This eliminates the need to manually copy and paste job descriptions.

* * *

## LLM-Based Job Information Extraction

The scraped text often contains navigation menus, advertisements, and formatting noise.

A **Groq LLM (Llama-3.3-70B-Versatile)** processes this raw text and extracts structured job information in JSON format:

```json
{
  "role": "Data Analyst",
  "experience": "3+ years",
  "skills": ["SQL", "Python", "Data Visualization"],
  "description": "Responsible for analyzing business data and building dashboards..."
}
```

This structured format allows downstream components to use the information more effectively.


* * *

## Resume Parsing

The resume is stored as a **PDF file** inside the project directory.

The extracted text is then passed to the LLM to produce structured resume data including:

- Skills
- Education
- Work Experience
- Projects

* * *

## Portfolio Semantic Search (ChromaDB and HuggingFace)

Portfolio projects are stored in a CSV file, converted into embeddings and indexed in **ChromaDB**

Job description are embedded, compared with stored project embeddings and **most relevant projects** returned.

* * *

## Skill Gap Analysis

 Skills from the resume and portfolio are aggregated to build a unified candidate profile, then compared against the required skills from the job description, using embeddings and semantic similarity to identify related or equivalent skills.

## Resume Tailoring

Using the extracted job data and structured resume information, the LLM generates a **tailored resume**.

The model focuses on:

- Highlighting relevant skills
- Emphasizing applicable experience
- Referencing relevant portfolio projects
- Aligning with job description keywords
- Maintaining ATS-friendly formatting

The output includes sections such as:

- Professional Summary
- Skills
- Experience
- Projects
- Education


---

## Cover Letter Generation

The system also generates a **custom cover letter** based on:

- Job description
- Candidate resume
- Relevant portfolio projects

The generated cover letter:

- References the specific role
- Highlights relevant skills
- Mentions applicable projects
- Maintains a professional tone

---

# Running the Application

### Clone the repository

``` bash
git clone https://github.com/sharlynmuturi/Career-Assistant-AI.git

cd Career-Assistant-AI
```

### Create a virtual environment and activate it

``` bash
python -m venv venv

venv\Scripts\activate # Windows

source venv/bin/activate # Mac/Linux
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory. Groq Link: [https://console.groq.com]

``` bash
GROQ_API_KEY=your_api_key_here
```

The API key is used to access the **Groq LLM service**.


### Preparing Portfolio Data

If you have your projects in a portfolio website, process them and export the cleaned dataset as portfolio.csv using the provided script.

Run:

``` bash
python scrape_portfolio.py
```

If you do not have a portfolio website, you can manually create a portfolio.csv with the columns project_name, description, tech_stack, link and all_text.

---


### Start the Streamlit app:

``` bash
streamlit run app.py
```

The app will open in your browser.

---

# Future Improvements

Possible extensions include:

- Improve job scraping robustness (Selenium fallback)
- Add fuzzy skill matching (semantic similarity)
- Allowing resume and portfolio csv uploads in the UI
- Supporting multiple job comparisons


# Limitations

- Some job pages block scraping → may return empty content
- LLM extraction depends on page quality
- Skill matching is rule-based (not fully semantic yet)

* * *
