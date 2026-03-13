# AI Career Assistant  
### Automated Resume Tailoring & Cover Letter Generation with LLMs

An AI-powered career assistant that helps tailor resume and generate personalized cover letters for specific job postings.

The application scrapes job descriptions from online listings, extracts structured job requirements using a Large Language Model (LLM), retrieves the most relevant portfolio projects using semantic vector search, and generates tailored resumes and cover letters.

The system integrates **LangChain, Groq LLM (Llama-3.3-70B-Versatile), ChromaDB, and Streamlit**.

---

# Project Overview

Job applications often require tailoring a resume and writing a custom cover letter for each role. This process is repetitive and time-consuming.

This project automates that workflow by:

1. Scraping job descriptions from career pages.
2. Extracting structured job requirements using an LLM.
3. Processing and parsing a resume.
4. Retrieving relevant portfolio projects using vector similarity search.
5. Generating a tailored resume aligned with the job posting.
6. Writing a personalized cover letter referencing relevant experience and projects.

The final output is presented through an interactive **Streamlit web interface**.

---

# Technology Stack

| Component | Technology |
|--------|--------|
| Programming Language | Python |
| LLM | Groq API – Llama-3.3-70B-Versatile |
| LLM Orchestration | LangChain |
| Vector Database | ChromaDB |
| Resume Processing | pdfplumber |
| Web Scraping | LangChain WebBaseLoader |
| Data Processing | Pandas |
| UI | Streamlit |

---

# Key Features

### Job Description Scraping
The system retrieves job postings directly from URLs using **LangChain's WebBaseLoader**, which extracts visible text from the webpage.

This eliminates the need to manually copy and paste job descriptions.

---

### Job Information Extraction
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

---

### Resume Processing

The resume is stored as a **PDF file** inside the project directory.

The application uses **pdfplumber** to extract text from each page of the PDF.

The extracted text is then passed to the LLM to produce structured resume data including:

- Skills
- Education
- Work Experience
- Projects

This structured representation helps the AI better understand the candidate profile.

---

### Portfolio Semantic Search (ChromaDB)

Portfolio projects are stored in a **CSV file**.

Each project contains:

- project_name
- description
- tech_stack
- link


These projects are inserted into **ChromaDB**, a vector database.

The database converts project descriptions into vector embeddings and stores them for semantic search.

When a job description is provided, the system retrieves the **most relevant portfolio projects** using similarity search.

This ensures the AI references the **most relevant projects** in the tailored resume and cover letter.

---

### Resume Tailoring

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

### Cover Letter Generation

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
git clone https://github.com/sharlynmuturi/ai-career-assistant.git

cd career-assistant-ai
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

Create a `.env` file in the root directory: (Groq Link)[https://console.groq.com]

``` bash
GROQ_API_KEY=your_api_key_here
```

The API key is used to access the **Groq LLM service**.


### Preparing Portfolio Data

If you have portfolio projects stored in a CSV file, you can process them using the provided script.

Run:

``` bash
python scrape_portfolio.py
```

This script:

1. Reads project descriptions from the CSV file.
2. Combines project metadata.
3. Prepares text for vector embedding.
4. Stores the projects in **ChromaDB**.

After this step, the vector database will be ready for semantic search.

---


### Start the Streamlit app:

``` bash
streamlit run app.py
```

The app will open in your browser.

---

# Future Improvements

Possible extensions include:

- Adding downloadable PDF resume output
- Allowing resume uploads in the UI
- Supporting multiple job comparisons


---
