#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install langchain openai python-docx beautifulsoup4 requests gradio langchain-community tiktoken fake-useragent langchain_openai pypdf


# In[18]:


import warnings
warnings.filterwarnings('ignore')


# In[34]:


import os
import requests

#Ensure you already set up openai_api_key as an environment parameter before use the model
openai_api_key = os.getenv("openai_api_key")
#os.environ["openai_api_key"] = openai_api_key
#os.environ["OPENAI_MODEL_NAME"]="gpt-4o"
# Check if the API key was retrieved successfully

if openai_api_key is None:
    raise ValueError("OpenAI API key not found in environment variables.")


# In[40]:


#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import docx
from bs4 import BeautifulSoup
import logging

import tiktoken
from openai import OpenAIError  # Make sure you handle exceptions from OpenAI
import openai
from fake_useragent import UserAgent
import random
import time


# In[36]:


llm_model = "gpt-4o"
llm = ChatOpenAI(temperature=0.0, model=llm_model, openai_api_key=openai_api_key)

# Token counting function with a check for non-string inputs
def count_tokens(text, model=llm_model):
    if not isinstance(text, str):
        return 0  # Return 0 tokens for non-string inputs
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# In[22]:


#Define prompt template
prompt_job_description = ChatPromptTemplate.from_template(
    "Your role is a tech job researcher. Your goal is to make sure to do amazing analysis on "
    "{job_description} to help job applicants."
    "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
)


# In[23]:


prompt_profiler = ChatPromptTemplate.from_template(
    "Your role is a Personal Profiler for Job Candidate. Your goal is to do incredible research on job candidates"
    "to help them stand out in the job market"
    "Equipped with analytical prowess, you dissect "
    "and synthesize information "
    "from diverse sources to craft comprehensive "
    "personal and professional profiles, laying the "
    "groundwork for personalized cover letter enhancements. "
    "The sources of the information you are gonna use " 
    "are candidate's resume  and Linkedin profile. \n\n"
    "Resume: {resume} "
    "Linkedin profile: {linkedin} "
    "If Linkedin is missing, use ONLY resume to craft the profile. "
)


# In[24]:


prompt_cover_letter_composer = ChatPromptTemplate.from_template(
    "Your role is a Cover Letter Writer for Job Candidates. "
    "Your goal is to compose a cover letter that is factually correct for the job application. "
    "With a strategic mind and an eye for detail, you "
    "excel at composing cover letters for applicants to tech companies. "
    "You understand what is important to recruiters "
    "in the tech space. "
    "You know how to highlight relevant skills and experiences, ensuring they "
    "resonate perfectly with the job's requirements. "
    "You are gonna use candidate's Personal Profile and the Job Description to craft an exceptional cover letter by using the Cover Letter Template provided below. \n\n"
    "Personal Profile: {personal_profile} \n "
    "Job Description: {job_summary} \n "
    "Cover Letter Template: {cover_letter_format} \n "
    "If Linkedin is missing, use ONLY resume to craft the profile. If Cover Letter Template is missing, you decide an appropriate format to use. "
)


# In[25]:


prompt_proof_reader = ChatPromptTemplate.from_template(
    "Your role is to proofread cover letters. "
    "Your goal is to ensure there are no grammatical errors "
    "and that the meaning of the cover letter is concise. "
    "With an eye for detail, you are the final gatekeeper " 
    "to ensure a high-quality cover letter is generated for job applications. "
    "You will use the Cover Letter Draft from previous work to complete your work. "
    "You will also ensure that the final cover letter follows the Cover Letter Template as closely as possibl. \n\n"
    "Cover Letter Draft: {cover_letter_draft} \n "
    "Cover Letter Template: {cover_letter_format} "
    "If Cover Letter Template is missing, you decide an appropriate format to use. "
)


# In[26]:


#Define individual chain and chain of thoughts
chain_one = LLMChain(llm=llm, prompt=prompt_job_description, output_key="job_summary")
chain_two = LLMChain(llm=llm, prompt=prompt_profiler, output_key="personal_profile")
chain_three = LLMChain(llm=llm, prompt=prompt_cover_letter_composer, output_key="cover_letter_draft")
chain_four = LLMChain(llm=llm, prompt= prompt_proof_reader, output_key="cover_letter_final")
#result = chain.run(input)
sequential_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["resume", "linkedin", "job_description", "cover_letter_format"],
    output_variables=["job_summary", "personal_profile", "cover_letter_draft", "cover_letter_final"],
    verbose=True
)


# In[27]:


# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        # Assuming 'file' is a file-like object
        file_path = file.name
        loader = PyPDFLoader(file_path)
        text = loader.load()
        return text
    except Exception as e:
        logging.error("Error extracting text from PDF", exc_info=True)
        return str(e)

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Rate limiter to slow down requests
def rate_limiter(min_wait=2, max_wait=5):
    delay = random.uniform(min_wait, max_wait)  # Random delay between requests
    time.sleep(delay)
    print(f"Rate limiter: Delaying for {delay:.2f} seconds.")

# Function to fetch job description from URL
def fetch_job_description(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        job_description = soup.get_text(strip=True)
        # Check if the fetched content is too short to be meaningful
        if len(job_description) < 100:
            raise ValueError("Fetched job description content is too short.")
        return job_description
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching job description: {e}")
        return None

# Function to fetch LinkedIn profile from URL
def fetch_linkedin_profile(url):
    ua = UserAgent()  # Create a UserAgent object
    headers = {
        'User-Agent': ua.random  # Randomly select a User-Agent for each request
    }
    try:
        # Apply rate limiter before making the request
        rate_limiter()  # Add a delay before each request
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Will raise an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract LinkedIn profile text - adjust based on the structure of the LinkedIn profile page
        linkedin_profile = soup.get_text()
        return linkedin_profile
    except requests.exceptions.RequestException as e:
        print(f"Error fetching LinkedIn profile: {e}")
        return None


# In[28]:


def cover_letter_gen(resume_file, linkedin_url, job_description_url, manual_job_description, cover_letter_file):
    # Extract text from the uploaded resume PDF
    resume_content = extract_text_from_pdf(resume_file)

    # Fetch job description from URL
    job_description = fetch_job_description(job_description_url)
    if not job_description:  # Check if job description is empty or too short
        job_description = manual_job_description.strip()
        if not job_description:
            return None, "Error: Job description could not be fetched and manual input is empty. Please provide a valid job description.", "", ""
        
    # Fetch LinkedIn profile from URL
    linkedin_profile = fetch_linkedin_profile(linkedin_url)
    if linkedin_profile is None or not linkedin_profile.strip():
        linkedin_profile = "LinkedIn profile could not be fetched."
        message = "Warning: LinkedIn profile could not be fetched. Continuing without LinkedIn profile data."
    else:
        message = "LinkedIn profile fetched successfully."
    
   
    # Check if cover letter file is provided
    if cover_letter_file is not None:
        # Determine the file type and extract text accordingly
        cover_letter_format = ""
        cover_letter_file_path = cover_letter_file.name
        if cover_letter_file_path.endswith(".pdf"):
            cover_letter_format = extract_text_from_pdf(cover_letter_file_path)
        elif cover_letter_file_path.endswith(".docx"):
            cover_letter_format = extract_text_from_docx(cover_letter_file)
        else:
            return None, "Error: Unsupported file format for cover letter. Please upload a .pdf or .docx file."
    else:
        cover_letter_format = ""
    
    
    inputs = {
        "resume": resume_content,
        "linkedin": linkedin_profile,
        "job_description": job_description,
        "cover_letter_format": cover_letter_format
    }
    
    # Calculate input tokens
    resume_tokens = count_tokens(resume_content)
    linkedin_tokens = count_tokens(linkedin_profile)
    job_description_tokens = count_tokens(job_description)
    cover_letter_format_tokens = count_tokens(cover_letter_format)
    input_tokens = resume_tokens + linkedin_tokens + job_description_tokens + cover_letter_format_tokens

    # Run the sequential chain
    try:
        outputs = sequential_chain(inputs)
        cover_letter_final = outputs["cover_letter_final"]
        output_tokens = count_tokens(cover_letter_final)
    except Exception as e:
        return None, f"Error during processing: {e}"
    
    # Total tokens
    total_tokens = input_tokens + output_tokens


     # Return the outputs: cover letter final, message, and LinkedIn profile content
    return cover_letter_final, message, input_tokens, output_tokens, total_tokens


# In[ ]:


#Use Gradio to generate web UI
import gradio as gr

gr.close_all()
demo = gr.Interface(fn=cover_letter_gen, 
                    inputs=[gr.File(label="Upload Resume (PDF) (Required)"),
                            gr.Textbox(lines=1, placeholder="Enter LinkedIn profile URL (Optional)", label="LinkedIn Profile URL"),
                            gr.Textbox(lines=1, placeholder="Enter job description URL (Required)", label="Job Description URL"),
                            gr.Textbox(lines=5, placeholder="Enter job description manually (if URL fetch fails)", label="Manual Job Description"),
                            gr.File(label="Upload Cover Letter Format (PDF or DOCX) (Optional)")
                    ],
                    outputs=[gr.Textbox(label="Cover Letter Final"),
                             gr.Textbox(label="Message"),
                             gr.Textbox(label="Input Tokens"),
                             gr.Textbox(label="Output Tokens"),
                             gr.Textbox(label="Total Tokens")
                    ],
                    title="Cover Letter Generator",
                    description="Upload your resume, Linkedin profile, and job description URL to generate your customized cover letter"
                   )
#demo.launch(share=True)
demo.launch()


# In[ ]:




