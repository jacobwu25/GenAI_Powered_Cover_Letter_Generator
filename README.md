# AI Powered Cover Letter Generator

## Project Description
An AI-powered tool that generates personalized cover letters based on user-provided resume, LinkedIn profile URL, job description URL, and optional cover letter format.

## Tech Stack
- **Python:** Main programming language
- **Gradio:** Interactive user interface
- **BeautifulSoup:** Web scraping for job descriptions and LinkedIn profiles
- **LangChain:** Managing sequential chains of operations
- **PyPDFLoader:** PDF file handling
- **docx:** DOCX file handling
- **OpenAI GPT-4:** Language model for generating text
- **Requests:** Library for making HTTP requests
- **LangChain Community:** Additional models and utilities for LangChain

## Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/jacobwu25/AI-Powered-Cover-Letter-Generator.git
    cd AI-Powered-Cover-Letter-Generator
    ```
2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```
4. **Set Up Environment Variables**:
    - Create a `.env` file in the root directory of your project:
      ```sh
      touch .env
      ```
    - Open the `.env` file and add your OpenAI API key:
      ```env
      OPENAI_API_KEY=your_openai_api_key
      ```

    Replace `your_openai_api_key` with your actual OpenAI API key.
## Usage
1. **Run the application**:
    ```sh
    python app.py
    ```
2. **Open the provided URL in your browser**.
3. **Upload your resume**, enter the LinkedIn profile URL, job description URL, and optionally upload a cover letter format.
4. **View the generated cover letter and messages**.

## High-Level Architecture

### Overview
The system is composed of several key components:
- **Input Handling:** Collects inputs from the user, including resume (PDF), LinkedIn profile URL, job description URL, manual job description text, and optional cover letter format (PDF or DOCX).
- **Processing:** Uses LangChain to manage sequential operations, including fetching data, text extraction, and generating the cover letter.
- **Output Generation:** Displays the final cover letter and messages to the user via the Gradio interface.

### Detailed Architecture

![Architecture Diagram](images/architecture-diagram.png)

1. **Gradio Interface:**
   - **Inputs:** Resume (PDF), LinkedIn profile URL, job description URL, manual job description text, optional cover letter format (PDF or DOCX)
   - **Outputs:** Final cover letter, messages

2. **Data Fetching and Text Extraction:**
   - **BeautifulSoup:** For scraping job descriptions and LinkedIn profiles
   - **PyPDFLoader:** For extracting text from PDFs
   - **docx:** For extracting text from DOCX files

3. **LangChain Sequential Chain:**
   - **LLM:** Uses OpenAI GPT-4
   - **Prompt Templates:** Used to define specific tasks for each chain
   - **Chains:**
     - **Chain One:** Summarizes the job description (`job_summary`)
     - **Chain Two:** Creates a personal profile based on the resume and LinkedIn profile (`personal_profile`)
     - **Chain Three:** Composes a draft of the cover letter (`cover_letter_draft`)
     - **Chain Four:** Proofreads and finalizes the cover letter (`cover_letter_final`)

### Chain of Thought
1. **Chain One: Job Summary Generation**
   - **Function:** Summarizes the job description to highlight key responsibilities and requirements.
   - **Prompt Template:** Uses a template to extract essential information from the job description.

2. **Chain Two: Personal Profile Creation**
   - **Function:** Generates a personal profile by combining data from the resume and LinkedIn profile.
   - **Prompt Template:** Uses a template to create a coherent and compelling profile summary.

3. **Chain Three: Cover Letter Draft Composition**
   - **Function:** Drafts the initial version of the cover letter by integrating the job summary and personal profile.
   - **Prompt Template:** Uses a template to compose the cover letter with relevant sections and personalized content.

4. **Chain Four: Cover Letter Finalization**
   - **Function:** Proofreads and finalizes the cover letter, ensuring it is polished and ready for submission.
   - **Prompt Template:** Uses a template to review and enhance the draft, correcting any errors and improving the overall flow.

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
