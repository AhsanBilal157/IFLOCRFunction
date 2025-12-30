import azure.functions as func
import datetime
import json
import logging
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
import os
from pydantic import BaseModel, Field
import operator
from typing import List, Optional, Any
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from datetime import datetime, date
import uuid
from pydantic import BaseModel
from data_extractor import extract_text_from_pdf,extract_text_from_docx,extract_text_from_txt,call_document_intelligence
from io import BytesIO
from typing import Optional
today_str = date.today().strftime("%B %d, %Y")

# groq_api = os.getenv("GROQ_API_KEY")

os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"   # must be "true" not the API key
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# os.environ["LANGCHAIN_PROJECT"] = 'langsmith-demo'
os.environ["OPENAI_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AzureWebJobsStorage"] = os.getenv("AzureWebJobsStorage")
APEX_URL = os.getenv("APEX_URL")
connection_string = os.getenv("connection_string")
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
import os
from typing import Annotated, TypedDict
# groq_api = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
import logging
import requests
import json
import uuid
from datetime import date, datetime
import json
import datetime
app = func.FunctionApp()
# ============================================
# --------- LLM Setup ------------
# ============================================

# groq_api = "YOUR_GROQ_API_KEY"

# model = ChatGroq(
#     api_key=groq_api,
#     model_name="llama-3.3-70b-versatile"
# )
@app.function_name(name="IFLOCRBlobFunction")
@app.blob_trigger(arg_name="readfile",path="resume/{name}",connection="AzureWebJobsStorage")
def main(readfile: func.InputStream):
    logging.info("[DEBG] ENTRY: IFLOCRBlobFunction invoked - starting.")
    model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # or your deployment
    api_version="2024-08-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    
    )
# ============================================
# --------- Resume Schemas & LLM ------------
# ============================================

    class Experience(BaseModel):
        job_title: str = Field(description="Job title of the candidate in this role")
        company: str = Field(description="Company where candidate worked")
        start_date: Optional[date] = Field(description="Start date of the job (YYYY-MM-DD)")
        end_date: Optional[date] = Field(description="End date of the job (YYYY-MM-DD, or null if Present)")
        description: Optional[str] = Field(description="Responsibilities or achievements in this role")

    class Education(BaseModel):
        degree: str = Field(description="Degree name, e.g., BS Computer Science")
        institution: str = Field(description="University or school name")
        start_date: Optional[date] = Field(description="Start date of the degree (YYYY-MM-DD)")
        end_date: Optional[date] = Field(description="End date of the degree (YYYY-MM-DD)")
        grade: Optional[str] = Field(description="Grade, GPA, or honors if mentioned")

    class Achievement(BaseModel):
        title: str = Field(description="Title or short description of the achievement")
        description: Optional[str] = Field(description="Detailed explanation if available")

    class Certification(BaseModel):
        name: str = Field(description="Certification name, e.g., AWS Certified Solutions Architect")
        issuer: Optional[str] = Field(description="Issuing authority or organization")
        certidate_date: Optional[date] = Field(description="Date of certification (YYYY-MM-DD)")

    class CandidateResume(BaseModel):
        # Personal Info
        name: str = Field(description="Full name of the candidate")
        years_of_experience: Optional[str] = Field(description="Total years of work experience")
        linkedin_profile: Optional[str] = Field(description="Candidate's LinkedIn profile URL")
        contact_number: Optional[str] = Field(description="Candidate's contact number")
        address: Optional[str] = Field(description="Candidate's address")
        email: str = Field(description="Candidate's email address")
        job_title: Optional[str] = Field(description="Job title candidate applied for")
        grade: Optional[str] = Field(description="Candidate grade or seniority level if mentioned")
        
        # ✅ New field added
        profile_summary: Optional[str] = Field(
            description="A short summary or professional overview of the candidate's experience and goals"
        )


        # Lists
        skills: List[str] = Field(description="List of skills mentioned in the resume")
        experiences: List[Experience] = Field(description="List of professional experiences")
        education: List[Education] = Field(description="List of educational qualifications")
        achievements: List[Achievement] = Field(description="List of achievements or awards")
        certifications: List[Certification] = Field(description="List of professional certifications")
        languages: Optional[List[str]] = Field(default=None, description="List of languages the candidate knows")
        projects: Optional[List[str]] = Field(default=None, description="List of notable projects mentioned")
        publications_research: Optional[List[str]] = Field(default=None, description="Publications or research entries")
        extra_curricular: Optional[List[str]] = Field(default=None, description="Extra curricular / professional activities")

    # Wrap LLM with structured output (Pydantic)
    resume_details_structured_model = model.with_structured_output(CandidateResume)

    # ============================================
    # --------- State ------------
    # ============================================ 

    class ResumeProcessingState(BaseModel):
        # From blob metadata
        blob_name: str
        blob_client: Optional[Any] = None
        subject: str
        body: str
        # email_date: datetime = None

        # Step 1 - Recruitment classifier
        is_recruitment: Optional[bool] = None

        # Step 2 - Requisition extractor
        requisition_id: Optional[str] = None
        job_title: Optional[str] = None

        # Step 3 - OCR result
        ocr_text: Optional[str] = None

        # Step 4 - Resume parser (structured)
        candidate_resume: Optional[CandidateResume] = None

        # Step 5 - Experience calculator
        calculated_experience: Optional[float] = None

        # Step 6 - Final aggregation
        final_output: Optional[dict] = None

        # Status / logging
        errors: List[str] = []
        processed: bool = False

    # ============================================
    # --------- Classifier Schema & LLM ----------
    # ============================================ 

    class RecruitmentClassifierOutput(BaseModel):
        is_recruitment: bool
        requisition_id: Optional[str] = None
        job_title: Optional[str] = None

    classifier_llm = model.with_structured_output(RecruitmentClassifierOutput)

    # ============================================
    # --------- Expereince Cal Schema & LLM ------
    # ============================================ 

    class ExperienceCalculatorOutput(BaseModel):
        years_of_experience: Optional[float] = None

    Experience_cal_llm = model.with_structured_output(ExperienceCalculatorOutput)

    # ============================================
    # --------- Helper Functions ------
    # ============================================ 

    class DateEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()  # converts to "YYYY-MM-DD"
            return super().default(obj)


    # Custom encoder for date/datetime objects
    def json_default(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        raise TypeError(f"Type {type(o)} not serializable")
    # ============================================
    # --------- Node Functions ------
    # ============================================ 



    def email_classifier(state: ResumeProcessingState):
        subject = state.subject
        body = state.body

        # prompt = f"""You are an expert email classifier. 
        # Your task is to decide if an email is related to *recruitment* or *not recruitment*.

        # Use only the SUBJECT and BODY of the email. 
        # Answer strictly with one word:  
        # - "Recruitment" if the email is about job opportunities, resumes, hiring, interviews, or related to recruitment.  
        # - "Not Recruitment" if the email is about anything else (e.g., invoices, marketing, newsletters, general communication).  

        # SUBJECT: {subject}
        # BODY: {body}

        # Your output must be only: True OR False.

        # """
        prompt = f"""SYSTEM / USER PROMPT (single string to send to the LLM)

        You are an expert email analyzer for recruitment pipelines. 
        Given an EMAIL SUBJECT and BODY, follow these steps exactly and then return ONLY a single JSON object that matches the schema:

        json start
        "is_recruitment": true|false,
        "requisition_id": "<digits>" | null,
        "job_title": "<clean job title>" | null
        json ends

        RULES (must-follow)
        1) Classification (is_recruitment):
        - Return true if the SUBJECT or BODY indicates the email is a job application or recruitment-related (keywords: "resume", "cv", "curriculum vitae", "application", "applying", "applied", "attached resume", "interview", "hiring", "candidate", "position", "vacancy", "recruitment", "test submission").
        - Return false for unrelated emails (examples: security alerts, invoices, newsletters, promotions, account notices). If you are confident it's not recruitment, return false.
        - Do NOT guess recruitment if subject/body contain only vague marketing or system notifications.

        2) Requisition ID extraction (requisition_id):
        - If there is an explicit numeric requisition/role ID, return it as a string of digits only (no letters, no punctuation).
        - Check patterns in priority order:
            a) `REQ` or `REQ-` or `REQ:` or `Requisition` followed by digits (e.g., "REQ-884", "Requisition 884").
            b) A numeric group that appears after a trailing hyphen or at end of subject like `... - 884`.
            c) Any standalone 3–6 digit number in subject or body that is obviously an ID.
        - If multiple candidates, prefer the match from (a) then (b) then (c). If still ambiguous, pick the numeric group most likely referenced in the SUBJECT (rightmost numeric group).
        - If none found or ambiguous, set requisition_id to null.
        - Always return only digits (e.g., "884"), not words.

        3) Job title extraction (job_title):
        - First try to extract the job title from the SUBJECT. Remove extraneous tokens like numeric IDs, words "Process", "Submission", "Application", "Test", and location tokens (if clearly not part of title).
        - If not clearly present, look for phrases in the BODY such as "applying for *<title>*", "interested in *<title>*", "position of *<title>*", or "for the role of *<title>*".
        - If you infer a title, ensure you do NOT invent or expand it. If uncertain, return null.
        - Clean the job title: trim whitespace, remove surrounding punctuation, collapse multiple spaces to one, and use Title Case for readability (preserve known acronyms, e.g., "QA", "HR").
        - If job_title can't be determined confidently, set it to null.

        4) No hallucination / no extra fields:
        - If you cannot find reliable information, use `null` rather than inventing values.
        - Do NOT return any additional fields. Only the three keys shown above.
        - Return a valid JSON object only (no commentary, no explanations).

        5) Output types:
        - `is_recruitment` must be boolean true/false (not string).
        - `requisition_id` must be string digits or null.
        - `job_title` must be string or null.

        FEW-SHOT EXAMPLES (Input -> Output)

        Example 1:
        Subject: "1 Lap Former Operator - Process - 884"
        Body: "Please find my CV attached."
        Output:
        "is_recruitment": true, "requisition_id": "884", "job_title": "Lap Former Operator"

        Example 2:
        Subject: "Application for Associate Manager - Recruitment Test Submission"
        Body: "Dear Hiring Manager, I am writing to confirm my interest in the Associate Manager position..."
        Output:
        "is_recruitment": true, "requisition_id": null, "job_title": "Associate Manager"

        Example 3:
        Subject: "Security alert"
        Body: "We noticed a new sign-in to your Google Account..."
        Output:
        "is_recruitment": false, "requisition_id": null, "job_title": null

        Example 4:
        Subject: "RE: REQ-12345 Application"
        Body: "Applying for the above role. CV attached."
        Output:
        "is_recruitment": true, "requisition_id": "12345", "job_title": null

        Example 5:
        Subject: "Interested in position as accountant"
        Body: "I am interested in working as an accountant. Please find my resume attached."
        Output:
        "is_recruitment": true, "requisition_id": null, "job_title": "Accountant"

        NOW: Process the following inputs and return ONLY the JSON object (no extra text):

        Subject: {subject}
        Body: {body}

        END"""


        output = classifier_llm.invoke(prompt)
        return {"is_recruitment" :output.is_recruitment ,"requisition_id": output.requisition_id,"job_title":output.job_title}



    def extract_data(state: ResumeProcessingState):
        blob_client = state.blob_client
        blob_name = state.blob_name
        file_extension = blob_name.split(".")[-1].lower()
        try:
            if file_extension == "pdf":
                # Read the blob content into memory
                blob_data = blob_client.download_blob()
                pdf_content = blob_data.readall()
                pdf_stream = BytesIO(pdf_content)
                content = extract_text_from_pdf(pdf_stream)
                print("########################")
                print(content)
                print("########################")
            elif file_extension == "txt" or file_extension == "docx":
                content = extract_text_from_txt(blob_name)
            elif file_extension in ["jpg", "jpeg", "png"]:
                logging.info("Image file detected, using document intelligence.")
                blob_data = blob_client.download_blob()
                content = blob_data.readall()
                content = call_document_intelligence(content)
            
            if content == None or not content.strip():
                blob_data = blob_client.download_blob()
                content = blob_data.readall()
                logging.warning(f"Parsing error or empty content: {content}")
                content = call_document_intelligence(content)

        except Exception as e:
            logging.error(f"Unexpected error during processing: {str(e)}")
            blob_data = blob_client.download_blob()
            content = blob_data.readall()
            content = call_document_intelligence(content)
        print(content)
        # logging.info(f"Final extracted content:\n{content}")
        return {"ocr_text": content}

    def resume_parser(state: ResumeProcessingState):
        content = state.ocr_text
        prompt = f"""You are an expert resume parsing agent. 
                Your task is to carefully analyze the given resume text and extract structured information following the provided schema.  

                - Always map the candidate’s details into the correct fields of the schema.  
                - Extract dates in the format YYYY-MM-DD (if the exact day is not available, approximate to the first day of the month, e.g., "2020-01-01").  
                - If any field is not available in the resume, return null or an empty list where appropriate.  
                - For experiences and education, preserve chronological order if possible (most recent first).  
                - For skills, return only unique keywords (not sentences).  
                - For contact info (email, phone, LinkedIn, address), extract exactly as written in the resume.  

                Input: {content}.  
                Output: JSON strictly conforming to the CandidateResume schema.  """
        output = resume_details_structured_model.invoke(prompt)
        return {"candidate_resume": output}

    def check_classifier(state: ResumeProcessingState):

        if state.is_recruitment == True:
            return True
        else:
            return False
        
    def aggregator(state: ResumeProcessingState):

        final_output = {
                "idempotency_key": str(uuid.uuid4()),
                "requisition_id": state.requisition_id,
                "job_title": state.job_title,
                "candidate_resume": (
                    state.candidate_resume.dict() if state.candidate_resume else None
                ),
                "calculated_experience": state.calculated_experience,
            }

        return {"final_output": final_output}

    def calc_expereince(state: ResumeProcessingState):
        content = state.ocr_text
        prompt2 = f"""
        You are an intelligent document parser. The following is the content extracted from a resume:
        {content}

        Your job is to calculate total professional experience from this resume. Follow these rules strictly:

        1. If the candidate mentions total experience (e.g., "5 years of experience"), extract only that number.
        2. If total experience is not mentioned, extract all experience durations from job start and end dates.
        3. If end date is written as "present" or "ongoing", use today's date: {today_str}.
        4. Sum all durations to calculate total experience in **years**, rounding to one decimal place.

        📌 Final Output Instruction:
        Give only a number. Do not include any units (like 'years') or explanations. Just return the number.

        Examples:
        ✅ 3.5
        ✅ 5

        ❌ Not allowed: "The candidate has 5 years of experience"
        ❌ Not allowed: "Experience: 4.2 years"
        """
        output = Experience_cal_llm.invoke(prompt2)
        return {"calculated_experience": output.years_of_experience}


    # ============================================
    # --------- Graph Defination ------
    # ============================================ 

    graph = StateGraph(ResumeProcessingState)

    graph.add_node("email_classifier",email_classifier)
    graph.add_node("extract_data",extract_data)
    graph.add_node("resume_parser",resume_parser)
    # graph.add_node("job_title_extractor",job_title_extractor)
    graph.add_node("aggregator",aggregator)
    # graph.add_node("fork_resume_job", fork_resume_and_job_title)
    graph.add_node("calc_expereince",calc_expereince)


    graph.add_edge(START,"email_classifier")
    graph.add_conditional_edges("email_classifier",check_classifier,  {False: END, True: "extract_data"})
    # graph.add_edge("fork_resume_job", "extract_data")
    # graph.add_edge("fork_resume_job", "job_title_extractor")
    graph.add_edge("extract_data", "resume_parser")
    graph.add_edge("extract_data", "calc_expereince")
    graph.add_edge("resume_parser","aggregator")
    graph.add_edge("calc_expereince","aggregator")
    # graph.add_edge("job_title_extractor","aggregator")
    graph.add_edge("aggregator",END)


    # checkpointer = InMemorySaver()

    compiled_graph = graph.compile()


# ============================================
# --------- Execution ------
# ============================================ 

    container_name = "resume"
    blob_path =readfile.name
    blob_name = blob_path.split("/", 1)[-1]
    
    try:
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        properties = blob_client.get_blob_properties()
        #     blob_url = blob_client.url
        #     logging.info(f"Blob URL: {blob_url}")
        #     return blob_client  # Return the blob client
    except Exception as e:
        logging.error(f"Error downloading blob: {e}")

    initial_state = {"blob_name":properties['name'],"blob_client": blob_client,"subject":properties['metadata']['subject'],"body":properties['metadata']['snippet']}

    output = compiled_graph.invoke(initial_state)
    logging.info(output['final_output'])
    # output = {"name": "ChatGPT"}


    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "PostmanRuntime/7.48.0"
    }
    try:
        response = requests.post(
            APEX_URL,
            headers=headers,
            # data=json.dumps(output, default=json_default),
            data=json.dumps(output['final_output'], default=json_default),  # ✅ use custom encoder
            timeout=30,
            verify=True,
            allow_redirects=True
        )
        # 3. Check for successful status codes (typically 200-299)
        # print("Status Code:", response.status_code)
        # print("Response:", response.text)
        if response.status_code >= 200 and response.status_code < 300:
                logging.info(f"✅ APEX POST successful. Status Code: {response.status_code}")
        else:
            # 4. Log detailed HTTP error for non-success codes
            error_message = f"❌ APEX POST FAILED: Status {response.status_code}. Response: {response.text[:500]}"
            logging.error(error_message)
            
            # Optionally, you can raise an exception here to mark the Azure Function as failed
            # raise Exception(error_message)

    except requests.exceptions.RequestException as e:
        # 5. Handle connection, timeout, or other request-level errors
        logging.critical(f"🔥 CRITICAL API CONNECTION ERROR: {e}")
        
        # Optionally, raise exception to fail the function and retry if configured
        # raise e
# "APEX_URL" : "https://oracleapex.com/ords/ifl/resume_api/insert_resume