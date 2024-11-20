import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
from typing import List, Optional, Dict
from typing_extensions import TypedDict
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_community.document_loaders import WebBaseLoader
# # Load environment variables
load_dotenv()

# Environment variables configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASS = os.getenv('GMAIL_PASS')
# Initialize services
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.5,
)
llm.invoke("hi how are you?")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Pydantic models for data validation
class SpecsComparison(BaseModel):
    processor: str = Field(..., description="Processor type and model")
    battery: str = Field(..., description="Battery capacity and type")
    camera: str = Field(..., description="Camera specs")
    display: str = Field(..., description="Display specifications")
    storage: str = Field(..., description="Storage options")

class RatingsComparison(BaseModel):
    overall_rating: float = Field(..., description="Overall rating out of 5")
    performance: float = Field(..., description="Performance rating")
    battery_life: float = Field(..., description="Battery life rating")
    camera_quality: float = Field(..., description="Camera quality rating")
    display_quality: float = Field(..., description="Display quality rating")

class Comparison(BaseModel):
    product_name: str = Field(..., description="Product name")
    specs_comparison: SpecsComparison
    ratings_comparison: RatingsComparison
    reviews_summary: str = Field(..., description="Review summary")

class BestProduct(BaseModel):
    product_name: str = Field(..., description="Best product name")
    justification: str = Field(..., description="Justification")

class ProductComparison(BaseModel):
    comparisons: List[Comparison]
    best_product: BestProduct

class SmartphoneReview(BaseModel):
    title: str = Field(..., description="Review title")
    url: Optional[str] = Field(None, description="Review URL")
    content: Optional[str] = Field(None, description="Review content")
    pros: Optional[List[str]] = Field(None, description="Pros")
    cons: Optional[List[str]] = Field(None, description="Cons")
    highlights: Optional[dict] = Field(None, description="Highlights")
    score: Optional[float] = Field(None, description="Score")

class ListOfSmartphoneReviews(BaseModel):
    reviews: List[SmartphoneReview]

class EmailRecommendation(BaseModel):
    subject: str
    heading: str
    justification_line: str

class State(TypedDict):
    query: str
    email: str
    products: list[dict]
    product_schema: list[SmartphoneReview]
    blogs_content: Optional[List[dict]]
    best_product: dict
    comparison: list
    youtube_link: str

# Email templates
email_template_prompt = """
You are an expert email content writer.

Generate an email recommendation based on the following inputs:
- Product Name: {product_name}
- Justification Line: {justification_line}
- User Query: "{user_query}" (a general idea of the user's interest, such as "a smartphone for photography" or "a premium gaming laptop").

Return your output in the following JSON format:
{format_instructions}

### Input Example:
Product Name: Google Pixel 8 Pro
Justification Line: Praised for its exceptional camera, advanced AI capabilities, and vibrant display.
User Query: a phone with an amazing camera

### Example Output:
{{
  "subject": "Capture Every Moment with Google Pixel 8 Pro",
  "heading": "Discover the Power of the Ultimate Photography Smartphone",
  "justification_line": "Known for its exceptional camera quality, cutting-edge AI features, and vibrant display, the Google Pixel 8 Pro is perfect for photography enthusiasts."
}}

Now generate the email recommendation based on the inputs provided.
"""
email_html_template = """
<!DOCTYPE html>
<html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }
            .email-container {
                max-width: 600px;
                margin: 20px auto;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            .header {
                background-color: #007BFF;
                color: #ffffff;
                padding: 20px;
                text-align: center;
            }
            .content {
                padding: 20px;
            }
            .button {
                display: inline-block;
                margin-top: 20px;
                background-color: #007BFF;
                color: #ffffff;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="email-container">
            <div class="header">
                <h1>{heading}</h1>
            </div>
            <div class="content">
                <h2>Our Top Pick: {product_name}</h2>
                <p>{justification}</p>
                <p>Watch our in-depth review:</p>
                <a href="{youtube_link}" class="button" target="_blank">Watch Review</a>
            </div>
        </div>
    </body>
</html>
"""

# Helper functions
def load_blog_content(page_url):
    try:
        loader = WebBaseLoader(web_paths=[page_url],bs_get_text_kwargs={"separator": " ", "strip": True})
        loaded_content = loader.load()
        return " ".join([doc.page_content for doc in loaded_content])
    except Exception as e:
        print(f"Error loading blog content: {e}")
        return ""

def send_email(recipient_email, subject, body):
    try:
        message = MIMEMultipart()
        message['From'] = GMAIL_USER
        message['To'] = recipient_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'html'))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(message)
            print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Node functions
def tavily_search_node(state):
    try:
        query = state.get('query')
        response = tavily_client.search(query=query, max_results=1)
        blogs_content = []
        
        for blog in response.get('results', []):
            content = load_blog_content(blog.get("url", ""))
            if content:
                blogs_content.append({
                    "title": blog.get("title", ""),
                    "url": blog.get("url", ""),
                    "content": content,
                    "score": blog.get("score", "")
                })
        
        return {"blogs_content": blogs_content}
    except Exception as e:
        print(f"Tavily search error: {e}")
        return {"blogs_content": []}
# Detailed prompt templates
schema_prompt_template = """
You are a professional assistant tasked with extracting structured information from blogs.

### Instructions:

1. **Product Details**: For each product mentioned in the blog post, populate the `products` array with structured data for each item, including:
   - `title`: The product name.
   - `url`: Link to the blog post or relevant page.
   - `content`: A concise summary of the product's main features or purpose.
   - `pros`: A list of positive aspects or advantages of the product. If available otherwise extract blog content.
   - `cons`: A list of negative aspects or disadvantages. If available otherwise extract blog content.
   - `highlights`: A dictionary containing notable features or specifications. If available otherwise extract blog content.
   - `score`: A numerical rating score if available; otherwise, use `0.0`.

### Blogs Contents: {blogs_content}

After extracting all information, just return the response in the JSON structure given below. Do not add any extracted information. The JSON should be in a valid structure with no extra characters inside, like Python's \\n.
"""

comparison_prompt_template = """
You are tasked with comparing different products and selecting the best one based on comprehensive analysis.

1. **List of Products for Comparison (`comparisons`):**
   - Each product should include:
     - **Product Name**: The name of the product (e.g., "Smartphone A").
     - **Specs Comparison**:
       - **Processor**: Type and model of the processor (e.g., "Snapdragon 888").
       - **Battery**: Battery capacity and type (e.g., "4500mAh").
       - **Camera**: Camera specifications (e.g., "108MP primary").
       - **Display**: Display type, size, and refresh rate (e.g., "6.5 inch OLED, 120Hz").
       - **Storage**: Storage options and whether it is expandable (e.g., "128GB, expandable").
     - **Ratings Comparison**:
       - **Overall Rating**: Overall rating out of 5 (e.g., 4.5).
       - **Performance**: Rating for performance out of 5 (e.g., 4.7).
       - **Battery Life**: Rating for battery life out of 5 (e.g., 4.3).
       - **Camera Quality**: Rating for camera quality out of 5 (e.g., 4.6).
       - **Display Quality**: Rating for display quality out of 5 (e.g., 4.8).
     - **Reviews Summary**: Summary of key points from user reviews that highlight the strengths and weaknesses of this product.

2. **Best Product Selection (`best_product`):**
   - **Product Name**: Select the best product among the compared items.
   - **Justification**: Provide a brief explanation of why this product is considered the best choice. This should be based on factors such as:
     - Balanced performance across all aspects
     - High user ratings and positive reviews
     - Advanced specifications and features
     - Value for money
     - Unique selling points or innovations

Please analyze the following product data and provide a structured comparison:

{product_data}

Return your analysis in the following format:
{format_instructions}
"""
# Update the node functions to use these detailed prompts
def schema_mapping_node(state: State):
    try:
        if not state.get("blogs_content"):
            return {"product_schema": []}

        parser = JsonOutputParser(pydantic_object=ListOfSmartphoneReviews)
        prompt = PromptTemplate(
            template=schema_prompt_template,
            input_variables=["blogs_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Implement retry mechanism
        max_retries = 2
        wait_time = 60
        
        for attempt in range(1, max_retries + 1):
            try:
                chain = prompt | llm | parser
                response = chain.invoke({"blogs_content": state["blogs_content"]})
                
                if response and len(response['products']) > 1:
                    return {"product_schema": response['products']}
                else:
                    print(f"Attempt {attempt}: Invalid response structure")
                    
                if attempt < max_retries:
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(wait_time)
        
        return {"product_schema": []}
        
    except Exception as e:
        print(f"Schema mapping error: {e}")
        return {"product_schema": []}

def product_comparison_node(state: State):
    try:
        if not state.get("product_schema"):
            return {"comparison": [], "best_product": {}}

        parser = JsonOutputParser(pydantic_object=ProductComparison)
        prompt = PromptTemplate(
            template=comparison_prompt_template,
            input_variables=["product_data"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | llm | parser
        response = chain.invoke({
            "product_data": json.dumps(state['product_schema'])
        })
        
        if not response:
            return {"comparison": [], "best_product": {}}
            
        return {
            "comparison": response["comparisons"],
            "best_product": response["best_product"]
        }
    except Exception as e:
        print(f"Product comparison error: {e}")
        return {"comparison": [], "best_product": {}}

def youtube_review_node(state: State):
    try:
        best_product_name = state.get("best_product", {}).get("product_name")
        if not best_product_name:
            return {"youtube_link": None}

        search_response = youtube.search().list(
            q=f"{best_product_name} review",
            part="snippet",
            type="video",
            maxResults=1
        ).execute()

        video_items = search_response.get("items", [])
        if not video_items:
            return {"youtube_link": None}

        video_id = video_items[0]["id"]["videoId"]
        return {"youtube_link": f"https://www.youtube.com/watch?v={video_id}"}
    except Exception as e:
        print(f"YouTube review error: {e}")
        return {"youtube_link": None}

def display_node(state: State):
    return {
        "products": state.get("product_schema", []),
        "best_product": state.get("best_product", {}),
        "comparison": state.get("comparison", []),
        "youtube_link": state.get("youtube_link")
    }

def send_email_node(state: State):
    try:
        if not state.get("best_product"):
            return state

        parser = JsonOutputParser(pydantic_object=EmailRecommendation)
        prompt = PromptTemplate(
            template=email_template_prompt,
            input_variables=["product_name", "justification_line", "user_query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | llm | parser
        response = chain.invoke({
            "product_name": state["best_product"]["product_name"],
            "justification_line": state["best_product"]["justification"],
            "user_query": state["query"]
        })
        
        html_content = email_html_template.format(
            product_name=state["best_product"]["product_name"],
            justification=response["justification_line"],
            youtube_link=state["youtube_link"],
            heading=response["heading"]
        )
        
        print("Sending email to:", state["email"])  # Debugging line
        send_email(state["email"], response['subject'], html_content)
        return state
    except Exception as e:
        print(f"Email sending error: {e}")  # This will show the error in the terminal
        return state

# Build the workflow
def build_workflow():
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("tavily_search", tavily_search_node)
    builder.add_node("schema_mapping", schema_mapping_node)
    builder.add_node("product_comparison", product_comparison_node)
    builder.add_node("youtube_review", youtube_review_node)
    builder.add_node("display", display_node)
    builder.add_node("send_email", send_email_node)
    
    # Add edges
    builder.add_edge(START, "tavily_search")
    builder.add_edge("tavily_search", "schema_mapping")
    builder.add_edge("schema_mapping", "product_comparison")
    builder.add_edge("product_comparison", "youtube_review")
    builder.add_edge("youtube_review", "send_email")
    builder.add_edge("send_email", "display")
    builder.add_edge("display", END)
    
    return builder.compile()

def main():
    st.set_page_config(page_title="ShopGenie", page_icon=":mag:")
    st.title("🔍ShopGenie")

    # User input
    query = st.text_input("Enter your product search query", placeholder="Best smartphones under $1000")
    email = st.text_input("Enter your email", placeholder="your.email@example.com")

    # Progress and result containers
    progress_container = st.container()
    result_container = st.container()

    if st.button("Search Products", type="primary"):
        if not query or not email:
            st.error("Please enter both a query and an email.")
            return

        # Clear previous results
        result_container.empty()
        
        # Show progress
        with progress_container:
            progress_text = st.empty()
            progress_bar = st.progress(0)

        try:
            # Initialize the workflow
            graph = build_workflow()
            initial_state = {
                "query": query,
                "email": email
            }

            # Run the workflow with progress updates
            full_results = {}
            for i, event in enumerate(graph.stream(input=initial_state, stream_mode="updates")):
                # Update progress
                progress = min((i + 1) * 20, 100)
                progress_bar.progress(progress)
                
                stages = {
                    0: "Searching for relevant content",
                    1: "Extracting product information",
                    2: "Comparing products",
                    3: "Finding video reviews",
                    4: "Preparing final results"
                }
                progress_text.text(stages.get(i, "Processing"))
                print(event)
                # Accumulate results
                full_results.update(event)

            # Clear progress
            progress_container.empty()

            # Display results in result container
            with result_container:
                if full_results.get("products"):
                    st.subheader("🏆 Product Comparisons")
                    for product in full_results["products"]:
                        with st.expander(f"{product.get('title', 'Unknown Product')}"):
                            st.write(f"**Pros:** {', '.join(product.get('pros', []))}")
                            st.write(f"**Cons:** {', '.join(product.get('cons', []))}")
                            st.write(f"**Score:** {product.get('score', 'N/A')}")

                if full_results.get("best_product"):
                    best_product = full_results["best_product"]
                    st.subheader("🌟 Top Recommendation")
                    st.markdown(f"### {best_product.get('product_name', 'Best Product')}")
                    st.write(f"**Why it's the best:** {best_product.get('justification', 'No details available')}")
                    
                    if full_results.get("youtube_link"):
                        st.video(full_results["youtube_link"])

                st.success("Analysis complete! An email with detailed results has been sent.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
