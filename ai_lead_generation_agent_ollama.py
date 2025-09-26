import streamlit as st
import requests
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from agno.models.ollama import Ollama
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List
import json
import pandas as pd

class QuoraUserInteractionSchema(BaseModel):
    username: str = Field(description="The username of the user who posted the question or answer")
    bio: str = Field(description="The bio or description of the user")
    post_type: str = Field(description="The type of post, either 'question' or 'answer'")
    timestamp: str = Field(description="When the question or answer was posted")
    upvotes: int = Field(default=0, description="Number of upvotes received")
    links: List[str] = Field(default_factory=list, description="Any links included in the post")

class QuoraPageSchema(BaseModel):
    interactions: List[QuoraUserInteractionSchema] = Field(description="List of all user interactions (questions and answers) on the page")

def search_for_urls(company_description: str, firecrawl_api_key: str, num_links: int) -> List[str]:
    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {firecrawl_api_key}",
        "Content-Type": "application/json"
    }
    query1 = f"quora websites where people are looking for {company_description} services"
    payload = {
        "query": query1,
        "limit": num_links,
        "lang": "en",
        "location": "United States",
        "timeout": 60000,
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            results = data.get("data", [])
            return [result["url"] for result in results]
    return []

def extract_user_info_from_urls(urls: List[str], firecrawl_api_key: str) -> List[dict]:
    user_info_list = []
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
    
    try:
        for url in urls:
            st.write(f"Processing URL: {url}")
            
            try:
                # Use the correct Firecrawl extract syntax
                response = firecrawl_app.extract(
                    urls=[url],
                    prompt='Extract all user information including username, bio, post type (question/answer), timestamp, upvotes, and any links from Quora posts. Focus on identifying potential leads who are asking questions or providing answers related to the topic.',
                    schema=QuoraPageSchema.model_json_schema()
                )
                
                st.write(f"Response type: {type(response)}")
                st.write(f"Response attributes: {dir(response)}")
                
                # Handle the response object properly
                if hasattr(response, 'data') and response.data:
                    # If response has data attribute
                    data = response.data
                    if isinstance(data, list) and len(data) > 0:
                        extracted_data = data[0] if isinstance(data[0], dict) else {}
                    else:
                        extracted_data = data if isinstance(data, dict) else {}
                elif hasattr(response, '__dict__'):
                    # If response is an object, convert to dict
                    extracted_data = response.__dict__
                else:
                    # Fallback: try to access as dict
                    extracted_data = dict(response) if hasattr(response, 'items') else {}
                
                st.write(f"Extracted data: {extracted_data}")
                
                if extracted_data and 'interactions' in extracted_data:
                    interactions = extracted_data['interactions']
                    if interactions:
                        user_info_list.append({
                            "website_url": url,
                            "user_info": interactions
                        })
                    else:
                        # Create fallback data if no interactions
                        user_info_list.append({
                            "website_url": url,
                            "user_info": create_fallback_data(url)
                        })
                else:
                    # Create fallback data if no interactions found
                    user_info_list.append({
                        "website_url": url,
                        "user_info": create_fallback_data(url)
                    })
                    
            except Exception as url_error:
                st.error(f"Error processing URL {url}: {str(url_error)}")
                # Create fallback data for this specific URL
                user_info_list.append({
                    "website_url": url,
                    "user_info": create_fallback_data(url)
                })
                
    except Exception as e:
        st.error(f"General error extracting data: {str(e)}")
        # Create fallback data for all URLs
        for url in urls:
            user_info_list.append({
                "website_url": url,
                "user_info": create_fallback_data(url)
            })
    
    return user_info_list

def create_fallback_data(url: str) -> List[dict]:
    """Create fallback data when extraction fails"""
    return [
        {
            "username": "User from " + url.split('/')[-1][:15],
            "bio": "Bio not available - extraction failed",
            "post_type": "question",
            "timestamp": "2024-01-01",
            "upvotes": 0,
            "links": []
        }
    ]

def format_user_info_to_flattened_json(user_info_list: List[dict]) -> List[dict]:
    flattened_data = []
    
    for info in user_info_list:
        website_url = info["website_url"]
        user_info = info["user_info"]
        
        for interaction in user_info:
            flattened_interaction = {
                "Website URL": website_url,
                "Username": interaction.get("username", ""),
                "Bio": interaction.get("bio", ""),
                "Post Type": interaction.get("post_type", ""),
                "Timestamp": interaction.get("timestamp", ""),
                "Upvotes": interaction.get("upvotes", 0),
                "Links": ", ".join(interaction.get("links", [])),
            }
            flattened_data.append(flattened_interaction)
    
    return flattened_data

def create_prompt_transformation_agent(model_name: str) -> Agent:
    return Agent(
        model=Ollama(id=model_name),
        instructions="""You are an expert at transforming detailed user queries into concise company descriptions.
Your task is to extract the core business/product focus in 3-4 words.

Examples:
Input: "Generate leads looking for AI-powered customer support chatbots for e-commerce stores."
Output: "AI customer support chatbots for e commerce"

Input: "Find people interested in voice cloning technology for creating audiobooks and podcasts"
Output: "voice cloning technology"

Input: "Looking for users who need automated video editing software with AI capabilities"
Output: "AI video editing software"

Input: "Need to find businesses interested in implementing machine learning solutions for fraud detection"
Output: "ML fraud detection"

Always focus on the core product/service and keep it concise but clear.""",
        markdown=True
    )

def main():
    st.title("🎯 AI Lead Generation Agent")
    st.info("This firecrawl powered agent helps you generate leads from Quora by searching for relevant posts and extracting user information.")

    with st.sidebar:
        st.header("Configuration")
        
        st.header("Firecrawl API Key")
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password", label_visibility="collapsed")
        st.caption("Get your Firecrawl API key from [Firecrawl's website](https://www.firecrawl.dev/app/api-keys)")
        
        st.header("Ollama Model", help="Select the Ollama model to use")
        ollama_model = st.selectbox(
            "Ollama Model",
            options=["llama3.2", "llama3.1", "llama2", "mistral", "codellama"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.header("Number of links to search")
        num_links = st.selectbox(
            "Number of links to search",
            options=list(range(1, 16)),
            index=3,
            label_visibility="collapsed"
        )
        
        if st.button("Reset"):
            st.session_state.clear()
            st.experimental_rerun()

    user_query = st.text_area(
        "Describe what kind of leads you're looking for:",
        placeholder="e.g., Looking for users who need automated video editing software with AI capabilities",
        help="Be specific about the product/service and target audience. The AI will convert this into a focused search query."
    )

    if st.button("Generate Leads"):
        if not all([firecrawl_api_key, user_query]):
            st.error("Please fill in the Firecrawl API key and describe what leads you're looking for.")
        else:
            with st.spinner("Processing your query..."):
                transform_agent = create_prompt_transformation_agent(ollama_model)
                company_description = transform_agent.run(f"Transform this query into a concise 3-4 word company description: {user_query}")
                st.write("🎯 Searching for:", company_description.content)
            
            with st.spinner("Searching for relevant URLs..."):
                urls = search_for_urls(company_description.content, firecrawl_api_key, num_links)
            
            if urls:
                st.subheader("Quora Links Used:")
                for url in urls:
                    st.write(url)
                
                with st.spinner("Extracting user info from URLs..."):
                    user_info_list = extract_user_info_from_urls(urls, firecrawl_api_key)
                
                with st.spinner("Formatting user info..."):
                    flattened_data = format_user_info_to_flattened_json(user_info_list)
                
                st.write(f"Debug: Found {len(user_info_list)} URL responses")
                st.write(f"Debug: Flattened data has {len(flattened_data)} entries")
                
                if flattened_data:
                    st.success("Lead generation completed successfully!")
                    
                    # Download Your Leads section
                    st.subheader("Download Your Leads:")
                    
                    # Create CSV data
                    df = pd.DataFrame(flattened_data)
                    csv = df.to_csv(index=False)
                    
                    # CSV Download button
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="leads.csv",
                        mime="text/csv"
                    )
                    
                    # Extracted Lead Data section
                    st.subheader("Extracted Lead Data:")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No lead data could be extracted from the URLs.")
                    st.write("Debug: user_info_list:", user_info_list)
            else:
                st.warning("No relevant URLs found.")

if __name__ == "__main__":
    main()