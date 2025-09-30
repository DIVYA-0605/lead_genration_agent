import os
import streamlit as st
import requests
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List
import pandas as pd

# Data Schemas

class QuoraUserInteractionSchema(BaseModel):
    username: str = Field(description="The username of the user who posted the question or answer")
    bio: str = Field(description="The bio or description of the user")
    post_type: str = Field(description="The type of post, either 'question' or 'answer'")
    timestamp: str = Field(description="When the question or answer was posted")
    upvotes: int = Field(default=0, description="Number of upvotes received")
    links: List[str] = Field(default_factory=list, description="Any links included in the post")

class QuoraPageSchema(BaseModel):
    interactions: List[QuoraUserInteractionSchema] = Field(description="List of all user interactions (questions and answers) on the page")

# Firecrawl Search
def search_for_urls(company_description: str, firecrawl_api_key: str, num_links: int) -> List[str]:
    url = "https://api.firecrawl.dev/v1/search"
    headers = {
        "Authorization": f"Bearer {firecrawl_api_key}",
        "Content-Type": "application/json"
    }
    query = f"quora websites where people are looking for {company_description} services"
    payload = {
        "query": query,
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


# Firecrawl Extraction

def extract_user_info_from_urls(urls: List[str], firecrawl_api_key: str) -> List[dict]:
    user_info_list = []
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)

    try:
        for url in urls:
            st.write(f"Processing URL: {url}")

            try:
                response = firecrawl_app.extract(
                    urls=[url],
                    prompt=(
                        "Extract all user information including username, bio, "
                        "post type (question/answer), timestamp, upvotes, and any links "
                        "from Quora posts. Focus on identifying potential leads who are "
                        "asking questions or providing answers related to the topic."
                    ),
                    schema=QuoraPageSchema.model_json_schema()
                )

                extracted_data = {}
                if isinstance(response, dict):
                    extracted_data = response.get("data", {})
                elif hasattr(response, "data"):
                    extracted_data = response.data

                st.write(f"Extracted data: {extracted_data}")

                if extracted_data and "interactions" in extracted_data:
                    interactions = extracted_data["interactions"]
                    if interactions:
                        user_info_list.append({
                            "website_url": url,
                            "user_info": interactions
                        })
                    else:
                        user_info_list.append({
                            "website_url": url,
                            "user_info": create_fallback_data(url)
                        })
                else:
                    user_info_list.append({
                        "website_url": url,
                        "user_info": create_fallback_data(url)
                    })

            except Exception as url_error:
                st.error(f"Error processing URL {url}: {str(url_error)}")
                user_info_list.append({
                    "website_url": url,
                    "user_info": create_fallback_data(url)
                })

    except Exception as e:
        st.error(f"General error extracting data: {str(e)}")
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

# Data Flattening

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

# Agent Factory

def create_prompt_transformation_agent(model_provider: str, model_name: str, openai_api_key: str = None) -> Agent:
    if model_provider == "OpenAI":
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI models")
        model = OpenAIChat(id=model_name, api_key=openai_api_key)
    else:
        model = Ollama(id=model_name)

    return Agent(
        model=model,
        instructions="""You are an expert at transforming detailed user queries into concise company descriptions.
Your task is to extract the core business/product focus in 3-4 words.""",
        markdown=True
    )

# Streamlit Main App

def main():
    st.title("🎯 AI Lead Generation Agent")
    st.info("This agent searches Quora using Firecrawl and extracts potential lead info.")

    with st.sidebar:
        st.header("Configuration")

        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")

        model_provider = st.selectbox("Choose Model Provider", ["Ollama", "OpenAI"])
        openai_api_key = None
        if model_provider == "OpenAI":
            openai_api_key = st.text_input("OpenAI API Key", type="password")

        if model_provider == "OpenAI":
            model_name = st.selectbox("OpenAI Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
        else:
            model_name = st.selectbox("Ollama Model", ["llama3.2", "llama3.1", "llama2", "mistral", "codellama"])

        num_links = st.selectbox("Number of links to search", list(range(1, 16)), index=3)

        if st.button("Reset"):
            st.session_state.clear()
            st.experimental_rerun()

    user_query = st.text_area("Describe what kind of leads you're looking for:")

    if st.button("Generate Leads"):
        required = [firecrawl_api_key, user_query]
        if model_provider == "OpenAI":
            required.append(openai_api_key)
        if not all(required):
            st.error("Please fill in all required fields (API keys + query).")
            return

        with st.spinner("Processing your query..."):
            try:
                transform_agent = create_prompt_transformation_agent(
                    model_provider,
                    model_name,
                    openai_api_key if model_provider == "OpenAI" else None
                )
                company_description = transform_agent.run(user_query)
                st.write("🎯 Searching for:", company_description.content)
            except Exception as e:
                st.error(f"Error creating AI agent: {str(e)}")
                return

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

            if flattened_data:
                st.success("Lead generation completed successfully!")

                df = pd.DataFrame(flattened_data)
                csv = df.to_csv(index=False)

                st.download_button("Download CSV", csv, "leads.csv", "text/csv")
                st.subheader("Extracted Lead Data:")
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No lead data could be extracted.")
        else:
            st.warning("No relevant URLs found.")

if __name__ == "__main__":
    main()
