# Shark Tank Analytics Platform

This Streamlit application provides a platform for analyzing and interacting with Shark Tank investment data.

## Features

* **Chat Interface:** Ask questions about Shark Tank data in natural language using Google's Gemini LLM and an ArangoDB graph database.
* **Analytics Dashboard:** Explore interactive visualizations of investment distributions, shark activity, and the investor-startup network.

## Setup Instructions

1. **Install Dependencies:**

   ```bash
   pip install streamlit arango langchain langchain-community langchain-google-genai pyArango plotly networkx

2. **Environment Variable**
[ARANGO_URL] ="your_url"
[ARANGO_USER] = "root"
[ARANGO_PASSWORD] = "your_password"
[DB_NAME] = "your_db"
[GOOGLE_API_KEY]="your_gemini_api_key
