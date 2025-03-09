import streamlit as st
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pyArango.connection import Connection
from arango import ArangoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import ArangoGraph
from langchain.chains import ArangoGraphQAChain
from langchain_core.prompts import ChatPromptTemplate
import os

# Initialize Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-think-exp",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ArangoDB Connection
def get_arangodb_connection():
    return Connection(
        arangoURL=os.environ["ARANGO_URL"],
        username=os.environ["ARANGO_USER"],
        password=os.environ["ARANGO_PASSWORD"]
        dbName=os.environ["DB_NAME"]
    )

# Streamlit App Configuration
st.set_page_config(page_title="Shark Tank Analytics", layout="wide")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page Layout
st.title("🦈 Shark Tank Analytics Platform")
tab1, tab2 = st.tabs(["Chat Interface", "Analytics Dashboard"])

# Chat Tab
with tab1:
    st.header("Chat with Shark Tank Data")
    
    # Chat Input
    user_query = st.chat_input("Ask about investments, sharks, or startups...")
    
    # Initialize ArangoGraph
    db = ArangoClient(hosts=arangoURL).db(dbName,
     username, password, verify=True)
   
    graph = ArangoGraph(db)
    
    # LangChain Q&A Chain
    chain = ArangoGraphQAChain.from_llm(llm, graph=graph)
    
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Process Query
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Process with LangChain
        try:
            response = chain.run(user_query)
            
            with st.chat_message("assistant"):
                # Enhanced response formatting
                if isinstance(response, dict):
                    if "result" in response:
                        if isinstance(response["result"], list):
                            df = pd.DataFrame(response["result"])
                            st.dataframe(df)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Found {len(df)} results"
                            })
                        else:
                            st.write(response["result"])
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response["result"]
                            })
                    else:
                        st.write(response)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": str(response)
                        })
                else:
                    st.write(response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {str(e)}"
            })

# Dashboard Tab
with tab2:
    st.header("Investment Analytics Dashboard")
    
    # Fetch data for visualizations
    conn = get_arangodb_connection()
    db = conn["shark_database"]
    
    # Investment Amount Distribution
    st.subheader("Investment Distribution")
    investment_query = """
    FOR investment IN investments
        RETURN investment.investment_amount
    """
    investments = db.AQLQuery(investment_query, rawResults=True)
    if investments:
        fig = px.histogram(
            x=investments,
            nbins=20,
            labels={"x": "Investment Amount"},
            title="Distribution of Investment Amounts"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Shark Activity
    st.subheader("Shark Investment Activity")
    shark_query = """
    FOR shark IN investors
        LET deals = (
            FOR v IN 1..1 OUTBOUND shark investments
                RETURN v
        )
        RETURN {
            shark: shark.name,
            deal_count: LENGTH(deals),
            total_invested: SUM(deals[*].investment_amount)
        }
    """
    shark_data = db.AQLQuery(shark_query, rawResults=True)
    if shark_data:
        df = pd.DataFrame(shark_data)
        fig = px.bar(
            df,
            x="shark",
            y=["deal_count", "total_invested"],
            title="Shark Investment Activity",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Network Visualization
    st.subheader("Investor-Startup Network")
    graph_query = """
    FOR edge IN investments
        LIMIT 50  # Limit for performance
        RETURN {
            source: SPLIT(edge._from, '/')[1],
            target: SPLIT(edge._to, '/')[1],
            amount: edge.investment_amount
        }
    """
    edges = db.AQLQuery(graph_query, rawResults=True)
    if edges:
        G = nx.from_pandas_edgelist(
            pd.DataFrame(edges),
            source="source",
            target="target",
            edge_attr="amount"
        )
        
        # Plotly network visualization
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=text,
            textposition="bottom center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[],
                line_width=2))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        st.plotly_chart(fig, use_container_width=True)

# Instructions for Setup
"""
To run this app:

1. Install dependencies:
pip install streamlit langchain-google-genai pyArango plotly networkx

2. Create a .streamlit/secrets.toml file with:
[ARANGO_URL]
ARANGO_USER = "root"
ARANGO_PASSWORD = "your_password"

3. Run with:
streamlit run shark_tank_app.py
"""