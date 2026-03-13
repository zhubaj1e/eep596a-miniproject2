import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone as PineconeClient
from langchain_openai import OpenAIEmbeddings

# ---- Constants ----
MODEL = "gpt-4.1-nano"
INDEX_NAME = "machine-learning-textbook"
NAMESPACE = "ns1000"


# ============================================================
# Agent Classes
# ============================================================

class Obnoxious_Agent:
    """Checks if a query is obnoxious. (No Langchain)"""
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = (
            "You are a content-moderation assistant. "
            "Determine whether the following user query is obnoxious, offensive, "
            "rude, hateful, or contains inappropriate language. "
            "Respond with ONLY 'Yes' if the query is obnoxious, or 'No' if it is not. "
            "Do not provide any explanation."
        )

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        text = response.choices[0].message.content.strip().lower()
        return text.startswith("yes")

    def check_query(self, query):
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=10,
        )
        return self.extract_action(response)


class Context_Rewriter_Agent:
    """Resolves ambiguities for multi-turn conversations."""
    def __init__(self, openai_client):
        self.client = openai_client

    def rephrase(self, user_history, latest_query):
        if not user_history:
            return latest_query
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in user_history]
        )
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query-rewriting assistant. Given the conversation "
                        "history and the user's latest message, rewrite the latest "
                        "message as a fully self-contained query that resolves all "
                        "pronouns and references. Output ONLY the rewritten query, "
                        "nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_text}\n\n"
                        f"Latest query: {latest_query}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()


class Query_Agent:
    """Checks query relevance and retrieves documents from Pinecone."""
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.prompt = (
            "You are a relevance-checking assistant for a Machine Learning textbook. "
            "Given a user query, decide if it is related to machine learning, "
            "data science, statistics, artificial intelligence, or any academic "
            "topic covered in a typical ML textbook. "
            "Respond with ONLY 'Relevant' or 'Irrelevant'. No explanation."
        )

    def query_vector_store(self, query, k=5):
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            namespace=NAMESPACE,
            include_metadata=True,
        )
        docs = []
        for match in results["matches"]:
            docs.append(
                {
                    "text": match["metadata"]["text"],
                    "page_number": match["metadata"].get("page_number", "N/A"),
                    "score": match["score"],
                }
            )
        return docs

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response, query=None):
        text = response.choices[0].message.content.strip().lower()
        return "relevant" in text


class Answering_Agent:
    """Generates answers using retrieved context."""
    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def generate_response(self, query, docs, conv_history, k=5):
        context = "\n\n".join([d["text"] for d in docs[:k]])
        history_msgs = []
        if conv_history:
            for m in conv_history:
                history_msgs.append({"role": m["role"], "content": m["content"]})
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the "
                    "provided context from a machine learning textbook. Use the context "
                    "to give accurate and informative answers. If the context doesn't "
                    "fully cover the question, say so while sharing what you can."
                ),
            },
        ]
        messages.extend(history_msgs)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Context from the textbook:\n{context}\n\n"
                    f"Question: {query}\n\nAnswer:"
                ),
            }
        )
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()


class Relevant_Documents_Agent:
    """Checks if retrieved documents are relevant to the query. (No Langchain)"""
    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def get_relevance(self, query, documents) -> str:
        docs_text = "\n\n".join(
            [f"[Doc {i+1}]: {d['text'][:500]}" for i, d in enumerate(documents)]
        )
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance-assessment assistant. Given a user query "
                        "and a set of retrieved document chunks, determine if the "
                        "documents are relevant to answering the query. "
                        "Respond with ONLY 'Relevant' or 'Irrelevant'. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nRetrieved Documents:\n{docs_text}"
                    ),
                },
            ],
            temperature=0,
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip().lower()
        return "relevant" in text


class Head_Agent:
    """Controller agent that orchestrates all sub-agents."""
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        self.client = OpenAI(api_key=openai_key)
        pc = PineconeClient(api_key=pinecone_key)
        self.pinecone_index = pc.Index(pinecone_index_name)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=openai_key
        )
        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.context_rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(
            self.pinecone_index, self.client, self.embeddings
        )
        self.relevant_docs_agent = Relevant_Documents_Agent(self.client)
        self.answering_agent = Answering_Agent(self.client)

    def process_query(self, query, history=None):
        if history is None:
            history = []
        agent_path = []

        # Step 1: Check for obnoxious content
        agent_path.append("Obnoxious_Agent")
        if self.obnoxious_agent.check_query(query):
            return {
                "response": (
                    "I'm sorry, but your query contains inappropriate language. "
                    "Please rephrase your question respectfully and I'll be happy to help."
                ),
                "agent_path": " -> ".join(agent_path),
            }

        # Step 2: Handle greetings / small talk
        agent_path.append("Greeting_Check")
        greetings = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "howdy", "greetings", "what's up", "whats up",
            "sup", "yo", "how are you", "how's it going", "nice to meet you",
        ]
        if query.strip().lower().rstrip("!?.,'\"") in greetings:
            resp = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a friendly assistant for a Machine Learning textbook. Respond to the greeting warmly and briefly mention that you can help with questions about machine learning."},
                    {"role": "user", "content": query},
                ],
                temperature=0.7, max_tokens=100,
            )
            return {"response": resp.choices[0].message.content.strip(), "agent_path": " -> ".join(agent_path)}

        # Step 3: Rewrite query for multi-turn context
        agent_path.append("Context_Rewriter_Agent")
        rewritten_query = self.context_rewriter.rephrase(history, query)

        # Step 4: Check query relevance to book topic
        agent_path.append("Query_Agent")
        relevance_response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.query_agent.prompt},
                {"role": "user", "content": rewritten_query},
            ],
            temperature=0, max_tokens=10,
        )
        if not self.query_agent.extract_action(relevance_response):
            return {
                "response": "This query does not appear to be related to machine learning or the topics covered in this textbook. I would be happy to answer questions based on the book's context.",
                "agent_path": " -> ".join(agent_path),
            }

        # Step 5: Retrieve documents from Pinecone
        docs = self.query_agent.query_vector_store(rewritten_query, k=5)

        # Step 6: Check if retrieved documents are relevant
        agent_path.append("Relevant_Documents_Agent")
        if not self.relevant_docs_agent.get_relevance(rewritten_query, docs):
            return {
                "response": "This query is not relevant to the context of this book. I would be happy to answer questions based on the book's context.",
                "agent_path": " -> ".join(agent_path),
            }

        # Step 7: Generate answer using context
        agent_path.append("Answering_Agent")
        answer = self.answering_agent.generate_response(rewritten_query, docs, history)
        return {"response": answer, "agent_path": " -> ".join(agent_path)}


# ============================================================
# Streamlit UI  (extended from Part 2: mp2_app_JiahaoXu.py)
# ============================================================

st.set_page_config(page_title="ML Textbook Multi-Agent Chatbot", page_icon="📚")
st.title("Mini Project 2: Multi-Agent Chatbot")

# ---- API Keys (same pattern as Part 2, extended with Pinecone) ----
api_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "open_ai_key.txt")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            api_key = f.read().strip()

if not pinecone_key:
    key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinecone_api_key.txt")
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            pinecone_key = f.read().strip()

if not api_key or not pinecone_key:
    st.error("Missing API keys. Set OPENAI_API_KEY and PINECONE_API_KEY as environment variables, "
             "or place open_ai_key.txt and pinecone_api_key.txt in the project root.")
    st.stop()

# ---- Initialize Head Agent (replaces the plain OpenAI client from Part 2) ----
@st.cache_resource
def get_head_agent():
    return Head_Agent(api_key, pinecone_key, INDEX_NAME)

head_agent = get_head_agent()


# ---- get_conversation() carried over from Part 2 (will be useful for Part 4 evaluation) ----
def get_conversation() -> str:
    lines = []
    for m in st.session_state.get("messages", []):
        role = m.get("role", "unknown")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---- Session state (same as Part 2) ----
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL  # upgraded from gpt-3.5-turbo

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---- Sidebar (new in Part 3) ----
with st.sidebar:
    st.header("About")
    st.markdown(
        "This chatbot uses a **multi-agent RAG pipeline** to answer questions "
        "about a Machine Learning textbook.\n\n"
        "**Agents:**\n"
        "1. **Obnoxious Agent** — filters offensive queries\n"
        "2. **Context Rewriter** — resolves multi-turn ambiguity\n"
        "3. **Query Agent** — retrieves relevant chunks from Pinecone\n"
        "4. **Relevant Documents Agent** — validates chunk relevance\n"
        "5. **Answering Agent** — generates the final answer\n"
    )
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()

# ---- Display conversation history (same pattern as Part 2) ----
for message in st.session_state["messages"]:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)
        if "agent_path" in message:
            st.caption(f"Agent path: {message['agent_path']}")

# ---- Chat input (same as Part 2, but response comes from Head_Agent instead of plain OpenAI) ----
if prompt := st.chat_input("What would you like to chat about?"):

    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history for the multi-agent pipeline
    conv_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"][:-1]
        if m["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Part 2 had: response = client.chat.completions.create(...)
            # Part 3 replaces it with the multi-agent pipeline:
            result = head_agent.process_query(prompt, conv_history)
            assistant_reply = result["response"]

        st.markdown(assistant_reply)
        st.caption(f"Agent path: {result['agent_path']}")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": assistant_reply,
        "agent_path": result["agent_path"],
    })
