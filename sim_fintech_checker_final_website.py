import streamlit as st
import pandas as pd
from rapidfuzz import fuzz
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load your OpenAI API key from secrets or set it directly here
import streamlit as st

# Safe way to load your OpenAI key
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    openai_key = st.secrets["openai"]["api_key"]
else:
    st.error("❌ OpenAI API key not found. Please set it in Streamlit Cloud → Secrets.")
    st.stop()

# Load data
@st.cache_data
def load_data():
    sim_df = pd.read_csv("sim_courses_clean.csv", sep=";")
    fintech_df = pd.read_csv("fintech_courses_clean.csv", sep=";")

    sim_df["source"] = "SIM"
    fintech_df["source"] = "FinTech"

    sim_df["text"] = sim_df.apply(
        lambda row: f"{row['event']} ({row['course_number']}, {row['lecturer']}, {row['classification']})", axis=1
    )
    fintech_df["text"] = fintech_df.apply(
        lambda row: f"{row['Name']} ({row['Number']}, {row['Lecturer']}, {row['Semester']}, {row['ECTS']} ECTS)", axis=1
    )

    return sim_df, fintech_df

# Build RAG system
@st.cache_resource
def build_chain(sim_df, fintech_df):
    df = pd.concat([sim_df[["text", "source"]], fintech_df[["text", "source"]]], ignore_index=True)
    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    llm = OpenAI(openai_api_key=openai_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Fuzzy matching
def fuzzy_match(a, b, threshold=70):
    score = fuzz.token_set_ratio(a.lower(), b.lower())
    return score >= threshold

# Streamlit UI
st.set_page_config(page_title="SIM/FinTech Course Checker")
st.title("📚 SIM/FinTech Course Checker")
st.write("Ask any question about which FinTech courses are accepted for SIM credit.")

query = st.text_area("💬 Your question or course name", placeholder="e.g. Which courses are accepted in both programs?")

if st.button("Submit") and query:
    sim_df, fintech_df = load_data()
    chain = build_chain(sim_df, fintech_df)

    query_lower = query.lower()

    # Fuzzy match against query
    sim_matches = [row for _, row in sim_df.iterrows() if fuzzy_match(query_lower, row["text"])]
    fintech_matches = [row for _, row in fintech_df.iterrows() if fuzzy_match(query_lower, row["text"])]

    general_list_keywords = ["which", "what", "accepted", "approved", "credited", "count", "shared", "overlap", "list", "courses"]
    specific_course = not any(k in query_lower for k in ["courses", "list"]) and len(query.split()) <= 15

    # General question → list all matches
    if any(k in query_lower for k in general_list_keywords) and not specific_course:
        accepted_courses = []
        for _, ft_row in fintech_df.iterrows():
            for _, sim_row in sim_df.iterrows():
                if fuzzy_match(ft_row["text"], sim_row["text"]):
                    classification = sim_row["classification"]
                    accepted_courses.append(f"{ft_row['Name']} — belongs to SIM classification: {classification}")
                    break

        if accepted_courses:
            st.markdown("### ✅ Accepted FinTech Courses in SIM")
            for course in accepted_courses:
                st.markdown(f"- {course}")
        else:
            st.markdown("⚠️ No FinTech courses are currently listed as accepted in SIM.")

    # Specific question → determine if the course is eligible
    else:
        if fintech_matches and sim_matches:
            matched_sim = sim_matches[0]
            matched_ft = fintech_matches[0]
            if fuzzy_match(matched_sim["text"], matched_ft["text"]):
                st.markdown("### ✅ This course is accepted in SIM:")
                st.markdown(f"- {matched_ft['Name']} — belongs to SIM classification: {matched_sim['classification']}")
            else:
                st.markdown("⚠️ This course appears in both datasets but does not match closely enough to confirm cross-listing.")
        elif fintech_matches and not sim_matches:
            st.markdown("⚠️ This course is **only available in FinTech** and does **not count toward SIM**.")
        elif sim_matches and not fintech_matches:
            st.markdown("⚠️ This course is **only available in SIM** and is **not part of the FinTech certificate**.")
        else:
            st.markdown("❌ This course is not found in either program.")
elif not query:
    st.info("Please enter a question.")
