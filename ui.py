import streamlit as st
import main

st.title("My RAG Chatbot")

# Store chat history in Streamlit
if "history" not in st.session_state:
    st.session_state.history = []  # list of ("user"/"assistant", text)

# Upload file
uploaded = st.file_uploader("Upload document", type=["pdf", "docx", "txt"])

if uploaded:
    # Save uploaded file temporarily
    with open("temp_file", "wb") as f:
        f.write(uploaded.getvalue())

    main.build_index_from_file("temp_file")
    st.success("Index built successfully!")

# Ask question
query = st.text_input("Ask a question")

if st.button("Ask"):
    if query.strip():
        # Add user message
        st.session_state.history.append(("user", query))

        # Get answer from backend
        ans = main.answer_query(query, st.session_state.history)

        # Add assistant message
        
        st.session_state.history.append(("assistant", ans))

# Show conversation
st.subheader("Conversation")
for role, text in st.session_state.history:
    if role == "user":
        st.write(f"**You:** {text}")
    else:
        st.write(f"**Bot:** {text}")
