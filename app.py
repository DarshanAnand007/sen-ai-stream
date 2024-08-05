import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(page_icon="neural.png", layout="wide", page_title="SEN")

st.subheader("SEN AI", anchor=False)
st.text("near real-time responses")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b", "tokens": 8192, "developer": "Google"},
    "llama3-70b-8192": {"name": "LLaMA3 70b", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3 8b", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral 8x7b", "tokens": 32768, "developer": "Mistral"},
}

with st.sidebar:
    st.title("Chat configuration")

    model_option = st.selectbox(
        "Choose a model you want to chat with:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=2  # Default to llama
    )

    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    role_name = "SEN-AI" if message["role"] == "assistant" else "SEN"
    st.markdown(f"**{role_name}:** {message['content']}")

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if prompt := st.chat_input("Enter your prompt here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.markdown(f"**User:** {prompt}")

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        chat_responses_generator = generate_chat_responses(chat_completion)
        full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(f"Error occurred: {e}")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
