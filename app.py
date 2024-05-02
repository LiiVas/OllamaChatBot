import torch
from sentence_transformers import SentenceTransformer, util
import os
import streamlit as st
from openai import OpenAI
import time

YELLOW = '\033[93m'
CYAN = '\033[96m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def load_vault():
    @st.cache_data(show_spinner=False)
    def _load_vault():
        vault_content = []
        try:
            with open("vault.txt", "r", encoding='utf-8') as vault_file:
                vault_content = vault_file.readlines()
        except Exception as e:
            st.error(f"Error loading vault: {e}")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vault_embeddings = model.encode(vault_content) if vault_content else []
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        return vault_content, vault_embeddings_tensor

    return _load_vault()

def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=3):
    if vault_embeddings.nelement() == 0:
        return []
    input_embedding = model.encode([user_input])
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model, ollama_model, conversation_history):
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        st.write("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        st.write(CYAN + "No relevant context found." + RESET_COLOR)

    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input

    conversation_history.append({"role": "user", "content": user_input_with_context})

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    try:
        # Send the completion request to the Ollama model
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages
        )
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in Ollama chat: {e}")
        return "An error occurred in Ollama chat. Please try again later."

def main():
    st.title("Ollama Chat")

    system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text"
    conversation_history = []

    vault_content, vault_embeddings_tensor = load_vault()

    while True:
        user_input = st.text_input("Ask a question about your documents (or type 'quit' to exit):", key="user_input")
        if user_input.lower() == 'quit':
            break

        if st.button("Submit"):
            response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model, "llama3", conversation_history)
            st.write(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
            break  # Exit the loop after submitting a question

if __name__ == "__main__":
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='llama3'
        )
        model = SentenceTransformer("all-MiniLM-L6-v2")
        main()
    except Exception as e:
        st.error(f"Error in main function: {e}")

