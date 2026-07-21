import os
import tempfile

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

st.set_page_config(page_title="Asistente de documentos", page_icon="📄", layout="wide")

st.title("📄 Asistente de documentos con LangChain + Groq")
st.caption("Sube archivos PDF, construye un almacén vectorial local y haz preguntas sobre ellos en español.")

PERSIST_DIR = os.path.join(os.getcwd(), "faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(PERSIST_DIR, exist_ok=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def load_vector_store(embeddings):
    index_path = os.path.join(PERSIST_DIR, "index.faiss")
    if os.path.exists(index_path):
        return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
    return None


def build_prompt_template():
    return PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "Eres un asistente experto que responde en español. "
            "Usa únicamente la información contenida en los documentos cargados. "
            "Si la respuesta no aparece explícitamente en los documentos, responde exactamente esta frase: "
            "La información solicitada no se encuentra en los documentos cargados.\n\n"
            "Pregunta del usuario: {question}\n\n"
            "Contexto relevante:\n{context}\n\n"
            "Respuesta:"
        ),
    )


with st.sidebar:
    st.header("🔐 Configuración")
    groq_api_key = st.text_input("API Key de Groq", type="password", placeholder="Ingresa tu clave")
    st.caption("La clave solo se usa en esta sesión.")

    uploaded_files = st.file_uploader(
        "Sube uno o varios PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Procesar documentos", type="primary"):
        if not groq_api_key:
            st.error("Introduce tu API key de Groq antes de procesar los documentos.")
        elif not uploaded_files:
            st.warning("Selecciona al menos un archivo PDF.")
        else:
            try:
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                with st.spinner("Extrayendo texto y creando el almacén vectorial..."):
                    split_docs = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        try:
                            loader = PyPDFLoader(temp_path)
                            raw_docs = loader.load()
                            for doc in raw_docs:
                                doc.metadata["source"] = uploaded_file.name

                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200,
                            )
                            chunks = splitter.split_documents(raw_docs)
                            split_docs.extend(chunks)
                        finally:
                            os.remove(temp_path)

                    if split_docs:
                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                        else:
                            st.session_state.vector_store.add_documents(split_docs)

                        st.session_state.vector_store.save_local(PERSIST_DIR)
                        st.success(f"Se cargaron {len(uploaded_files)} PDF(s) y {len(split_docs)} fragmentos de texto.")
                    else:
                        st.warning("No se pudo extraer texto útil de los archivos PDF.")
            except Exception as exc:
                st.error(f"Ocurrió un error al procesar los PDFs: {exc}")

    if st.button("Limpiar chat"):
        st.session_state.chat_history = []
        st.success("Historial del chat limpiado.")


if st.session_state.vector_store is None:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        st.session_state.vector_store = load_vector_store(embeddings)
    except Exception:
        st.session_state.vector_store = None

st.subheader("💬 Chat sobre los documentos")
if st.session_state.vector_store is None:
    st.info("Carga documentos PDF para habilitar el chat.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Escribe tu pregunta en español...")
    if user_question:
        if not groq_api_key:
            st.error("Introduce tu API key de Groq para consultar los documentos.")
        else:
            try:
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=groq_api_key,
                    temperature=0,
                )
                prompt = build_prompt_template()

                docs = st.session_state.vector_store.similarity_search(user_question, k=4)
                context = "\n\n".join(doc.page_content for doc in docs)

                if not docs:
                    answer = "La información solicitada no se encuentra en los documentos cargados."
                else:
                    formatted_prompt = prompt.format(question=user_question, context=context)
                    answer = llm.invoke(formatted_prompt).content

                with st.chat_message("user"):
                    st.write(user_question)
                with st.chat_message("assistant"):
                    st.write(answer)

                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"No se pudo responder la pregunta: {exc}")
