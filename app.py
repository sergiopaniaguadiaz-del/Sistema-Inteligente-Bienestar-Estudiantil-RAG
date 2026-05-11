
import streamlit as st
import ollama

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==================================================
# CONFIGURACIÓN
# ==================================================

st.set_page_config(
    page_title="Bienestar Estudiantil AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CSS PROFESIONAL
# ==================================================

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #F4F7FB;
}

/* Header */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1E3A8A;
    margin-bottom: 0;
}

.subtitle {
    font-size: 18px;
    color: #475569;
    margin-top: -10px;
    margin-bottom: 25px;
}

/* Cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #64748B;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #0F172A;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3A8A 0%, #0F172A 100%);
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Chat */
.stChatMessage {
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 10px;
}

/* Texto respuestas chatbot */
.stChatMessage p,
.stChatMessage div {
    color: black !important;
}

""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("# 🎓 Bienestar AI")

    st.markdown("""
    Sistema inteligente basado en:

    - Machine Learning
    - Clustering
    - Predicción de riesgo
    - Embeddings
    - RAG
    - Ollama + Llama3
    """)

    st.markdown("---")

    st.subheader("👩‍💼 Decisor")

    st.write("""
    **Dra. Fernanda López Martínez**

    Coordinadora de Bienestar Estudiantil
    """)

    st.markdown("---")

    st.subheader("📌 Objetivo")

    st.write("""
    Identificar perfiles de estudiantes con mayor riesgo emocional para priorizar estrategias preventivas.
    """)

    st.markdown("---")

    st.subheader("💡 Preguntas sugeridas")

    suggested_questions = [
        "¿Qué segmento presenta mayor riesgo emocional?",
        "¿Qué variables influyen más en ansiedad?",
        "¿Cómo afectan las redes sociales?",
        "¿Qué hábitos tienen estudiantes vulnerables?",
        "¿Cuál fue el accuracy del modelo?"
    ]

    for q in suggested_questions:
        st.write(f"• {q}")

# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<p class="main-title"> Asistente Inteligente de Bienestar Estudiantil</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Consulta resultados del análisis de Machine Learning usando lenguaje natural.</p>',
    unsafe_allow_html=True
)

# ==================================================
# MÉTRICAS
# ==================================================

#col1, col2, col3, col4 = st.columns(4)

#with col1:
#    st.markdown("""
#    <div class="metric-card">
#        <div class="metric-title">Accuracy</div>
#        <div class="metric-value">87%</div>
#    </div>
#    """, unsafe_allow_html=True)

#with col2:
#    st.markdown("""
#    <div class="metric-card">
#        <div class="metric-title">Recall</div>
#        <div class="metric-value">91%</div>
#    </div>
#    """, unsafe_allow_html=True)

#with col3:
#    st.markdown("""
#    <div class="metric-card">
#        <div class="metric-title">Segmentos</div>
#        <div class="metric-value">3</div>
#    </div>
#    """, unsafe_allow_html=True)

#with col4:
#    st.markdown("""
#    <div class="metric-card">
#        <div class="metric-title">Modelo</div>
#        <div class="metric-value">RF</div>
#    </div>
#    """, unsafe_allow_html=True)

#st.markdown("<br>", unsafe_allow_html=True)

# ==================================================
# CARGAR EMBEDDINGS
# ==================================================

@st.cache_resource
def load_embeddings():

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings_model

embeddings = load_embeddings()

# ==================================================
# CARGAR VECTOR DB
# ==================================================

@st.cache_resource
def load_vector_db():

    db_local = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db_local

db = load_vector_db()

retriever = db.as_retriever(search_kwargs={"k": 3})

# ==================================================
# HISTORIAL
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==================================================
# INPUT
# ==================================================

user_question = st.chat_input(
    "Escribe una pregunta sobre bienestar estudiantil..."
)


# ==================================================
# PROCESAR
# ==================================================

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(user_question)

    # ==========================================
    # RECUPERAR CONTEXTO
    # ==========================================

    docs = retriever.get_relevant_documents(
        user_question
    )
    
    print(docs)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # ==========================================
    # PROMPT
    # ==========================================

    prompt = f"""
    Eres un asistente especializado en bienestar estudiantil.

    Debes responder únicamente usando la información proporcionada.

    NO inventes información.

    Si no existe información suficiente responde:

    'No se encontró evidencia suficiente en el análisis realizado.'

    Contexto:
    {context}

    Pregunta:
    {user_question}

    Respuesta:
    """
    # ==========================================
    # LOADER
    # ==========================================

    with st.spinner("🧠 Analizando resultados del modelo..."):
    
        print("Enviando pregunta a Ollama...")
    
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print("Respuesta recibida")
        
        answer = response["message"]["content"]

    # ==========================================
    # RESPUESTA
    # ==========================================

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.markdown("---")

        with st.expander("📚 Evidencia utilizada"):

            for doc in docs:

                st.markdown(
                    f"✅ {doc.metadata['source']}"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Proyecto Integrador · Minería de Datos · Machine Learning + RAG"
)