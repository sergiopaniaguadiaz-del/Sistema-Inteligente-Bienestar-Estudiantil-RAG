# Sistema Inteligente de Bienestar Estudiantil basado en Machine Learning y RAG

## Descripción General

Este proyecto desarrolla un sistema inteligente orientado a la detección preventiva de riesgo emocional estudiantil mediante técnicas de Machine Learning y Retrieval-Augmented Generation (RAG).

La solución integra análisis predictivo, segmentación de estudiantes, almacenamiento analítico y recuperación semántica basada en embeddings y FAISS para permitir consultas en lenguaje natural sobre resultados reales del análisis.

---

## Objetivo

Transformar datos de bienestar digital y salud emocional estudiantil en conocimiento útil para apoyar la toma de decisiones preventivas dentro de una institución educativa.

---

## Arquitectura General

Dataset CSV
→ Machine Learning
→ Data Warehouse Analítico
→ Transformación Semántica
→ Embeddings
→ FAISS Vector Database
→ Ollama + Llama 3
→ Streamlit

---

## Tecnologías Utilizadas

| Tecnología            | Uso                      |
| --------------------- | ------------------------ |
| Python                | Desarrollo principal     |
| Pandas                | Procesamiento de datos   |
| Scikit-learn          | Machine Learning         |
| LangChain             | Integración RAG          |
| Sentence Transformers | Embeddings               |
| FAISS                 | Base vectorial           |
| Ollama                | LLM local                |
| Llama 3               | Generación de respuestas |
| Streamlit             | Interfaz web             |

---

## Modelos Implementados

### Clustering

* K-Means

### Modelo Supervisado

* Random Forest Classifier

---

## Componentes del Sistema

### generar_documentos_rag.py

Transforma resultados analíticos CSV en documentos semánticos interpretativos optimizados para retrieval.

### crear_vector_db.py

Genera embeddings y construye la base vectorial FAISS utilizada por el sistema RAG.

### app.py

Interfaz conversacional desarrollada en Streamlit que permite realizar consultas en lenguaje natural utilizando recuperación contextual y Llama 3.

---

## Flujo Operativo

1. Ejecutar notebook de Machine Learning
2. Generar CSV analíticos
3. Ejecutar:
   python generar_documentos_rag.py
4. Ejecutar:
   python crear_vector_db.py
5. Ejecutar:
   streamlit run app.py

---

## Ejemplos de Preguntas

* ¿Qué segmento presenta mayor riesgo emocional?
* ¿Qué variables influyen más en ansiedad?
* ¿Cuál fue el accuracy del modelo?
* ¿Qué hábitos digitales son más riesgosos?

---

## Capturas del Sistema

Agregar screenshots del sistema aquí.

---

## Autor

Sergio Paniagua Díaz

Proyecto de Maestría — Sistema Inteligente de Bienestar Estudiantil basado en Machine Learning y RAG.
