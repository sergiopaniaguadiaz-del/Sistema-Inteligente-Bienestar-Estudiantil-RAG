import pandas as pd
from pathlib import Path


# =========================
# CONFIGURACIÓN DE RUTAS
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATAWAREHOUSE_DIR = BASE_DIR / "datawarehouse"
DATA_RAG_DIR = BASE_DIR / "data_rag"

DATA_RAG_DIR.mkdir(exist_ok=True)


# =========================
# FUNCIONES AUXILIARES
# =========================

def pct(value):
    """Convierte decimal a porcentaje legible."""
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "No disponible"


def num(value):
    """Formatea números."""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "No disponible"


def guardar_txt(nombre_archivo, contenido):
    ruta = DATA_RAG_DIR / nombre_archivo
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido.strip())
    print(f"Documento generado: {ruta}")


# =========================
# 1. SEGMENTOS
# =========================

def generar_resumen_segmentos():
    df = pd.read_csv(DATAWAREHOUSE_DIR / "dim_segment.csv")

    texto = """
ANÁLISIS DE SEGMENTOS DE ESTUDIANTES

Este documento resume los perfiles identificados mediante segmentación de estudiantes.
Los segmentos permiten priorizar acciones preventivas de bienestar estudiantil.
"""

    for _, row in df.iterrows():
        texto += f"""

SEGMENTO: {row.get('segment_name', 'No disponible')}
ID DEL SEGMENTO: {row.get('segment_id', 'No disponible')}
NIVEL DE PRIORIDAD: {row.get('priority_level', 'No disponible')}

Cantidad de estudiantes: {row.get('cantidad_estudiantes', 'No disponible')}

Características promedio del segmento:
- Horas diarias en redes sociales: {num(row.get('daily_social_media_hours'))}
- Horas de sueño: {num(row.get('sleep_hours'))}
- Tiempo frente a pantalla antes de dormir: {num(row.get('screen_time_before_sleep'))}
- Desempeño académico: {num(row.get('academic_performance'))}
- Actividad física: {num(row.get('physical_activity'))}
- Nivel de estrés: {num(row.get('stress_level'))}
- Nivel de ansiedad: {num(row.get('anxiety_level'))}
- Nivel de adicción digital: {num(row.get('addiction_level'))}
- Proporción de riesgo alto de ansiedad: {pct(row.get('high_anxiety_risk'))}

Interpretación:
{row.get('segment_description', 'No disponible')}

Implicación para bienestar estudiantil:
Este segmento debe considerarse en la toma de decisiones preventivas según su nivel de prioridad.
"""

    guardar_txt("resumen_segmentos.txt", texto)


# =========================
# 2. MÉTRICAS DEL MODELO
# =========================

def generar_metricas_modelo():
    df = pd.read_csv(DATAWAREHOUSE_DIR / "dim_model.csv")
    row = df.iloc[0]

    texto = f"""
MÉTRICAS DEL MODELO SUPERVISADO

Este documento describe el desempeño del modelo predictivo utilizado para estimar riesgo alto de ansiedad.

Modelo utilizado:
{row.get('model_name', 'No disponible')}

Variable objetivo:
{row.get('target_variable', 'No disponible')}

Fecha del análisis:
{row.get('analysis_date', 'No disponible')}

Resultados del modelo:
- Accuracy: {pct(row.get('accuracy'))}
- Precision: {pct(row.get('precision'))}
- Recall: {pct(row.get('recall'))}
- F1-score: {pct(row.get('f1_score'))}

Resultado de sobreajuste:
{row.get('overfitting_result', 'No disponible')}

Interpretación:
El modelo permite identificar estudiantes con posible riesgo alto de ansiedad.
Sin embargo, las métricas muestran un desempeño moderado, por lo que sus resultados deben utilizarse como apoyo a la toma de decisiones y no como diagnóstico definitivo.

Conclusión ejecutiva:
El modelo aporta valor como herramienta preventiva, pero requiere validación, mejora y monitoreo antes de utilizarse en decisiones críticas.
"""

    guardar_txt("metricas_modelo.txt", texto)


# =========================
# 3. VARIABLES IMPORTANTES
# =========================

def generar_variables_importantes(top_n=10):
    df = pd.read_csv(DATAWAREHOUSE_DIR / "feature_importance.csv")
    df = df.sort_values(by="importance", ascending=False).head(top_n)

    texto = """
VARIABLES MÁS IMPORTANTES DEL MODELO

Este documento resume las variables con mayor peso en el modelo predictivo.
Estas variables ayudan a explicar qué factores influyen más en la predicción del riesgo emocional.
"""

    for i, row in enumerate(df.itertuples(index=False), start=1):
        texto += f"""
{i}. Variable: {row.variable}
   Importancia relativa: {num(row.importance)}
"""

    texto += """

Interpretación:
Las variables con mayor importancia deben considerarse como factores relevantes para analizar el riesgo estudiantil.
Estas variables pueden orientar intervenciones preventivas, análisis institucionales y estrategias de acompañamiento.

Conclusión ejecutiva:
El modelo identifica patrones asociados al bienestar digital, emocional, académico y conductual de los estudiantes.
"""

    guardar_txt("variables_importantes.txt", texto)


# =========================
# 4. SOBREAJUSTE
# =========================

def generar_analisis_sobreajuste():
    df = pd.read_csv(DATAWAREHOUSE_DIR / "overfitting_analysis.csv")

    texto = """
ANÁLISIS DE SOBREAJUSTE DEL MODELO

Este documento compara el desempeño del modelo en entrenamiento y prueba.
El objetivo es identificar si el modelo generaliza correctamente o si presenta señales de sobreajuste.
"""

    for _, row in df.iterrows():
        texto += f"""

Métrica: {row.get('metrica', 'No disponible')}
- Resultado en entrenamiento: {pct(row.get('entrenamiento'))}
- Resultado en prueba: {pct(row.get('prueba'))}
- Diferencia: {pct(row.get('diferencia'))}
"""

    max_diff = df.loc[df["diferencia"].idxmax()]

    texto += f"""

Hallazgo principal:
La mayor diferencia se observa en la métrica {max_diff.get('metrica')}, con una brecha de {pct(max_diff.get('diferencia'))} entre entrenamiento y prueba.

Interpretación:
Una diferencia elevada entre entrenamiento y prueba indica posible sobreajuste.
Esto significa que el modelo puede estar aprendiendo patrones específicos del conjunto de entrenamiento y perdiendo capacidad de generalización sobre datos nuevos.

Conclusión ejecutiva:
El modelo es útil como aproximación inicial, pero debe mejorarse antes de considerarse plenamente robusto.
Se recomienda revisar variables, ajustar hiperparámetros, balancear clases y validar con nuevos datos.
"""

    guardar_txt("analisis_sobreajuste.txt", texto)


# =========================
# 5. TENDENCIAS POR EDAD
# =========================

def generar_tendencias_por_edad():
    df = pd.read_csv(DATAWAREHOUSE_DIR / "dim_age_trends.csv")

    edad_mayor_ansiedad = df.loc[df["avg_anxiety_level"].idxmax()]
    edad_mayor_riesgo = df.loc[df["high_anxiety_risk_rate"].idxmax()]
    edad_menor_sueno = df.loc[df["avg_sleep_hours"].idxmin()]
    edad_mayor_redes = df.loc[df["avg_social_media_hours"].idxmax()]

    texto = """
TENDENCIAS DE BIENESTAR ESTUDIANTIL POR EDAD

Este documento resume patrones de ansiedad, sueño, uso de redes sociales y riesgo emocional por edad.
"""

    for _, row in df.iterrows():
        texto += f"""

Edad: {row.get('age')}
Cantidad de estudiantes: {row.get('students_count')}

Promedios observados:
- Horas diarias en redes sociales: {num(row.get('avg_social_media_hours'))}
- Horas de sueño: {num(row.get('avg_sleep_hours'))}
- Tiempo frente a pantalla antes de dormir: {num(row.get('avg_screen_time_before_sleep'))}
- Nivel de estrés: {num(row.get('avg_stress_level'))}
- Nivel de ansiedad: {num(row.get('avg_anxiety_level'))}
- Nivel de adicción digital: {num(row.get('avg_addiction_level'))}
- Tasa de riesgo alto de ansiedad: {pct(row.get('high_anxiety_risk_rate'))}
- Score promedio de riesgo: {num(row.get('avg_risk_score'))}
"""

    texto += f"""

Hallazgos principales:
- La edad con mayor nivel promedio de ansiedad es {edad_mayor_ansiedad.get('age')}, con un promedio de {num(edad_mayor_ansiedad.get('avg_anxiety_level'))}.
- La edad con mayor tasa de riesgo alto de ansiedad es {edad_mayor_riesgo.get('age')}, con una tasa de {pct(edad_mayor_riesgo.get('high_anxiety_risk_rate'))}.
- La edad con menor promedio de sueño es {edad_menor_sueno.get('age')}, con {num(edad_menor_sueno.get('avg_sleep_hours'))} horas promedio.
- La edad con mayor uso promedio de redes sociales es {edad_mayor_redes.get('age')}, con {num(edad_mayor_redes.get('avg_social_media_hours'))} horas diarias.

Conclusión ejecutiva:
Las tendencias por edad permiten identificar grupos demográficos que podrían requerir mayor atención preventiva.
"""

    guardar_txt("tendencias_por_edad.txt", texto)


# =========================
# 6. RESUMEN DE ESTUDIANTES EN RIESGO
# =========================

def generar_resumen_estudiantes_riesgo():
    fact = pd.read_csv(DATAWAREHOUSE_DIR / "fact_student_analysis.csv")

    total = len(fact)
    reales_alto = fact["high_anxiety_risk_real"].sum()
    predichos_alto = fact["high_anxiety_risk_predicted"].sum()

    niveles = fact["risk_score_level"].value_counts().to_dict()

    score_promedio = fact["risk_score"].mean()
    score_max = fact["risk_score"].max()
    score_min = fact["risk_score"].min()

    texto = f"""
RESUMEN DE ESTUDIANTES EN RIESGO

Este documento resume los resultados individuales agregados del análisis de riesgo estudiantil.
No se listan todos los estudiantes para evitar saturar el sistema RAG; se presentan estadísticas consolidadas.

Total de estudiantes analizados:
{total}

Estudiantes con riesgo alto real:
{int(reales_alto)} estudiantes, equivalente a {pct(reales_alto / total)}

Estudiantes con riesgo alto predicho por el modelo:
{int(predichos_alto)} estudiantes, equivalente a {pct(predichos_alto / total)}

Distribución por nivel de score de riesgo:
"""

    for nivel, cantidad in niveles.items():
        texto += f"""
- {nivel}: {cantidad} estudiantes, equivalente a {pct(cantidad / total)}
"""

    texto += f"""

Score de riesgo:
- Score promedio: {num(score_promedio)}
- Score mínimo: {num(score_min)}
- Score máximo: {num(score_max)}

Interpretación:
El análisis permite identificar estudiantes con mayor probabilidad de presentar riesgo alto de ansiedad.
Estos resultados pueden utilizarse para priorizar acciones preventivas de acompañamiento, siempre considerando que el modelo no sustituye una evaluación clínica o psicológica.

Conclusión ejecutiva:
El sistema permite transformar resultados individuales en información agregada útil para la Coordinación de Bienestar Estudiantil.
"""

    guardar_txt("resumen_estudiantes_riesgo.txt", texto)


# =========================
# EJECUCIÓN PRINCIPAL
# =========================

if __name__ == "__main__":
    print("Generando documentos RAG...")

    generar_resumen_segmentos()
    generar_metricas_modelo()
    generar_variables_importantes()
    generar_analisis_sobreajuste()
    generar_tendencias_por_edad()
    generar_resumen_estudiantes_riesgo()

    print("Proceso finalizado correctamente.")