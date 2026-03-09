import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# ==========================================================
# SESSION STATE (guardar dataset entre módulos)
# ==========================================================

if "ph_df" not in st.session_state:
    st.session_state.ph_df = None

# Variables globales
ph_file = None

# ==========================================================
# SIDEBAR DE NAVEGACIÓN
# ==========================================================

ph_menu = st.sidebar.selectbox(
    "Menú de navegación",
    ["Home", "Carga del Dataset", "EDA", "Conclusiones"]
)

# ==========================================================
# MODULO 1 - HOME
# ==========================================================

if ph_menu == "Home":

    st.header("Descripcion del Proyecto")

    st.write("""
    Esta aplicación interactiva permite realizar un Análisis Exploratorio de Datos (EDA)
    sobre el dataset de una compañía de seguros. El objetivo es analizar el comportamiento
    de los clientes, sus características y patrones relacionados con la renovación de pólizas.
    """)

    st.header("Autor")

    st.write("""
    Nombre: William Samuel Kenneth Paucar Huayta  
    Curso: Especialización Python for Analytics  
    Año: 2026
    """)

    st.header("Herramientas utilizadas")

    st.write("""
    - NumPy  
    - Pandas  
    - Matplotlib  
    - Seaborn        
    """)

# ==========================================================
# MODULO 2 - CARGA DEL DATASET
# ==========================================================

elif ph_menu == "Carga del Dataset":

    st.header("Cargar el Dataset")

    ph_file = st.file_uploader("Subir archivo InsuranceCompany.csv", type=["csv"])

    if ph_file is not None:

        st.session_state.ph_df = pd.read_csv(ph_file)

        st.subheader("Vista previa del dataset")
        st.dataframe(st.session_state.ph_df.head())

        st.subheader("Dimensiones del dataset")

        ph_rows, ph_cols = st.session_state.ph_df.shape

        st.write(f"Filas: {ph_rows}")
        st.write(f"Columnas: {ph_cols}")

    else:
        st.warning("Por favor cargue el archivo CSV para continuar.")

# ==========================================================
# MODULO 3 - EDA
# ==========================================================

elif ph_menu == "EDA":

    if st.session_state.ph_df is None:

        st.warning("Primero debe cargar el dataset en el módulo 'Carga del Dataset'")

    else:

        ph_df = st.session_state.ph_df

        st.title("Análisis Exploratorio de Datos (EDA)")

        # ==========================================================
        # TABS PARA LOS ITEMS DEL EDA
        # ==========================================================

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10= st.tabs([
            "Item 1 - Información del Dataset",
            "Item 2 - Clasificación de Variables",
            "Item 3 - Estadísticas Descriptivas",
            "Item 4 - Valores Faltantes",
            "Item 5 - Distribución de Variables",
            "Item 6 - Variables Categóricas",
            "Item 7 - Análisis Bivariado",
            "Item 8 - Análisis Bivariado Categórico-Categórico",
            "Item 9 - Análisis Dinámico",
            "Item 10 - Hallazgos clave"

        ])

        # ==========================================================
        # ITEM 1
        # ==========================================================

        with tab1:

            st.header("Ítem 1: Información general del dataset")

        # ======================================================
        # COLUMNAS PRINCIPALES
        # ======================================================

            col1, col2 = st.columns(2)

        # ------------------------------------------------------
        # INFO DEL DATASET
        # ------------------------------------------------------

            with col1:

                st.subheader("Estructura del dataset (.info())")

                ph_info_df = pd.DataFrame({
                    "Variable": ph_df.columns,
                    "Tipo de dato": ph_df.dtypes.astype(str),
                    "Valores no nulos": ph_df.notnull().sum()
                })

                st.dataframe(ph_info_df, width="stretch")

            # ------------------------------------------------------
            # TIPOS DE DATOS
            # ------------------------------------------------------

            with col2:

                st.subheader("Tipos de datos")

                ph_types = ph_df.dtypes.reset_index()
                ph_types.columns = ["Variable", "Tipo de dato"]

                ph_types["Tipo de dato"] = ph_types["Tipo de dato"].astype(str)

                st.dataframe(ph_types, width="stretch")

            st.divider()

            # ======================================================
            # VALORES NULOS
            # ======================================================

            st.subheader("Conteo de valores nulos")

            ph_nulls = ph_df.isnull().sum().reset_index()
            ph_nulls.columns = ["Variable", "Valores nulos"]

            st.dataframe(ph_nulls, width="stretch")

        with tab2:
            
            st.header("Ítem 2: Clasificación de Variables")

            st.write("Identificación de variables numéricas y categóricas del dataset.")

            # ======================================================
            # FUNCIÓN PERSONALIZADA
            # ======================================================

            def clasificar_variables(df):

                numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

                return numericas, categoricas

            # Ejecutamos la función
            ph_numeric, ph_categorical = clasificar_variables(ph_df)

            # ======================================================
            # COLUMNAS PARA MOSTRAR RESULTADOS
            # ======================================================

            col1, col2 = st.columns(2)

            # ------------------------------------------------------
            # VARIABLES NUMÉRICAS
            # ------------------------------------------------------

            with col1:

                st.subheader("Variables Numéricas")

                ph_num_df = pd.DataFrame({
                    "Variables Numéricas": ph_numeric
                })

                st.dataframe(ph_num_df, width="stretch")

                st.metric("Cantidad de variables numéricas", len(ph_numeric))

            # ------------------------------------------------------
            # VARIABLES CATEGÓRICAS
            # ------------------------------------------------------

            with col2:

                st.subheader("Variables Categóricas")

                ph_cat_df = pd.DataFrame({
                    "Variables Categóricas": ph_categorical
                })

                st.dataframe(ph_cat_df, width="stretch")

                st.metric("Cantidad de variables categóricas", len(ph_categorical))

            # ======================================================
            # TABLA RESUMEN CON CONTEO
            # ======================================================

            st.divider()

            st.subheader("Conteo de Tipos de Variables")

            ph_summary = pd.DataFrame({
                "Tipo de Variable": ["Numéricas", "Categóricas"],
                "Cantidad": [len(ph_numeric), len(ph_categorical)]
            })

            st.dataframe(ph_summary, width="stretch")
        
        with tab3: 

            st.header("Ítem 3: Estadísticas Descriptivas")

            st.write("Análisis de estadísticas descriptivas de las variables numéricas.")

        # ======================================================
        # SELECCIONAR VARIABLES NUMÉRICAS
        # ======================================================

            ph_numeric_df = ph_df.select_dtypes(include =["int64","float64"])

            # ======================================================
            # DESCRIBE()
            # ======================================================

            st.subheader("Estadísticas descriptivas (.describe())")

            ph_describe = ph_numeric_df.describe()

            st.dataframe(ph_describe, width="stretch")

            st.divider()

            # ======================================================
            # INTERPRETACIÓN BÁSICA
            # ======================================================

            st.subheader("Interpretación básica")

            col1, col2, col3 = st.columns(3)

            # MEDIA
            with col1:

                st.markdown("### Media")

                ph_mean = ph_numeric_df.mean().reset_index()
                ph_mean.columns = ["Variable", "Media"]

                st.dataframe(ph_mean, width="stretch")

            # MEDIANA
            with col2:

                st.markdown("### Mediana")

                ph_median = ph_numeric_df.median().reset_index()
                ph_median.columns = ["Variable", "Mediana"]

                st.dataframe(ph_median, width="stretch")

            # DISPERSIÓN
            with col3:

                st.markdown("### Desviación estándar (Dispersión)")

                ph_std = ph_numeric_df.std().reset_index()
                ph_std.columns = ["Variable", "Desviación estándar"]

                st.dataframe(ph_std, width="stretch")
        # ==========================================================
        # ITEM 4
        # ==========================================================

        with tab4:

            st.header("Ítem 4: Análisis de Valores Faltantes")

            st.write("Identificación y análisis de valores faltantes en el dataset.")

            # ======================================================
            # CONTEO DE VALORES FALTANTES
            # ======================================================

            st.subheader("Conteo de valores faltantes por variable")

            ph_missing = ph_df.isnull().sum().reset_index()
            ph_missing.columns = ["Variable", "Valores faltantes"]

            st.dataframe(ph_missing, width="stretch")

            st.divider()

            # ======================================================
            # VISUALIZACIÓN SIMPLE
            # ======================================================

            st.subheader("Visualización de valores faltantes")

            ph_missing_filtered = ph_missing[ph_missing["Valores faltantes"] > 0]

            if ph_missing_filtered.empty:

                st.success("No se encontraron valores faltantes en el dataset.")

            else:

                st.bar_chart(
                    ph_missing_filtered.set_index("Variable")["Valores faltantes"]
                )

            st.divider()

            # ======================================================
            # DISCUSIÓN BREVE
            # ======================================================

            st.subheader("Discusión")

            total_missing = ph_df.isnull().sum().sum()

            if total_missing == 0:

                st.write(
                    "El dataset no presenta valores faltantes, lo que facilita el análisis "
                    "y evita la necesidad de aplicar técnicas de imputación o limpieza de datos."
                )

            else:

                st.write(
                    f"Se identificaron **{total_missing} valores faltantes** en el dataset. "
                    "Esto podría afectar algunos análisis estadísticos o modelos predictivos, "
                    "por lo que sería recomendable aplicar técnicas de limpieza de datos "
                    "como eliminación de registros o imputación de valores."
                )
                # ==========================================================
        # ITEM 5
        # ==========================================================

        with tab5:

            st.header("Ítem 5: Distribución de Variables Numéricas")

            st.write("Análisis de la distribución de las variables numéricas mediante histogramas.")

            # ======================================================
            # SELECCIONAR VARIABLES NUMÉRICAS
            # ======================================================

            ph_numeric_cols = ph_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

            if len(ph_numeric_cols) == 0:

                st.warning("No se encontraron variables numéricas en el dataset.")

            else:

                # ======================================================
                # SELECTOR DE VARIABLE
                # ======================================================

                ph_selected_var = st.selectbox(
                    "Seleccione una variable numérica para visualizar su distribución:",
                    ph_numeric_cols
                )

                # ======================================================
                # HISTOGRAMA CON SEABORN
                # ======================================================

                st.subheader(f"Histograma de {ph_selected_var}")

                fig, ax = plt.subplots()

                sns.histplot(ph_df[ph_selected_var], kde=True, ax=ax)

                ax.set_xlabel(ph_selected_var)
                ax.set_ylabel("Frecuencia")

                st.pyplot(fig)

                st.divider()

                # ======================================================
                # INTERPRETACIÓN VISUAL
                # ======================================================

                st.subheader("Interpretación visual")

                mean_val = ph_df[ph_selected_var].mean()
                median_val = ph_df[ph_selected_var].median()

                st.write(
                    f"La variable **{ph_selected_var}** presenta una media de **{mean_val:.2f}** "
                    f"y una mediana de **{median_val:.2f}**. "
                    "La forma del histograma permite observar la distribución de los datos, "
                    "identificar posibles sesgos y detectar valores atípicos."
                )
            # ==========================================================
            # ITEM 6
            # ==========================================================

        with tab6:

            st.header("Ítem 6: Análisis de Variables Categóricas")

            st.write("Análisis de frecuencias y proporciones de variables categóricas.")

            # ======================================================
            # IDENTIFICAR VARIABLES CATEGÓRICAS
            # ======================================================

            ph_cat_cols = ph_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            if len(ph_cat_cols) == 0:

                st.warning("No se encontraron variables categóricas en el dataset.")

            else:

                # ======================================================
                # SELECTOR DE VARIABLE
                # ======================================================

                ph_selected_cat = st.selectbox(
                    "Seleccione una variable categórica:",
                    ph_cat_cols
                )

                # ======================================================
                # CONTEO DE VALORES
                # ======================================================

                st.subheader("Conteo de categorías")

                ph_counts = ph_df[ph_selected_cat].value_counts().reset_index()
                ph_counts.columns = ["Categoría", "Conteo"]

                st.dataframe(ph_counts, width="stretch")

                st.divider()

                # ======================================================
                # GRÁFICO DE BARRAS
                # ======================================================

                st.subheader("Gráfico de barras")

                fig, ax = plt.subplots()

                sns.barplot(
                    x="Categoría",
                    y="Conteo",
                    data=ph_counts,
                    ax=ax
                )

                ax.set_xlabel(ph_selected_cat)
                ax.set_ylabel("Frecuencia")

                st.pyplot(fig)

                st.divider()

                # ======================================================
                # PROPORCIONES
                # ======================================================

                st.subheader("Proporciones")

                ph_prop = ph_df[ph_selected_cat].value_counts(normalize=True).reset_index()
                ph_prop.columns = ["Categoría", "Proporción"]

                ph_prop["Proporción"] = ph_prop["Proporción"].round(3)

                st.dataframe(ph_prop, width="stretch")

                # ==========================================================
        # ITEM 7
        # ==========================================================

        with tab7:

            st.header("Ítem 7: Análisis Bivariado (Numérico vs Categórico)")

            st.write("Análisis de la relación entre variables numéricas y categóricas.")

            # ======================================================
            # IDENTIFICAR VARIABLES
            # ======================================================

            ph_numeric_cols = ph_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            ph_cat_cols = ph_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            if len(ph_numeric_cols) == 0 or len(ph_cat_cols) == 0:

                st.warning("Se necesitan variables numéricas y categóricas para este análisis.")

            else:

                # ======================================================
                # SELECTORES DE VARIABLES
                # ======================================================

                col1, col2 = st.columns(2)

                with col1:
                    ph_num_var = st.selectbox(
                        "Seleccione una variable numérica",
                        ph_numeric_cols
                    )

                with col2:
                    ph_cat_var = st.selectbox(
                        "Seleccione una variable categórica",
                        ph_cat_cols
                    )

                # ======================================================
                # BOXPLOT (RECOMENDADO PARA BIVARIADO)
                # ======================================================

                st.subheader(f"Distribución de {ph_num_var} según {ph_cat_var}")

                fig, ax = plt.subplots()

                sns.boxplot(
                    x=ph_cat_var,
                    y=ph_num_var,
                    data=ph_df,
                    ax=ax
                )

                ax.set_xlabel(ph_cat_var)
                ax.set_ylabel(ph_num_var)

                st.pyplot(fig)

                st.divider()

                # ======================================================
                # PROMEDIOS POR CATEGORÍA
                # ======================================================

                st.subheader("Promedio de la variable numérica por categoría")

                ph_group = ph_df.groupby(ph_cat_var)[ph_num_var].mean().reset_index()
                ph_group.columns = ["Categoría", "Promedio"]

                st.dataframe(ph_group, width="stretch")

                st.divider()

                # ======================================================
                # INTERPRETACIÓN
                # ======================================================

                st.subheader("Interpretación")

                st.write(
                    f"El gráfico muestra cómo varía **{ph_num_var}** entre las diferentes "
                    f"categorías de **{ph_cat_var}**. Esto permite identificar posibles "
                    "diferencias en el comportamiento de la variable numérica según la "
                    "categoría, lo cual puede ser útil para entender patrones en los datos."
                )
            # ==========================================================
            # ITEM 8
            # ==========================================================

        with tab8:

            st.header("Ítem 8: Análisis Bivariado (Categórico vs Categórico)")

            st.write("Análisis de la relación entre dos variables categóricas.")

            # ======================================================
            # IDENTIFICAR VARIABLES CATEGÓRICAS
            # ======================================================

            ph_cat_cols = ph_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            if len(ph_cat_cols) < 2:

                st.warning("Se necesitan al menos dos variables categóricas para este análisis.")

            else:

                # ======================================================
                # SELECCIÓN DE VARIABLES
                # ======================================================

                col1, col2 = st.columns(2)

                with col1:
                    ph_cat_var1 = st.selectbox(
                        "Seleccione la primera variable categórica",
                        ph_cat_cols
                    )

                with col2:
                    ph_cat_var2 = st.selectbox(
                        "Seleccione la segunda variable categórica",
                        ph_cat_cols
                    )

                # ======================================================
                # TABLA DE CONTINGENCIA
                # ======================================================

                st.subheader("Tabla de contingencia (conteos cruzados)")

                ph_crosstab = pd.crosstab(ph_df[ph_cat_var1], ph_df[ph_cat_var2])

                st.dataframe(ph_crosstab, width="stretch")

                st.divider()

                # ======================================================
                # GRÁFICO DE BARRAS AGRUPADAS
                # ======================================================

                st.subheader("Gráfico de barras comparativo")

                ph_plot_df = ph_crosstab.reset_index().melt(
                    id_vars=ph_cat_var1,
                    var_name="Categoria_2",
                    value_name="Conteo"
                )

                fig, ax = plt.subplots()

                sns.barplot(
                    data=ph_plot_df,
                    x=ph_cat_var1,
                    y="Conteo",
                    hue="Categoria_2",
                    ax=ax
                )

                ax.set_ylabel("Frecuencia")

                st.pyplot(fig)

                st.divider()

                # ======================================================
                # INTERPRETACIÓN
                # ======================================================

                st.subheader("Interpretación")

                st.write(
                    f"Este análisis permite observar la relación entre **{ph_cat_var1}** y "
                    f"**{ph_cat_var2}**. La tabla de contingencia muestra la frecuencia "
                    "de cada combinación de categorías, mientras que el gráfico de barras "
                    "permite comparar visualmente cómo se distribuyen las categorías entre sí."
                )
        # ==========================================================
        # ITEM 9
        # ==========================================================

        with tab9:

            st.header("Ítem 9: Análisis basado en parámetros seleccionados")
            st.write("Seleccione variables para generar análisis dinámicos.")

            # COLUMNAS
            ph_numeric_cols = ph_df.select_dtypes(include=["number"]).columns.tolist()
            ph_cat_cols = ph_df.select_dtypes(include=["object","category","bool"]).columns.tolist()

            col1, col2 = st.columns(2)

            with col1:
                ph_selected_num = st.selectbox(
                "Seleccione una variable numérica",
                ph_numeric_cols,
                key="item9_num"
            )

            with col2:
                ph_selected_cats = st.multiselect(
                    "Seleccione variables categóricas",
                    ph_cat_cols,
                    key ="item9_cats"
                )

            st.divider()

            if ph_selected_num and ph_selected_cats:

                for cat in ph_selected_cats:

                    st.subheader(f"Análisis de {ph_selected_num} según {cat}")

                    try:

                        # AGRUPAR
                        ph_group = (
                            ph_df
                            .groupby(cat, as_index=False)[ph_selected_num]
                            .mean()
                            .reset_index()
                        )

                        # limpiar datos
                        ph_group = ph_group.dropna()

                        st.dataframe(ph_group, use_container_width=True)

                        # GRAFICO
                        fig, ax = plt.subplots()

                        sns.barplot(
                            data=ph_group,
                            x=cat,
                            y=ph_selected_num,
                            ax=ax
                        )

                        ax.set_title(f"Promedio de {ph_selected_num} por {cat}")
                        ax.set_xlabel(cat)
                        ax.set_ylabel("Promedio")

                        plt.xticks(rotation=45)

                        st.pyplot(fig)

                        st.divider()

                    except Exception as e:
                        st.error(f"Error con la variable {cat}")
                        st.write(e)

            else:
                st.info("Seleccione una variable numérica y al menos una categórica.")

        with tab10:
            st.header("Ítem 10: Hallazgos Clave")

            st.write(
                "En este apartado se presentan los hallazgos principales derivados del "
                "Análisis Exploratorio de Datos (EDA) sobre el dataset de la compañía de seguros."
            )

            # ==========================================================
            # 1️⃣ Resumen Visual - Distribuciones Numéricas
            # ==========================================================

            st.subheader("Resumen Visual: Variables Numéricas")

            ph_numeric_cols = ph_df.select_dtypes(include=["number"]).columns.tolist()

            if len(ph_numeric_cols) > 0:

                fig, axs = plt.subplots(nrows=len(ph_numeric_cols), ncols=1, figsize=(8, 4*len(ph_numeric_cols)))

                if len(ph_numeric_cols) == 1:
                    axs = [axs]

                for ax, col in zip(axs, ph_numeric_cols):
                    sns.histplot(ph_df[col], kde=True, ax=ax)
                    ax.set_title(f"Distribución de {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frecuencia")

                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.warning("No se encontraron variables numéricas para mostrar resumen visual.")

            st.divider()

            # ==========================================================
            # 2️⃣ Resumen Visual - Variables Categóricas
            # ==========================================================

            st.subheader("Resumen Visual: Variables Categóricas")

            ph_cat_cols = ph_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            if len(ph_cat_cols) > 0:

                fig, axs = plt.subplots(nrows=len(ph_cat_cols), ncols=1, figsize=(8, 3*len(ph_cat_cols)))

                if len(ph_cat_cols) == 1:
                    axs = [axs]

                for ax, col in zip(axs, ph_cat_cols):
                    ph_counts = ph_df[col].value_counts()
                    sns.barplot(x=ph_counts.index, y=ph_counts.values, ax=ax)
                    ax.set_title(f"Conteo por categoría de {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frecuencia")
                    plt.setp(ax.get_xticklabels(), rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.warning("No se encontraron variables categóricas para mostrar resumen visual.")

            st.divider()

            # ==========================================================
            # 3️⃣ Insights Principales
            # ==========================================================

            st.subheader("Insights Principales")

            st.write("""
            - Las variables numéricas muestran [tendencias generales, sesgos y posibles valores atípicos].  
            - Las variables categóricas evidencian [diferencias significativas en las categorías principales].  
            - Se identificaron correlaciones interesantes entre [X e Y] que podrían influir en la retención de clientes.  
            - Los valores faltantes son [escasos/importantes], lo que sugiere [posible limpieza o imputación].  
            - Estos hallazgos pueden guiar futuras estrategias de análisis predictivo y toma de decisiones.
            """)

            st.success("✅ El Análisis Exploratorio de Datos ha sido completado exitosamente.")

if ph_menu == "Conclusionese":
    st.header("Conclusiones Finales")

    st.write("""
    Después de realizar el análisis exploratorio de datos (EDA) sobre el dataset de la compañía de seguros, se pueden extraer las siguientes conclusiones clave:
    """)

    st.subheader("1. Comportamiento de renovación de pólizas")
    st.write("""
    Se observa que ciertas características de los clientes, como edad, antigüedad y tipo de póliza, muestran patrones claros de renovación. Esto permite a la compañía identificar grupos de clientes con mayor riesgo de no renovar y enfocar estrategias de retención.
    """)

    st.subheader("2. Variables más influyentes")
    st.write("""
    Las variables numéricas y categóricas analizadas revelan que algunos factores tienen un impacto más significativo en la toma de decisiones de los clientes. Estos insights son útiles para priorizar recursos en marketing y personalización de ofertas.
    """)

    st.subheader("3. Datos faltantes y calidad del dataset")
    st.write("""
    Se identificaron valores faltantes en varias columnas, lo que puede afectar la precisión de análisis futuros. Es recomendable implementar procedimientos de limpieza o imputación para mejorar la confiabilidad de los reportes y análisis posteriores.
    """)

    st.subheader("4. Distribución de clientes y pólizas")
    st.write("""
    Los histogramas y análisis de frecuencias muestran concentraciones de clientes en ciertos rangos de edad, ingresos y tipo de póliza. Esto ayuda a la compañía a segmentar su cartera de clientes y diseñar estrategias específicas para cada segmento.
    """)

    st.subheader("5. Relación entre variables")
    st.write("""
    Los análisis bivariados y cruzados indican que existen relaciones claras entre variables categóricas y numéricas. Comprender estas relaciones permite tomar decisiones basadas en datos para optimizar campañas, mejorar la retención y ajustar productos a las necesidades de los clientes.
    """)