# APP - CLASSISOLO

# BIBLIOTECAS
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import gc

from datetime import date
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)

# CONFIGURAÇÃO
st.set_page_config(
    page_title="ClassiSolo",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.main{
background-color:#0d1117;
color:white;
}
h1,h2,h3,h4{
color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 ClassiSolo")
st.markdown("""
Sistema de Classificação da Degradação do Solo utilizando:
- Sentinel-2
- NDVI (Vegetação)
- Machine Learning (RF, SVM, KNN, XGBoost)
- Área de estudo: Estado de Pernambuco
""")

# SIDEBAR
st.sidebar.image("Logo - UFRPE.png", width=400)
st.sidebar.title("⚙️ Configurações")

data_maxima = date(2026, 6, 16)
start_date = st.sidebar.date_input(
    "Data inicial",
    value=date(2020,1,1),
    max_value=data_maxima
)
end_date = st.sidebar.date_input(
    "Data final",
    value=date(2026,6,14),
    max_value=data_maxima
)
run_btn = st.sidebar.button("▶️ Executar", use_container_width=True)

# EARTH ENGINE
@st.cache_resource
def init_ee():
    try:
        ee.Initialize(project="earth-engine-denilson")
        return True
    except Exception as e:
        st.error(e)
        return False

if not init_ee():
    st.stop()

# SHAPEFILE
try:
    gdf = gpd.read_file("Pernambuco_FN.shp")
    gdf = gdf.to_crs(epsg=4326)
    gdf_union = gdf.unary_union
    gdf_simple = gdf_union.simplify(tolerance=0.02)
    geojson = json.loads(gpd.GeoSeries([gdf_simple], crs="EPSG:4326").to_json())
    region = ee.FeatureCollection(geojson).geometry()
    st.success("✅ Pernambuco carregado")
    del gdf, gdf_union, gdf_simple, geojson
    gc.collect()
except Exception as e:
    st.error(e)
    st.stop()

# FUNÇÕES
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloudBitMask = (1 << 10) | (1 << 11)
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    return image.updateMask(mask).divide(10000)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI')
    ndmi = image.normalizedDifference(['B8','B11']).rename('NDMI')
    ndbi = image.normalizedDifference(['B11','B8']).rename('NDBI')
    return image.addBands([ndvi, ndmi, ndbi])

# PROCESSAMENTO
if run_btn:
    with st.spinner("🛰️ Processando imagens..."):
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(mask_s2_clouds)
            .map(add_indices)
        )
        size = collection.size().getInfo()
        st.success(f"✅ Imagens: {size}")

        if size == 0:
            st.warning("Nenhuma imagem disponível para o período.")
            st.stop()

        composite = collection.median().clip(region)

    # CLASSIFICAÇÃO POR NDVI
    with st.spinner("Classificando degradação..."):
        ndvi_img = composite.select('NDVI')
        labels = ndvi_img.expression(
            "b(0) > 0.5 ? 1 : (b(0) > 0.2 ? 2 : 3)",
            {'b0': ndvi_img}
        ).rename('classe')

        # Percentuais
        hist = labels.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=500,
            maxPixels=1e9
        ).getInfo()
        hist_data = hist.get('classe', {})
        baixa_pix = hist_data.get('1', 0)
        moderada_pix = hist_data.get('2', 0)
        alta_pix = hist_data.get('3', 0)
        total_pix = baixa_pix + moderada_pix + alta_pix
        perc_baixa = (baixa_pix / total_pix) * 100 if total_pix else 0
        perc_moderada = (moderada_pix / total_pix) * 100 if total_pix else 0
        perc_alta = (alta_pix / total_pix) * 100 if total_pix else 0

    # AMOSTRAGEM PARA TREINO
    with st.spinner("Amostrando dados para treino dos modelos..."):
        bandas = ['B2','B3','B4','B8','B11','NDVI','NDMI','NDBI']
        img_treino = composite.select(bandas).addBands(labels.rename('classe'))

        amostra = img_treino.sample(
            region=region,
            scale=30,
            numPixels=5000,
            seed=42,
            geometries=False
        ).getInfo()

        X, y = [], []
        for f in amostra['features']:
            p = f['properties']
            if 'classe' not in p:
                continue
            classe = int(p['classe'])
            if classe == 0:
                continue
            X.append([p['B2'], p['B3'], p['B4'], p['B8'], p['B11'],
                      p['NDVI'], p['NDMI'], p['NDBI']])
            y.append(classe)

        del amostra, img_treino
        gc.collect()

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int8)

        st.success(f"✅ Amostras válidas: {len(X)}")

        if len(X) < 100:
            st.warning("Número insuficiente de amostras. Usando apenas classificação NDVI (sem ML).")
            tem_amostras = False
        else:
            tem_amostras = True

    # TREINO E AVALIAÇÃO DOS MODELOS
    if tem_amostras:
        y_enc = y - 1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.40, stratify=y_enc, random_state=42
        )
        y_test_orig = y_test + 1

        modelos = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            "SVM": SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=9,
                weights='distance'
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="mlogloss"
            )
        }

        resultados = []
        melhor_modelo = None
        melhor_nome = None
        melhor_acc = 0

        for nome, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            pred_enc = modelo.predict(X_test)
            pred_orig = pred_enc + 1
            acc = accuracy_score(y_test_orig, pred_orig)
            prec = precision_score(y_test_orig, pred_orig, average='weighted')
            rec = recall_score(y_test_orig, pred_orig, average='weighted')
            f1 = f1_score(y_test_orig, pred_orig, average='weighted')
            cv = cross_val_score(modelo, X, y_enc, cv=5).mean()
            resultados.append({
                "Modelo": nome,
                "Accuracy": acc,
                "Precisão": prec,
                "Recall": rec,
                "F1": f1,
                "CV": cv
            })
            if acc > melhor_acc:
                melhor_acc = acc
                melhor_modelo = modelo
                melhor_nome = nome

        resultados_df = pd.DataFrame(resultados).sort_values(by="Accuracy", ascending=False)

    else:
        # Sem amostras: placeholders
        melhor_modelo = None
        melhor_nome = "Nenhum modelo treinado"
        melhor_acc = 0
        resultados_df = pd.DataFrame(columns=["Modelo", "Accuracy", "Precisão", "Recall", "F1", "CV"])

    # ABAS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 CLASSIFICAÇÃO",
        "📊 MÉTRICAS",
        "📈 DISTRIBUIÇÃO",
        "📘 SOBRE"
    ])

    # ABA 1 - CLASSIFICAÇÃO
    with tab1:
        st.subheader(f"Classificação da Degradação do Solo")
        st.markdown(f"**Período:** {start_date} a {end_date}")
        st.markdown("**Critério:** Baixa (NDVI > 0,5), Moderada (0,2–0,5), Alta (< 0,2)")
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Baixa", f"{perc_baixa:.2f}%")
        col2.metric("🟡 Moderada", f"{perc_moderada:.2f}%")
        col3.metric("🔴 Alta", f"{perc_alta:.2f}%")

    # ABA 2 - MÉTRICAS (APENAS TABELA DE COMPARAÇÃO)
    with tab2:
        st.subheader("Comparação dos Modelos")
        if tem_amostras:
            st.success(f"🏆 Melhor Modelo: {melhor_nome}")
            st.dataframe(resultados_df, use_container_width=True)
            # Removidos os 4 cards de métricas e a tabela de relatório por classe
        else:
            st.info("ℹ️ Sem amostras suficientes para treinar modelos. A classificação é baseada apenas no NDVI.")

    # ABA 3 - DISTRIBUIÇÃO (gráfico de barras e boxplot)
        # ==================================================
    # ABA 3 - DISTRIBUIÇÃO (HISTOGRAMA + BOXPLOT POR CLASSE)
    # ==================================================
    with tab3:
        st.subheader("📊 Distribuição do NDVI por Classe de Degradação")

        # Amostragem de NDVI para cada classe (reutilizada para ambos os gráficos)
        with st.spinner("Amostrando valores de NDVI por classe..."):
            ndvi_by_class = {1: [], 2: [], 3: []}
            for classe in [1, 2, 3]:
                mask = labels.eq(classe)
                masked_ndvi = ndvi_img.updateMask(mask)
                sample = masked_ndvi.sample(
                    region=region,
                    scale=500,
                    numPixels=500,
                    seed=42,
                    geometries=False
                ).getInfo()
                samples = sample['features']
                values = [f['properties']['NDVI'] for f in samples 
                          if 'NDVI' in f['properties'] and f['properties']['NDVI'] is not None]
                ndvi_by_class[classe] = values

            # Preparar dados para os gráficos
            dados_classes = {
                'Baixa': ndvi_by_class[1],
                'Moderada': ndvi_by_class[2],
                'Alta': ndvi_by_class[3]
            }
            cores = {'Baixa': 'green', 'Moderada': 'yellow', 'Alta': 'red'}
            labels_ordem = ['Baixa', 'Moderada', 'Alta']

        # Criar duas colunas
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Histograma do NDVI por Classe")
            if any(len(v) > 0 for v in dados_classes.values()):
                fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
                # Plotar histogramas sobrepostos com transparência
                for nome_classe in labels_ordem:
                    valores = dados_classes[nome_classe]
                    if len(valores) > 0:
                        ax_hist.hist(valores, bins=20, alpha=0.5, label=nome_classe,
                                     color=cores[nome_classe], edgecolor='black', linewidth=0.5)
                ax_hist.set_xlabel("NDVI")
                ax_hist.set_ylabel("Frequência")
                ax_hist.set_title(f"Distribuição do NDVI\n{start_date} a {end_date}")
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                st.pyplot(fig_hist)
                plt.close(fig_hist)
            else:
                st.warning("Sem dados para histograma.")

        with col2:
            st.markdown("#### 📦 Boxplot do NDVI por Classe")
            # Preparar dados para boxplot 
            data_box = []
            labels_box = []
            cores_box = []
            for nome_classe in labels_ordem:
                valores = dados_classes[nome_classe]
                if len(valores) > 0:
                    data_box.append(valores)
                    labels_box.append(nome_classe)
                    cores_box.append(cores[nome_classe])

            if data_box:
                fig_box, ax_box = plt.subplots(figsize=(5, 3.5))
                bp = ax_box.boxplot(data_box, labels=labels_box, patch_artist=True,
                                    boxprops=dict(facecolor='lightblue'))
                # Aplica as cores nas caixas
                for patch, cor in zip(bp['boxes'], cores_box):
                    patch.set_facecolor(cor)
                    patch.set_alpha(0.6)
                ax_box.set_ylabel("NDVI")
                ax_box.set_title("Distribuição por nível de degradação")
                ax_box.grid(True, alpha=0.3)
                st.pyplot(fig_box)
                plt.close(fig_box)
            else:
                st.warning("Não foi possível obter amostras para todas as classes.")

    # ABA 4 - SOBRE
    with tab4:
        st.markdown("""
        # 🌱 ClassiSolo

        **Como funciona?**
        - Utiliza imagens Sentinel-2 do período selecionado.
        - Calcula o NDVI (Índice de Vegetação) para cada pixel.
        - Classifica o indice de Vegetação do solo em três níveis:
          - 🟢 **Baixa**: NDVI > 0,5 (vegetação densa)
          - 🟡 **Moderada**: NDVI entre 0,2 e 0,5 (vegetação rala)
          - 🔴 **Alta**: NDVI < 0,2 (solo exposto ou urbano)
     **Algoritmos testados:** Random Forest, SVM, KNN, XGBoost.

        """)

else:
    st.info("👈 Clique em Executar para iniciar a análise.")