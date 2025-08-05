import streamlit as st
import pandas as pd
import io
from unidecode import unidecode
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import gc
import os
from datetime import datetime
import hashlib

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Monitoramento de Equipamentos Climáticos",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Injeção de CSS para Tema Escuro e Ajuste de Abas ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# CSS adicional para garantir abas horizontais e layout
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            flex-wrap: nowrap;
            overflow-x: auto;
            justify-content: flex-start;
            background-color: #121212;
            border-bottom: 1px solid #424242;
        }
        .stTabs [data-baseweb="tab"] {
            color: #FFFFFF;
            background-color: #121212;
            border: 1px solid #424242;
            margin-right: 8px;
            padding: 8px 16px;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [data-baseweb="tab"].stTabs--active {
            background-color: #2E7D32;
            color: #FFFFFF;
            border-bottom: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Diretório para Armazenamento de Arquivos ---
DATA_DIR = "uploaded_files"
os.makedirs(DATA_DIR, exist_ok=True)
EXCEL_PATH = os.path.join(DATA_DIR, "equipamentos.xlsx")
KML_PATH = os.path.join(DATA_DIR, "fazendas.kml")

# --- Funções Auxiliares ---
@st.cache_data
def get_file_hash(file):
    """Calcula o hash SHA-256 de um arquivo."""
    hasher = hashlib.sha256()
    file.seek(0)
    hasher.update(file.read())
    file.seek(0)
    return hasher.hexdigest()

@st.cache_data
def save_uploaded_file(uploaded_file, file_path):
    """Salva o arquivo enviado no disco."""
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return get_file_hash(uploaded_file)

@st.cache_data
def check_file_updates(uploaded_file, file_path):
    """Verifica se o arquivo enviado é diferente do salvo."""
    if not os.path.exists(file_path):
        return True
    with open(file_path, "rb") as f:
        saved_hash = hashlib.sha256(f.read()).hexdigest()
    return saved_hash != get_file_hash(uploaded_file)

@st.cache_data
def extrair_dados_kml(kml_content):
    """Extrai metadados e geometria de um conteúdo KML."""
    try:
        tree = ET.fromstring(kml_content)
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        dados = []
        for placemark in tree.findall('.//kml:Placemark', ns):
            props = {}
            name_elem = placemark.find('kml:name', ns)
            props['Name'] = name_elem.text if name_elem is not None else None
            for simple_data in placemark.findall('.//kml:SimpleData', ns):
                props[simple_data.get('name')] = simple_data.text

            geometry = None
            polygon_elem = placemark.find('.//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
            if polygon_elem is not None:
                coords_text = polygon_elem.text.strip()
                coords = [tuple(map(float, c.split(','))) for c in coords_text.split()]
                try:
                    geometry = Polygon([(c[0], c[1]) for c in coords])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria Polygon: {geom_e}")
                    geometry = None

            line_elem = placemark.find('.//kml:LineString/kml:coordinates', ns)
            if line_elem is not None:
                coords_text = line_elem.text.strip()
                coords = [tuple(map(float, c.split(','))) for c in coords_text.split()]
                try:
                    geometry = LineString([(c[0], c[1]) for c in coords])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria LineString: {geom_e}")
                    geometry = None

            point_elem = placemark.find('.//kml:Point/kml:coordinates', ns)
            if point_elem is not None:
                coords_text = point_elem.text.strip()
                coords = tuple(map(float, coords_text.split(',')))
                try:
                    geometry = Point(coords[0], coords[1])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria Point: {geom_e}")
                    geometry = None

            if geometry:
                dados.append({**props, 'geometry': geometry})

        if not dados:
            st.warning("Nenhuma geometria válida encontrada no KML.")
            return gpd.GeoDataFrame(columns=['Name', 'geometry'])

        gdf = gpd.GeoDataFrame(dados, crs="EPSG:4326")
        return gdf

    except Exception as e:
        st.error(f"Erro ao processar KML: {e}")
        return gpd.GeoDataFrame(columns=['Name', 'geometry'])

@st.cache_data
def formatar_nome(nome):
    """Padroniza nomes removendo acentos e convertendo para maiúsculas."""
    return unidecode(str(nome).upper()) if isinstance(nome, (str, bytes)) else nome

@st.cache_data
def normalizar_coordenadas(valor, scale_factor=1000000000):
    """Normaliza e valida coordenadas."""
    if isinstance(valor, str):
        try:
            valor_float = float(valor.replace(',', '')) / scale_factor
            return round(valor_float, 6)
        except ValueError:
            return None
    elif isinstance(valor, (int, float)):
        return round(float(valor) / scale_factor, 6)
    return None

@st.cache_data
def classificar_dbm(valor):
    """Classifica o valor DBM em categorias de sinal."""
    if pd.isna(valor):
        return np.nan
    try:
        valor_float = float(valor)
        if valor_float > -70:
            return 4  # ótimo
        elif valor_float > -85:
            return 3  # bom
        elif valor_float > -100:
            return 2  # regular
        else:
            return 1  # ruim
    except (ValueError, TypeError):
        valor_str = unidecode(str(valor)).strip().lower()
        mapping = {"otimo": 4, "bom": 3, "regular": 2, "ruim": 1}
        return mapping.get(valor_str, np.nan)

@st.cache_data(show_spinner="Calculando interpolação IDW...")
def interpolacao_idw(_df, x_col='VL_LONGITUDE', y_col='VL_LATITUDE', val_col='DBM', resolution=0.002, buffer=0.05, _geom_mask=None):
    _df = _df.dropna(subset=[val_col]).copy()
    if _df.empty:
        st.warning("Nenhum dado válido para interpolação.")
        return None, None, None, None

    if _df[val_col].dropna().between(1, 4).all():
        _df['class_num'] = _df[val_col].astype(int)
    else:
        _df['class_num'] = _df[val_col].apply(classificar_dbm)

    _df = _df.dropna(subset=['class_num'])
    if _df.empty:
        st.warning("Nenhum dado válido após classificação.")
        return None, None, None, None

    minx, miny = _df[x_col].min() - buffer, _df[y_col].min() - buffer
    maxx, maxy = _df[x_col].max() + buffer, _df[y_col].max() + buffer

    x_grid = np.arange(minx, maxx, resolution)
    y_grid = np.arange(miny, maxy, resolution)

    if len(x_grid) * len(y_grid) > 1_000_000:
        st.warning("Grade de interpolação muito grande.")
        return None, None, None, None

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    pontos = _df[[x_col, y_col]].values
    valores = _df['class_num'].values

    distances = cdist(grid_points, pontos)
    epsilon = 1e-9
    weights = 1 / (distances**2 + epsilon)
    denom = weights.sum(axis=1)
    numer = (weights * valores).sum(axis=1)
    interpolated = np.clip(np.round(numer / denom), 1, 4)

    if _geom_mask is not None:
        grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_points[:, 0], grid_points[:, 1]), crs="EPSG:4326")
        mask_series = grid_gdf.within(_geom_mask)
        interpolated[~mask_series.values] = np.nan

    grid_numerico = interpolated.reshape(len(y_grid), len(x_grid))
    del distances, weights, interpolated
    gc.collect()

    return grid_x.reshape(len(y_grid), len(x_grid)), grid_y.reshape(len(y_grid), len(x_grid)), grid_numerico, (minx, maxx, miny, maxy)

def estilo_ponto(row):
    mapa_frota_icones = {
        "pluviometro": ("#4CAF50", "o"),  # Verde
        "estacao": ("#9C27B0", "o")      # Roxo
    }
    frota = str(row.get("DESC_TIPO_EQUIPAMENTO", "")).strip().lower()
    return mapa_frota_icones.get(frota, ("#F44336", "o"))  # Vermelho padrão

def plotar_interpolacao(grid_x, grid_y, grid_numerico, geom_fazenda, bounds, df_pontos, unidade):
    minx, maxx, miny, maxy = bounds
    colors = {1: '#F44336', 2: '#FFCA28', 3: '#4CAF50', 4: '#1B5E20'}  # Vermelho, Amarelo, Verde, Verde Escuro
    cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(1, 5)])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#121212')  # Tamanho reduzido
    ax.set_facecolor('#121212')

    # Plotar a camada raster (interpolação IDW)
    im = ax.imshow(grid_numerico, extent=(minx, maxx, miny, maxy), origin='lower', cmap=cmap, interpolation='nearest', alpha=0.8)

    # Plotar os limites da fazenda sem legenda
    if geom_fazenda is not None and not geom_fazenda.is_empty:
        try:
            if isinstance(geom_fazenda, MultiPolygon):
                for part in geom_fazenda.geoms:
                    if part.is_valid:
                        gpd.GeoSeries([part]).boundary.plot(ax=ax, color='#FFFFFF', linewidth=2)
            else:
                if geom_fazenda.is_valid:
                    gpd.GeoSeries([geom_fazenda]).boundary.plot(ax=ax, color='#FFFFFF', linewidth=2)
                else:
                    st.warning(f"Geometria inválida para a fazenda '{unidade}'. Tentando corrigir...")
                    geom_fazenda = geom_fazenda.buffer(0)
                    if geom_fazenda.is_valid:
                        gpd.GeoSeries([geom_fazenda]).boundary.plot(ax=ax, color='#FFFFFF', linewidth=2)
                    else:
                        st.error(f"Não foi possível corrigir a geometria para a fazenda '{unidade}'.")
        except Exception as e:
            st.error(f"Erro ao plotar limites da fazenda '{unidade}': {e}")

    # Plotar os pontos dos equipamentos
    legenda = {}
    for _, row in df_pontos.iterrows():
        cor, marcador = estilo_ponto(row)
        label = row.get("DESC_TIPO_EQUIPAMENTO", "N/A")
        if label not in legenda:
            legenda[label] = ax.scatter(
                row["VL_LONGITUDE"], row["VL_LATITUDE"], c=cor, marker=marcador, s=100,
                edgecolor="#FFFFFF", linewidth=0.7, label=label, alpha=0.9
            )
        else:
            ax.scatter(
                row["VL_LONGITUDE"], row["VL_LATITUDE"], c=cor, marker=marcador, s=100,
                edgecolor="#FFFFFF", linewidth=0.7, alpha=0.9
            )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("Longitude", color='#E0E0E0')
    ax.set_ylabel("Latitude", color='#E0E0E0')
    ax.set_title(f"Intensidade do Sinal - {unidade}", color='#FFFFFF')
    ax.tick_params(colors='#E0E0E0')

    cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
    cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'], color='#FFFFFF')
    cbar.set_label('Classe de Sinal', color='#FFFFFF')
    cbar.ax.set_facecolor('#424242')

    ax.legend(title="Legenda", loc='upper right', markerscale=1.2, facecolor='#424242', edgecolor='#FFFFFF', labelcolor='#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#424242')

    st.pyplot(fig)
    plt.close(fig)
    gc.collect()

# --- Função para Página de Upload (Ocultada das Abas) ---
def upload_page():
    st.markdown("<h2 class='sidebar-header'>Gerenciamento de Arquivos</h2>", unsafe_allow_html=True)

    excel_file = st.file_uploader("Selecione o arquivo Excel (.xlsx)", type=["xlsx"], key="excel_uploader")
    kml_file = st.file_uploader("Selecione o arquivo KML (.kml - Opcional)", type=["kml"], key="kml_uploader")

    # Estado para armazenar hashes dos arquivos
    if 'excel_hash' not in st.session_state:
        st.session_state.excel_hash = None
    if 'kml_hash' not in st.session_state:
        st.session_state.kml_hash = None

    # Processar uploads
    global df_csv, gdf_kml
    if excel_file:
        if check_file_updates(excel_file, EXCEL_PATH):
            st.session_state.excel_hash = save_uploaded_file(excel_file, EXCEL_PATH)
            st.success("Arquivo Excel salvo com sucesso!")
        else:
            st.info("Arquivo Excel já está atualizado.")

    if kml_file:
        if check_file_updates(kml_file, KML_PATH):
            st.session_state.kml_hash = save_uploaded_file(kml_file, KML_PATH)
            st.success("Arquivo KML salvo com sucesso!")
        else:
            st.info("Arquivo KML já está atualizado.")

    if st.button("Atualizar Dados", key="update_button"):
        if os.path.exists(EXCEL_PATH):
            try:
                df_csv = pd.read_excel(EXCEL_PATH, dtype={'VL_LATITUDE': str, 'VL_LONGITUDE': str, 'VL_FIRMWARE_EQUIPAMENTO': str})
                df_csv.columns = df_csv.columns.str.strip()
                df_csv["VL_LATITUDE"] = df_csv["VL_LATITUDE"].apply(normalizar_coordenadas)
                df_csv["VL_LONGITUDE"] = df_csv["VL_LONGITUDE"].apply(normalizar_coordenadas)
                df_csv["UNIDADE"] = df_csv["UNIDADE"].apply(formatar_nome)
                df_csv["VL_FIRMWARE_EQUIPAMENTO"] = df_csv["VL_FIRMWARE_EQUIPAMENTO"].astype(str).replace('nan', None)
                df_csv = df_csv.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"])
                df_csv = df_csv[(df_csv["VL_LATITUDE"].between(-90, 90)) & (df_csv["VL_LONGITUDE"].between(-180, 180))]
                st.success("Dados Excel atualizados!")
            except Exception as e:
                st.error(f"Erro ao atualizar Excel: {e}")
        else:
            st.error("Nenhum arquivo Excel salvo para atualizar.")

        if os.path.exists(KML_PATH):
            try:
                with open(KML_PATH, 'r', encoding='utf-8') as f:
                    kml_content = f.read()
                gdf_kml = extrair_dados_kml(kml_content)
                if gdf_kml is not None and not gdf_kml.empty:
                    gdf_kml['NomeFazendaExtraido'] = gdf_kml.get('NOME_FAZ', gdf_kml.get('Name', 'sem_nome'))
                    gdf_kml['NomeFazendaKML_Padronizada'] = gdf_kml['NomeFazendaExtraido'].apply(formatar_nome)
                    gdf_kml['geometry'] = gdf_kml['geometry'].apply(lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom)
                    gdf_kml = gdf_kml[gdf_kml['geometry'].notna() & ~gdf_kml['geometry'].is_empty]
                    st.success("Dados KML atualizados!")
                else:
                    gdf_kml = None
                    st.warning("KML atualizado, mas nenhuma geometria válida encontrada.")
            except Exception as e:
                st.error(f"Erro ao atualizar KML: {e}")
                gdf_kml = None
        else:
            st.info("Nenhum arquivo KML salvo para atualizar.")

# --- Sidebar com Botão para Upload ---
st.sidebar.markdown("<h2 class='sidebar-header'>Navegação</h2>", unsafe_allow_html=True)
if st.sidebar.button("Gerenciar Arquivos", key="upload_page_button"):
    st.session_state.page = "upload"
else:
    st.session_state.page = st.session_state.get("page", "dashboard")

# --- Carregar Arquivos Salvos na Inicialização ---
df_csv = None
gdf_kml = None
if os.path.exists(EXCEL_PATH):
    try:
        df_csv = pd.read_excel(EXCEL_PATH, dtype={'VL_LATITUDE': str, 'VL_LONGITUDE': str, 'VL_FIRMWARE_EQUIPAMENTO': str})
        df_csv.columns = df_csv.columns.str.strip()
        df_csv["VL_LATITUDE"] = df_csv["VL_LATITUDE"].apply(normalizar_coordenadas)
        df_csv["VL_LONGITUDE"] = df_csv["VL_LONGITUDE"].apply(normalizar_coordenadas)
        df_csv["UNIDADE"] = df_csv["UNIDADE"].apply(formatar_nome)
        df_csv["VL_FIRMWARE_EQUIPAMENTO"] = df_csv["VL_FIRMWARE_EQUIPAMENTO"].astype(str).replace('nan', None)
        df_csv = df_csv.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"])
        df_csv = df_csv[(df_csv["VL_LATITUDE"].between(-90, 90)) & (df_csv["VL_LONGITUDE"].between(-180, 180))]
    except Exception as e:
        st.error(f"Erro ao carregar Excel salvo: {e}")

if os.path.exists(KML_PATH):
    try:
        with open(KML_PATH, 'r', encoding='utf-8') as f:
            kml_content = f.read()
        gdf_kml = extrair_dados_kml(kml_content)
        if gdf_kml is not None and not gdf_kml.empty:
            gdf_kml['NomeFazendaExtraido'] = gdf_kml.get('NOME_FAZ', gdf_kml.get('Name', 'sem_nome'))
            gdf_kml['NomeFazendaKML_Padronizada'] = gdf_kml['NomeFazendaExtraido'].apply(formatar_nome)
            gdf_kml['geometry'] = gdf_kml['geometry'].apply(lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom)
            gdf_kml = gdf_kml[gdf_kml['geometry'].notna() & ~gdf_kml['geometry'].is_empty]
    except Exception as e:
        st.error(f"Erro ao carregar KML salvo: {e}")
        gdf_kml = None

# --- Dashboard Principal ---
if st.session_state.page == "upload":
    upload_page()
else:
    st.markdown("<h1 class='main-title'>Monitoramento de Equipamentos Climáticos</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p class='description'>Visualize dados de equipamentos climáticos e analise a intensidade do sinal em suas fazendas.</p>
    """, unsafe_allow_html=True)

    if df_csv is not None and not df_csv.empty:
        # Abas no topo
        tab_labels = ["📊 Visão Geral", "📈 Firmware", "📡 Sinal", "🌎 Mapa de Equipamentos"]
        tab_visao, tab_firmware, tab_sinal, tab_mapa = st.tabs(tab_labels)

        with tab_visao:
            st.markdown("<h2 class='section-title'>Visão Geral</h2>", unsafe_allow_html=True)
            # Paleta de cores para gráficos
            cores_personalizadas = ["#2E7D32", "#1565C0", "#FFCA28", "#E64A19"]  # Verde, Azul, Amarelo, Laranja
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h4 class='sub-subsection-title'>Percentual de Equipamentos com Dados Móveis</h4>", unsafe_allow_html=True)
                if 'D_MOVEIS_AT' in df_csv.columns:
                    # Padronizar valores de D_MOVEIS_AT
                    df_csv['D_MOVEIS_AT'] = df_csv['D_MOVEIS_AT'].str.upper().replace({'SIM': 'Sim', 'NAO': 'Não'})
                    contagem_moveis = df_csv['D_MOVEIS_AT'].value_counts()
                    if not contagem_moveis.empty:
                        fig_4g = px.pie(
                            values=contagem_moveis.values,
                            names=contagem_moveis.index,
                            title='<b>Percentual de Equipamentos com Dados Móveis</b>',
                            hole=0.5,
                            color_discrete_sequence=
