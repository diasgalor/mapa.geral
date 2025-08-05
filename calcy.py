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

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#121212')
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
                    contagem_moveis = df_csv['D_MOVEIS_AT'].value_counts()
                    fig_4g = px.pie(
                        values=contagem_moveis.values,
                        names=contagem_moveis.index,
                        title='<b>Percentual de Equipamentos com Dados Móveis</b>',
                        hole=0.5,
                        color_discrete_sequence=cores_personalizadas
                    )
                    fig_4g.update_traces(textinfo='percent+label', textfont_size=14, textfont_color='#FFFFFF')
                    fig_4g.update_layout(
                        showlegend=True,
                        legend_title='Dados Móveis',
                        plot_bgcolor='#121212',
                        paper_bgcolor='#121212',
                        font_color='#FFFFFF',
                        title_font_color='#FFFFFF',
                        legend=dict(bgcolor='#424242', font=dict(color='#FFFFFF')),
                        height=400
                    )
                    st.plotly_chart(fig_4g, use_container_width=True)
                else:
                    st.warning("Coluna 'D_MOVEIS_AT' não encontrada.")

            with col2:
                st.markdown("<h4 class='sub-subsection-title'>Distribuição de Dados Móveis e Solinfnet por Unidade</h4>", unsafe_allow_html=True)
                if 'TIPO_COMUNICACAO' in df_csv.columns and 'UNIDADE' in df_csv.columns:
                    df_contagem_com = df_csv[df_csv['TIPO_COMUNICACAO'].isin(['Dados Móveis', 'Solinfnet'])].groupby(['UNIDADE', 'TIPO_COMUNICACAO']).size().reset_index(name='Quantidade')
                    fig_com = px.bar(
                        df_contagem_com,
                        x='UNIDADE',
                        y='Quantidade',
                        color='TIPO_COMUNICACAO',
                        title='<b>Dados Móveis e Solinfnet por Unidade</b>',
                        text='Quantidade',
                        barmode='stack',
                        color_discrete_sequence=cores_personalizadas
                    )
                    fig_com.update_layout(
                        plot_bgcolor='#121212',
                        paper_bgcolor='#121212',
                        font_color='#FFFFFF',
                        title_font_color='#FFFFFF',
                        legend=dict(bgcolor='#424242', font=dict(color='#FFFFFF')),
                        height=400
                    )
                    st.plotly_chart(fig_com, use_container_width=True)
                else:
                    st.warning("Colunas 'TIPO_COMUNICACAO' ou 'UNIDADE' não encontradas.")

        with tab_firmware:
            st.markdown("<h2 class='section-title'>Distribuição de Firmwares por Unidade</h2>", unsafe_allow_html=True)
            if 'VL_FIRMWARE_EQUIPAMENTO' in df_csv.columns and 'UNIDADE' in df_csv.columns:
                # Filtrar valores nulos ou vazios
                df_firmware_fazenda = df_csv.dropna(subset=['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE'])
                df_firmware_fazenda = df_firmware_fazenda[df_firmware_fazenda['VL_FIRMWARE_EQUIPAMENTO'] != '']
                if not df_firmware_fazenda.empty:
                    df_firmware_fazenda = df_firmware_fazenda.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
                    fig_firmware = px.bar(
                        df_firmware_fazenda,
                        x='Quantidade',
                        y='UNIDADE',
                        color='VL_FIRMWARE_EQUIPAMENTO',
                        title='<b>Distribuição de Firmwares por Unidade</b>',
                        labels={'VL_FIRMWARE_EQUIPAMENTO': 'Firmware', 'UNIDADE': 'Unidade', 'Quantidade': 'Qtd.'},
                        orientation='h',
                        text='Quantidade',
                        color_discrete_sequence=cores_personalizadas
                    )
                    fig_firmware.update_layout(
                        plot_bgcolor='#121212',
                        paper_bgcolor='#121212',
                        font_color='#FFFFFF',
                        title_font_color='#FFFFFF',
                        legend=dict(bgcolor='#424242', font=dict(color='#FFFFFF')),
                        height=600,
                        xaxis=dict(title='Quantidade de Equipamentos'),
                        yaxis=dict(title='')
                    )
                    st.plotly_chart(fig_firmware, use_container_width=True)
                else:
                    st.warning("Nenhum dado válido encontrado nas colunas 'VL_FIRMWARE_EQUIPAMENTO' ou 'UNIDADE' após filtragem.")
            else:
                st.warning("Colunas 'VL_FIRMWARE_EQUIPAMENTO' ou 'UNIDADE' não encontradas.")

        with tab_sinal:
            st.markdown("<h2 class='section-title'>Mapa de Intensidade do Sinal</h2>", unsafe_allow_html=True)
            signal_cols_ok = all(col in df_csv.columns for col in ['VL_LATITUDE', 'VL_LONGITUDE', 'UNIDADE', 'DESC_TIPO_EQUIPAMENTO'])
            
            if signal_cols_ok and gdf_kml is not None and not gdf_kml.empty:
                gdf_equipamentos = gpd.GeoDataFrame(
                    df_csv,
                    geometry=gpd.points_from_xy(df_csv['VL_LONGITUDE'], df_csv['VL_LATITUDE']),
                    crs="EPSG:4326"
                )
                gdf_equipamentos['UNIDADE_Padronizada'] = gdf_equipamentos['UNIDADE'].apply(formatar_nome)
                unidades_disponiveis = sorted(list(set(gdf_kml['NomeFazendaKML_Padronizada'].dropna()) & set(gdf_equipamentos['UNIDADE_Padronizada'].dropna())))

                if unidades_disponiveis:
                    if 'selected_unidade_sinal' not in st.session_state:
                        st.session_state.selected_unidade_sinal = unidades_disponiveis[0]

                    selected_unidade = st.selectbox(
                        "Selecione a Fazenda para Interpolação:",
                        unidades_disponiveis,
                        key="fazenda_sinal",
                        index=unidades_disponiveis.index(st.session_state.selected_unidade_sinal) if st.session_state.selected_unidade_sinal in unidades_disponiveis else 0,
                        on_change=lambda: st.session_state.update(selected_unidade_sinal=st.session_state.fazenda_sinal)
                    )

                    with st.spinner("Gerando mapa de sinal..."):
                        cache_key = f"interpolacao_idw_{selected_unidade}"
                        interpolacao_idw.clear()
                        df_fazenda = gdf_equipamentos[gdf_equipamentos['UNIDADE_Padronizada'] == selected_unidade].copy()
                        geom_df = gdf_kml[gdf_kml['NomeFazendaKML_Padronizada'] == selected_unidade]

                        if not df_fazenda.empty and not geom_df.empty:
                            try:
                                fazenda_geom = geom_df.unary_union
                                if not fazenda_geom.is_valid:
                                    fazenda_geom = fazenda_geom.buffer(0)
                                    if not fazenda_geom.is_valid:
                                        st.error(f"Geometria inválida para a fazenda '{selected_unidade}' após tentativa de correção.")
                                        fazenda_geom = None
                            except Exception as e:
                                st.error(f"Erro ao processar geometria da fazenda '{selected_unidade}': {e}")
                                fazenda_geom = None

                            if fazenda_geom is not None:
                                df_fazenda['DBM'] = pd.to_numeric(df_fazenda['DBM'], errors='coerce')
                                has_dbm = 'DBM' in df_fazenda.columns and not df_fazenda['DBM'].dropna().empty
                                has_intensidade = 'INTENSIDADE' in df_fazenda.columns and not df_fazenda['INTENSIDADE'].dropna().empty
                                mapping = {"ruim": 1, "regular": 2, "bom": 3, "otimo": 4}

                                val_col_used = None
                                if has_dbm and has_intensidade:
                                    df_fazenda['INTENSIDADE_MAP'] = df_fazenda['INTENSIDADE'].apply(
                                        lambda x: mapping.get(unidecode(str(x)).strip().lower(), np.nan)
                                    )
                                    for idx, row in df_fazenda.iterrows():
                                        if pd.isna(row['DBM']) and pd.notna(row.get('INTENSIDADE_MAP')):
                                            df_fazenda.loc[idx, 'DBM'] = row['INTENSIDADE_MAP']
                                    val_col_used = 'DBM'
                                elif has_dbm:
                                    val_col_used = 'DBM'
                                elif has_intensidade:
                                    df_fazenda['DBM'] = df_fazenda['INTENSIDADE'].apply(
                                        lambda x: mapping.get(unidecode(str(x)).strip().lower(), np.nan)
                                    )
                                    val_col_used = 'DBM'

                                if val_col_used:
                                    df_fazenda_filtered = df_fazenda.dropna(subset=[val_col_used])
                                    if not df_fazenda_filtered.empty:
                                        grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(
                                            _df=df_fazenda_filtered,
                                            x_col='VL_LONGITUDE',
                                            y_col='VL_LATITUDE',
                                            val_col=val_col_used,
                                            resolution=0.002,
                                            buffer=0.05,
                                            _geom_mask=fazenda_geom
                                        )
                                        if grid_x is not None:
                                            plotar_interpolacao(grid_x, grid_y, grid_numerico, fazenda_geom, bounds, df_fazenda_filtered, selected_unidade)
                                        else:
                                            st.warning("Não foi possível gerar a interpolação para esta fazenda.")
                                    else:
                                        st.warning("Nenhum dado de sinal válido para esta fazenda.")
                                else:
                                    st.info("Colunas 'DBM' ou 'INTENSIDADE' necessárias para interpolação.")
                            else:
                                st.warning(f"Nenhuma geometria válida para a fazenda '{selected_unidade}'.")
                        else:
                            st.warning(f"Nenhum equipamento ou geometria encontrada para a fazenda '{selected_unidade}'.")
                else:
                    st.warning("Nenhuma unidade correspondente encontrada entre os arquivos Excel e KML.")
            else:
                st.info("Faça o upload dos arquivos Excel e KML para habilitar o mapa de sinal.")

        with tab_mapa:
            st.markdown("<h2 class='section-title'>Mapa Interativo de Equipamentos</h2>", unsafe_allow_html=True)
            map_cols_ok = all(col in df_csv.columns for col in ['DESC_TIPO_EQUIPAMENTO', 'FROTA', 'STATUS'])
            if map_cols_ok:
                mapa = folium.Map(location=[df_csv["VL_LATITUDE"].mean(), df_csv["VL_LONGITUDE"].mean()], zoom_start=6)
                marker_cluster = MarkerCluster().add_to(mapa)

                df_estacoes = df_csv[df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO", case=False, na=False)]
                for _, row in df_estacoes.iterrows():
                    if pd.notna(row["VL_LATITUDE"]) and pd.notna(row["VL_LONGITUDE"]):
                        folium.Marker(
                            location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                            popup=f"<b>Frota:</b> {row['FROTA']}<br><b>Unidade:</b> {row['UNIDADE']}<br><b>Tipo:</b> {row['DESC_TIPO_EQUIPAMENTO']}",
                            icon=folium.Icon(color="purple", icon="cloud", prefix="fa")
                        ).add_to(marker_cluster)

                df_pluviometros_ativos = df_csv[
                    (df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("PLUVIOMETRO", case=False, na=False)) &
                    (df_csv["STATUS"].str.upper() == "ATIVO")
                ]
                for _, row in df_pluviometros_ativos.iterrows():
                    if pd.notna(row["VL_LATITUDE"]) and pd.notna(row["VL_LONGITUDE"]):
                        folium.Marker(
                            location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                            popup=f"<b>Frota:</b> {row['FROTA']}<br><b>Unidade:</b> {row['UNIDADE']}<br><b>Tipo:</b> {row['DESC_TIPO_EQUIPAMENTO']}",
                            icon=folium.Icon(color="green", icon="tint", prefix="fa")
                        ).add_to(marker_cluster)

                if gdf_kml is not None and not gdf_kml.empty:
                    folium.GeoJson(
                        gdf_kml,
                        name="Limites das Fazendas",
                        tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Fazenda:"]),
                        style_function=lambda x: {"fillColor": "#1565C0", "color": "#FFFFFF", "weight": 1, "fillOpacity": 0.2}
                    ).add_to(mapa)

                folium.LayerControl().add_to(mapa)
                st.components.v1.html(mapa._repr_html_(), height=500)
            else:
                st.error("Colunas necessárias para o mapa de equipamentos não encontradas.")

    else:
        st.info("Faça o upload de um arquivo Excel para começar.")

    st.markdown("<div class='footer'>Desenvolvido para Monitoramento Agrícola</div>", unsafe_allow_html=True)
