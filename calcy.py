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

# --- Injeção de CSS para Tema Escuro ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    return None

@st.cache_data
def classificar_dbm(valor):
    """Classifica o valor DBM em categorias de sinal."""
    if pd.isna(valor):
        return np.nan
    elif valor > -70:
        return 4  # ótimo
    elif valor > -85:
        return 3  # bom
    elif valor > -100:
        return 2  # regular
    else:
        return 1  # ruim

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

# Função ajustada para plotar a interpolação
def plotar_interpolacao(grid_x, grid_y, grid_numerico, geom_fazenda, bounds, df_pontos, unidade):
    minx, maxx, miny, maxy = bounds
    colors = {1: '#F44336', 2: '#FFCA28', 3: '#4CAF50', 4: '#1B5E20'}  # Vermelho, Amarelo, Verde, Verde Escuro
    cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(1, 5)])

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#121212')
    ax.set_facecolor('#121212')

    # Plotar a camada raster (interpolação IDW)
    im = ax.imshow(grid_numerico, extent=(minx, maxx, miny, maxy), origin='lower', cmap=cmap, interpolation='nearest', alpha=0.8)

    # Plotar os limites da fazenda
    if geom_fazenda is not None and not geom_fazenda.is_empty:
        if isinstance(geom_fazenda, MultiPolygon):
            for part in geom_fazenda.geoms:
                if part.is_valid:
                    gpd.GeoSeries([part]).boundary.plot(ax=ax, color='#FFFFFF', linewidth=2, label=f'Limites de {unidade}')
        else:
            if geom_fazenda.is_valid:
                gpd.GeoSeries([geom_fazenda]).boundary.plot(ax=ax, color='#FFFFFF', linewidth=2, label=f'Limites de {unidade}')

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
    ax.set_title(f"Intensidade do Sinal - {unidade}", color='#E0E0E0')
    ax.tick_params(colors='#E0E0E0')

    cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
    cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'], color='#E0E0E0')
    cbar.set_label('Classe de Sinal', color='#E0E0E0')

    ax.legend(title="Legenda", loc='upper right', markerscale=1.2, facecolor='#1E1E1E', edgecolor='#424242', labelcolor='#E0E0E0')
    ax.grid(True, linestyle='--', alpha=0.5, color='#424242')
    
    st.pyplot(fig)
    plt.close(fig)
    gc.collect()

# Aba do Mapa de Sinal (substitua na seção correspondente)
with tab_sinal:
    st.markdown("<h3 class='subsection-title'>Mapa de Intensidade do Sinal</h3>", unsafe_allow_html=True)
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
            # Gerenciar estado da selectbox
            if 'selected_unidade_sinal' not in st.session_state:
                st.session_state.selected_unidade_sinal = unidades_disponiveis[0]

            selected_unidade = st.selectbox(
                "Selecione a Fazenda para Interpolação:",
                unidades_disponiveis,
                key="fazenda_sinal",
                index=unidades_disponiveis.index(st.session_state.selected_unidade_sinal) if st.session_state.selected_unidade_sinal in unidades_disponiveis else 0,
                on_change=lambda: st.session_state.update(selected_unidade_sinal=st.session_state.fazenda_sinal)
            )

            # Limpar cache ao mudar a unidade
            cache_key = f"interpolacao_idw_{selected_unidade}"
            interpolacao_idw.clear()  # Limpar cache global da função
            df_fazenda = gdf_equipamentos[gdf_equipamentos['UNIDADE_Padronizada'] == selected_unidade].copy()
            geom_df = gdf_kml[gdf_kml['NomeFazendaKML_Padronizada'] == selected_unidade]

            if not df_fazenda.empty and not geom_df.empty:
                fazenda_geom = geom_df.unary_union
                if not fazenda_geom.is_valid:
                    fazenda_geom = fazenda_geom.buffer(0)

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
                        # Usar uma chave única para cache baseada na unidade
                        grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(
                            _df=df_fazenda_filtered,
                            x_col='VL_LONGITUDE',
                            y_col='VL_LATITUDE',
                            val_col=val_col_used,
                            resolution=0.002,
                            buffer=0.05,
                            _geom_mask=fazenda_geom,
                            _hash=cache_key  # Parâmetro adicional para forçar recálculo
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
                st.warning("Nenhum equipamento ou geometria encontrada para a fazenda selecionada.")
        else:
            st.warning("Nenhuma unidade correspondente encontrada entre os arquivos Excel e KML.")
    else:
        st.info("Faça o upload dos arquivos Excel e KML para habilitar o mapa de sinal.")

    with tab_firmware:
        st.markdown("<h3 class='subsection-title'>Distribuição de Firmwares por Unidade</h3>", unsafe_allow_html=True)
        if 'VL_FIRMWARE_EQUIPAMENTO' in df_csv.columns and 'UNIDADE' in df_csv.columns:
            df_firmware_fazenda = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
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
                font_color='#E0E0E0',
                height=600,
                xaxis=dict(title='Quantidade de Equipamentos'),
                yaxis=dict(title=''),
                legend_title='Versão do Firmware'
            )
            st.plotly_chart(fig_firmware, use_container_width=True)
        else:
            st.warning("Colunas 'VL_FIRMWARE_EQUIPAMENTO' ou 'UNIDADE' não encontradas.")

    with tab_comunicacao:
        st.markdown("<h3 class='subsection-title'>Relação de Equipamentos e Comunicação</h3>", unsafe_allow_html=True)
        if 'DESC_TIPO_EQUIPAMENTO' in df_csv.columns and 'UNIDADE' in df_csv.columns:
            df_contagem_1 = df_csv[
                df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO|PLUVIOMETRO", case=False, na=False)
            ].groupby(['UNIDADE', 'DESC_TIPO_EQUIPAMENTO']).size().reset_index(name='Quantidade')
            fig1 = px.bar(
                df_contagem_1,
                x='UNIDADE',
                y='Quantidade',
                color='DESC_TIPO_EQUIPAMENTO',
                title='Pluviômetros e Estações por Unidade',
                text='Quantidade',
                barmode='stack',
                color_discrete_sequence=cores_personalizadas
            )
            fig1.update_layout(plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#E0E0E0', height=450)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Colunas 'DESC_TIPO_EQUIPAMENTO' ou 'UNIDADE' não encontradas.")

        if 'TIPO_COMUNICACAO' in df_csv.columns and 'UNIDADE' in df_csv.columns:
            df_contagem_2 = df_csv[df_csv['TIPO_COMUNICACAO'] != '4G'].groupby(['UNIDADE', 'TIPO_COMUNICACAO']).size().reset_index(name='Quantidade')
            fig2 = px.bar(
                df_contagem_2,
                x='UNIDADE',
                y='Quantidade',
                color='TIPO_COMUNICACAO',
                title='Tipos de Comunicação por Unidade (Excluindo 4G)',
                text='Quantidade',
                barmode='stack',
                color_discrete_sequence=cores_personalizadas
            )
            fig2.update_layout(plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='#E0E0E0', height=450)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Colunas 'TIPO_COMUNICACAO' ou 'UNIDADE' não encontradas.")

else:
    st.info("Faça o upload de um arquivo Excel para começar.")

st.markdown("<div class='footer'>Desenvolvido para Monitoramento Agrícola</div>", unsafe_allow_html=True)
