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

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Monitoramento de Equipamentos Climáticos",
    page_icon="🌿",
    layout="wide", # Usa a largura total da página
    initial_sidebar_state="expanded"
)

# --- Injeção de CSS para um Design Limpo ---
# Carrega o CSS customizado
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Título e Descrição da Aplicação ---
st.markdown("<h1 class='main-title'>Monitoramento de Equipamentos Climáticos</h1>", unsafe_allow_html=True)
st.markdown("""
    <p class='description'>Esta aplicação permite visualizar dados de equipamentos climáticos, como estações e pluviômetros,
    e analisar a intensidade do sinal em suas fazendas. Faça o upload dos arquivos Excel e KML para começar.
    </p>
""", unsafe_allow_html=True)

# --- Funções Auxiliares ---

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
                    st.warning(f"Erro ao criar geometria Polygon para placemark {props.get('Name', 'Sem Nome')}: {geom_e}")
                    geometry = None

            line_elem = placemark.find('.//kml:LineString/kml:coordinates', ns)
            if line_elem is not None:
                coords_text = line_elem.text.strip()
                coords = [tuple(map(float, c.split(','))) for c in coords_text.split()]
                try:
                    geometry = LineString([(c[0], c[1]) for c in coords])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria LineString para placemark {props.get('Name', 'Sem Nome')}: {geom_e}")
                    geometry = None

            point_elem = placemark.find('.//kml:Point/kml:coordinates', ns)
            if point_elem is not None:
                coords_text = point_elem.text.strip()
                coords = tuple(map(float, coords_text.split(',')))
                try:
                    geometry = Point(coords[0], coords[1])
                except Exception as geom_e:
                    st.warning(f"Erro ao criar geometria Point para placemark {props.get('Name', 'Sem Nome')}: {geom_e}")
                    geometry = None

            if geometry:
                dados.append({**props, 'geometry': geometry})

        if not dados:
            st.warning("Aviso: Nenhuma geometria válida encontrada no KML.")
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
            valor_normalizado = round(valor_float, 6)
            return valor_normalizado
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
def interpolacao_idw(_df, x_col='VL_LONGITUDE', y_col='VL_LATITUDE', val_col='DBM',
                     resolution=0.002, buffer=0.05, _geom_mask=None):
    _df = _df.dropna(subset=[val_col]).copy()
    if _df.empty:
        st.warning("Nenhum dado válido para interpolação após filtrar DBM/Intensidade.")
        return None, None, None, None

    # Verifica se a coluna val_col já está entre 1 e 4 (classes de sinal)
    # Se não, aplica a classificação de DBM
    if _df[val_col].dropna().between(1, 4).all():
        _df['class_num'] = _df[val_col].astype(int)
    else:
        _df['class_num'] = _df[val_col].apply(classificar_dbm)

    # Filtrar NaN da coluna 'class_num' após a classificação
    _df = _df.dropna(subset=['class_num'])
    if _df.empty:
        st.warning("Nenhum dado válido para interpolação após classificação de sinal.")
        return None, None, None, None

    minx, miny = _df[x_col].min() - buffer, _df[y_col].min() - buffer
    maxx, maxy = _df[x_col].max() + buffer, _df[y_col].max() + buffer

    x_grid = np.arange(minx, maxx, resolution)
    y_grid = np.arange(miny, maxy, resolution)

    if len(x_grid) * len(y_grid) > 1_000_000:
        st.warning("Grade de interpolação muito grande, reduzindo resolução ou buffer.")
        # Poderia adicionar lógica para ajustar resolution/buffer automaticamente aqui
        return None, None, None, None

    x_grid = np.arange(minx, maxx, resolution)
    y_grid = np.arange(miny, maxy, resolution)

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)  # Sem o .ravel()
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))


    pontos = _df[[x_col, y_col]].values
    valores = _df['class_num'].values

    distances = cdist(grid_points, pontos)
    epsilon = 1e-9
    weights = 1 / (distances**2 + epsilon)
    denom = weights.sum(axis=1)
    numer = (weights * valores).sum(axis=1)
    interpolated = numer / denom
    interpolated = np.clip(np.round(interpolated), 1, 4)

    if _geom_mask is not None:
        # Criar um GeoDataFrame temporário para os pontos da grade
        grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_points[:, 0], grid_points[:, 1]), crs="EPSG:4326")
        # Realizar um join espacial com a geometria da fazenda
        # Use .sindex.query para otimizar se _geom_mask for um MultiPolygon grande
        if isinstance(_geom_mask, MultiPolygon):
             # Para MultiPolygon, iterar sobre cada polígono ou usar sindex.query_bulk
             # Para simplificar e evitar complexidade excessiva, usaremos 'within' diretamente para geometrias razoáveis
             # Para datasets muito grandes, uma abordagem mais otimizada seria necessária aqui
             # Isso cria uma série booleana que é True se o ponto está DENTRO de *qualquer* parte do MultiPolygon
            mask_series = grid_gdf.within(_geom_mask)
        else:
            mask_series = grid_gdf.within(_geom_mask)

        interpolated[~mask_series.values] = np.nan

    # Remodelar a grade para as dimensões originais de (len(y_grid), len(x_grid))
    grid_numerico = interpolated.reshape(len(y_grid), len(x_grid))

    del distances, weights, interpolated
    gc.collect()

    return grid_x.reshape(len(y_grid), len(x_grid)), grid_y.reshape(len(y_grid), len(x_grid)), grid_numerico, (minx, maxx, miny, maxy)

def estilo_ponto(row):
    mapa_frota_icones = {
        "pluviometro": ("blue", "o"),
        "estacao": ("purple", "o")
    }
    frota = str(row.get("DESC_TIPO_EQUIPAMENTO", "")).strip().lower()
    if frota in mapa_frota_icones:
        cor, marcador = mapa_frota_icones[frota]
    else:
        cor, marcador = ("red", "o")
    return cor, marcador

def plotar_interpolacao(grid_x, grid_y, grid_numerico, geom_fazenda, bounds, df_pontos):
    minx, maxx, miny, maxy = bounds

    colors = {
        1: '#fc8d59',  # ruim (vermelho/laranja)
        2: '#fee08b',  # regular (amarelo claro)
        3: '#91cf60',  # bom (verde claro)
        4: '#1a9850'   # ótimo (verde escuro)
    }
    cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(1, 5)])

    # Ajuste aqui para diminuir o tamanho do mapa
    fig, ax = plt.subplots(figsize=(10, 8)) # Reduzido de (12, 10) para (10, 8)

    # Note: imshow expects grid_numerico to have dimensions (rows, columns)
    # which corresponds to (y_grid_len, x_grid_len).
    # The extent is (left, right, bottom, top) -> (minx, maxx, miny, maxy)
    im = ax.imshow(grid_numerico,
                   extent=(minx, maxx, miny, maxy),
                   origin='lower', # Para garantir que (0,0) esteja no canto inferior esquerdo
                   cmap=cmap,
                   interpolation='nearest',
                   alpha=0.8)

    if geom_fazenda is not None and not geom_fazenda.is_empty:
        # Se for um MultiPolygon, plotar cada parte individualmente
        if isinstance(geom_fazenda, MultiPolygon):
            for part in geom_fazenda.geoms:
                if part.is_valid:
                    gpd.GeoSeries([part]).boundary.plot(ax=ax, color='black', linewidth=2)
        else:
            if geom_fazenda.is_valid:
                gpd.GeoDataFrame(geometry=[geom_fazenda]).boundary.plot(ax=ax, color='black', linewidth=2)

    legenda = {}
    for _, row in df_pontos.iterrows():
        cor, marcador = estilo_ponto(row)
        label = row.get("DESC_TIPO_EQUIPAMENTO", "N/A")
        if label not in legenda:
            legenda[label] = ax.scatter(
                row["VL_LONGITUDE"],
                row["VL_LATITUDE"],
                c=cor,
                marker=marcador,
                s=100,
                edgecolor="k",
                linewidth=0.7,
                label=label,
                alpha=0.9
            )
        else:
            ax.scatter(
                row["VL_LONGITUDE"],
                row["VL_LATITUDE"],
                c=cor,
                marker=marcador,
                s=100,
                edgecolor="k",
                linewidth=0.7,
                alpha=0.9
            )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Interpolação IDW da Intensidade do Sinal com Equipamentos")

    cbar = plt.colorbar(im, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
    cbar.ax.set_yticklabels(['Ruim', 'Regular', 'Bom', 'Ótimo'])
    cbar.set_label('Classe de Sinal')

    ax.legend(title="DESC_TIPO_EQUIPAMENTO", loc='upper right', markerscale=1.2)

    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig) # Usa st.pyplot para exibir o gráfico Matplotlib
    plt.close(fig) # Fecha a figura para liberar memória
    gc.collect()

# --- Carregamento de Dados na Sidebar ---
st.sidebar.markdown("<h2 class='sidebar-header'>Upload de Arquivos</h2>", unsafe_allow_html=True) # Título estilizado

excel_file = st.sidebar.file_uploader(
    "Selecione o arquivo Excel (.xlsx)", # Corrigido aqui: removidas as tags <span>
    type=["xlsx"],
    help="Faça o upload do arquivo Excel contendo os dados dos equipamentos.",
    key="excel_uploader" # Adicionado key para evitar avisos futuros
)
kml_file = st.sidebar.file_uploader(
    "Selecione o arquivo KML (.kml - Opcional)", # Corrigido aqui: removidas as tags <span>
    type=["kml"],
    help="Opcional: Faça o upload do arquivo KML com os limites das fazendas.",
    key="kml_uploader" # Adicionado key
)

df_csv = None
gdf_kml = None

if excel_file:
    try:
        df_csv = pd.read_excel(io.BytesIO(excel_file.read()), dtype={'VL_LATITUDE': str, 'VL_LONGITUDE': str})
        df_csv.columns = df_csv.columns.str.strip()
        st.sidebar.success("Arquivo Excel carregado com sucesso!")

        # Aplicar normalização de coordenadas e padronização de unidade
        df_csv["VL_LATITUDE"] = df_csv["VL_LATITUDE"].apply(normalizar_coordenadas)
        df_csv["VL_LONGITUDE"] = df_csv["VL_LONGITUDE"].apply(normalizar_coordenadas)
        df_csv["UNIDADE"] = df_csv["UNIDADE"].apply(formatar_nome)

        # Filtrar coordenadas válidas
        df_csv = df_csv.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"])
        df_csv = df_csv[
            (df_csv["VL_LATITUDE"].between(-90, 90)) &
            (df_csv["VL_LONGITUDE"].between(-180, 180))
        ]

        # Verificar colunas essenciais para os gráficos e mapas
        required_cols_excel = ['VL_LATITUDE', 'VL_LONGITUDE', 'UNIDADE', 'DESC_TIPO_EQUIPAMENTO', 'FROTA', 'STATUS', 'VL_FIRMWARE_EQUIPAMENTO', 'TIPO_COMUNICACAO', 'D_MOVEIS_AT']
        missing_excel_cols = [col for col in required_cols_excel if col not in df_csv.columns]
        if missing_excel_cols:
            st.warning(f"Algumas colunas essenciais para os gráficos podem estar faltando no seu Excel: {', '.join(missing_excel_cols)}. Verifique o modelo esperado.")

    except Exception as e:
        st.sidebar.error(f"Erro ao ler arquivo Excel: {e}")

if kml_file:
    try:
        kml_content = kml_file.read().decode('utf-8')
        gdf_kml = extrair_dados_kml(kml_content)
        if gdf_kml is not None and not gdf_kml.empty:
            st.sidebar.success("Arquivo KML carregado e processado com sucesso!")
            # Preparar o gdf_kml_com_nomes para a interpolação do sinal
            gdf_kml['NomeFazendaExtraido'] = gdf_kml.get('NOME_FAZ', gdf_kml.get('Name', 'sem_nome'))
            gdf_kml['NomeFazendaKML_Padronizada'] = gdf_kml['NomeFazendaExtraido'].apply(formatar_nome)
            # Validar e corrigir geometrias (buffer(0) pode ajudar com geometrias inválidas)
            gdf_kml['geometry'] = gdf_kml['geometry'].apply(
                lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom if geom else None
            )
            # Remove geometrias None ou vazias
            gdf_kml = gdf_kml[gdf_kml['geometry'].notna()]
            gdf_kml = gdf_kml[~gdf_kml['geometry'].is_empty]
        else:
            st.sidebar.warning("Arquivo KML processado, mas nenhuma geometria válida foi encontrada.")
            gdf_kml = None # Garante que gdf_kml seja None se estiver vazio
    except Exception as e:
        st.sidebar.error(f"Erro ao processar arquivo KML: {e}")
        gdf_kml = None

# --- Exibição dos Conteúdos Principais ---

if df_csv is not None and not df_csv.empty:
    st.markdown("<h2 class='section-title'>📊 Dashboard de Equipamentos</h2>", unsafe_allow_html=True)

    # --- Gerenciamento de estado para as abas ---
    # Definição das etiquetas das abas
    tab_labels = ["🌎 Mapa de Equipamentos", "📈 Firmware por Unidade", "📊 Relação & Comunicação", "📡 Mapa de Sinal"]

    # Inicializa o estado da aba ativa se não existir
    if 'active_tab_label' not in st.session_state:
        st.session_state.active_tab_label = tab_labels[0] # Padrão para a primeira aba

    # Função para atualizar o estado da aba ativa
    def update_active_tab():
        # st.session_state.main_dashboard_tabs guarda o índice da aba selecionada
        st.session_state.active_tab_label = tab_labels[st.session_state.main_dashboard_tabs]

    # Encontra o índice da aba ativa no session_state para definir como padrão
    try:
        default_tab_index = tab_labels.index(st.session_state.active_tab_label)
    except ValueError:
        default_tab_index = 0 # Fallback caso a etiqueta não seja encontrada

    # Cria as abas com a key, index inicial e callback on_change
    tab_mapa, tab_firmware, tab_relacao, tab_sinal = st.tabs(tab_labels)

    with tab_mapa:
        st.markdown("<h3 class='subsection-title'>Mapa Interativo de Equipamentos Climáticos</h3>", unsafe_allow_html=True)
        # Requer colunas específicas para o mapa
        map_cols_ok = True
        if 'DESC_TIPO_EQUIPAMENTO' not in df_csv.columns:
            st.error("Coluna 'DESC_TIPO_EQUIPAMENTO' é necessária para o mapa de equipamentos.")
            map_cols_ok = False
        if 'FROTA' not in df_csv.columns:
            st.error("Coluna 'FROTA' é necessária para o mapa de equipamentos.")
            map_cols_ok = False
        if 'STATUS' not in df_csv.columns:
            st.error("Coluna 'STATUS' é necessária para o mapa de pluviômetros.")
            map_cols_ok = False

        if map_cols_ok:
            if not df_csv.empty:
                # Criar o mapa centralizado na média das coordenadas válidas
                try:
                    mapa = folium.Map(
                        location=[df_csv["VL_LATITUDE"].mean(), df_csv["VL_LONGITUDE"].mean()],
                        zoom_start=6
                    )
                except Exception as e:
                    st.error(f"Erro ao criar o mapa base. Verifique as coordenadas. Detalhes: {e}")
                    mapa = folium.Map(location=[-14.235, -51.925], zoom_start=4) # Centraliza no Brasil

                marker_cluster = MarkerCluster().add_to(mapa)

                # Estações meteorológicas
                df_estacoes = df_csv[df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("ESTACAO", case=False, na=False)]
                if not df_estacoes.empty:
                    for _, row in df_estacoes.iterrows():
                        if pd.notna(row["VL_LATITUDE"]) and pd.notna(row["VL_LONGITUDE"]):
                            folium.Marker(
                                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                                popup=f"<b>Frota:</b> {row['FROTA']}<br><b>Unidade:</b> {row['UNIDADE']}<br><b>Tipo:</b> {row['DESC_TIPO_EQUIPAMENTO']}",
                                icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
                            ).add_to(marker_cluster)

                # Pluviômetros ativos
                df_pluviometros_ativos = df_csv[
                    (df_csv["DESC_TIPO_EQUIPAMENTO"].str.contains("PLUVIOMETRO", case=False, na=False)) &
                    (df_csv["STATUS"].str.upper() == "ATIVO")
                ]
                if not df_pluviometros_ativos.empty:
                    for _, row in df_pluviometros_ativos.iterrows():
                        if pd.notna(row["VL_LATITUDE"]) and pd.notna(row["VL_LONGITUDE"]):
                            folium.Marker(
                                location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
                                popup=f"<b>Frota:</b> {row['FROTA']}<br><b>Unidade:</b> {row['UNIDADE']}<br><b>Tipo:</b> {row['DESC_TIPO_EQUIPAMENTO']}",
                                icon=folium.Icon(color="green", icon="tint", prefix="fa")
                            ).add_to(marker_cluster)

                # Adicionar limites do KML (se disponível e válido)
                if gdf_kml is not None and not gdf_kml.empty:
                    try:
                        folium.GeoJson(
                            gdf_kml,
                            name="Limites das Fazendas",
                            tooltip=folium.GeoJsonTooltip(fields=["Name"], aliases=["Fazenda:"]),
                            style_function=lambda x: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.1}
                        ).add_to(mapa)
                    except Exception as e:
                        st.warning(f"Erro ao adicionar limites do KML ao mapa: {e}. Verifique o formato do KML.")

                folium.LayerControl().add_to(mapa)
                st.components.v1.html(mapa._repr_html_(), height=500)
            else:
                st.warning("Nenhum dado de equipamento válido para exibir no mapa após filtragem.")
        else:
            st.error("Não é possível gerar o mapa devido à falta de colunas essenciais no Excel.")

    with tab_firmware:
        st.markdown("<h3 class='subsection-title'>Distribuição de Firmwares por Unidade</h3>", unsafe_allow_html=True)
        if 'VL_FIRMWARE_EQUIPAMENTO' in df_csv.columns and 'UNIDADE' in df_csv.columns:
            df_firmware_fazenda = df_csv.groupby(['VL_FIRMWARE_EQUIPAMENTO', 'UNIDADE']).size().reset_index(name='Quantidade')
            df_firmware_fazenda = df_firmware_fazenda.sort_values(by='Quantidade', ascending=True)

            fig_firmware = px.bar(
                df_firmware_fazenda,
                x='Quantidade',
                y='UNIDADE',
                color='VL_FIRMWARE_EQUIPAMENTO',
                title='<b>Distribuição de Firmwares por Unidade</b>',
                labels={
                    'VL_FIRMWARE_EQUIPAMENTO': 'Versão do Firmware',
                    'UNIDADE': 'Unidade',
                    'Quantidade': 'Qtd. de Equipamentos'
                },
                orientation='h',
                text='Quantidade',
                color_discrete_sequence=px.colors.qualitative.Dark2 # Pode ser ajustado para a nova paleta se desejar
            )

            fig_firmware.update_layout(
                title_font_size=20,
                title_font_family='Arial',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(250,250,250,1)',
                bargap=0.25,
                height=600,
                xaxis=dict(title='Quantidade de Equipamentos'),
                yaxis=dict(title=''),
                legend_title='Versão do Firmware'
            )
            fig_firmware.update_traces(textposition='outside', textfont=dict(size=12, color='black'))
            st.plotly_chart(fig_firmware, use_container_width=True)
        else:
            st.warning("Colunas 'VL_FIRMWARE_EQUIPAMENTO' ou 'UNIDADE' não encontradas para o gráfico de firmware.")

    with tab_relacao:
        st.markdown("<h3 class='subsection-title'>Relação de Equipamentos e Tipos de Comunicação</h3>", unsafe_allow_html=True)
        # Usando cores da nova paleta
        cores_personalizadas_plotly = ["#2196F3", "#4CAF50", "#FFC107", "#FF5722"] # Azul, Verde, Amarelo, Laranja

        # Gráfico 1: Pluviômetros e Estações por Fazenda
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
                color_discrete_sequence=cores_personalizadas_plotly
            )
            fig1.update_layout(height=450, legend_title='Tipo de Equipamento', plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Colunas 'DESC_TIPO_EQUIPAMENTO' ou 'UNIDADE' não encontradas para o gráfico de pluviômetros/estações.")

        # Gráfico 2: Tipo de Comunicação por Unidade (excluindo '4G')
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
                color_discrete_sequence=cores_personalizadas_plotly
            )
            fig2.update_layout(height=450, legend_title='Tipo de Comunicação', plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Colunas 'TIPO_COMUNICACAO' ou 'UNIDADE' não encontradas para o gráfico de comunicação.")

        # Gráfico de Rosca: Percentual de Equipamentos com Dados Móveis
        if 'D_MOVEIS_AT' in df_csv.columns:
            contagem_moveis = df_csv['D_MOVEIS_AT'].value_counts()
            fig3 = px.pie(
                values=contagem_moveis.values,
                names=contagem_moveis.index,
                title='Percentual de Equipamentos com Dados Móveis',
                hole=0.5,
                color_discrete_sequence=cores_personalizadas_plotly
            )
            fig3.update_traces(textinfo='percent+label', textfont_size=14)
            fig3.update_layout(showlegend=True, legend_title='Dados Móveis')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Coluna 'D_MOVEIS_AT' não encontrada para o gráfico de dados móveis.")

    with tab_sinal:
        st.markdown("<h3 class='subsection-title'>Mapa Interativo de Intensidade do Sinal</h3>", unsafe_allow_html=True)
        required_cols_signal = ['VL_LATITUDE', 'VL_LONGITUDE', 'UNIDADE', 'DESC_TIPO_EQUIPAMENTO']
        # DBM e INTENSIDADE não são estritamente 'requeridas' se uma delas existir
        # Mas para o proposito da interpolação, pelo menos uma delas precisa existir.
        # Ajustei a lógica para verificar isso mais abaixo.

        signal_cols_ok = True
        for col in required_cols_signal:
            if col not in df_csv.columns:
                st.error(f"Coluna '{col}' é necessária para o mapa de sinal.")
                signal_cols_ok = False

        if signal_cols_ok and not df_csv.empty and gdf_kml is not None and not gdf_kml.empty:
            gdf_equipamentos = gpd.GeoDataFrame(
                df_csv,
                geometry=gpd.points_from_xy(df_csv['VL_LONGITUDE'], df_csv['VL_LATITUDE']),
                crs="EPSG:4326"
            )
            gdf_equipamentos['UNIDADE_Padronizada'] = gdf_equipamentos['UNIDADE'].apply(formatar_nome)

            # Garante que as unidades no KML também são padronizadas
            kml_unidades_padronizadas = gdf_kml['NomeFazendaKML_Padronizada'].dropna().unique()
            equip_unidades_padronizadas = gdf_equipamentos['UNIDADE_Padronizada'].dropna().unique()

            # Apenas mostra as unidades que existem em ambos (equipamentos e KML)
            unidades_disponiveis = sorted(list(set(kml_unidades_padronizadas) & set(equip_unidades_padronizadas)))

            if not unidades_disponiveis:
                st.warning("Nenhuma unidade encontrada que tenha equipamentos e geometria KML correspondente para gerar o mapa de sinal.")
            else:
                # Gerencia o estado da selectbox também, se necessário
                if 'selected_unidade_sinal' not in st.session_state and unidades_disponiveis:
                    st.session_state.selected_unidade_sinal = unidades_disponiveis[0]

                selected_unidade = st.selectbox(
                    "Selecione a Fazenda para Interpolação de Sinal:",
                    unidades_disponiveis,
                    key="fazenda_sinal",
                    index=unidades_disponiveis.index(st.session_state.selected_unidade_sinal) if st.session_state.selected_unidade_sinal in unidades_disponiveis else 0,
                    on_change=lambda: st.session_state.update(selected_unidade_sinal=st.session_state.fazenda_sinal)
                )

                if selected_unidade:
                    df_fazenda = gdf_equipamentos[gdf_equipamentos['UNIDADE_Padronizada'] == selected_unidade].copy()

                    if df_fazenda.empty:
                        st.warning(f"Nenhum equipamento encontrado para a fazenda '{selected_unidade}'.")
                    else:
                        geom_df = gdf_kml[gdf_kml['NomeFazendaKML_Padronizada'] == selected_unidade]
                        if geom_df.empty:
                            st.warning(f"Nenhuma geometria KML encontrada para a fazenda '{selected_unidade}'. Verifique se o nome da fazenda no Excel corresponde ao KML.")
                        else:
                            # Tentar combinar geometrias em MultiPolygon se houver múltiplas
                            try:
                                fazenda_geom = geom_df.unary_union
                                if not fazenda_geom.is_valid:
                                    fazenda_geom = fazenda_geom.buffer(0) # Tenta corrigir geometria inválida
                            except Exception as e:
                                st.error(f"Erro ao combinar geometrias da fazenda: {e}. Certifique-se de que o KML contém geometrias válidas.")
                                fazenda_geom = None

                            if fazenda_geom is None or fazenda_geom.is_empty:
                                st.warning(f"A geometria combinada para a fazenda '{selected_unidade}' está vazia ou inválida. Não é possível gerar o mapa de sinal.")
                            else:
                                df_fazenda['DBM'] = pd.to_numeric(df_fazenda['DBM'], errors='coerce')
                                has_dbm = 'DBM' in df_fazenda.columns and not df_fazenda['DBM'].dropna().empty
                                has_intensidade = 'INTENSIDADE' in df_fazenda.columns and not df_fazenda['INTENSIDADE'].dropna().empty

                                mapping = {"ruim": 1, "regular": 2, "bom": 3, "otimo": 4}

                                val_col_used = None
                                if has_dbm and has_intensidade:
                                    # Priorizar DBM, mas preencher com INTENSIDADE se DBM for NaN
                                    df_com_dbm = df_fazenda.dropna(subset=['DBM'])
                                    if not df_com_dbm.empty:
                                        # Calcular médias DBM por INTENSIDADE_MAP apenas para os valores DBM existentes
                                        # O ideal é mapear INTENSIDADE para DBM, se não tiver DBM direto.
                                        # Se INTENSIDADE_MAP já é a classe 1-4, pode ser usada diretamente.
                                        df_fazenda['INTENSIDADE_MAP'] = df_fazenda['INTENSIDADE'].apply(
                                            lambda x: mapping.get(unidecode(str(x)).strip().lower(), np.nan)
                                        )
                                        # Tentar inferir DBM a partir de INTENSIDADE para NaNs em DBM
                                        for idx, row in df_fazenda.iterrows():
                                            if pd.isna(row['DBM']) and pd.notna(row.get('INTENSIDADE_MAP')):
                                                # Esta parte precisa de uma regra de negócio mais definida.
                                                # Por simplicidade, se INTENSIDADE_MAP está presente e DBM não,
                                                # e não há um mapeamento direto de intensidade para DBM,
                                                # usaremos INTENSIDADE_MAP como o valor de sinal (1-4).
                                                df_fazenda.loc[idx, 'DBM'] = row['INTENSIDADE_MAP']
                                    val_col_used = 'DBM'
                                elif has_dbm:
                                    val_col_used = 'DBM'
                                elif has_intensidade:
                                    df_fazenda['DBM'] = df_fazenda['INTENSIDADE'].apply(
                                        lambda x: mapping.get(unidecode(str(x)).strip().lower(), np.nan)
                                    )
                                    val_col_used = 'DBM'
                                else:
                                    st.warning("Nenhuma coluna de sinal (DBM ou INTENSIDADE) válida encontrada para interpolação nesta fazenda.")
                                    val_col_used = None

                                if val_col_used:
                                    df_fazenda_filtered = df_fazenda.dropna(subset=[val_col_used])
                                    if not df_fazenda_filtered.empty:
                                        grid_x, grid_y, grid_numerico, bounds = interpolacao_idw(
                                            _df=df_fazenda_filtered,
                                            x_col='VL_LONGITUDE',
                                            y_col='VL_LATITUDE',
                                            val_col=val_col_used,
                                            resolution=0.002, # Ajuste conforme a necessidade de detalhe vs. performance
                                            buffer=0.05,
                                            _geom_mask=fazenda_geom
                                        )

                                        if grid_x is not None:
                                            plotar_interpolacao(grid_x, grid_y, grid_numerico, fazenda_geom, bounds, df_fazenda_filtered)
                                        else:
                                            st.warning("Não foi possível gerar a interpolação para esta fazenda. Verifique os dados ou a resolução da grade.")
                                    else:
                                        st.warning("Nenhum dado de sinal válido após filtragem para esta fazenda.")
                                else:
                                    st.info("Para gerar o mapa de sinal, o Excel deve ter as colunas 'DBM' ou 'INTENSIDADE'.")
        elif not signal_cols_ok:
            st.error("Não é possível gerar o mapa de sinal devido à falta de colunas essenciais no Excel.")
        else:
            st.info("Por favor, faça o upload dos arquivos Excel e KML para habilitar o mapa de sinal.")

else:
    st.info("Por favor, faça o upload de um arquivo Excel para começar.")

st.markdown("<div class='footer'>Desenvolvido para Monitoramento Agrícola</div>", unsafe_allow_html=True)