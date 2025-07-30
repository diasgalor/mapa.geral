# 🌿 Monitoramento Agrícola Inteligente com Streamlit

Esta é uma aplicação interativa desenvolvida com Streamlit para visualização e análise de dados de equipamentos climáticos e intensidade de sinal em propriedades agrícolas.

## Funcionalidades

* **Upload de Dados:** Carregue arquivos Excel (.xlsx) contendo dados de equipamentos (coordenadas, tipo, firmware, comunicação) e arquivos KML (.kml) com os limites das fazendas.
* **Mapa Interativo de Equipamentos:** Visualize estações meteorológicas e pluviômetros ativos em um mapa Folium interativo, com clusterização de marcadores.
* **Gráficos Analíticos:**
    * Distribuição de Firmwares por Unidade.
    * Relação de Pluviômetros e Estações por Unidade.
    * Tipos de Comunicação por Unidade (excluindo 4G).
    * Percentual de Equipamentos com Dados Móveis.
* **Mapa de Interpolação de Sinal (IDW):** Selecione uma fazenda para visualizar a intensidade do sinal interpolada (IDW) com base nos dados DBM/Intensidade dos equipamentos, com os limites da fazenda sobrepostos.

## Como Executar Localmente

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
    cd SEU_REPOSITORIO/streamlit_app
    ```
    (Substitua `SEU_USUARIO` e `SEU_REPOSITORIO` pelo seu.)

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    # venv\Scripts\activate  # No Windows
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```
    A aplicação será aberta automaticamente no seu navegador padrão.

## Como Fazer Deploy no Streamlit Cloud

1.  **Faça o push do seu código para o GitHub:** Certifique-se de que a pasta `streamlit_app` (ou a pasta que contém `app.py`, `style.css` e `requirements.txt`) esteja no seu repositório GitHub.
2.  **Vá para Streamlit Cloud:** Acesse [share.streamlit.io](https://share.streamlit.io/).
3.  **Conecte sua conta GitHub** (se ainda não o fez).
4.  **Clique em "New app"** e selecione o seu repositório e a branch onde o código está.
5.  **Defina o "Main file path"** para `streamlit_app/app.py` (ou o caminho correto para o seu arquivo `app.py`).
6.  **Clique em "Deploy!"**

O Streamlit Cloud irá automaticamente ler seu `requirements.txt`, instalar as dependências e publicar sua aplicação.

## Estrutura do Projeto
├── app.py              # Código principal da aplicação Streamlit
├── style.css           # Estilos CSS customizados para a interface
├── requirements.txt    # Lista de dependências Python
└── README.md           # Este arquivo de documentação

## Contato

Se tiver dúvidas ou sugestões, sinta-se à vontade para abrir uma issue no GitHub.