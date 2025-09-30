import os
import pandas as pd
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from werkzeug.utils import secure_filename
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import traceback
import json
import time

# --- CONFIGURAÇÃO INICIAL ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# --- VARIÁVEIS GLOBAIS ---
df_global = None
df_reparos_global = None
df_instalacoes_global = None
CACHE_FILE = 'geocache.json'

# --- FUNÇÕES DE CACHE ---
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=4)

# --- FUNÇÕES AUXILIARES ---
def formatar_dataframe_para_json(df):
    if df is None or df.empty:
        return []
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
        df_copy[col] = df_copy[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else None)
    records = df_copy.to_dict(orient='records')
    cleaned_records = []
    for record in records:
        cleaned_record = {}
        for key, value in record.items():
            if isinstance(value, float) and pd.isna(value):
                cleaned_record[key] = None
            else:
                cleaned_record[key] = value
        cleaned_records.append(cleaned_record)
    return cleaned_records

# --- ROTA PRINCIPAL ---
@app.route("/")
def index():
    return render_template("index.html")

# --- ROTAS DA API ---

# VERSÃO FINAL E CORRIGIDA DA FUNÇÃO upload_dados
@app.route("/api/upload-dados", methods=["POST"])
def upload_dados():
    global df_global
    try:
        if "custos" not in request.files or "tarefas" not in request.files:
            return jsonify({"erro": "Arquivos não enviados"}), 400
        
        file_custos = request.files["custos"]
        file_tarefas = request.files["tarefas"]
        
        # --- DIAGNÓSTICO ---
        print("--- INICIANDO DIAGNÓSTICO ---")
        df_custos_check = pd.read_excel(file_custos)
        df_tarefas_check = pd.read_excel(file_tarefas)
        print(f"Arquivo 'custos' ({file_custos.filename}) tem {df_custos_check.shape[1]} colunas.")
        print(f"Arquivo 'tarefas' ({file_tarefas.filename}) tem {df_tarefas_check.shape[1]} colunas.")
        print("-----------------------------")
        # --- FIM DO DIAGNÓSTICO ---

        # --- PROCESSAMENTO DA PLANILHA DE CUSTOS ---
        df_custos = pd.read_excel(file_custos, header=None, skiprows=1, dtype=str)
        nomes_custos = ['codigo_chamado', 'codigo_projeto', 'data_conclusao_instalacao', 'codigo_os', 'abertura', 'cliente', 'cidade', 'categoria', 'tipo_de_servico', 'resp_financeira', 'valor_custo', 'valor_venda', 'data_baixa_os', 'data_conclusao_servico', 'data_validacao_custo', 'data_faturamento', 'instalador']
        df_custos = df_custos.iloc[:, :len(nomes_custos)]
        df_custos.columns = nomes_custos
        df_custos.dropna(subset=['codigo_chamado'], inplace=True)
        df_custos['codigo_chamado'] = df_custos['codigo_chamado'].astype(str).str.strip()

        # --- PROCESSAMENTO DA PLANILHA DE TAREFAS ---
        df_tarefas = pd.read_excel(file_tarefas, header=None, skiprows=1, dtype=str)
        nomes_tarefas = ['numero_sequencial', 'id_os', 'codigo_chamado', 'codigo_projeto_tarefa', 'data_instalacao', 'instalador_tarefa', 'categoria_tarefa', 'status_os', 'tipo_reclamacao', 'tipo_servico_tarefa', 'respons_financeira_tarefa', 'cliente_tarefa', 'cidade_tarefa', 'usuario_abertura', 'emissao', 'previsao_inicio', 'previsao_conclusao', 'qtd_reagendamentos', 'data_reagendada', 'data_pre_baixa', 'baixa_os', 'conclusao_servico_tarefa', 'dias_atraso', 'obs_os']
        
        # A linha que causa o erro está aqui. O print acima vai nos mostrar o porquê.
        df_tarefas = df_tarefas.iloc[:, :len(nomes_tarefas)]
        df_tarefas.columns = nomes_tarefas
        
        df_tarefas.dropna(subset=['codigo_chamado'], inplace=True)
        df_tarefas['codigo_chamado'] = df_tarefas['codigo_chamado'].astype(str).str.strip()
        df_tarefas.drop_duplicates(subset=['codigo_chamado'], keep='first', inplace=True)

        # --- JUNÇÃO (MERGE) ---
        df_final = pd.merge(df_custos, df_tarefas, on='codigo_chamado', how='left')

        # ... (o resto da função continua igual)
        if df_final.empty:
            return jsonify({"erro": "As planilhas foram lidas, mas a junção não produziu resultados. Verifique se os 'códigos de chamado' correspondem."}), 400

        for col_orig, col_fallback in [('instalador', 'instalador_tarefa'), ('categoria', 'categoria_tarefa'), ('tipo_de_servico', 'tipo_servico_tarefa'), ('resp_financeira', 'respons_financeira_tarefa'), ('cliente', 'cliente_tarefa'), ('cidade', 'cidade_tarefa')]:
            if col_fallback in df_final.columns:
                df_final[col_orig + '_final'] = df_final[col_orig].fillna(df_final[col_fallback])
            else:
                df_final[col_orig + '_final'] = df_final[col_orig]

        df_final['valor_custo'] = pd.to_numeric(df_final['valor_custo'], errors='coerce').fillna(0)
        df_final['valor_venda'] = pd.to_numeric(df_final['valor_venda'], errors='coerce').fillna(0)
        df_final['lucratividade'] = df_final['valor_venda'] - df_final['valor_custo']
        df_final['margem_lucro'] = (df_final['lucratividade'] / df_final['valor_venda'] * 100).where(df_final['valor_venda'] > 0, 0)
        df_final['data_referencia'] = pd.to_datetime(df_final['data_baixa_os'], errors='coerce', dayfirst=True).fillna(pd.to_datetime(df_final['abertura'], errors='coerce', dayfirst=True))
        
        df_global = df_final.copy()
        
        opcoes_filtros = {
            'instaladores': sorted(df_global['instalador_final'].astype(str).dropna().unique().tolist()),
            'usuarios': sorted(df_global['usuario_abertura'].astype(str).dropna().unique().tolist()) if 'usuario_abertura' in df_global.columns else [],
            'clientes': sorted(df_global['cliente_final'].astype(str).dropna().unique().tolist()),
            'resp_financeira': sorted(df_global['resp_financeira_final'].astype(str).dropna().unique().tolist()),
            'tipo_servico': sorted(df_global['tipo_de_servico_final'].astype(str).dropna().unique().tolist()),
            'categorias': sorted(df_global['categoria_final'].astype(str).dropna().unique().tolist()),
            'status_os': sorted(df_global['status_os'].astype(str).dropna().unique().tolist()) if 'status_os' in df_global.columns else []
        }
        
        return jsonify({"mensagem": "Planilhas carregadas e unidas com sucesso!", "filtros": opcoes_filtros})
    
    except Exception as e:
        app.logger.error(f"Erro no upload: {str(e)}")
        traceback.print_exc()
        return jsonify({"erro": f"Erro ao processar as planilhas: {str(e)}"}), 500
@app.route("/api/estatisticas", methods=["POST"])
def estatisticas():
    if df_global is None:
        return jsonify({"erro": "Nenhum dado carregado."}), 400
    
    try:
        filtros = request.json
        df_filtrado = df_global.copy()

        # Aplica os filtros
        if filtros.get('data_inicio'):
            df_filtrado = df_filtrado[df_filtrado['data_referencia'] >= pd.to_datetime(filtros['data_inicio'])]
        if filtros.get('data_fim'):
            df_filtrado = df_filtrado[df_filtrado['data_referencia'] <= pd.to_datetime(filtros['data_fim'])]
        if filtros.get('instalador'): df_filtrado = df_filtrado[df_filtrado['instalador_final'] == filtros['instalador']]
        if filtros.get('usuario'): df_filtrado = df_filtrado[df_filtrado['usuario_abertura'] == filtros['usuario']]
        if filtros.get('cliente'): df_filtrado = df_filtrado[df_filtrado['cliente_final'] == filtros['cliente']]
        if filtros.get('resp_financeira'): df_filtrado = df_filtrado[df_filtrado['resp_financeira_final'] == filtros['resp_financeira']]
        if filtros.get('tipo_servico'): df_filtrado = df_filtrado[df_filtrado['tipo_de_servico_final'] == filtros['tipo_servico']]
        if filtros.get('categoria'): df_filtrado = df_filtrado[df_filtrado['categoria_final'] == filtros['categoria']]
        if filtros.get('status_os'): df_filtrado = df_filtrado[df_filtrado['status_os'] == filtros['status_os']]
        
        lucro_total = df_filtrado['lucratividade'].sum()
        venda_total = df_filtrado['valor_venda'].sum()

        def get_top_sum(df, group_col, sum_col): return df.groupby(group_col)[sum_col].sum().nlargest(5).to_dict()
        def get_top_count(df, group_col): return df[group_col].value_counts().nlargest(5).to_dict()
        def get_top_mean(df, group_col, mean_col): return df.groupby(group_col)[mean_col].mean().nlargest(5).to_dict()

        # --- LÓGICA DE RETRABALHO REFINADA E BLINDADA ---
        total_retrabalhos_por_equipe = {}
        colunas_retrabalho = ['cliente_final', 'instalador_final', 'tipo_de_servico_final']
        if all(col in df_filtrado.columns for col in colunas_retrabalho):
            # 1. Remove linhas onde qualquer uma das colunas chave é nula
            df_retrabalho = df_filtrado.dropna(subset=colunas_retrabalho).copy()
            
            # 2. Garante que os dados são strings para um agrupamento consistente
            for col in colunas_retrabalho:
                df_retrabalho[col] = df_retrabalho[col].astype(str)

            # 3. Agrupa pela combinação única
            combinacoes = df_retrabalho.groupby(colunas_retrabalho).size()
            
            # 4. Filtra apenas as combinações que ocorreram mais de uma vez (retrabalhos)
            retrabalhos = combinacoes[combinacoes > 1].reset_index(name='contagem')
            
            if not retrabalhos.empty:
                # 5. Para cada retrabalho, subtrai 1 para contar apenas as visitas extras.
                retrabalhos['contagem'] = retrabalhos['contagem'] - 1
                # 6. Agrupa por instalador e soma os retrabalhos
                total_retrabalhos_por_equipe = retrabalhos.groupby('instalador_final')['contagem'].sum()

        # --- LÓGICA DE CUSTO MÉDIO TOPSUN REFINADA ---
        df_topsun = df_filtrado[df_filtrado['resp_financeira_final'].str.strip().str.upper() == 'TOPSUN'].copy()

        resultado = {
            "kpi_lucro_total": lucro_total,
            "kpi_margem_lucro_geral": (lucro_total / venda_total * 100) if venda_total > 0 else 0,
            "kpi_valor_venda_total": venda_total,
            "kpi_total_ordens": len(df_filtrado),
            "top_instaladores_margem_lucro": get_top_mean(df_filtrado, 'instalador_final', 'margem_lucro'),
            "top_tipos_servico_margem_lucro": get_top_mean(df_filtrado, 'tipo_de_servico_final', 'margem_lucro'),
            "top_instaladores_lucratividade": get_top_sum(df_filtrado, 'instalador_final', 'lucratividade'),
            "top_instaladores_ordens": get_top_count(df_filtrado, 'instalador_final'),
            "top_clientes_ordens": get_top_count(df_filtrado, 'cliente_final'),
            "top_clientes_custo": get_top_sum(df_filtrado, 'cliente_final', 'valor_custo'),
            "ordens_por_responsavel": df_filtrado['resp_financeira_final'].value_counts().to_dict(),
            "top_servicos_custo_medio": get_top_mean(df_filtrado, 'tipo_de_servico_final', 'valor_custo'),
            "top_equipes_custo_topsun": get_top_mean(df_topsun, 'instalador_final', 'valor_custo'),
            "top_equipes_retrabalhos": total_retrabalhos_por_equipe.nlargest(5).to_dict(),
        }

        resultado_limpo = pd.DataFrame([resultado]).where(pd.notnull(pd.DataFrame([resultado])), None).to_dict('records')[0]
        
        return jsonify(resultado_limpo)

    except Exception as e:
        app.logger.error(f"Erro em /api/estatisticas: {str(e)}")
        traceback.print_exc()
        return jsonify({"erro": f"Erro ao calcular estatísticas: {str(e)}"}), 500


@app.route("/api/ranking-completo", methods=["POST"])
def ranking_completo():
    if df_global is None: return jsonify({"erro": "Nenhum dado carregado."}), 400
    
    try:
        request_data = request.json
        filtros = request_data.get('filtros', {})
        ranking_desejado = request_data.get('ranking')
        limite = request_data.get('limite', 30)

        df = df_global.copy()

        # Aplica filtros de data e outros
        if filtros.get('data_inicio'): df = df[df['data_referencia'] >= pd.to_datetime(filtros['data_inicio'])]
        if filtros.get('data_fim'): df = df[df['data_referencia'] <= pd.to_datetime(filtros['data_fim'])]
        if filtros.get('instalador'): df = df[df['instalador_final'] == filtros['instalador']]
        if filtros.get('usuario'): df = df[df['usuario_abertura'] == filtros['usuario']]
        if filtros.get('cliente'): df = df[df['cliente_final'] == filtros['cliente']]
        if filtros.get('resp_financeira'): df = df[df['resp_financeira_final'] == filtros['resp_financeira']]
        if filtros.get('tipo_servico'): df = df[df['tipo_de_servico_final'] == filtros['tipo_servico']]
        if filtros.get('categoria'): df = df[df['categoria_final'] == filtros['categoria']]
        if filtros.get('status_os'): df = df[df['status_os'] == filtros['status_os']]

        resultado = {}
        
        # --- CORREÇÃO PRINCIPAL APLICADA AQUI ---
        # Lógica específica para o ranking de retrabalhos
        if ranking_desejado == 'top_equipes_retrabalhos':
            colunas_retrabalho = ['cliente_final', 'instalador_final', 'tipo_de_servico_final']
            df_retrabalho = df.dropna(subset=colunas_retrabalho).copy()
            for col in colunas_retrabalho:
                df_retrabalho[col] = df_retrabalho[col].astype(str)
            
            combinacoes = df_retrabalho.groupby(colunas_retrabalho).size()
            retrabalhos = combinacoes[combinacoes > 1].reset_index(name='contagem')
            
            if not retrabalhos.empty:
                retrabalhos['contagem'] = retrabalhos['contagem'] - 1
                total_retrabalhos_por_equipe = retrabalhos.groupby('instalador_final')['contagem'].sum()
                resultado = total_retrabalhos_por_equipe.to_dict()

        # Lógica para todos os outros rankings
        else:
            rankings_map = {
                'top_instaladores_lucratividade': ('sum', 'instalador_final', 'lucratividade'),
                'top_instaladores_ordens': ('count', 'instalador_final', None),
                'top_instaladores_margem_lucro': ('mean', 'instalador_final', 'margem_lucro'),
                'top_tipos_servico_margem_lucro': ('mean', 'tipo_de_servico_final', 'margem_lucro'),
                'top_clientes_ordens': ('count', 'cliente_final', None),
                'top_clientes_custo': ('sum', 'cliente_final', 'valor_custo'),
                'top_servicos_custo_medio': ('mean', 'tipo_de_servico_final', 'valor_custo'),
                'top_equipes_custo_topsun': ('mean', 'instalador_final', 'valor_custo'),
            }

            if ranking_desejado == 'top_equipes_custo_topsun':
                df = df[df['resp_financeira_final'].str.strip().str.upper() == 'TOPSUN'].copy()

            if ranking_desejado in rankings_map:
                op, group_col, val_col = rankings_map[ranking_desejado]
                
                if op == 'sum': series = df.groupby(group_col)[val_col].sum()
                elif op == 'count': series = df[group_col].value_counts()
                elif op == 'mean': series = df.groupby(group_col)[val_col].mean()
                
                resultado = {k: v for k, v in series.items() if pd.notna(v)}

        ranking_ordenado = sorted(resultado.items(), key=lambda item: item[1], reverse=True)
        return jsonify(ranking_ordenado[:limite])

    except Exception as e:
        app.logger.error(f"Erro em /api/ranking-completo: {str(e)}")
        traceback.print_exc()
        return jsonify({"erro": f"Erro ao gerar ranking completo: {str(e)}"}), 500


@app.route("/api/detalhes-ranking", methods=["POST"])
def detalhes_ranking():
    if df_global is None: return jsonify({"erro": "Nenhum dado carregado."}), 400
    coluna = request.json.get('coluna')
    valor = request.json.get('valor')
    filtros_ativos = request.json.get('filtros', {})
    df_filtrado = df_global.copy()
    if filtros_ativos.get('instalador'): df_filtrado = df_filtrado[df_filtrado['instalador_final'] == filtros_ativos['instalador']]
    if coluna in df_filtrado.columns:
        df_detalhes = df_filtrado[df_filtrado[coluna] == valor]
        return jsonify(formatar_dataframe_para_json(df_detalhes))
    return jsonify({"erro": f"Coluna '{coluna}' não encontrada."}), 400

@app.route("/api/dados-completos", methods=["GET"])
def dados_completos():
    if df_global is None: return jsonify([]), 200
    return jsonify(formatar_dataframe_para_json(df_global))

@app.route("/api/processar-mapa", methods=["POST"])
def processar_mapa():
    global df_reparos_global, df_instalacoes_global
    try:
        if 'reparosFile' not in request.files:
            return jsonify({"erro": "Arquivo de reparos não enviado."}), 400
        reparos_file = request.files["reparosFile"]
        df_reparos_bruto = pd.read_excel(reparos_file, header=None, skiprows=1, dtype=str)
        colunas_necessarias = 17
        if df_reparos_bruto.shape[1] >= colunas_necessarias:
            df_reparos = df_reparos_bruto.iloc[:, [2, 3, 5, 6, 11, 12, 16]].copy()
            df_reparos.columns = ['id_chamado', 'codigo_projeto', 'instalador', 'categoria', 'cliente', 'cidade', 'previsao_conclusao']
        else:
            return jsonify({"erro": f"A planilha de reparos precisa ter pelo menos {colunas_necessarias} colunas."}), 400
        df_reparos.fillna({'cidade': 'N/A', 'instalador': 'N/A', 'cliente': 'N/A'}, inplace=True)
        df_reparos['cidade'] = df_reparos['cidade'].astype(str).str.strip()
        coordenadas_cache = load_cache()
        geolocator = Nominatim(user_agent=f"top_sun_dashboard_{time.time()}", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=5.0, max_retries=2, swallow_exceptions=True)
        cidades_para_buscar = [c for c in df_reparos['cidade'].unique() if c and c != 'N/A' and c not in coordenadas_cache]
        if cidades_para_buscar:
            for cidade in cidades_para_buscar:
                location = geocode(cidade + ', Brazil')
                coordenadas_cache[cidade] = (location.latitude, location.longitude) if location else (None, None)
            save_cache(coordenadas_cache)
        df_reparos['latitude'] = df_reparos['cidade'].map(lambda c: coordenadas_cache.get(c, (None, None))[0])
        df_reparos['longitude'] = df_reparos['cidade'].map(lambda c: coordenadas_cache.get(c, (None, None))[1])
        df_reparos_global = df_reparos.copy()
        
        dados_mapa = formatar_dataframe_para_json(df_reparos_global.dropna(subset=['latitude', 'longitude']))
        dados_tabela = formatar_dataframe_para_json(df_reparos_global)
        
        todos_instaladores = set(df_reparos_global['instalador'].astype(str).dropna().unique())
        if df_instalacoes_global is not None:
            todos_instaladores.update(df_instalacoes_global['instalador'].astype(str).dropna().unique())

        filtros_mapa = {'instaladores': sorted(list(todos_instaladores)), 'status': []}
        resposta = {"dados_mapa": dados_mapa, "dados_tabela": dados_tabela, "mensagem": "Reparos carregados!", "filtros_disponiveis": filtros_mapa}
        return jsonify(resposta)
    except Exception as e:
        app.logger.error(f"Erro na rota do mapa de reparos: {str(e)}")
        traceback.print_exc()
        return jsonify({"erro": f"Erro no servidor ao processar mapa: {str(e)}"}), 500

@app.route("/api/upload-instalacoes", methods=["POST"])
def upload_instalacoes():
    global df_instalacoes_global, df_reparos_global
    try:
        if 'instalacoesFile' not in request.files:
            return jsonify({"erro": "Arquivo de instalações não enviado"}), 400
        instalacoes_file = request.files["instalacoesFile"]
        df_instalacoes_bruto = pd.read_excel(instalacoes_file, header=1, dtype=str)
        if df_instalacoes_bruto.shape[1] < 9:
            return jsonify({"erro": "A planilha de instalações precisa ter pelo menos 9 colunas (A-I)."}), 400
        df_instalacoes = df_instalacoes_bruto.iloc[:, [1, 3, 4, 5, 8]].copy()
        df_instalacoes.columns = ['projeto', 'cliente', 'cidade', 'instalador', 'data_previsao_instalacao']
        df_instalacoes.dropna(how='all', inplace=True)
        df_instalacoes.fillna({'cidade': 'N/A', 'instalador': 'N/A', 'cliente': 'N/A', 'data_previsao_instalacao': 'N/A'}, inplace=True)
        df_instalacoes['cidade'] = df_instalacoes['cidade'].astype(str).str.strip()
        df_instalacoes['instalador'] = df_instalacoes['instalador'].astype(str).str.strip()
        coordenadas_cache = load_cache()
        geolocator = Nominatim(user_agent=f"top_sun_dashboard_{time.time()}", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=5.0, max_retries=2, swallow_exceptions=True)
        cidades_para_buscar = [c for c in df_instalacoes['cidade'].unique() if c and c != 'N/A' and c not in coordenadas_cache]
        if cidades_para_buscar:
            for cidade in cidades_para_buscar:
                location = geocode(cidade + ', Brazil')
                coordenadas_cache[cidade] = (location.latitude, location.longitude) if location else (None, None)
            save_cache(coordenadas_cache)
        df_instalacoes['latitude'] = df_instalacoes['cidade'].map(lambda c: coordenadas_cache.get(c, (None, None))[0])
        df_instalacoes['longitude'] = df_instalacoes['cidade'].map(lambda c: coordenadas_cache.get(c, (None, None))[1])
        df_instalacoes['codigo_projeto'] = df_instalacoes['projeto']
        df_instalacoes_global = df_instalacoes.copy()
        
        dados_mapa = formatar_dataframe_para_json(df_instalacoes_global.dropna(subset=['latitude', 'longitude']))
        dados_tabela = formatar_dataframe_para_json(df_instalacoes_global)
        todos_instaladores = set()
        if df_reparos_global is not None:
            todos_instaladores.update(df_reparos_global['instalador'].astype(str).dropna().unique())
        if df_instalacoes_global is not None:
            todos_instaladores.update(df_instalacoes_global['instalador'].astype(str).dropna().unique())
        filtros_mapa = {'instaladores': sorted(list(todos_instaladores)), 'status': []}
        resposta = {"dados_mapa": dados_mapa, "dados_tabela": dados_tabela, "filtros_disponiveis": filtros_mapa, "mensagem": "Instalações carregadas!"}
        return jsonify(resposta)
    except Exception as e:
        app.logger.error(f"Erro no upload de instalações: {str(e)}")
        traceback.print_exc()
        return jsonify({"erro": f"Erro no servidor ao fazer upload de instalações: {str(e)}"}), 500

@app.route("/api/buscar-proximidade", methods=["POST"])
def buscar_proximidade():
    data = request.json
    endereco = data.get('endereco')
    if not endereco:
        return jsonify({"erro": "Endereço não fornecido."}), 400
    try:
        geolocator = Nominatim(user_agent=f"top_sun_dashboard_proximity_{time.time()}", timeout=10)
        origem_location = geolocator.geocode(endereco + ', Brazil')
        if not origem_location:
            return jsonify({"erro": "Não foi possível geocodificar o endereço de origem."}), 400
        origem_lat, origem_lon = origem_location.latitude, origem_location.longitude
        todos_os_pontos = []
        if df_reparos_global is not None:
            df_reparos_copy = df_reparos_global.copy()
            df_reparos_copy['tipo'] = 'reparo'
            todos_os_pontos.extend(df_reparos_copy.dropna(subset=['latitude', 'longitude']).to_dict(orient='records'))
        if df_instalacoes_global is not None:
            df_instalacoes_copy = df_instalacoes_global.copy()
            df_instalacoes_copy['tipo'] = 'instalacao'
            todos_os_pontos.extend(df_instalacoes_copy.dropna(subset=['latitude', 'longitude']).to_dict(orient='records'))
        if not todos_os_pontos:
            return jsonify({"erro": "Nenhum ponto de reparo ou instalação disponível para calcular proximidade."}), 400
        pontos_com_distancia = []
        for ponto in todos_os_pontos:
            distancia_km = ((origem_lat - ponto['latitude'])**2 + (origem_lon - ponto['longitude'])**2)**0.5 * 111
            tempo_viagem_min = distancia_km * 1.5
            pontos_com_distancia.append({**ponto, 'distancia_km': distancia_km, 'distancia_texto': f"{distancia_km:.2f} km", 'tempo_viagem_min': tempo_viagem_min, 'tempo_viagem_texto': f"{int(tempo_viagem_min // 60)}h {int(tempo_viagem_min % 60)}min"})
        pontos_com_distancia.sort(key=lambda x: x['distancia_km'])
        top_5_proximos = pontos_com_distancia[:5]
        return jsonify({"mensagem": "Busca de proximidade concluída!", "origem_lat": origem_lat, "origem_lon": origem_lon, "reparos_proximos": top_5_proximos})
    except Exception as e:
        app.logger.error(f"Erro na busca de proximidade: {str(e)}")
        traceback.print_exc()
        return jsonify({"erro": f"Erro ao buscar proximidade: {str(e)}"}), 500
@app.route("/api/detalhes-retrabalho-equipe", methods=["POST"])
def detalhes_retrabalho_equipe():
    if df_global is None:
        return jsonify({"erro": "Nenhum dado carregado."}), 400
    
    try:
        data = request.json
        equipe_selecionada = data.get('equipe')
        filtros = data.get('filtros', {})

        if not equipe_selecionada:
            return jsonify({"erro": "Nenhuma equipe selecionada."}), 400

        df_filtrado = df_global.copy()

        # Aplica os mesmos filtros de data que estão ativos no dashboard
        if filtros.get('data_inicio'):
            df_filtrado = df_filtrado[df_filtrado['data_referencia'] >= pd.to_datetime(filtros['data_inicio'])]
        if filtros.get('data_fim'):
            df_filtrado = df_filtrado[df_filtrado['data_referencia'] <= pd.to_datetime(filtros['data_fim'])]

        # Filtra apenas para a equipe selecionada
        df_equipe = df_filtrado[df_filtrado['instalador_final'] == equipe_selecionada]

        # --- LÓGICA DE RETRABALHO REFINADA ---
        # Agrupa pela combinação de cliente e tipo de serviço para contar as ocorrências
        retrabalhos_por_combinacao = df_equipe.groupby(['cliente_final', 'tipo_de_servico_final']).size()
        
        # Filtra para manter apenas combinações com mais de 1 ocorrência (retrabalhos)
        retrabalhos_reais = retrabalhos_por_combinacao[retrabalhos_por_combinacao > 1].reset_index(name='contagem')
        
        # Formata para o frontend
        resultado_formatado = [
            {"cliente": row['cliente_final'], "servico": row['tipo_de_servico_final'], "visitas_totais": row['contagem']}
            for index, row in retrabalhos_reais.iterrows()
        ]
        
        # Ordena do maior para o menor
        resultado_formatado.sort(key=lambda x: x['visitas_totais'], reverse=True)

        return jsonify(resultado_formatado)

    except Exception as e:
        app.logger.error(f"Erro em /api/detalhes-retrabalho-equipe: {str(e)}")
        traceback.print_exc()
        return jsonify({"erro": f"Erro ao buscar detalhes de retrabalho: {str(e)}"}), 500

# --- INICIALIZAÇÃO DO APP ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
