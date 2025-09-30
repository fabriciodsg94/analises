import pandas as pd
import os
import time
import traceback

# --- CONFIGURAÇÃO ---
# Coloque aqui os nomes exatos dos seus arquivos de teste
NOME_ARQUIVO_CUSTOS = "custos.xlsx"  # <--- MUDE AQUI para o nome do seu arquivo de custos
NOME_ARQUIVO_TAREFAS = "tarefas.xlsx" # <--- MUDE AQUI para o nome do seu arquivo de tarefas

def run_test():
    try:
        # --- PASSO 1: Ler a planilha de tarefas (apenas colunas necessárias) ---
        print("PASSO 1: Lendo planilha de tarefas... ", end="", flush=True)
        start_time = time.time()
        colunas_tarefas_necessarias = {
            2: 'codigo_chamado', 5: 'instalador_tarefa', 6: 'categoria_tarefa', 7: 'status_os',
            9: 'tipo_servico_tarefa', 10: 'respons_financeira_tarefa', 11: 'cliente_tarefa',
            12: 'cidade_tarefa', 13: 'usuario_abertura'
        }
        df_tarefas = pd.read_excel(
            NOME_ARQUIVO_TAREFAS, header=None, skiprows=1, dtype=str, engine='openpyxl',
            usecols=colunas_tarefas_necessarias.keys()
        ).rename(columns=colunas_tarefas_necessarias)
        df_tarefas.dropna(subset=['codigo_chamado'], inplace=True)
        df_tarefas.drop_duplicates(subset=['codigo_chamado'], keep='last', inplace=True)
        print(f"OK! (Levou {time.time() - start_time:.2f} segundos)")

        # --- PASSO 2: Criar mapas de consulta ---
        print("PASSO 2: Criando mapas de consulta a partir das tarefas... ", end="", flush=True)
        start_time = time.time()
        mapas_tarefas = {
            col: df_tarefas.set_index('codigo_chamado')[col]
            for col in df_tarefas.columns if col != 'codigo_chamado'
        }
        # Liberar memória
        del df_tarefas
        print(f"OK! (Levou {time.time() - start_time:.2f} segundos)")

        # --- PASSO 3: Ler a planilha de custos ---
        print("PASSO 3: Lendo planilha de custos... ", end="", flush=True)
        start_time = time.time()
        df_custos = pd.read_excel(NOME_ARQUIVO_CUSTOS, header=None, skiprows=1, dtype=str, engine='openpyxl')
        nomes_custos = ['codigo_chamado', 'codigo_projeto', 'data_conclusao_instalacao', 'codigo_os', 'abertura', 'cliente', 'cidade', 'categoria', 'tipo_de_servico', 'resp_financeira', 'valor_custo', 'valor_venda', 'data_baixa_os', 'data_conclusao_servico', 'data_validacao_custo', 'data_faturamento', 'instalador']
        num_cols_custos = len(nomes_custos)
        if df_custos.shape[1] < num_cols_custos:
            nomes_custos = nomes_custos[:df_custos.shape[1]]
        df_custos = df_custos.iloc[:, :len(nomes_custos)]
        df_custos.columns = nomes_custos
        df_custos.dropna(subset=['codigo_chamado'], inplace=True)
        print(f"OK! (Levou {time.time() - start_time:.2f} segundos)")

        # --- PASSO 4: Aplicar os mapas ---
        print("PASSO 4: Aplicando mapas na planilha de custos... ", end="", flush=True)
        start_time = time.time()
        df_final = df_custos.copy()
        for nome_coluna, mapa in mapas_tarefas.items():
            df_final[nome_coluna] = df_final['codigo_chamado'].map(mapa)
        # Liberar memória
        del df_custos
        del mapas_tarefas
        print(f"OK! (Levou {time.time() - start_time:.2f} segundos)")

        # --- PASSO 5: Cálculos finais ---
        print("PASSO 5: Executando cálculos finais... ", end="", flush=True)
        start_time = time.time()
        colunas_fallback = {'instalador': 'instalador_tarefa', 'categoria': 'categoria_tarefa', 'tipo_de_servico': 'tipo_servico_tarefa', 'resp_financeira': 'respons_financeira_tarefa', 'cliente': 'cliente_tarefa', 'cidade': 'cidade_tarefa'}
        for col_orig, col_fallback in colunas_fallback.items():
            df_final[col_orig + '_final'] = df_final[col_orig].fillna(df_final.get(col_fallback))

        df_final['valor_custo'] = pd.to_numeric(df_final.get('valor_custo'), errors='coerce').fillna(0)
        df_final['valor_venda'] = pd.to_numeric(df_final.get('valor_venda'), errors='coerce').fillna(0)
        df_final['lucratividade'] = df_final['valor_venda'] - df_final['valor_custo']
        df_final['margem_lucro'] = (df_final['lucratividade'] / df_final['valor_venda'].replace(0, pd.NA) * 100)

        df_final['data_referencia'] = pd.to_datetime(df_final.get('data_baixa_os'), errors='coerce', dayfirst=True)
        df_final['data_referencia'].fillna(pd.to_datetime(df_final.get('abertura'), errors='coerce', dayfirst=True), inplace=True)
        print(f"OK! (Levou {time.time() - start_time:.2f} segundos)")

        print("\n--- DIAGNÓSTICO CONCLUÍDO COM SUCESSO! ---")
        print(f"Total de linhas processadas: {len(df_final)}")

    except Exception as e:
        print("\n--- OCORREU UM ERRO! ---")
        print(f"O processo falhou com o seguinte erro: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
