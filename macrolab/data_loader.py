import os
import pandas as pd


def load_and_merge_macrolab_backups(data_dir="dados"):
    """
    Consolida os arquivos CSV de backup do MacroLab em um único DataFrame estruturado.
    Espera uma coluna comum de tempo (ex: 'Year' ou 'Date') em cada arquivo.
    """
    # Mapeamento dos arquivos de backup comuns no projeto
    backup_files = {
        'Divida_PIB': 'backup_Divida_PIB.csv',
        'Risco_Brasil': 'backup_Risco_Brasil.csv',
        # Adicione outros arquivos conforme necessário (ex: inflação, juros, câmbio)
    }
    
    merged_df = None
    time_col = 'Year'  # Pode ser ajustado para 'Date' conforme o padrão dos seus arquivos
    
    for indicator_name, filename in backup_files.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"[Aviso] Arquivo não encontrado: {filepath}. Pulando...")
            continue
            
        try:
            # Leitura do CSV (ajuste o separador se necessário, ex: sep=';')
            df = pd.read_csv(filepath)
            
            # Padronização básica de nomes de colunas se necessário
            # Exemplo: renomear valor para o nome do indicador
            if 'Value' in df.columns:
                df = df.rename(columns={'Value': indicator_name})
            elif indicator_name not in df.columns and len(df.columns) >= 2:
                # Assume que a segunda coluna é o valor do indicador
                val_col = [c for c in df.columns if c != time_col][0]
                df = df.rename(columns={val_col: indicator_name})
                
            if time_col not in df.columns:
                print(f"[Erro] Coluna temporal '{time_col}' não encontrada em {filename}.")
                continue
                
            # Seleciona apenas o tempo e a coluna de interesse para evitar conflitos
            df_subset = df[[time_col, indicator_name]].dropna(subset=[time_col])
            
            # Realiza o merge progressivo (outer join para preservar toda a linha temporal)
            if merged_df is None:
                merged_df = df_subset
            else:
                merged_df = pd.merge(merged_df, df_subset, on=time_col, how='outer')
                
            print(f"[Sucesso] Integrado: {filename}")
            
        except Exception as e:
            print(f"[Erro] Falha ao processar {filename}: {e}")
            
    if merged_df is not None:
        # Ordenação temporal e limpeza de índices duplicados
        if time_col in merged_df.columns:
            try:
                # tentar converter Year para inteiro/ano para ordenar corretamente
                merged_df[time_col] = pd.to_numeric(merged_df[time_col], errors='coerce')
            except Exception:
                pass
            merged_df = merged_df.sort_values(by=time_col).reset_index(drop=True)
            
        print(f"\n[Info] Base consolidada com sucesso. Dimensões: {merged_df.shape}")
        return merged_df
    else:
        print("[Erro] Nenhum dado foi consolidado.")
        return pd.DataFrame()


if __name__ == "__main__":
    # Teste rápido do loader
    df_cleaned = load_and_merge_macrolab_backups(data_dir="dados")
    print(df_cleaned.head())
