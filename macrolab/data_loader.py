import os
import pandas as pd


def _default_backup_files():
    return {
        'Divida_PIB': 'backup_Divida_PIB.csv',
        'Risco_Brasil': 'backup_Risco_Brasil.csv',
        # Adicione outros arquivos conforme necessário
    }


def _detect_time_column(df, preferred=('Year', 'Date')):
    for col in preferred:
        if col in df.columns:
            return col
    # fallback: try to find a column that looks like a year or date
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        s = sample.iloc[0]
        # heurística simples: coluna com 4 dígitos -> ano
        if len(s) >= 4 and s[:4].isdigit():
            return col
        # tentar detectar formato de data com '-','/' ou ano-mes
        if any(ch in s for ch in ['-', '/']) and any(d.isdigit() for d in s):
            return col
    return None


def load_and_merge_macrolab_backups(
    data_dir="dados",
    backup_files=None,
    preferred_time_cols=('Year', 'Date'),
    sep=',',
    save_merged=True,
    merged_filename='merged_macrolab_backups.csv',
):
    """
    Consolida os arquivos CSV de backup do MacroLab em um único DataFrame estruturado.

    Parâmetros:
    - data_dir: diretório onde os CSVs estão armazenados
    - backup_files: dict mapping nome_indicador -> filename. Se None, usa um conjunto padrão
    - preferred_time_cols: tupla com nomes de colunas temporais preferidas
    - sep: separador do CSV
    - save_merged: se True, salva o CSV consolidado dentro de data_dir
    - merged_filename: nome do arquivo a ser salvo

    Retorna:
    - pandas.DataFrame consolidado (com coluna 'Year' sempre que possível)
    """
    if backup_files is None:
        backup_files = _default_backup_files()

    merged_df = None

    for indicator_name, filename in backup_files.items():
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"[Aviso] Arquivo não encontrado: {filepath}. Pulando...")
            continue

        try:
            df = pd.read_csv(filepath, sep=sep)

            # detectar coluna temporal no arquivo (pode ser Year ou Date ou outra)
            time_col = _detect_time_column(df, preferred=preferred_time_cols)
            if time_col is None:
                print(f"[Erro] Não foi possível detectar coluna temporal em {filename}. Pulando...")
                continue

            # padronizar nome da coluna temporal para 'Year' quando fizer sentido
            if time_col != 'Year':
                # se for Date, tentar extrair o ano
                if time_col.lower() == 'date':
                    try:
                        df['Year'] = pd.to_datetime(df[time_col], errors='coerce').dt.year
                    except Exception:
                        df['Year'] = pd.to_datetime(df[time_col], infer_datetime_format=True, errors='coerce').dt.year
                else:
                    # criar Year tentando converter para numérico (ano) ou extraindo parte inicial
                    df['Year'] = pd.to_numeric(df[time_col], errors='coerce')
                    if df['Year'].isnull().all():
                        # tentar parse de data e extrair ano
                        df['Year'] = pd.to_datetime(df[time_col], errors='coerce').dt.year
            else:
                # garantir que Year seja numérico/inteiro
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                if df['Year'].isnull().all():
                    df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year

            # Padronização básica de nomes de colunas de valor
            if 'Value' in df.columns and indicator_name not in df.columns:
                df = df.rename(columns={'Value': indicator_name})
            elif indicator_name not in df.columns:
                # tentativa heurística: escolher a primeira coluna que não é a temporal
                non_time_cols = [c for c in df.columns if c not in ['Year', time_col]]
                if non_time_cols:
                    df = df.rename(columns={non_time_cols[0]: indicator_name})

            if 'Year' not in df.columns or indicator_name not in df.columns:
                print(f"[Erro] Coluna temporal 'Year' ou indicador '{indicator_name}' não encontrada/normalizada em {filename}. Pulando...")
                continue

            df_subset = df[['Year', indicator_name]].dropna(subset=['Year'])

            # converter Year para inteiro para evitar duplicidade de tipos
            df_subset['Year'] = df_subset['Year'].astype(int)

            if merged_df is None:
                merged_df = df_subset
            else:
                merged_df = pd.merge(merged_df, df_subset, on='Year', how='outer')

            print(f"[Sucesso] Integrado: {filename}")

        except Exception as e:
            print(f"[Erro] Falha ao processar {filename}: {e}")

    if merged_df is not None and not merged_df.empty:
        merged_df = merged_df.sort_values(by='Year').reset_index(drop=True)

        if save_merged:
            os.makedirs(data_dir, exist_ok=True)
            out_path = os.path.join(data_dir, merged_filename)
            try:
                merged_df.to_csv(out_path, index=False)
                print(f"[Info] Base consolidada salva em: {out_path}")
            except Exception as e:
                print(f"[Aviso] Falha ao salvar base consolidada: {e}")

        print(f"\n[Info] Base consolidada com sucesso. Dimensões: {merged_df.shape}")
        return merged_df

    print("[Erro] Nenhum dado foi consolidado.")
    return pd.DataFrame()


if __name__ == '__main__':
    df_cleaned = load_and_merge_macrolab_backups(data_dir='dados')
    print(df_cleaned.head())
