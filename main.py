import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
from linearmodels.panel import PanelOLS

# Importar o novo módulo de carregamento
from macrolab.data_loader import load_and_merge_macrolab_backups


def main(data_dir="dados"):
    print("--- Iniciando Pipeline de Dados ---")

    # 1. Ingestão e Consolidação (usa data_loader; também procura merged CSV salvo)
    merged_path = os.path.join(data_dir, "merged_macrolab_backups.csv")
    if os.path.exists(merged_path):
        print(f"[Info] Encontrado arquivo consolidado: {merged_path}. Carregando...")
        df_cleaned = pd.read_csv(merged_path)
    else:
        df_cleaned = load_and_merge_macrolab_backups(data_dir=data_dir)
        # Se loader salvou, reponha merged_path; senão, salvar cópia local
        if df_cleaned is not None and not df_cleaned.empty and not os.path.exists(merged_path):
            try:
                os.makedirs(data_dir, exist_ok=True)
                df_cleaned.to_csv(merged_path, index=False)
                print(f"[Info] Salvou cópia consolidada em: {merged_path}")
            except Exception as e:
                print(f"[Aviso] Não foi possível salvar merged CSV: {e}")

    if df_cleaned is None or df_cleaned.empty:
        raise ValueError("O DataFrame consolidado está vazio. Verifique os arquivos na pasta 'dados'.")

    # Normalizar coluna temporal
    if 'Year' not in df_cleaned.columns and 'Date' in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={'Date': 'Year'})

    # Tentar converter Year para datetime/ano inteiro
    df_cleaned['Year'] = pd.to_datetime(df_cleaned['Year'], errors='coerce').dt.year
    df_cleaned = df_cleaned.dropna(subset=['Year']).copy()
    df_cleaned['Year'] = df_cleaned['Year'].astype(int)

    print("\n--- Iniciando Estimação VAR ---")
    # Preparar e rodar VAR
    var_cols = ['Year', 'GDP', 'Inflation', 'Interest_Rate']
    var_cols_present = [col for col in var_cols if col in df_cleaned.columns]

    # Necessitamos ao menos uma variável além do tempo para VAR
    if 'Year' in var_cols_present and len(var_cols_present) > 1:
        var_data = df_cleaned[var_cols_present].dropna().copy()
        if not var_data.empty:
            # Garantir que index é temporal
            if 'Year' in var_data.columns:
                var_data = var_data.set_index('Year')
            try:
                model = VAR(var_data)
                res = model.fit(maxlags=1, ic='aic')
                print(res.summary())
            except Exception as e:
                print(f"Erro na estimação do VAR: {e}")
        else:
            print("var_data ficou vazio após dropna() — dados insuficientes para VAR.")
    else:
        print("Dados insuficientes para rodar o VAR (verifique as colunas disponíveis).")

    print("\n--- Iniciando Estimação de Painel (PanelOLS/SPJ) ---")
    # Preparar e rodar Modelo de Painel
    df_panel = df_cleaned[[c for c in var_cols_present if c in df_cleaned.columns]].dropna().copy()
    if not df_panel.empty and 'GDP' in df_panel.columns:
        # Criar colunas de painel (ajuste entity conforme seus dados reais)
        df_panel['entity'] = 'BRA'
        df_panel['time'] = pd.to_datetime(df_panel['Year'], format='%Y', errors='coerce').dt.year
        df_panel = df_panel.dropna(subset=['time']).copy()
        df_panel = df_panel.set_index(['entity', 'time']).sort_index()

        y = df_panel['GDP']
        X_cols = [c for c in df_panel.columns if c not in ['Year', 'GDP', 'entity', 'time']]
        if not X_cols:
            print("Sem regressoras disponíveis para o modelo de painel.")
        else:
            X = df_panel[X_cols].astype(float)
            X = sm.add_constant(X)
            try:
                panel_model = PanelOLS(y, X, entity_effects=False, time_effects=False)
                panel_res = panel_model.fit()
                print(panel_res.summary)
            except Exception as e:
                print(f"Erro na estimação do Painel: {e}")
    else:
        print("Dados insuficientes para rodar o modelo de painel.")


if __name__ == "__main__":
    main(data_dir="dados")
