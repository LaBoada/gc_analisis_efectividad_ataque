import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
import os
import glob

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="🏀 ACB Basketball Analysis",
    page_icon="🏀",
    layout="wide"
)

# ==============================================================================
# DATA FUNCTIONS
# ==============================================================================
@st.cache_data
def load_from_drive(folder_path):
    files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    all_data = []
    for file in files:
        try:
            df = pd.read_excel(file)
            df['SOURCE'] = os.path.basename(file)
            all_data.append(df)
        except Exception as e:
            st.warning(f"Error en {os.path.basename(file)}: {e}")
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined['TIME_TO_HALFCOURT'] = combined['INICIO_ACCION'] - combined['MEDIO_CAMPO']
        combined['TOTAL_POSSESSION_TIME'] = combined['INICIO_ACCION'] - combined['FIN_ACCION']
        combined['TIME_BUCKET'] = pd.cut(combined['TOTAL_POSSESSION_TIME'], 
                                         bins=[0, 8, 14, 24], 
                                         labels=['Fast (0-8s)', 'Medium (8-14s)', 'Slow (14-24s)'])
        return combined, len(files)
    return None, 0

@st.cache_data
def load_uploaded_files(uploaded_files):
    all_data = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        df['SOURCE'] = file.name
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    combined['TIME_TO_HALFCOURT'] = combined['INICIO_ACCION'] - combined['MEDIO_CAMPO']
    combined['TOTAL_POSSESSION_TIME'] = combined['INICIO_ACCION'] - combined['FIN_ACCION']
    combined['TIME_BUCKET'] = pd.cut(combined['TOTAL_POSSESSION_TIME'], 
                                     bins=[0, 8, 14, 24], 
                                     labels=['Fast (0-8s)', 'Medium (8-14s)', 'Slow (14-24s)'])
    return combined

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================
def calculate_general_stats(df):
    t2_att = df['T2'].sum()
    t3_att = df['T3'].sum()
    tc_att = df['TC'].sum() if 'TC' in df.columns else t2_att + t3_att
    t2_pts = df['PUNTOS T2'].sum()
    t3_pts = df['PUNTOS T3'].sum()
    tc_pts = df['PUNTOS TC'].sum() if 'PUNTOS TC' in df.columns else t2_pts + t3_pts
    
    return {
        'Posesiones': len(df),
        'Puntos Totales': int(df['PUNTOS'].sum()),
        'PPA': round(df['PUNTOS'].mean(), 2),
        'PPT2': round(t2_pts / t2_att, 2) if t2_att > 0 else 0,
        'PPT3': round(t3_pts / t3_att, 2) if t3_att > 0 else 0,
        'PPT': round(tc_pts / tc_att, 2) if tc_att > 0 else 0,
        'Total TO': int(df['TO'].sum()),
        'Puntos TO': int(df['PUNTOS TO'].sum()) if 'PUNTOS TO' in df.columns else 0,
        '%TO': round(df['TO'].sum() / len(df) * 100, 1) if len(df) > 0 else 0
    }

def analyze_sets_performance(df):
    set_stats = df.groupby('CATEGORIA').agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'T2': 'sum', 'PUNTOS T2': 'sum',
        'T3': 'sum', 'PUNTOS T3': 'sum',
        'TC': 'sum', 'PUNTOS TC': 'sum',
        'TO': 'sum'
    }).reset_index()
    set_stats.columns = ['Categoria', 'Posesiones', 'Puntos', 'PPA', 
                         'T2_Att', 'T2_Pts', 'T3_Att', 'T3_Pts', 'TC_Att', 'TC_Pts', 'TO']
    set_stats['PPA'] = set_stats['PPA'].round(2)
    set_stats['PPT2'] = (set_stats['T2_Pts'] / set_stats['T2_Att']).round(2).fillna(0)
    set_stats['PPT3'] = (set_stats['T3_Pts'] / set_stats['T3_Att']).round(2).fillna(0)
    set_stats['PPT'] = (set_stats['TC_Pts'] / set_stats['TC_Att']).round(2).fillna(0)
    set_stats['%TO'] = (set_stats['TO'] / set_stats['Posesiones'] * 100).round(1)
    return set_stats[['Categoria', 'Posesiones', 'Puntos', 'PPA', 'PPT2', 'PPT3', 'PPT', '%TO']].sort_values('PPA', ascending=False)

def analyze_with_stats(df, group_col, group_name):
    stats = df.groupby(group_col).agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'T2': 'sum', 'PUNTOS T2': 'sum',
        'T3': 'sum', 'PUNTOS T3': 'sum',
        'TC': 'sum', 'PUNTOS TC': 'sum',
        'TO': 'sum'
    }).reset_index()
    stats.columns = [group_name, 'Posesiones', 'Puntos', 'PPA', 
                     'T2_Att', 'T2_Pts', 'T3_Att', 'T3_Pts', 'TC_Att', 'TC_Pts', 'TO']
    stats['PPA'] = stats['PPA'].round(2)
    stats['PPT2'] = (stats['T2_Pts'] / stats['T2_Att']).round(2).fillna(0)
    stats['PPT3'] = (stats['T3_Pts'] / stats['T3_Att']).round(2).fillna(0)
    stats['PPT'] = (stats['TC_Pts'] / stats['TC_Att']).round(2).fillna(0)
    stats['%TO'] = (stats['TO'] / stats['Posesiones'] * 100).round(1)
    return stats[[group_name, 'Posesiones', 'Puntos', 'PPA', 'PPT2', 'PPT3', 'PPT', '%TO']].sort_values('PPA', ascending=False)

def analyze_lineups(df):
    stats = df.groupby(['J1', 'J2', 'J3', 'J4', 'J5']).agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'T2': 'sum', 'PUNTOS T2': 'sum',
        'T3': 'sum', 'PUNTOS T3': 'sum',
        'TC': 'sum', 'PUNTOS TC': 'sum',
        'TO': 'sum'
    }).reset_index()
    stats.columns = ['J1', 'J2', 'J3', 'J4', 'J5', 'Posesiones', 'Puntos', 'PPA', 
                     'T2_Att', 'T2_Pts', 'T3_Att', 'T3_Pts', 'TC_Att', 'TC_Pts', 'TO']
    stats['PPA'] = stats['PPA'].round(2)
    stats['PPT2'] = (stats['T2_Pts'] / stats['T2_Att']).round(2).fillna(0)
    stats['PPT3'] = (stats['T3_Pts'] / stats['T3_Att']).round(2).fillna(0)
    stats['PPT'] = (stats['TC_Pts'] / stats['TC_Att']).round(2).fillna(0)
    stats['%TO'] = (stats['TO'] / stats['Posesiones'] * 100).round(1)
    return stats[['J1', 'J2', 'J3', 'J4', 'J5', 'Posesiones', 'Puntos', 'PPA', 'PPT2', 'PPT3', 'PPT', '%TO']].sort_values('PPA', ascending=False)

def analyze_players(df):
    """Analizar eficiencia de jugadores (finalizadores)"""
    stats = df.groupby('FINALIZADOR').agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'T2': 'sum', 'PUNTOS T2': 'sum',
        'T3': 'sum', 'PUNTOS T3': 'sum',
        'TC': 'sum', 'PUNTOS TC': 'sum',
        'TO': 'sum'
    }).reset_index()
    stats.columns = ['Jugador', 'Acciones', 'Puntos', 'PPA', 
                     'T2_Att', 'T2_Pts', 'T3_Att', 'T3_Pts', 'TC_Att', 'TC_Pts', 'TO']
    stats['PPA'] = stats['PPA'].round(2)
    stats['PPT2'] = (stats['T2_Pts'] / stats['T2_Att']).round(2).fillna(0)
    stats['PPT3'] = (stats['T3_Pts'] / stats['T3_Att']).round(2).fillna(0)
    stats['PPT'] = (stats['TC_Pts'] / stats['TC_Att']).round(2).fillna(0)
    stats['%TO'] = (stats['TO'] / stats['Acciones'] * 100).round(1)
    return stats[['Jugador', 'Acciones', 'Puntos', 'PPA', 'PPT2', 'PPT3', 'PPT', '%TO']].sort_values('PPA', ascending=False)

def analyze_duos(df, min_n=3):
    """Analizar eficiencia de dúos (combinaciones de 2 jugadores en pista)"""
    duos_data = []
    
    for idx, row in df.iterrows():
        players = [row['J1'], row['J2'], row['J3'], row['J4'], row['J5']]
        players = [p for p in players if pd.notna(p)]
        
        for duo in combinations(sorted(players), 2):
            duos_data.append({
                'Duo': f"{duo[0]} + {duo[1]}",
                'PUNTOS': row['PUNTOS'],
                'T2': row['T2'],
                'PUNTOS T2': row['PUNTOS T2'],
                'T3': row['T3'],
                'PUNTOS T3': row['PUNTOS T3'],
                'TC': row['TC'] if 'TC' in df.columns else row['T2'] + row['T3'],
                'PUNTOS TC': row['PUNTOS TC'] if 'PUNTOS TC' in df.columns else row['PUNTOS T2'] + row['PUNTOS T3'],
                'TO': row['TO']
            })
    
    if not duos_data:
        return pd.DataFrame()
    
    duos_df = pd.DataFrame(duos_data)
    stats = duos_df.groupby('Duo').agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'T2': 'sum', 'PUNTOS T2': 'sum',
        'T3': 'sum', 'PUNTOS T3': 'sum',
        'TC': 'sum', 'PUNTOS TC': 'sum',
        'TO': 'sum'
    }).reset_index()
    
    stats.columns = ['Duo', 'Posesiones', 'Puntos', 'PPA', 
                     'T2_Att', 'T2_Pts', 'T3_Att', 'T3_Pts', 'TC_Att', 'TC_Pts', 'TO']
    stats['PPA'] = stats['PPA'].round(2)
    stats['PPT2'] = (stats['T2_Pts'] / stats['T2_Att']).round(2).fillna(0)
    stats['PPT3'] = (stats['T3_Pts'] / stats['T3_Att']).round(2).fillna(0)
    stats['PPT'] = (stats['TC_Pts'] / stats['TC_Att']).round(2).fillna(0)
    stats['%TO'] = (stats['TO'] / stats['Posesiones'] * 100).round(1)
    
    return stats[['Duo', 'Posesiones', 'Puntos', 'PPA', 'PPT2', 'PPT3', 'PPT', '%TO']].sort_values('PPA', ascending=False)

def analyze_trios(df, min_n=3):
    """Analizar eficiencia de tríos (combinaciones de 3 jugadores en pista)"""
    trios_data = []
    
    for idx, row in df.iterrows():
        players = [row['J1'], row['J2'], row['J3'], row['J4'], row['J5']]
        players = [p for p in players if pd.notna(p)]
        
        for trio in combinations(sorted(players), 3):
            trios_data.append({
                'Trio': f"{trio[0]} + {trio[1]} + {trio[2]}",
                'PUNTOS': row['PUNTOS'],
                'T2': row['T2'],
                'PUNTOS T2': row['PUNTOS T2'],
                'T3': row['T3'],
                'PUNTOS T3': row['PUNTOS T3'],
                'TC': row['TC'] if 'TC' in df.columns else row['T2'] + row['T3'],
                'PUNTOS TC': row['PUNTOS TC'] if 'PUNTOS TC' in df.columns else row['PUNTOS T2'] + row['PUNTOS T3'],
                'TO': row['TO']
            })
    
    if not trios_data:
        return pd.DataFrame()
    
    trios_df = pd.DataFrame(trios_data)
    stats = trios_df.groupby('Trio').agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'T2': 'sum', 'PUNTOS T2': 'sum',
        'T3': 'sum', 'PUNTOS T3': 'sum',
        'TC': 'sum', 'PUNTOS TC': 'sum',
        'TO': 'sum'
    }).reset_index()
    
    stats.columns = ['Trio', 'Posesiones', 'Puntos', 'PPA', 
                     'T2_Att', 'T2_Pts', 'T3_Att', 'T3_Pts', 'TC_Att', 'TC_Pts', 'TO']
    stats['PPA'] = stats['PPA'].round(2)
    stats['PPT2'] = (stats['T2_Pts'] / stats['T2_Att']).round(2).fillna(0)
    stats['PPT3'] = (stats['T3_Pts'] / stats['T3_Att']).round(2).fillna(0)
    stats['PPT'] = (stats['TC_Pts'] / stats['TC_Att']).round(2).fillna(0)
    stats['%TO'] = (stats['TO'] / stats['Posesiones'] * 100).round(1)
    
    return stats[['Trio', 'Posesiones', 'Puntos', 'PPA', 'PPT2', 'PPT3', 'PPT', '%TO']].sort_values('PPA', ascending=False)

def get_best_performers(df, min_n=3):
    """Obtener el mejor generador, finalizador y bloqueador"""
    results = {}
    
    # Mejor Finalizador
    fin_stats = analyze_players(df)
    fin_filtered = fin_stats[fin_stats['Acciones'] >= min_n]
    if len(fin_filtered) > 0:
        best_fin = fin_filtered.iloc[0]
        results['finalizador'] = {
            'nombre': best_fin['Jugador'],
            'ppa': best_fin['PPA'],
            'acciones': best_fin['Acciones']
        }
    
    # Mejor Generador
    df_gen = df[df['GENERADOR'].notna()]
    if len(df_gen) > 0:
        gen_stats = analyze_with_stats(df_gen, 'GENERADOR', 'Generador')
        gen_filtered = gen_stats[gen_stats['Posesiones'] >= min_n]
        if len(gen_filtered) > 0:
            best_gen = gen_filtered.iloc[0]
            results['generador'] = {
                'nombre': best_gen['Generador'],
                'ppa': best_gen['PPA'],
                'acciones': best_gen['Posesiones']
            }
    
    # Mejor Bloqueador
    bloq_stats = analyze_with_stats(df, 'BLOQUEADOR', 'Bloqueador')
    bloq_filtered = bloq_stats[(bloq_stats['Posesiones'] >= min_n) & (bloq_stats['Bloqueador'] != 'NADA')]
    if len(bloq_filtered) > 0:
        best_bloq = bloq_filtered.iloc[0]
        results['bloqueador'] = {
            'nombre': best_bloq['Bloqueador'],
            'ppa': best_bloq['PPA'],
            'acciones': best_bloq['Posesiones']
        }
    
    return results

def analyze_player_by_concept(df, min_n=1):
    """Analizar eficiencia de cada jugador por categoría/concepto"""
    player_concept = df.groupby(['FINALIZADOR', 'CATEGORIA']).agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'TO': 'sum'
    }).reset_index()
    
    player_concept.columns = ['Jugador', 'Concepto', 'Acciones', 'Puntos', 'PPA', 'TO']
    player_concept['PPA'] = player_concept['PPA'].round(2)
    player_concept['%TO'] = (player_concept['TO'] / player_concept['Acciones'] * 100).round(1)
    
    return player_concept[['Jugador', 'Concepto', 'Acciones', 'Puntos', 'PPA', '%TO']].sort_values(['Jugador', 'PPA'], ascending=[True, False])

def analyze_concept_by_defense(df, min_n=1):
    """Analizar eficiencia del concepto según la defensa rival"""
    if 'DEFENSA' not in df.columns:
        return None
    
    concept_defense = df.groupby(['CATEGORIA', 'DEFENSA']).agg({
        'PUNTOS': ['count', 'sum', 'mean'],
        'T2': 'sum', 'PUNTOS T2': 'sum',
        'T3': 'sum', 'PUNTOS T3': 'sum',
        'TC': 'sum', 'PUNTOS TC': 'sum',
        'TO': 'sum'
    }).reset_index()
    
    concept_defense.columns = ['Concepto', 'Defensa', 'Posesiones', 'Puntos', 'PPA',
                               'T2_Att', 'T2_Pts', 'T3_Att', 'T3_Pts', 'TC_Att', 'TC_Pts', 'TO']
    concept_defense['PPA'] = concept_defense['PPA'].round(2)
    concept_defense['PPT2'] = (concept_defense['T2_Pts'] / concept_defense['T2_Att']).round(2).fillna(0)
    concept_defense['PPT3'] = (concept_defense['T3_Pts'] / concept_defense['T3_Att']).round(2).fillna(0)
    concept_defense['PPT'] = (concept_defense['TC_Pts'] / concept_defense['TC_Att']).round(2).fillna(0)
    concept_defense['%TO'] = (concept_defense['TO'] / concept_defense['Posesiones'] * 100).round(1)
    
    return concept_defense[['Concepto', 'Defensa', 'Posesiones', 'Puntos', 'PPA', 'PPT2', 'PPT3', 'PPT', '%TO']].sort_values(['Concepto', 'PPA'], ascending=[True, False])

def analyze_shot_types(df):
    """Analizar volumen y eficiencia por Tipo_tiro"""
    if 'TIPO_TIRO' not in df.columns:
        return None
    
    shot_stats = df.groupby('TIPO_TIRO').agg({
        'PUNTOS': ['count', 'sum', 'mean']
    }).reset_index()
    
    shot_stats.columns = ['Tipo_Tiro', 'Acciones', 'Puntos', 'PPA']
    shot_stats['PPA'] = shot_stats['PPA'].round(2)
    
    return shot_stats.sort_values('Acciones', ascending=False)

# ==============================================================================
# STYLING FUNCTION
# ==============================================================================
def style_dataframe(df, columns_to_style):
    styler = df.style
    for col in columns_to_style:
        if col in df.columns:
            if col == '%TO':
                styler = styler.background_gradient(subset=[col], cmap='RdYlGn_r', vmin=0, vmax=df[col].max() if df[col].max() > 0 else 1)
            else:
                styler = styler.background_gradient(subset=[col], cmap='RdYlGn', vmin=0, vmax=df[col].max() if df[col].max() > 0 else 1)
    return styler.format(precision=2)

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================
def create_player_efficiency_map(df, min_n):
    """Mapa de eficiencia de jugadores: PPA vs Volumen"""
    player_stats = analyze_players(df)
    player_stats = player_stats[player_stats['Acciones'] >= min_n]
    
    if len(player_stats) == 0:
        return None
    
    avg_ppa = player_stats['PPA'].mean()
    avg_acciones = player_stats['Acciones'].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=player_stats['Acciones'],
        y=player_stats['PPA'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=player_stats['PPA'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='PPA')
        ),
        text=player_stats['Jugador'],
        textposition='top center',
        textfont=dict(size=9),
        hovertemplate='<b>%{text}</b><br>Acciones: %{x}<br>PPA: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=avg_ppa, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Media PPA: {avg_ppa:.2f}",
        annotation_position="right"
    )
    
    fig.add_vline(
        x=avg_acciones, 
        line_dash="dash", 
        line_color="blue",
        annotation_text=f"Media Acciones: {avg_acciones:.1f}",
        annotation_position="top"
    )
    
    fig.add_annotation(x=player_stats['Acciones'].max() * 0.85, y=player_stats['PPA'].max() * 0.95,
                       text="⭐ Alto Vol + Alta Efic", showarrow=False, font=dict(size=10, color='green'))
    fig.add_annotation(x=player_stats['Acciones'].min() + (avg_acciones - player_stats['Acciones'].min()) * 0.3,
                       y=player_stats['PPA'].max() * 0.95,
                       text="💎 Bajo Vol + Alta Efic", showarrow=False, font=dict(size=10, color='blue'))
    fig.add_annotation(x=player_stats['Acciones'].max() * 0.85,
                       y=player_stats['PPA'].min() + (avg_ppa - player_stats['PPA'].min()) * 0.3,
                       text="⚠️ Alto Vol + Baja Efic", showarrow=False, font=dict(size=10, color='orange'))
    fig.add_annotation(x=player_stats['Acciones'].min() + (avg_acciones - player_stats['Acciones'].min()) * 0.3,
                       y=player_stats['PPA'].min() + (avg_ppa - player_stats['PPA'].min()) * 0.3,
                       text="❌ Bajo Vol + Baja Efic", showarrow=False, font=dict(size=10, color='red'))
    
    fig.update_layout(
        title='🗺️ Mapa de Eficiencia de Jugadores',
        xaxis_title='Volumen de Acciones',
        yaxis_title='Puntos por Acción (PPA)',
        height=500,
        margin=dict(l=20, r=20, t=80, b=30)
    )
    
    return fig

def create_shot_type_chart(df):
    """Gráfico de Volumen vs Eficiencia por Tipo de Tiro (TIPO_TIRO)"""
    shot_data = analyze_shot_types(df)
    
    if shot_data is None or len(shot_data) == 0:
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Colores variados para cada tipo de tiro
    colors = ['#4472C4', '#70AD47', '#FFC000', '#C00000', '#7030A0', '#00B0F0', '#FF6600', '#92D050']
    bar_colors = colors[:len(shot_data)]
    
    fig.add_trace(go.Bar(
        x=shot_data['Tipo_Tiro'], 
        y=shot_data['Acciones'],
        name='Acciones', 
        marker_color=bar_colors,
        text=[f"n={int(n)}" for n in shot_data['Acciones']], 
        textposition='outside'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=shot_data['Tipo_Tiro'], 
        y=shot_data['PPA'],
        name='PPA', 
        mode='lines+markers+text',
        line=dict(color='#C00000', width=3), 
        marker=dict(size=12, symbol='diamond'),
        text=[f"{v:.2f}" for v in shot_data['PPA']], 
        textposition='top center',
        textfont=dict(size=12, color='#C00000')
    ), secondary_y=True)
    
    fig.update_layout(
        title='🎯 Volumen vs Eficiencia por Tipo de Tiro',
        height=400, 
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=20, r=20, t=80, b=30)
    )
    fig.update_yaxes(title_text="Número de Acciones", secondary_y=False)
    fig.update_yaxes(title_text="Puntos por Acción (PPA)", secondary_y=True)
    
    return fig

# ==============================================================================
# MAIN APP
# ==============================================================================

# Sidebar - Fuente de datos
st.sidebar.header("📁 Fuente de Datos")
data_source = st.sidebar.radio("Selecciona fuente:", ["📂 Google Drive", "📤 Subir archivos"])

df = None
num_files = 0

if data_source == "📂 Google Drive":
    folder_path = st.sidebar.text_input("Ruta:", value="/content/drive/MyDrive/complet")
    if st.sidebar.button("🔄 Cargar datos", type="primary"):
        if os.path.exists(folder_path):
            df, num_files = load_from_drive(folder_path)
            if df is not None:
                st.session_state['df'] = df
                st.session_state['num_files'] = num_files
                st.sidebar.success(f"✅ {num_files} archivos!")
    if 'df' in st.session_state:
        df = st.session_state['df']
        num_files = st.session_state['num_files']
else:
    uploaded = st.sidebar.file_uploader("Subir Excel", type=['xlsx'], accept_multiple_files=True)
    if uploaded:
        df = load_uploaded_files(uploaded)
        num_files = len(uploaded)

if df is not None:
    # Logo y título
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("https://i0.wp.com/basketinsular.net/wp-content/uploads/2014/10/cb-gran-canaria.jpg?ssl=1", width=100)
    with col_title:
        st.title("🏀 ACB Basketball Efficiency Analysis")
    
    # Filters con multiselect
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Filtros")
    min_n = st.sidebar.slider("Muestra mínima (n)", 1, 20, 3)

    if 'COMPETICION' in df.columns:
        competiciones = sorted(df['COMPETICION'].unique().tolist())
        sel_comp = st.sidebar.multiselect("🏆 Competición", competiciones, default=competiciones)
        if sel_comp:
            df = df[df['COMPETICION'].isin(sel_comp)]

    if 'PISTA' in df.columns:
        pistas = sorted(df['PISTA'].unique().tolist())
        sel_pista = st.sidebar.multiselect("🏠 Pista", pistas, default=pistas)
        if sel_pista:
            df = df[df['PISTA'].isin(sel_pista)]

    if 'VICTORIA' in df.columns:
        victorias = sorted(df['VICTORIA'].unique().tolist())
        sel_victoria = st.sidebar.multiselect("✅ Victoria", victorias, default=victorias)
        if sel_victoria:
            df = df[df['VICTORIA'].isin(sel_victoria)]

    if 'JORNADA' in df.columns:
        jornadas = sorted(df['JORNADA'].unique().tolist())
        sel_jornada = st.sidebar.multiselect("📅 Jornada", jornadas, default=jornadas)
        if sel_jornada:
            df = df[df['JORNADA'].isin(sel_jornada)]

    if 'RIVAL' in df.columns:
        rivales = sorted(df['RIVAL'].unique().tolist())
        sel_rival = st.sidebar.multiselect("🆚 Rival", rivales, default=rivales)
        if sel_rival:
            df = df[df['RIVAL'].isin(sel_rival)]

    if 'CUARTO' in df.columns:
        cuartos = sorted(df['CUARTO'].unique().tolist())
        sel_cuarto = st.sidebar.multiselect("⏱️ Cuarto", cuartos, default=cuartos)
        if sel_cuarto:
            df = df[df['CUARTO'].isin(sel_cuarto)]
    
    # ==========================================================================
    # PESTAÑAS PRINCIPALES
    # ==========================================================================
    tab_general, tab_individual, tab_combinaciones, tab_sistemas, tab_bloqueos, tab_quintetos = st.tabs([
        "📊 General", 
        "👤 Individual", 
        "👥 Combinaciones",
        "📋 Sistemas", 
        "🧱 Bloqueos",
        "🏀 Quintetos"
    ])
    
    # ==========================================================================
    # TAB: GENERAL
    # ==========================================================================
    with tab_general:
        st.subheader("📊 Contexto General")
        stats = calculate_general_stats(df)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📊 Posesiones", stats['Posesiones'])
        c2.metric("🏀 Puntos Totales", stats['Puntos Totales'])
        c3.metric("📈 PPA", stats['PPA'])
        c4.metric("🎯 PPT2", stats['PPT2'])
        c5.metric("🎯 PPT3", stats['PPT3'])
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🎯 PPT", stats['PPT'])
        c2.metric("❌ Total TO", stats['Total TO'])
        c3.metric("📉 Puntos TO", stats['Puntos TO'])
        c4.metric("📊 %TO", f"{stats['%TO']}%")
        c5.metric("📁 Archivos", num_files)
        
        # Mejores performers
        st.markdown("---")
        st.subheader("⭐ Mejores Rendimientos")
        
        best = get_best_performers(df, min_n)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'finalizador' in best:
                st.metric(
                    "🎯 Mejor Finalizador",
                    best['finalizador']['nombre'],
                    f"PPA: {best['finalizador']['ppa']} (n={best['finalizador']['acciones']})"
                )
            else:
                st.info("No hay datos suficientes")
        
        with col2:
            if 'generador' in best:
                st.metric(
                    "🎨 Mejor Generador",
                    best['generador']['nombre'],
                    f"PPA: {best['generador']['ppa']} (n={best['generador']['acciones']})"
                )
            else:
                st.info("No hay datos suficientes")
        
        with col3:
            if 'bloqueador' in best:
                st.metric(
                    "🧱 Mejor Bloqueador",
                    best['bloqueador']['nombre'],
                    f"PPA: {best['bloqueador']['ppa']} (n={best['bloqueador']['acciones']})"
                )
            else:
                st.info("No hay datos suficientes")
    
    # ==========================================================================
    # TAB: INDIVIDUAL
    # ==========================================================================
    with tab_individual:
        st.subheader("👤 Análisis Individual de Jugadores")
        
        # Mapa de eficiencia
        st.markdown("### 🗺️ Mapa de Eficiencia de Jugadores")
        st.caption("PPA (Puntos por Acción) vs Volumen de Acciones")
        
        player_map = create_player_efficiency_map(df, min_n)
        if player_map:
            st.plotly_chart(player_map, use_container_width=True)
        else:
            st.warning("No hay suficientes datos con la muestra mínima seleccionada")
        
        st.markdown("---")
        
        # Tabla detallada de jugadores
        st.markdown("### 📋 Tabla Detallada por Jugador")
        player_df = analyze_players(df)
        player_filtered = player_df[player_df['Acciones'] >= min_n]
        
        if len(player_filtered) > 0:
            st.dataframe(
                style_dataframe(player_filtered, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.warning("No hay datos con la muestra mínima seleccionada")
        
        st.markdown("---")
        
        # Gráfico de tipo de tiro
        st.markdown("### 🎯 Volumen vs Eficiencia por Tipo de Tiro")
        shot_chart = create_shot_type_chart(df)
        if shot_chart:
            st.plotly_chart(shot_chart, use_container_width=True)
        else:
            st.warning("No hay columna 'TIPO_TIRO' en los datos")
        
        st.markdown("---")
        
        # Eficiencia por jugador y concepto
        st.markdown("### 📊 Eficiencia por Jugador y Concepto")
        
        # Selector de jugador
        players_list = df['FINALIZADOR'].dropna().unique().tolist()
        selected_player = st.selectbox("Selecciona jugador:", sorted(players_list))
        
        player_concept_df = analyze_player_by_concept(df, min_n)
        player_concept_filtered = player_concept_df[player_concept_df['Jugador'] == selected_player]
        
        if len(player_concept_filtered) > 0:
            st.dataframe(
                style_dataframe(player_concept_filtered, ['PPA', '%TO']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No hay datos para este jugador")
    
    # ==========================================================================
    # TAB: COMBINACIONES (Dúos y Tríos)
    # ==========================================================================
    with tab_combinaciones:
        st.subheader("👥 Análisis de Combinaciones")
        
        # Dúos
        st.markdown("### 👫 Eficiencia de Dúos")
        duos_df = analyze_duos(df, min_n)
        duos_filtered = duos_df[duos_df['Posesiones'] >= min_n]
        
        if len(duos_filtered) > 0:
            st.dataframe(
                style_dataframe(duos_filtered.head(20), ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No hay dúos con la muestra mínima seleccionada")
        
        st.markdown("---")
        
        # Tríos
        st.markdown("### 👨‍👩‍👦 Eficiencia de Tríos")
        trios_df = analyze_trios(df, min_n)
        trios_filtered = trios_df[trios_df['Posesiones'] >= min_n]
        
        if len(trios_filtered) > 0:
            st.dataframe(
                style_dataframe(trios_filtered.head(20), ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No hay tríos con la muestra mínima seleccionada")
    
    # ==========================================================================
    # TAB: SISTEMAS
    # ==========================================================================
    with tab_sistemas:
        st.subheader("📋 Rendimiento por Sistema (Categoría)")
        
        set_df = analyze_sets_performance(df)
        set_filtered = set_df[set_df['Posesiones'] >= min_n]
        
        if len(set_filtered) > 0:
            st.dataframe(
                style_dataframe(set_filtered, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.warning("No hay datos con la muestra mínima seleccionada")
        
        st.markdown("---")
        
        # Eficacia del concepto según la defensa
        st.markdown("### 🛡️ Eficacia del Concepto según la Defensa")
        
        concept_defense_df = analyze_concept_by_defense(df, min_n)
        
        if concept_defense_df is not None:
            # Selector de concepto
            concepts_list = concept_defense_df['Concepto'].unique().tolist()
            selected_concept = st.selectbox("Selecciona concepto/sistema:", sorted(concepts_list))
            
            concept_filtered = concept_defense_df[concept_defense_df['Concepto'] == selected_concept]
            concept_filtered = concept_filtered[concept_filtered['Posesiones'] >= min_n]
            
            if len(concept_filtered) > 0:
                st.dataframe(
                    style_dataframe(concept_filtered, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No hay datos suficientes para este concepto")
        else:
            st.warning("No hay columna 'DEFENSA' en los datos")
    
    # ==========================================================================
    # TAB: BLOQUEOS
    # ==========================================================================
    with tab_bloqueos:
        st.subheader("🧱 Análisis de Bloqueos")
        
        # Tipo de bloqueo
        st.markdown("### Tipo de Bloqueo")
        screen_df = analyze_with_stats(df, 'BLOQUEO', 'Tipo_Bloqueo')
        st.dataframe(
            style_dataframe(screen_df, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
            use_container_width=True, 
            hide_index=True
        )
        
        st.markdown("---")
        
        # Bloqueadores
        st.markdown("### Bloqueadores")
        screener_df = analyze_with_stats(df, 'BLOQUEADOR', 'Bloqueador')
        screener_filtered = screener_df[(screener_df['Posesiones'] >= min_n) & (screener_df['Bloqueador'] != 'NADA')]
        
        if len(screener_filtered) > 0:
            st.dataframe(
                style_dataframe(screener_filtered, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.warning("No hay datos con la muestra mínima seleccionada")
        
        st.markdown("---")
        
        # Generadores
        st.markdown("### Generadores")
        df_gen = df[df['GENERADOR'].notna()]
        if len(df_gen) > 0:
            gen_df = analyze_with_stats(df_gen, 'GENERADOR', 'Generador')
            gen_filtered = gen_df[gen_df['Posesiones'] >= min_n]
            if len(gen_filtered) > 0:
                st.dataframe(
                    style_dataframe(gen_filtered, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                    use_container_width=True, 
                    hide_index=True
                )
        
        st.markdown("---")
        
        # Tiempo
        st.markdown("### ⏱️ Por Tiempo de Posesión")
        time_df = analyze_with_stats(df, 'TIME_BUCKET', 'Tiempo')
        st.dataframe(
            style_dataframe(time_df, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
            use_container_width=True, 
            hide_index=True
        )
    
    # ==========================================================================
    # TAB: QUINTETOS
    # ==========================================================================
    with tab_quintetos:
        st.subheader("🏀 Análisis de Quintetos")
        
        lineup_df = analyze_lineups(df)
        lineup_filtered = lineup_df[lineup_df['Posesiones'] >= min_n]
        
        if len(lineup_filtered) > 0:
            st.dataframe(
                style_dataframe(lineup_filtered, ['PPA', 'PPT2', 'PPT3', 'PPT', '%TO']),
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.warning("No hay quintetos con la muestra mínima seleccionada")
    
    # ==========================================================================
    # DOWNLOAD
    # ==========================================================================
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Descargar")
    
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        analyze_players(df).to_excel(writer, sheet_name='Jugadores', index=False)
        analyze_sets_performance(df).to_excel(writer, sheet_name='Sistemas', index=False)
        analyze_duos(df, 1).to_excel(writer, sheet_name='Duos', index=False)
        analyze_trios(df, 1).to_excel(writer, sheet_name='Trios', index=False)
        analyze_with_stats(df, 'BLOQUEO', 'Tipo_Bloqueo').to_excel(writer, sheet_name='Bloqueos', index=False)
        analyze_with_stats(df, 'BLOQUEADOR', 'Bloqueador').to_excel(writer, sheet_name='Bloqueadores', index=False)
        analyze_lineups(df).to_excel(writer, sheet_name='Quintetos', index=False)
        analyze_player_by_concept(df, 1).to_excel(writer, sheet_name='Jugador_Concepto', index=False)
        if analyze_concept_by_defense(df, 1) is not None:
            analyze_concept_by_defense(df, 1).to_excel(writer, sheet_name='Concepto_Defensa', index=False)
    
    st.sidebar.download_button(
        "📥 Descargar Excel",
        output.getvalue(),
        "ACB_Analisis_Completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    # Pantalla de bienvenida
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("https://i0.wp.com/basketinsular.net/wp-content/uploads/2014/10/cb-gran-canaria.jpg?ssl=1", width=100)
    with col_title:
        st.title("🏀 ACB Basketball Efficiency Analysis")
    
    st.info("👈 Selecciona fuente de datos en el sidebar")
    st.markdown("""
    ### 📊 Este dashboard incluye:
    
    **📊 General:**
    - Métricas globales: PPA, PPT2, PPT3, PPT, %TO
    - Mejores performers: Finalizador, Generador, Bloqueador
    
    **👤 Individual:**
    - Mapa de eficiencia de jugadores (PPA vs Volumen)
    - Tabla detallada por jugador
    - Volumen vs Eficiencia por Tipo de Tiro
    - Eficiencia por jugador y concepto
    
    **👥 Combinaciones:**
    - Eficiencia de Dúos
    - Eficiencia de Tríos
    
    **📋 Sistemas:**
    - Rendimiento por categoría/sistema de juego
    - Eficacia del concepto según la defensa
    
    **🧱 Bloqueos:**
    - Análisis por tipo de bloqueo
    - Eficiencia de bloqueadores y generadores
    - Análisis por tiempo de posesión
    
    **🏀 Quintetos:**
    - Eficiencia de combinaciones de 5 jugadores
    
    ### 🎨 Código de colores:
    - 🟢 Verde = Mayor eficiencia
    - 🔴 Rojo = Menor eficiencia
    - Para %TO: 🟢 = Menos pérdidas (mejor)
    """)
