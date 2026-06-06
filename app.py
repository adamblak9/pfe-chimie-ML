"""
============================================================
  ML DASHBOARD — Conductivité GO/rGO
  PFE Master — Ahimi Nisrine — FSTT 2025/2026
  Améliorations : oversampling (SMOGN-like via noise), standardisation,
                  pipeline sklearn, meilleure évaluation
============================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ML GO/rGO — Conductivité",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main { background: #0a0e1a; }

.metric-card {
    background: #111827;
    border: 1px solid #2a3a5c;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.metric-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.blue  { color: #3b82f6; }
.green { color: #10b981; }
.orange{ color: #f59e0b; }
.red   { color: #ef4444; }

.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 0.4rem;
}

.info-box {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    font-size: 0.83rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}
.warn-box {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid #f59e0b;
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    font-size: 0.83rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}
.pred-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1));
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.pred-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #10b981;
}
.pred-label { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; }
.pred-ci    { font-size: 0.85rem; color: #f59e0b; margin-top: 0.5rem; font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_data():
    raw = {
        'ID_IG': [0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25,1.30,
                  1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50,1.55,
                  0.80,0.88,0.92,1.02,1.08,1.18,1.28,1.38,1.48,1.52],
        'C_O':   [2.1,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,
                  10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,
                  2.0,2.3,2.8,3.8,4.5,8.5,11.5,14.3,17.5,20.0],
        'Conductivite': [0.05,0.08,0.12,0.18,0.25,0.40,0.65,1.20,2.50,4.80,
                         25,48,95,180,320,580,950,1500,2800,4500,
                         0.03,0.06,0.10,0.22,0.35,1.80,62,420,2100,5200],
        'Materiau': ['GO','GO','GO','GO','GO','GO','GO','GO','GO','GO',
                     'rGO','rGO','rGO','rGO','rGO','rGO','rGO','rGO','rGO','rGO',
                     'GO','GO','GO','GO','GO','GO','rGO','rGO','rGO','rGO'],
        'Reference': [
            'Dreyer et al., 2010','Lerf et al., 1998','Marcano et al., 2010','Hummers et al., 1958',
            'Kudin et al., 2008','Ferrari et al., 2006','Dreyer et al., 2010','Marcano et al., 2010',
            'Kudin et al., 2008','Ferrari et al., 2006',
            'Stankovich et al., 2007','Chua & Pumera, 2014','Stankovich et al., 2007','Bae et al., 2010',
            'Reina et al., 2009','Stankovich et al., 2007','Chua & Pumera, 2014','Bae et al., 2010',
            'Reina et al., 2009','Novoselov et al., 2004',
            'Lerf et al., 1998','Hummers et al., 1958','Marcano et al., 2010','Dreyer et al., 2010',
            'Kudin et al., 2008','Ferrari et al., 2006','Stankovich et al., 2007','Chua & Pumera, 2014',
            'Bae et al., 2010','Novoselov et al., 2004',
        ]
    }
    return pd.DataFrame(raw)

# ============================================================
# OVERSAMPLING (Gaussian noise on log-space — SMOGN-like)
# ============================================================
@st.cache_data
def oversample(df, n_synthetic=40, noise_std=0.04, seed=42):
    """
    Régression oversampling : génère n_synthetic points synthétiques
    en ajoutant un bruit gaussien sur chaque point existant (dans l'espace log).
    Equivalent simplifié de SMOGN pour la régression.
    """
    rng = np.random.default_rng(seed)
    rows = []
    log_cond = np.log10(df['Conductivite'].values)

    for _ in range(n_synthetic):
        idx = rng.integers(0, len(df))
        row = df.iloc[idx].copy()
        # Bruit gaussien dans l'espace features et log-conductivité
        new_idiq = np.clip(row['ID_IG'] + rng.normal(0, noise_std), 0.80, 1.60)
        new_co   = np.clip(row['C_O']   + rng.normal(0, 0.3),       2.0,  20.0)
        new_logc = log_cond[idx]        + rng.normal(0, noise_std * 2)
        rows.append({
            'ID_IG': new_idiq,
            'C_O':   new_co,
            'Conductivite': 10 ** new_logc,
            'Materiau': row['Materiau'],
            'Reference': 'Synthétique (oversampling)',
        })
    synthetic = pd.DataFrame(rows)
    augmented = pd.concat([df, synthetic], ignore_index=True)
    return augmented, synthetic

# ============================================================
# PIPELINE BUILDER
# ============================================================
def build_pipeline(model_name):
    models = {
        'Random Forest':        RandomForestRegressor(n_estimators=200, random_state=42, max_features='sqrt'),
        'Gradient Boosting':    GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, random_state=42),
        'Ridge Regression':     Ridge(alpha=1.0),
        'Régression Linéaire':  LinearRegression(),
    }
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  models[model_name]),
    ])

# ============================================================
# TRAIN + EVALUATE
# ============================================================
@st.cache_data
def train_evaluate(use_oversampling, model_name, n_synthetic, noise_std):
    df_orig = load_data()

    if use_oversampling:
        df_aug, df_synth = oversample(df_orig, n_synthetic=n_synthetic, noise_std=noise_std)
        df_train_pool = df_aug
    else:
        df_train_pool = df_orig
        df_synth = pd.DataFrame()

    X = df_train_pool[['ID_IG', 'C_O']].values
    y = np.log10(df_train_pool['Conductivite'].values)

    # Always test on ORIGINAL data only (no synthetic in test set)
    X_orig = df_orig[['ID_IG', 'C_O']].values
    y_orig = np.log10(df_orig['Conductivite'].values)
    X_train, X_test, y_train, y_test = train_test_split(
        X_orig, y_orig, test_size=0.25, random_state=42
    )

    # If oversampling: add synthetic to train set only
    if use_oversampling and len(df_synth) > 0:
        X_synth = df_synth[['ID_IG', 'C_O']].values
        y_synth = np.log10(df_synth['Conductivite'].values)
        X_train = np.vstack([X_train, X_synth])
        y_train = np.concatenate([y_train, y_synth])

    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)

    y_pred_test = pipe.predict(X_test)
    y_pred_all  = pipe.predict(X_orig)

    r2_test  = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test= np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Cross-val on original data
    cv   = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe_cv = build_pipeline(model_name)
    cv_scores = cross_val_score(pipe_cv, X_orig, y_orig, cv=cv, scoring='r2')

    # Prediction for our sample
    notre = np.array([[1.11, 14.3]])
    log_pred = pipe.predict(notre)[0]
    pred_cond = 10 ** log_pred

    # Confidence interval from tree variance (RF/GB only) or ±1 MAE
    if hasattr(pipe['model'], 'estimators_'):
        tree_preds = np.array([
            tree.predict(pipe['scaler'].transform(notre))[0]
            for tree in pipe['model'].estimators_
        ]) if hasattr(pipe['model'], 'estimators_') else np.array([log_pred])
        std_log = tree_preds.std() if len(tree_preds) > 1 else mae_test
    else:
        std_log = mae_test

    ci_low  = 10 ** (log_pred - std_log)
    ci_high = 10 ** (log_pred + std_log)

    # Feature importance
    try:
        imp = pipe['model'].feature_importances_
    except AttributeError:
        coef = np.abs(pipe['model'].coef_)
        imp  = coef / coef.sum()

    return {
        'pipe': pipe,
        'X_orig': X_orig, 'y_orig': y_orig,
        'X_test': X_test, 'y_test': y_test,
        'y_pred_test': y_pred_test,
        'y_pred_all': y_pred_all,
        'r2_test': r2_test, 'mae_test': mae_test, 'rmse_test': rmse_test,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        'pred_cond': pred_cond,
        'ci_low': ci_low, 'ci_high': ci_high,
        'importances': imp,
        'df_synth': df_synth,
        'n_train': len(X_train),
    }

# ============================================================
# MATPLOTLIB THEME
# ============================================================
plt.rcParams.update({
    'figure.facecolor':  '#111827',
    'axes.facecolor':    '#111827',
    'axes.edgecolor':    '#2a3a5c',
    'axes.labelcolor':   '#e2e8f0',
    'xtick.color':       '#94a3b8',
    'ytick.color':       '#94a3b8',
    'text.color':        '#e2e8f0',
    'grid.color':        '#1e2d4a',
    'grid.linewidth':    0.8,
    'legend.facecolor':  '#1a2236',
    'legend.edgecolor':  '#2a3a5c',
    'font.family':       'monospace',
})

COLOR_GO    = '#3b82f6'
COLOR_RGO   = '#ef4444'
COLOR_SYNTH = '#6b7280'
COLOR_NOTRE = '#f59e0b'
COLOR_GREEN = '#10b981'
COLOR_LINE  = '#10b981'

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🔬 Modèle")
    model_name = st.selectbox(
        "Algorithme",
        ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Régression Linéaire'],
        index=0
    )

    st.markdown("### 🔄 Data Augmentation")
    use_oversampling = st.toggle("Oversampling (Gaussian noise)", value=True)

    if use_oversampling:
        n_synthetic = st.slider("Points synthétiques", 20, 100, 40, 10)
        noise_std   = st.slider("Niveau de bruit (σ)", 0.01, 0.10, 0.04, 0.01)
    else:
        n_synthetic, noise_std = 40, 0.04

    st.markdown("### ⚡ Notre Échantillon")
    idiq_val = st.slider("I(D)/I(G)", 0.80, 1.60, 1.11, 0.01)
    co_val   = st.slider("C/O",       2.0,  20.0, 14.3, 0.1)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#4b5563;'>PFE Master — FSTT 2025/2026<br/>Materials Informatics</div>",
        unsafe_allow_html=True
    )

# ============================================================
# RUN MODEL
# ============================================================
res = train_evaluate(use_oversampling, model_name, n_synthetic, noise_std)

# Custom prediction for sidebar sliders
notre_custom = np.array([[idiq_val, co_val]])
log_custom   = res['pipe'].predict(notre_custom)[0]
pred_custom  = 10 ** log_custom

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style='background:#111827;border:1px solid #2a3a5c;border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;'>
  <div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#3b82f6,#f59e0b,#ef4444);'></div>
  <h1 style='font-size:1.4rem;font-weight:700;color:#fff;margin:0 0 0.3rem;'>
    ⚡ ML — Prédiction de la Conductivité GO/rGO
  </h1>
  <p style='color:#94a3b8;font-size:0.88rem;margin:0;'>
    Matériaux à base de graphène oxydé — Materials Informatics | Standardisation + Oversampling + Pipelines sklearn
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP METRICS
# ============================================================
df_orig = load_data()
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-val blue">30</div>
    <div class="metric-label">Points originaux</div>
    </div>""", unsafe_allow_html=True)

with c2:
    extra = n_synthetic if use_oversampling else 0
    st.markdown(f"""<div class="metric-card">
    <div class="metric-val orange">{res['n_train']}</div>
    <div class="metric-label">Points entraînement (+{extra} synth.)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    color = "green" if res['r2_test'] > 0.95 else "orange"
    st.markdown(f"""<div class="metric-card">
    <div class="metric-val {color}">{res['r2_test']:.3f}</div>
    <div class="metric-label">R² test (8 pts)</div>
    </div>""", unsafe_allow_html=True)

with c4:
    cv_color = "green" if res['cv_mean'] > 0.80 else ("orange" if res['cv_mean'] > 0.60 else "red")
    st.markdown(f"""<div class="metric-card">
    <div class="metric-val {cv_color}">{res['cv_mean']:.3f}</div>
    <div class="metric-label">CV R² 5-fold (±{res['cv_std']:.2f})</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-val green">{res['pred_cond']:.0f} S/m</div>
    <div class="metric-label">Conductivité GO/Al/GO</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# PREDICTION PANEL
# ============================================================
col_pred, col_info = st.columns([1, 2])

with col_pred:
    st.markdown('<div class="section-header">🔮 Prédiction interactive</div>', unsafe_allow_html=True)

    mat_class = "rGO (modérément réduit)" if pred_custom > 50 else ("GO (modérément oxydé)" if pred_custom > 1 else "GO (fortement oxydé)")
    mat_color = "#ef4444" if "rGO" in mat_class else "#3b82f6"

    ci_lo_c = 10 ** (log_custom - res['mae_test'])
    ci_hi_c = 10 ** (log_custom + res['mae_test'])

    st.markdown(f"""
    <div class="pred-box">
      <div class="pred-val">{pred_custom:.1f} S/m</div>
      <div class="pred-label">Conductivité prédite — {model_name}</div>
      <div class="pred-ci">IC ±1 MAE : [{ci_lo_c:.1f} – {ci_hi_c:.1f}] S/m</div>
      <div style="margin-top:0.8rem;">
        <span style="background:rgba(0,0,0,0.3);color:{mat_color};border:1px solid {mat_color};
                     padding:0.2rem 0.8rem;border-radius:20px;font-size:0.78rem;font-family:monospace;">
          {mat_class}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Notre échantillon fixe
    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.8rem 1rem;background:#1a2236;border-radius:8px;border-left:3px solid #f59e0b;font-size:0.82rem;color:#94a3b8;">
      <strong style="color:#f59e0b;">★ GO/Al/GO (I(D)/I(G)=1,11 · C/O=14,3)</strong><br/>
      RF : <strong style="color:#10b981;">{res['pred_cond']:.0f} S/m</strong>&nbsp;&nbsp;
      IC : [{res['ci_low']:.0f} – {res['ci_high']:.0f}] S/m
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="section-header">📋 Pipeline de traitement</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
      <strong style="color:#e2e8f0;">StandardScaler</strong> — Centrage et réduction (µ=0, σ=1) appliqués avant chaque modèle via <code>sklearn.Pipeline</code>.
      Garantit que I(D)/I(G) et C/O contribuent équitablement indépendamment de leur échelle.
    </div>
    """, unsafe_allow_html=True)

    if use_oversampling:
        st.markdown(f"""
        <div class="info-box">
          <strong style="color:#e2e8f0;">Oversampling Gaussien (SMOGN-like)</strong> — {n_synthetic} points synthétiques générés
          par ajout de bruit gaussien (σ={noise_std}) sur les features et σ={noise_std*2:.2f} sur log(conductivité).
          Les points synthétiques restent dans le jeu d'<em>entraînement uniquement</em> — jamais dans le test.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warn-box">
          Oversampling désactivé — le modèle s'entraîne uniquement sur les 22 points originaux.
        </div>
        """, unsafe_allow_html=True)

    # CV bar chart inline
    fig_cv, ax_cv = plt.subplots(figsize=(5, 1.8))
    fig_cv.patch.set_facecolor('#1a2236')
    ax_cv.set_facecolor('#1a2236')
    folds = [f"Fold {i+1}" for i in range(5)]
    colors_cv = [COLOR_GREEN if s > 0.8 else ('#f59e0b' if s > 0.6 else '#ef4444') for s in res['cv_scores']]
    ax_cv.bar(folds, res['cv_scores'], color=colors_cv, edgecolor='none', width=0.5)
    ax_cv.axhline(res['cv_mean'], color='#f59e0b', linewidth=1.5, linestyle='--', label=f"Moy. {res['cv_mean']:.3f}")
    ax_cv.set_ylim(-0.1, 1.1)
    ax_cv.set_ylabel("R²", fontsize=9)
    ax_cv.set_title(f"Validation croisée 5-fold — {model_name}", fontsize=9, pad=4)
    ax_cv.legend(fontsize=8, loc='lower right')
    ax_cv.grid(True, axis='y', alpha=0.3)
    ax_cv.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig_cv, use_container_width=True)
    plt.close()

st.markdown("---")

# ============================================================
# MAIN CHARTS — Row 1
# ============================================================
st.markdown('<div class="section-header">📊 Analyse des données & Performance du modèle</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#111827')

# Helper
def scatter_mat(ax, df_, xcol, ycol, show_synth=False):
    for mat, col, lbl in [('GO', COLOR_GO, 'GO (littérature)'), ('rGO', COLOR_RGO, 'rGO (littérature)')]:
        m = df_orig['Materiau'] == mat
        ax.scatter(df_orig[m][xcol], df_orig[m][ycol], color=col, label=lbl, s=70, zorder=3, alpha=0.9)
    if show_synth and use_oversampling and len(res['df_synth']) > 0:
        ax.scatter(res['df_synth'][xcol], res['df_synth'][ycol],
                   color=COLOR_SYNTH, label='Synthétique', s=30, zorder=2, alpha=0.5, marker='+')
    # Notre échantillon fixe
    ax.scatter(1.11 if xcol == 'ID_IG' else 14.3,
               res['pred_cond'],
               color=COLOR_NOTRE, s=200, marker='*', zorder=5, label='GO/Al/GO (prédit)')

# --- Chart 1 : ID/IG vs Conductivité ---
ax1 = axes[0]
scatter_mat(ax1, df_orig, 'ID_IG', 'Conductivite', show_synth=True)
ax1.set_yscale('log')
ax1.set_xlabel('I(D)/I(G)', fontsize=10)
ax1.set_ylabel('Conductivité (S/m)', fontsize=10)
ax1.set_title('I(D)/I(G) vs Conductivité', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=9)

# --- Chart 2 : C/O vs Conductivité ---
ax2 = axes[1]
scatter_mat(ax2, df_orig, 'C_O', 'Conductivite', show_synth=True)
ax2.set_yscale('log')
ax2.set_xlabel('C/O', fontsize=10)
ax2.set_ylabel('Conductivité (S/m)', fontsize=10)
ax2.set_title('C/O vs Conductivité', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=9)

# --- Chart 3 : Predicted vs Real ---
ax3 = axes[2]
y_all  = res['y_orig']
yp_all = res['y_pred_all']
ax3.scatter(y_all, yp_all, color=COLOR_GREEN, s=70, zorder=3, alpha=0.85, label='Données originales')
# Test points highlighted
ax3.scatter(res['y_test'], res['y_pred_test'], color=COLOR_NOTRE, s=100, zorder=4,
            edgecolors='white', linewidth=0.8, label='Test set')
lim = [min(y_all.min(), yp_all.min()) - 0.3, max(y_all.max(), yp_all.max()) + 0.3]
ax3.plot(lim, lim, color='#4b5563', linewidth=1.5, linestyle='--', label='Idéal')
ax3.set_xlim(lim); ax3.set_ylim(lim)
ax3.set_xlabel('log₁₀(Conductivité réelle)', fontsize=10)
ax3.set_ylabel('log₁₀(Conductivité prédite)', fontsize=10)
ax3.set_title(f'{model_name} — R²={res["r2_test"]:.3f}', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=9)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

# ============================================================
# MAIN CHARTS — Row 2
# ============================================================
col_imp, col_res, col_comp = st.columns(3)

with col_imp:
    st.markdown('<div class="section-header">🎯 Importance des variables</div>', unsafe_allow_html=True)
    fig2, ax = plt.subplots(figsize=(4, 3))
    fig2.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    feats = ['I(D)/I(G)', 'C/O']
    colors_imp = [COLOR_GO, COLOR_RGO]
    bars = ax.bar(feats, res['importances'], color=colors_imp, edgecolor='none', width=0.5)
    for bar, imp in zip(bars, res['importances']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{imp:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#e2e8f0')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Importance relative", fontsize=9)
    ax.set_title(model_name, fontsize=9, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

with col_res:
    st.markdown('<div class="section-header">📉 Résidus (log-space)</div>', unsafe_allow_html=True)
    fig3, ax = plt.subplots(figsize=(4, 3))
    fig3.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    residuals = res['y_orig'] - res['y_pred_all']
    ax.scatter(res['y_pred_all'], residuals, color=COLOR_GREEN, s=60, alpha=0.8)
    ax.axhline(0, color='#f59e0b', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Valeur prédite (log)", fontsize=9)
    ax.set_ylabel("Résidu (log)", fontsize=9)
    ax.set_title("Analyse des résidus", fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

with col_comp:
    st.markdown('<div class="section-header">📊 Comparaison modèles (CV R²)</div>', unsafe_allow_html=True)

    # Quick CV comparison (cached for speed)
    @st.cache_data
    def compare_models(use_os, n_s, ns):
        df_o = load_data()
        X_o = df_o[['ID_IG', 'C_O']].values
        y_o = np.log10(df_o['Conductivite'].values)

        if use_os:
            df_a, _ = oversample(df_o, n_s, ns)
            Xa = df_a[['ID_IG', 'C_O']].values
            ya = np.log10(df_a['Conductivite'].values)
        else:
            Xa, ya = X_o, y_o

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        for nm in ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Régression Linéaire']:
            p = build_pipeline(nm)
            sc = cross_val_score(p, X_o, y_o, cv=cv, scoring='r2')
            results[nm] = sc.mean()
        return results

    comp = compare_models(use_oversampling, n_synthetic, noise_std)
    fig4, ax = plt.subplots(figsize=(4, 3))
    fig4.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    names_short = ['RF', 'GB', 'Ridge', 'LR']
    vals = list(comp.values())
    cols_bar = [COLOR_GREEN if v > 0.80 else ('#f59e0b' if v > 0.60 else '#ef4444') for v in vals]
    brs = ax.bar(names_short, vals, color=cols_bar, width=0.5, edgecolor='none')
    for b, v in zip(brs, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold', color='#e2e8f0')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("CV R² moyen", fontsize=9)
    ax.set_title("Comparaison 4 modèles", fontsize=9, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close()

# ============================================================
# DATA TABLE
# ============================================================
st.markdown("---")
st.markdown('<div class="section-header">📚 Base de données bibliographiques</div>', unsafe_allow_html=True)

df_display = df_orig.copy()
df_display['log₁₀(σ)'] = np.log10(df_display['Conductivite']).round(3)
df_display.index = range(1, len(df_display) + 1)

# Highlight our sample row if present
def style_row(row):
    if abs(row['ID_IG'] - 1.11) < 0.001 and abs(row['C_O'] - 14.3) < 0.001:
        return ['background-color: rgba(245,158,11,0.12)'] * len(row)
    return [''] * len(row)

styled = (
    df_display[['Materiau', 'ID_IG', 'C_O', 'Conductivite', 'log₁₀(σ)', 'Reference']]
    .style
    .apply(style_row, axis=1)
    .format({'ID_IG': '{:.2f}', 'C_O': '{:.1f}', 'Conductivite': '{:.2f}', 'log₁₀(σ)': '{:.3f}'})
)
st.dataframe(styled, use_container_width=True, height=400)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown(f"""
    <div style="font-size:0.78rem;color:#4b5563;font-family:monospace;">
    <strong style="color:#64748b;">Modèle actif</strong><br/>
    {model_name}<br/>
    Standardisation : ✅ StandardScaler<br/>
    Oversampling : {'✅ Gaussien ×' + str(n_synthetic) if use_oversampling else '❌ désactivé'}
    </div>""", unsafe_allow_html=True)
with col_f2:
    st.markdown(f"""
    <div style="font-size:0.78rem;color:#4b5563;font-family:monospace;">
    <strong style="color:#64748b;">Métriques</strong><br/>
    R² test = {res['r2_test']:.3f}<br/>
    MAE = {res['mae_test']:.3f} (log)<br/>
    CV R² = {res['cv_mean']:.3f} ± {res['cv_std']:.3f}
    </div>""", unsafe_allow_html=True)
with col_f3:
    st.markdown(f"""
    <div style="font-size:0.78rem;color:#4b5563;font-family:monospace;">
    <strong style="color:#64748b;">Notre échantillon</strong><br/>
    GO/Al/GO | I(D)/I(G)=1,11 | C/O=14,3<br/>
    σ prédite = {res['pred_cond']:.0f} S/m<br/>
    IC : [{res['ci_low']:.0f} – {res['ci_high']:.0f}] S/m
    </div>""", unsafe_allow_html=True)