import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import time

st.set_page_config(
    page_title="UC4 - Content Clustering",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] label {
    color: #8b949e !important; font-size: 0.75rem !important;
    font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #0d1117; border: 1px solid #30363d; color: #e6edf3;
}
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: #0d1117; border: 1px solid #30363d; color: #e6edf3;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
}

h1 { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; color: #e6edf3 !important; letter-spacing: -0.02em !important; }
h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important; color: #c9d1d9 !important; }

[data-testid="stFileUploader"] {
    background-color: #161b22; border: 1px dashed #30363d; border-radius: 6px; padding: 12px;
}
[data-testid="stFileUploader"] label {
    color: #8b949e !important; font-size: 0.75rem !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important;
}

.stButton > button {
    background-color: #e08e45 !important; color: #0d1117 !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
}
.stButton > button:hover { background-color: #f0a060 !important; }
.stDownloadButton > button {
    background-color: #238636 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 500 !important;
}

[data-testid="stMetric"] {
    background-color: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem !important; font-weight: 600 !important; }

[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }

.stAlert { border-radius: 6px !important; border-left-width: 3px !important; }
[data-testid="stInfo"] { background-color: #1f1408 !important; border-left-color: #e08e45 !important; color: #f0c070 !important; }
[data-testid="stSuccess"] { background-color: #0a2217 !important; border-left-color: #238636 !important; color: #3fb950 !important; }
[data-testid="stWarning"] { background-color: #231c00 !important; border-left-color: #e3b341 !important; color: #e3b341 !important; }
[data-testid="stError"] { background-color: #280d11 !important; border-left-color: #da3633 !important; color: #f85149 !important; }

hr { border-color: #21262d; }

.streamlit-expanderHeader {
    background-color: #161b22 !important; border: 1px solid #21262d !important;
    border-radius: 6px !important; color: #c9d1d9 !important; font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background-color: #0d1117 !important; border: 1px solid #21262d !important; border-top: none !important;
}

code {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: #161b22 !important; color: #f0c070 !important;
    padding: 2px 6px !important; border-radius: 3px !important; font-size: 0.82em !important;
}

.section-bar {
    background: linear-gradient(90deg, #e08e4522, transparent);
    border-left: 3px solid #e08e45;
    padding: 8px 16px; border-radius: 0 6px 6px 0;
    margin: 20px 0 12px 0;
    font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #f0c070;
}

.cluster-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.cluster-name {
    font-weight: 600;
    color: #e6edf3;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.cluster-count {
    font-family: 'IBM Plex Mono', monospace;
    color: #e08e45;
    font-size: 0.78rem;
    font-weight: 600;
}
.cluster-sample {
    color: #8b949e;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.outlier-row {
    background: #280d1122;
    border-left: 2px solid #f85149;
    padding: 6px 12px;
    border-radius: 0 4px 4px 0;
    margin: 3px 0;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #c9d1d9;
}

.stProgress > div > div { background-color: #e08e45 !important; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────

def load_file(f):
    if f is None: return None
    try:
        if f.name.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(f)
        return pd.read_csv(f)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def detect_emb_col(df):
    for col in df.columns:
        if col.lower() in ['embedding', 'embeddings']:
            return col
    for col in df.columns:
        if col == 'Address': continue
        sample = df[col].dropna()
        if len(sample) > 0 and isinstance(sample.iloc[0], str) and sample.iloc[0].count(',') > 50:
            return col
    return None

def parse_vec(s):
    try:
        return np.array([float(x) for x in str(s).strip().strip('[]').split(',')], dtype=np.float32)
    except:
        return None

def section(title):
    st.markdown(f'<div class="section-bar">{title}</div>', unsafe_allow_html=True)

# Colour palette for cluster cards (cycling)
CLUSTER_COLORS = [
    "#1f6feb", "#238636", "#9b59b6", "#e08e45", "#da3633",
    "#0d7a6b", "#b36fd1", "#3fb950", "#388bfd", "#e3b341",
    "#79c0ff", "#56d364", "#d2a8ff", "#ffa657", "#ffa198",
]


# ─── SIDEBAR ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    provider = st.selectbox("Embedding Provider", ["Gemini", "OpenAI"])
    api_key  = st.text_input("API Key (for cluster naming)", type="password",
        help="Used to name clusters with plain-English labels. Optional - clusters will still be created without it.")

    st.markdown("---")
    st.markdown("**Clustering**")
    auto_detect = st.checkbox("Auto-detect cluster count", value=False,
        help="Uses silhouette scoring. Slower but more accurate. Otherwise set manually below.")
    n_clusters = st.slider("Number of clusters", 3, 60, 15, 1,
        disabled=auto_detect,
        help="Rule of thumb: total pages ÷ 20. E.g. 400 pages → 20 clusters.")

    st.markdown("---")
    st.markdown("**Outlier Detection**")
    outlier_std = st.slider("Sensitivity (std deviations)", 1.0, 3.5, 2.0, 0.5,
        help="Lower = more outliers flagged. 2.0 is a good default.")

    st.markdown("---")
    st.markdown("**Exclusions**")
    exclude_raw = st.text_area("URL patterns to exclude",
        value="?page=, /page/, ?filter=, /tag/, /author/",
        height=80, help="Comma separated. Paginated and filtered URLs add noise.")

    pages_for_naming = st.slider("Pages per cluster for naming", 3, 20, 8,
        help="How many URLs the AI reads to name each cluster.")


# ─── MAIN ───────────────────────────────────────────────────────────

st.markdown("# Content Clustering & Outlier Detection")
st.markdown("---")

section("01 - Upload Files")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**SF Embeddings Export** `required`")
    st.caption("Screaming Frog → Bulk Export → AI Tab → Export")
    emb_file = st.file_uploader("Drop embeddings file", type=["csv", "xlsx", "xls"], key="emb", label_visibility="collapsed")

with col2:
    st.markdown("**Internal Sheet** `optional`")
    st.caption("Must include Address column. Any of these are auto-detected: Title 1, Clicks, Impressions")
    internal_file = st.file_uploader("Drop internal sheet", type=["csv", "xlsx", "xls"], key="internal", label_visibility="collapsed")

emb_df = load_file(emb_file)
internal_df = load_file(internal_file)
emb_col = None
title_col = None
clicks_col = None
impressions_col = None

if emb_df is not None:
    if 'Address' not in emb_df.columns:
        st.error("No 'Address' column found.")
    else:
        # Apply exclusions
        exclude_list = [p.strip() for p in exclude_raw.split(',') if p.strip()]
        mask = pd.Series([True] * len(emb_df))
        for pat in exclude_list:
            mask = mask & ~emb_df['Address'].str.contains(re.escape(pat), na=False)
        emb_df = emb_df[mask].reset_index(drop=True)

        # Join internal sheet if provided
        if internal_df is not None and 'Address' in internal_df.columns:
            extra_cols = [c for c in internal_df.columns if c != 'Address']
            emb_df = emb_df.merge(internal_df[['Address'] + extra_cols], on='Address', how='left')

        emb_col = detect_emb_col(emb_df)
        title_col = next((c for c in emb_df.columns if 'title' in c.lower() and c != emb_col), None)
        clicks_col = next((c for c in emb_df.columns if c.lower() in ['clicks', 'click', 'gsc clicks']), None)
        impressions_col = next((c for c in emb_df.columns if c.lower() in ['impressions', 'impression', 'gsc impressions']), None)
        position_col = next((c for c in emb_df.columns if c.lower() in ['position', 'avg position', 'average position', 'gsc position']), None)
        inlinks_col = next((c for c in emb_df.columns if c.lower() in ['inlinks', 'unique inlinks', 'inlink count']), None)
        wordcount_col = next((c for c in emb_df.columns if c.lower() in ['word count', 'wordcount', 'words']), None)
        depth_col = next((c for c in emb_df.columns if c.lower() in ['crawl depth', 'depth', 'folder depth']), None)
        indexability_col = next((c for c in emb_df.columns if c.lower() in ['indexability', 'indexable']), None)

        c1, c2, c3 = st.columns(3)
        c1.success(f"✓ Embeddings - {len(emb_df):,} pages")
        c2.success(f"✓ Internal sheet - {len(internal_df):,} rows") if internal_df is not None else c2.info("○ Internal sheet - optional")
        c3.info(f"Embedding col: `{emb_col}`" if emb_col else "❌ No embedding column found")

        detected = []
        if title_col: detected.append(f"Title: `{title_col}`")
        if clicks_col: detected.append(f"Clicks: `{clicks_col}`")
        if impressions_col: detected.append(f"Impressions: `{impressions_col}`")
        if position_col: detected.append(f"Position: `{position_col}`")
        if inlinks_col: detected.append(f"Inlinks: `{inlinks_col}`")
        if wordcount_col: detected.append(f"Word Count: `{wordcount_col}`")
        if depth_col: detected.append(f"Crawl Depth: `{depth_col}`")
        if indexability_col: detected.append(f"Indexability: `{indexability_col}`")
        if detected:
            st.info("Detected columns: " + "  |  ".join(detected))

        if emb_col is None:
            st.error("Could not detect the embedding vector column. Check your export.")



# ── RUN ──────────────────────────────────────────────────────────────
section("02 - Run Clustering")

ready = emb_df is not None and emb_col is not None
run_btn = st.button("▶  Run Clustering", disabled=not ready)

if run_btn:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    import warnings
    warnings.filterwarnings('ignore')

    with st.spinner("Parsing embedding vectors..."):
        emb_df['_vec'] = emb_df[emb_col].apply(parse_vec)
        bad = emb_df['_vec'].isna().sum()
        emb_df = emb_df[emb_df['_vec'].notna()].reset_index(drop=True)
        if bad > 0:
            st.warning(f"Dropped {bad} rows with unparseable embeddings")
        X = np.stack(emb_df['_vec'].values)
        X_norm = normalize(X)

    if auto_detect:
        st.info("Auto-detecting optimal cluster count... this may take 1–2 minutes.")
        prog = st.progress(0)
        best_k, best_score = 5, -1
        search_range = list(range(max(3, min(5, len(emb_df)//20)), min(40, len(emb_df)//5) + 1, 2))
        status = st.empty()
        for idx, k in enumerate(search_range):
            prog.progress((idx + 1) / len(search_range))
            status.caption(f"Testing k={k}...")
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=200)
            labels = km.fit_predict(X_norm)
            score = silhouette_score(X_norm, labels, sample_size=min(1000, len(X_norm)))
            if score > best_score:
                best_score, best_k = score, k
        prog.empty()
        status.empty()
        k_final = best_k
        st.success(f"Auto-detected {k_final} clusters (silhouette score: {best_score:.3f})")
    else:
        k_final = min(n_clusters, len(emb_df) - 1)

    with st.spinner(f"Running KMeans with {k_final} clusters..."):
        km = KMeans(n_clusters=k_final, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_norm)
        emb_df['_cluster'] = labels
        centroids = km.cluster_centers_
        distances = np.array([np.linalg.norm(X_norm[i] - centroids[labels[i]]) for i in range(len(X_norm))])
        emb_df['_distance'] = distances
        mean_d = distances.mean()
        std_d  = distances.std()
        emb_df['_outlier'] = distances > (mean_d + outlier_std * std_d)

    # ── Name clusters via AI (if API key provided)
    cluster_names = {i: f"Cluster {i+1}" for i in range(k_final)}

    if api_key:
        st.info("Naming clusters via AI...")
        prog2 = st.progress(0)
        title_col = next((c for c in emb_df.columns if 'title' in c.lower() and c != emb_col), None)

        def _name_cluster(cluster_id, provider, api_key, pages_for_naming, title_col):
            subset = emb_df[emb_df['_cluster'] == cluster_id].head(pages_for_naming)
            if title_col:
                lines = [f"{r['Address']} | {r[title_col]}" for _, r in subset.iterrows()]
            else:
                lines = subset['Address'].tolist()
            prompt = (
                "Below are page URLs" + (" and titles" if title_col else "") + " from a single topic cluster. "
                "Give it a short, clear 2–5 word label. Return only the label.\n\n" + "\n".join(lines)
            )
            try:
                if provider == "OpenAI":
                    import openai
                    client = openai.OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=20
                    )
                    return resp.choices[0].message.content.strip()
                else:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    return model.generate_content(prompt).text.strip()
            except:
                return f"Cluster {cluster_id+1}"

        for idx, cid in enumerate(sorted(emb_df['_cluster'].unique())):
            prog2.progress((idx + 1) / k_final)
            cluster_names[cid] = _name_cluster(cid, provider, api_key, pages_for_naming, title_col)
            time.sleep(0.2)
        prog2.empty()

    emb_df['_cluster_name'] = emb_df['_cluster'].map(cluster_names)

    # Build output df
    base_cols = ['Address', '_cluster', '_cluster_name', '_distance', '_outlier']
    if title_col:
        base_cols = ['Address', title_col, '_cluster', '_cluster_name', '_distance', '_outlier']

    enrich_map = [
        (clicks_col,       'Clicks'),
        (impressions_col,  'Impressions'),
        (position_col,     'Avg Position'),
        (inlinks_col,      'Inlinks'),
        (wordcount_col,    'Word Count'),
        (depth_col,        'Crawl Depth'),
        (indexability_col, 'Indexability'),
    ]
    active_enrich = [(src, dst) for src, dst in enrich_map if src and src in emb_df.columns]

    out_cols = base_cols + [src for src, _ in active_enrich]
    out_df = emb_df[out_cols].copy()

    rename_map = (['Address'] + ([title_col] if title_col else []) +
                  ['Cluster ID', 'Cluster Name', 'Centroid Distance', 'Is Outlier'] +
                  [dst for _, dst in active_enrich])
    out_df.columns = rename_map
    out_df['Centroid Distance'] = out_df['Centroid Distance'].round(4)

    int_cols = ['Clicks', 'Impressions', 'Inlinks', 'Word Count', 'Crawl Depth']
    float_cols = ['Avg Position']
    for col in int_cols:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors='coerce').fillna(0).astype(int)
    for col in float_cols:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors='coerce').round(1)

    st.session_state['uc4_results'] = out_df
    st.session_state['uc4_cluster_names'] = cluster_names
    st.session_state['uc4_k'] = k_final
    st.session_state['uc4_has_clicks'] = 'Clicks' in out_df.columns
    st.session_state['uc4_has_impressions'] = 'Impressions' in out_df.columns
    st.session_state['uc4_has_position'] = 'Avg Position' in out_df.columns
    st.session_state['uc4_has_inlinks'] = 'Inlinks' in out_df.columns
    st.session_state['uc4_has_wordcount'] = 'Word Count' in out_df.columns
    st.session_state['uc4_has_depth'] = 'Crawl Depth' in out_df.columns
    st.session_state['uc4_has_indexability'] = 'Indexability' in out_df.columns


# ── RESULTS ──────────────────────────────────────────────────────────
if 'uc4_results' in st.session_state:
    out_df = st.session_state['uc4_results']
    cluster_names = st.session_state['uc4_cluster_names']
    k_final = st.session_state['uc4_k']
    has_clicks       = st.session_state.get('uc4_has_clicks', False)
    has_impressions  = st.session_state.get('uc4_has_impressions', False)
    has_position     = st.session_state.get('uc4_has_position', False)
    has_inlinks      = st.session_state.get('uc4_has_inlinks', False)
    has_wordcount    = st.session_state.get('uc4_has_wordcount', False)
    has_depth        = st.session_state.get('uc4_has_depth', False)
    has_indexability = st.session_state.get('uc4_has_indexability', False)

    section("03 - Results")

    total = len(out_df)
    outliers = out_df['Is Outlier'].sum()
    avg_size = total / k_final if k_final > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Pages", f"{total:,}")
    m2.metric("Clusters", f"{k_final:,}")
    m3.metric("Outliers Flagged", f"{int(outliers):,}")
    m4.metric("Avg Cluster Size", f"{avg_size:.0f}")

    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["Cluster Overview", "All Pages", "🚨 Outliers"])

    with tab1:
        agg_dict = {'Pages': ('Address', 'count'), 'Avg_Distance': ('Centroid Distance', 'mean')}
        if has_clicks:       agg_dict['Total_Clicks']      = ('Clicks', 'sum')
        if has_impressions:  agg_dict['Total_Impressions'] = ('Impressions', 'sum')
        if has_position:     agg_dict['Avg_Position']      = ('Avg Position', 'mean')
        if has_inlinks:      agg_dict['Avg_Inlinks']       = ('Inlinks', 'mean')
        if has_wordcount:    agg_dict['Avg_WordCount']      = ('Word Count', 'mean')
        if has_depth:        agg_dict['Avg_Depth']          = ('Crawl Depth', 'mean')
        if has_indexability:
            out_df['_indexed'] = out_df['Indexability'].str.lower().eq('indexable').astype(int)
            agg_dict['Pct_Indexed'] = ('_indexed', 'mean')

        cluster_summary = (out_df.groupby(['Cluster ID', 'Cluster Name'])
                           .agg(**agg_dict)
                           .reset_index()
                           .sort_values('Pages', ascending=False))

        cols_per_row = 3
        rows = [cluster_summary.iloc[i:i+cols_per_row] for i in range(0, len(cluster_summary), cols_per_row)]
        for row in rows:
            rcols = st.columns(cols_per_row)
            for col_idx, (_, r) in enumerate(row.iterrows()):
                color = CLUSTER_COLORS[int(r['Cluster ID']) % len(CLUSTER_COLORS)]
                sample_urls = out_df[out_df['Cluster ID'] == r['Cluster ID']]['Address'].head(1).tolist()
                sample = sample_urls[0] if sample_urls else ''

                gsc_parts = []
                if has_clicks:      gsc_parts.append(f"{int(r.get('Total_Clicks', 0)):,} clicks")
                if has_impressions: gsc_parts.append(f"{int(r.get('Total_Impressions', 0)):,} impr")
                if has_position:    gsc_parts.append(f"pos {r.get('Avg_Position', 0):.1f}")
                gsc_line = (f'<div class="cluster-count" style="color:#58a6ff">{" &nbsp;·&nbsp; ".join(gsc_parts)}</div>'
                            if gsc_parts else '')

                sf_parts = []
                if has_inlinks:   sf_parts.append(f"avg {r.get('Avg_Inlinks', 0):.0f} inlinks")
                if has_wordcount: sf_parts.append(f"avg {r.get('Avg_WordCount', 0):.0f} words")
                if has_depth:     sf_parts.append(f"depth {r.get('Avg_Depth', 0):.1f}")
                if has_indexability:
                    pct = int(r.get('Pct_Indexed', 0) * 100)
                    sf_parts.append(f"{pct}% indexed")
                sf_line = (f'<div class="cluster-count" style="color:#3fb950">{" &nbsp;·&nbsp; ".join(sf_parts)}</div>'
                           if sf_parts else '')

                with rcols[col_idx]:
                    st.markdown(f"""
<div class="cluster-card" style="border-top: 3px solid {color}">
  <div class="cluster-name">{r['Cluster Name']}</div>
  <div class="cluster-count">{int(r['Pages'])} pages &nbsp;·&nbsp; avg dist {r['Avg_Distance']:.3f}</div>
  {gsc_line}
  {sf_line}
  <div class="cluster-sample">{sample}</div>
</div>""", unsafe_allow_html=True)

    with tab2:
        st.dataframe(
            out_df.sort_values(['Cluster ID', 'Centroid Distance']),
            use_container_width=True, hide_index=True,
            column_config={
                'Centroid Distance': st.column_config.ProgressColumn(min_value=0, max_value=float(out_df['Centroid Distance'].max()), format="%.4f"),
                'Is Outlier': st.column_config.CheckboxColumn()
            }
        )

    with tab3:
        outlier_df = out_df[out_df['Is Outlier'] == True].sort_values('Centroid Distance', ascending=False)
        if len(outlier_df):
            st.caption(f"{len(outlier_df)} pages are significantly distant from their assigned cluster centre.")
            for _, row in outlier_df.head(30).iterrows():
                title = row.get('Title 1', row.get(title_col, '')) if title_col else ''
                label = f"{row['Address']}" + (f"  |  {title}" if title else "")
                meta_parts = [f"Cluster: {row['Cluster Name']}", f"Distance: {row['Centroid Distance']:.4f}"]
                if has_clicks:       meta_parts.append(f"Clicks: {int(row.get('Clicks', 0)):,}")
                if has_impressions:  meta_parts.append(f"Impr: {int(row.get('Impressions', 0)):,}")
                if has_position:     meta_parts.append(f"Pos: {row.get('Avg Position', '')}")
                if has_inlinks:      meta_parts.append(f"Inlinks: {int(row.get('Inlinks', 0))}")
                if has_wordcount:    meta_parts.append(f"Words: {int(row.get('Word Count', 0))}")
                if has_depth:        meta_parts.append(f"Depth: {int(row.get('Crawl Depth', 0))}")
                if has_indexability: meta_parts.append(f"{row.get('Indexability', '')}")
                meta_str = "  &nbsp;·&nbsp;  ".join(meta_parts)
                st.markdown(f'<div class="outlier-row">{label}<br><span style="color:#8b949e">{meta_str}</span></div>', unsafe_allow_html=True)
        else:
            st.success("No outliers detected at the current sensitivity setting.")

    with st.expander("📋 Cluster action guide"):
        st.markdown("""
| Finding | Action |
|---------|--------|
| Very small clusters (1–3 pages) | Nascent topic worth building out, or off-topic content to prune |
| Large cluster with no commercial page | Create a hub or service page for that topic |
| Pages in the wrong cluster | Content drift - page title and body cover different topics |
| Outlier with traffic | Do NOT prune. Rewrite to align with a core cluster |
| Outlier with zero traffic (12+ months) | Strong prune candidate. Redirect and remove. |
""")

    section("04 - Export")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv1 = out_df.to_csv(index=False).encode()
        st.download_button("⬇  content_clusters.csv", data=csv1,
            file_name="content_clusters.csv", mime="text/csv")
        st.caption("All pages with cluster assignment")
    with col_dl2:
        outlier_export = out_df[out_df['Is Outlier'] == True].sort_values('Centroid Distance', ascending=False)
        csv2 = outlier_export.to_csv(index=False).encode()
        st.download_button("⬇  content_outliers.csv", data=csv2,
            file_name="content_outliers.csv", mime="text/csv")
        st.caption("Outlier pages only - prune/rewrite candidates")
