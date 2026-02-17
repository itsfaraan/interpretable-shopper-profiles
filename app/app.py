import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

st.set_page_config(page_title="Shopper Profiles (XAI)", layout="wide")

DATA_PATH = "data/processed/customer_with_clusters.csv"
SURROGATE_PATH = "models/surrogate_rf.joblib"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(SURROGATE_PATH)

# 1. WE LOAD THE DATA FIRST
df = load_data()
clf = load_model()

# 2. WE DEFINE THE FEATURES
feature_cols = [c for c in df.columns if c not in ["CustomerID", "ClusterID"]]

# 3. NOW WE CAN ANALYZE THE DATA TO NAME THE CLUSTERS
def assign_personas_dynamically(dataframe):
    """Automatically assigns persona names based on the cluster's mathematical averages."""
    cluster_means = dataframe.groupby('ClusterID')[feature_cols].mean()
    
    p_map = {}
    a_map = {}
    
    for c_id, row in cluster_means.iterrows():
        if row['Monetary'] > 50000: 
            p_map[c_id] = "The True VIPs"
            a_map[c_id] = "Assign dedicated account manager. Invite to premium early-access events."
        elif row['Return_Rate'] > 0.20: 
            p_map[c_id] = "High Return Risk"
            a_map[c_id] = "Investigate return reasons. Restrict free return privileges if abusive."
        elif row['Night_Shopper'] > 0.50: 
            p_map[c_id] = "Night Shopper / At-Risk"
            a_map[c_id] = "Send a win-back campaign with a strong discount. Target late at night."
        elif row['Weekend_Ratio'] > 0.50: 
            p_map[c_id] = "Weekend Shopper"
            a_map[c_id] = "Target with Friday evening / Saturday morning push notifications."
        elif row['Basket_Diversity'] > 100:
            p_map[c_id] = "High Diversity Explorer"
            a_map[c_id] = "Recommend new product categories. High potential for cross-selling."
        else:
            p_map[c_id] = "Standard Occasional Shopper"
            a_map[c_id] = "Send standard weekly promotional emails to increase frequency."
            
    return p_map, a_map

# 4. GENERATE MAPS AND APPLY TO DATAFRAME
PERSONA_MAP, ACTION_MAP = assign_personas_dynamically(df)
df['Persona'] = df['ClusterID'].map(PERSONA_MAP).fillna(df['ClusterID'].astype(str))


st.title("AI-Driven Customer Personas and Behavioral Insights")
st.caption("Empowering CRM with Interpretable Machine Learning Segments")

with st.sidebar:
    st.header("Customer Selection")
    cid = st.selectbox("Select CustomerID", df["CustomerID"].tolist())

# ==========================================
# CREATE TABS
# ==========================================
tab_individual, tab_global, tab_data = st.tabs([
    "👤 Individual Profile & Simulation", 
    "🌍 Global Audience Overview", 
    "🔍 Data Explorer"
])

# ==========================================
# TAB 1: INDIVIDUAL PROFILE & SIMULATOR
# ==========================================
with tab_individual:
    row = df[df["CustomerID"] == cid].iloc[0]
    cluster_id = int(row["ClusterID"])
    persona_name = PERSONA_MAP.get(cluster_id, f"Cluster {cluster_id}")

    st.subheader(f"Customer Profiling: {cid} | Persona: {persona_name}")

    st.markdown("### Customer vs. Persona Average")
    seg_mean = df[df["ClusterID"] == cluster_id][feature_cols].mean()

    m_cols = st.columns(4)
    for i, col_name in enumerate(feature_cols):
        val = float(row[col_name])
        mean_val = float(seg_mean[col_name])
        
        if 'Ratio' in col_name or 'Rate' in col_name or 'Shopper' in col_name:
            delta_val = val - mean_val
            val_fmt = f"{val:.1%}"
            delta_fmt = f"{delta_val:+.1%}"
        else:
            delta_pct = (val - mean_val) / (mean_val + 1e-5) * 100
            val_fmt = f"${val:,.2f}" if col_name == 'Monetary' else f"{val:,.0f}"
            delta_fmt = f"{delta_pct:.0f}% vs Avg"

        with m_cols[i % 4]:
            st.metric(label=col_name, value=val_fmt, delta=delta_fmt)

    st.divider()

    st.markdown(f"## Key Behaviors Driving the '{persona_name}' Profile")

    X_one = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
    pred_class = int(clf.predict(X_one)[0])

    explainer = shap.TreeExplainer(clf)
    try:
        exp = explainer(X_one)
        values = np.array(exp.values)
        if values.ndim == 3:
            vals_1d = values[0, :, pred_class].astype(float)
        else:
            vals_1d = values[0, :].astype(float)
    except Exception:
        shap_values = explainer.shap_values(X_one)
        if isinstance(shap_values, list):
            vals_1d = np.array(shap_values[pred_class][0], dtype=float)
        else:
            vals_1d = np.array(shap_values[0], dtype=float)

    col_chart, col_text = st.columns([2, 1])

    with col_chart:
        fig = plt.figure(figsize=(6, 4))
        try:
            order = np.argsort(np.abs(vals_1d))[::-1][:6]
            colors = ['green' if x > 0 else 'red' for x in vals_1d[order][::-1]]
            plt.barh([feature_cols[i] for i in order][::-1], vals_1d[order][::-1], color=colors)
            plt.title("Impact of specific features on this persona")
            plt.xlabel("SHAP Value (Impact)")
            plt.tight_layout()
        except Exception as e:
            plt.text(0.1, 0.5, f"Plot failed: {e}")
        st.pyplot(fig, clear_figure=True)

    with col_text:
        st.markdown("### Behavioral Summary")
        st.write(f"The model confidently identified Customer **{cid}** as a **{persona_name}**.")
        
        top_idx = np.argsort(np.abs(vals_1d))[::-1][:3]
        st.write("**Top deciding factors:**")
        for i in top_idx:
            feat = feature_cols[i]
            direction = "raised" if vals_1d[i] > 0 else "lowered"
            st.write(f"- Their **{feat}** significantly {direction} the likelihood of this profile.")
        
        st.info(f"🎯 **Recommended Action:**\n\n{ACTION_MAP.get(pred_class, 'Review profile manually.')}")

    # --- SIMULATOR ---
    st.markdown("---")
    st.markdown("### 🎛️ 'What-If' Scenario Simulator")
    st.write("Adjust this customer's behaviors to see what it would take to change their Persona.")

    with st.expander("Open Scenario Simulator"):
        sim_cols = st.columns(3)
        simulated_features = {}
        
        for i, col_name in enumerate(feature_cols):
            current_val = float(row[col_name])
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            
            with sim_cols[i % 3]:
                if 'Ratio' in col_name or 'Rate' in col_name or 'Shopper' in col_name:
                    simulated_features[col_name] = st.slider(f"Simulate {col_name}", min_value=0.0, max_value=1.0, value=current_val, step=0.05)
                else:
                    simulated_features[col_name] = st.slider(f"Simulate {col_name}", min_value=min_val, max_value=max_val, value=current_val, step=float((max_val-min_val)/100))
        
        X_simulated = pd.DataFrame([simulated_features], columns=feature_cols)
        new_pred_class = int(clf.predict(X_simulated)[0])
        new_persona_name = PERSONA_MAP.get(new_pred_class, f"Cluster {new_pred_class}")
        
        st.markdown("#### Simulation Result:")
        if new_pred_class == cluster_id:
            st.info(f"The customer remains in the **{persona_name}** persona.")
        else:
            st.success(f"🎉 The customer's persona changed to: **{new_persona_name}**!")


# ==========================================
# TAB 2: GLOBAL AUDIENCE OVERVIEW
# ==========================================
with tab_global:
    st.header("Macro Segment Analytics")
    st.write("Understand the distribution and financial impact of your AI-driven customer segments.")
    
    col_pie, col_bar = st.columns(2)
    
    with col_pie:
        st.subheader("Audience Distribution")
        dist = df['Persona'].value_counts()
        
        # Make the figure slightly wider to accommodate a legend
        fig_pie, ax_pie = plt.subplots(figsize=(8, 5)) 
        
        # Draw the pie chart without direct labels, and only show % if > 2%
        wedges, texts, autotexts = ax_pie.pie(
            dist, 
            autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "", 
            startangle=140, 
            colors=plt.cm.Set3.colors,
            pctdistance=0.75
        )
        ax_pie.axis('equal')
        
        # Add a clean legend to the right of the chart
        ax_pie.legend(wedges, dist.index,
                      title="Personas",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))
                      
        st.pyplot(fig_pie, clear_figure=True)
        
    with col_bar:
        st.subheader("Average Spend per Persona")
        avg_spend = df.groupby('Persona')['Monetary'].mean().sort_values()
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        avg_spend.plot(kind='barh', ax=ax_bar, color='steelblue')
        
        # --- NEW: Formatter to convert 200000 to $200K ---
        formatter = ticker.FuncFormatter(lambda x, pos: f'${x/1000:,.0f}K' if x >= 1000 else f'${x:,.0f}')
        ax_bar.xaxis.set_major_formatter(formatter)
        
        ax_bar.set_xlabel("Average Monetary Value")
        ax_bar.set_ylabel("")
        
        # Optional padding adjustment so the labels don't get cut off
        plt.tight_layout() 
        st.pyplot(fig_bar, clear_figure=True)

    st.markdown("---")
    st.subheader("Segment Archetypes (Averages)")
    st.write("A macro view of the behavioral fingerprint for each persona.")
    
    # Create a clean summary table grouped by persona
    segment_means_df = df.groupby('Persona')[feature_cols].mean()
    
    # Format the table for readability using Streamlit's native dataframe styling
    formatted_df = segment_means_df.style.format({
        'Recency': "{:.0f}",
        'Frequency': "{:.1f}",
        'Monetary': "${:,.2f}",
        'Weekend_Ratio': "{:.1%}",
        'Night_Shopper': "{:.1%}",
        'Basket_Diversity': "{:.0f}",
        'Return_Rate': "{:.1%}"
    }).background_gradient(cmap='Blues', axis=0)
    
    st.dataframe(formatted_df, use_container_width=True)

    # ==========================================
# TAB 3: DATA EXPLORER
# ==========================================
with tab_data:
    st.header("Raw Customer Data Explorer")
    st.write("Customers within each persona.")
    
    # Create filters
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        selected_persona = st.selectbox("Filter by Persona:", ["All"] + list(df['Persona'].unique()))
    with col_filter2:
        sort_by = st.selectbox("Sort by:", ["Monetary", "Frequency", "Recency", "Return_Rate"])
        
    # Apply filters
    filtered_df = df.copy()
    if selected_persona != "All":
        filtered_df = filtered_df[filtered_df['Persona'] == selected_persona]
        
    # Apply sorting
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
    
    # Display metrics for the current view
    st.markdown(f"**Showing {len(filtered_df)} customers.**")
    
    # Display the dataframe cleanly
    st.dataframe(
        filtered_df[["CustomerID", "Persona"] + feature_cols].style.format({
            'Recency': "{:.0f}",
            'Frequency': "{:.0f}",
            'Monetary': "${:,.2f}",
            'Weekend_Ratio': "{:.1%}",
            'Night_Shopper': "{:.1%}",
            'Basket_Diversity': "{:.0f}",
            'Return_Rate': "{:.1%}"
        }), 
        use_container_width=True,
        height=600
    )

