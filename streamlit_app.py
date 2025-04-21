import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from scipy.constants import Planck, elementary_charge

st.set_page_config(page_title="Analyse de conductance", layout="wide")

st.title("🧪 Analyse de plateaux quantiques (G / G₀)")
st.markdown("Charge un fichier CSV issu de l'oscilloscope pour analyser les plateaux quantifiés de conductance.")

uploaded_file = st.file_uploader("Dépose un fichier CSV", type=["csv"])

if uploaded_file:

    G0 = 2 * elementary_charge**2 / Planck
    resistance = 1000
    residual_resistance = 250

    def compute_conductance(voltage, source_voltage, resistance=1, resistance_residuelle=250):
        G = (1 / resistance) * (source_voltage - voltage) / voltage
        G_corrige = 1 / (1 / G - resistance_residuelle)
        return G_corrige

    # Lecture du fichier
    df_raw = pd.read_csv(uploaded_file, header=None)
    df_raw = df_raw.dropna(axis=1, how='all')
    data = df_raw[pd.to_numeric(df_raw.iloc[:, -2], errors='coerce').notna() &
                  pd.to_numeric(df_raw.iloc[:, -1], errors='coerce').notna()]
    data = data.iloc[:, -2:]
    data.columns = ["Time", "Voltage"]
    data = data.dropna()
    data["Time"] = pd.to_numeric(data["Time"])
    data["Voltage"] = pd.to_numeric(data["Voltage"])

    t = data["Time"].values
    voltage = data["Voltage"].values

    # Détection du Sample Interval
    sample_interval = t[1] - t[0] if len(t) > 1 else 1e-10

    # Lissage
    window_size = 10
    voltage_smoothed = np.convolve(voltage, np.ones(window_size) / window_size, mode='same')
    dv = np.diff(voltage_smoothed)
    threshold = 2 * np.std(dv)

    # Estimation Vsource
    if abs(min(voltage_smoothed)) > abs(max(voltage_smoothed)):
        signal_inverted = True
        source_voltage = abs(min(voltage_smoothed))
    else:
        signal_inverted = False
        source_voltage = max(voltage_smoothed)

    def expected_voltage(n, Rres, V, R):
        return V / (1 + R / (1 / (n * G0) + Rres))

    expected_min = np.array([expected_voltage(n+1, 0, source_voltage, resistance) for n in range(5)])
    expected_max = np.array([expected_voltage(n+1, 600, source_voltage, resistance) for n in range(5)])

    # Détection des plateaux
    plateaus_with_conductance = []
    in_plateau = False
    start_index = 0

    for i, d in enumerate(dv):
        if abs(d) < threshold:
            if not in_plateau:
                in_plateau = True
                start_index = i
        else:
            if in_plateau:
                plateau_data = voltage[start_index:i+1]
                avg = np.mean(plateau_data)
                val = abs(avg) if signal_inverted else avg

                for j in range(5):
                    if expected_min[j] <= val <= expected_max[j]:
                        G = compute_conductance(abs(avg), source_voltage, resistance, residual_resistance)
                        G_order = G / G0
                        plateau_time = (t[start_index] + t[i]) / 2
                        plateau_voltage = -abs(avg) if signal_inverted else avg
                        plateaus_with_conductance.append({
                            "time": plateau_time,
                            "voltage": plateau_voltage,
                            "G_order": G_order,
                            "start": start_index,
                            "end": i
                        })
                        break
                in_plateau = False

    # Tracé interactif avec Plotly
    fig = go.Figure()

    # Signal brut
    fig.add_trace(go.Scatter(
        x=t, y=voltage, mode='lines',
        name='Signal brut',
        line=dict(color='lightgray', width=1), opacity=0.4
    ))

    # Signal lissé
    fig.add_trace(go.Scatter(
        x=t, y=voltage_smoothed, mode='lines',
        name='Signal lissé',
        line=dict(color='black', width=1.2), opacity=0.7
    ))

    # Colormap
    norm = mcolors.Normalize(
        vmin=min(p["G_order"] for p in plateaus_with_conductance),
        vmax=max(p["G_order"] for p in plateaus_with_conductance)
    )
    colormap = cm.get_cmap('viridis')

    for p in plateaus_with_conductance:
        color_rgb = colormap(norm(p["G_order"]))
        color_str = f"rgba({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)}, 1)"

        fig.add_trace(go.Scatter(
            x=t[p["start"]:p["end"]+1],
            y=voltage_smoothed[p["start"]:p["end"]+1],
            mode='lines',
            line=dict(width=3, color=color_str),
            name=f"{p['G_order']:.2f} G₀",
            showlegend=False,
            hovertemplate=f"G / G₀ = {p['G_order']:.2f}<br>V = {p['voltage']:.3f} V<br>t = {p['time']:.6f} s"
        ))

    # Colorbar (fake trace)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            color=[p["G_order"] for p in plateaus_with_conductance],
            cmin=min(p["G_order"] for p in plateaus_with_conductance),
            cmax=max(p["G_order"] for p in plateaus_with_conductance),
            colorbar=dict(title="G / G₀"),
            showscale=True,
            size=0
        ),
        showlegend=False
    ))

    fig.update_layout(
        title="Signal + plateaux détectés (colorés selon G / G₀)",
        xaxis_title="Temps (s)",
        yaxis_title="Tension (V)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
