"""Portal de presentación para la defensa del Trabajo de Grado ULA.

Esta interfaz presenta exclusivamente los análisis directamente alineados con
los objetivos, resultados y limitaciones descritos en la tesis. Utiliza el
mismo corpus procesado de la campaña experimental; no realiza mediciones en
tiempo real ni constituye una fuente experimental independiente.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import ipaddress
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Configuración de presentación
# -----------------------------------------------------------------------------
MODO_JURADO = True

OUT_CANDIDATES = [
    Path("cap4_data"),
    Path("salidas_tesis_v3"),
    Path("salidas_tesis"),
    Path("."),
]

IXP_NETWORKS = [
    (ipaddress.ip_network("206.41.108.0/24"), "IXP · FL-IX (Miami)"),
    (ipaddress.ip_network("206.223.124.0/24"), "IXP · NAP Colombia (Bogotá)"),
]

SERVICE_ORDER = [
    "Biblioteca Digital / Vereda",
    "Intranet ULA",
    "Sistema de Grados",
    "Saber ULA",
]

PROBE_ORDER = [
    "ULA (Mérida)",
    "CANTV (Caracas)",
    "NetUno (Caracas)",
    "Movistar (Maracay)",
    "Inter (Barquisimeto)",
    "Airtek (Maracaibo)",
    "Tiggee (Bogotá)",
    "Satnet (Quito)",
    "AT&T (Miami)",
]

NATIONAL_PROBES = [
    "ULA (Mérida)",
    "CANTV (Caracas)",
    "NetUno (Caracas)",
    "Movistar (Maracay)",
    "Inter (Barquisimeto)",
    "Airtek (Maracaibo)",
]

SERVICE_INSIGHT = {
    "Biblioteca Digital / Vereda": (
        "Fue el servicio con mayor continuidad observable durante la campaña y sirve como "
        "referencia para comparar diferencias de RTT y enrutamiento entre redes de origen."
    ),
    "Intranet ULA": (
        "Presentó el patrón temporal más marcado: la alcanzabilidad ICMP observada fue "
        "considerablemente mayor en la franja matutina que en la nocturna."
    ),
    "Sistema de Grados": (
        "Mostró una alternancia longitudinal entre sesiones de respuesta generalizada y "
        "sesiones sin respuesta, por lo que el RTT de las ejecuciones exitosas debe leerse "
        "junto con su alcanzabilidad."
    ),
    "Saber ULA": (
        "Presentó el comportamiento más dependiente de la red de origen: durante gran parte "
        "de la campaña coexistieron redes con respuesta y redes sin respuesta."
    ),
}

HEATMAP_SCALE = [
    [0.0, "#d73027"],
    [0.5, "#fee08b"],
    [1.0, "#1a9850"],
]

OONI_COLOR_MAP = {
    "OK": "#1a9850",
    "Anomaly": "#f1c40f",
    "Failure": "#d73027",
    "Confirmed": "#7f0000",
}

st.set_page_config(
    page_title="Análisis de conectividad ULA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Oculta elementos de administración que no aportan a la defensa.
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      [data-testid="stToolbar"] {visibility: hidden; height: 0px;}
      [data-testid="stDecoration"] {display: none;}
      .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def plotly_config() -> dict:
    if MODO_JURADO:
        return {"displaylogo": False, "displayModeBar": False, "scrollZoom": False, "responsive": True}
    return {"displaylogo": False, "displayModeBar": True, "scrollZoom": False, "responsive": True}


def sankey_config() -> dict:
    """Mantiene hover y arrastre de nodos sin exponer controles innecesarios."""
    return {
        "displaylogo": False,
        "displayModeBar": False,
        "scrollZoom": False,
        "responsive": True,
        "staticPlot": False,
    }


def _buscar_out_dir() -> Path:
    for candidate in OUT_CANDIDATES:
        if (candidate / "pings_ejecuciones.csv").exists() or (candidate / "pings_ejecuciones.csv.gz").exists():
            return candidate
    return OUT_CANDIDATES[0]


OUT = _buscar_out_dir()


@st.cache_data(show_spinner=False)
def load_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path_csv = OUT / f"{name}.csv"
    path_gz = OUT / f"{name}.csv.gz"
    if path_csv.exists():
        path = path_csv
    elif path_gz.exists():
        path = path_gz
    else:
        raise FileNotFoundError(path_csv)
    return pd.read_csv(path, parse_dates=parse_dates or [])


@st.cache_data(show_spinner=False)
def cargar_datos():
    p = load_csv("pings_ejecuciones", ["timestamp"])
    t = load_csv("traceroutes_resumen", ["timestamp"])
    try:
        h = load_csv("traceroutes_hops_app", ["timestamp"])
    except FileNotFoundError:
        h = load_csv("traceroutes_hops", ["timestamp"])
    o = load_csv("ooni_diario")
    c_sondas = load_csv("tabla_3_4_cobertura_sondas", ["Inicio observado", "Fin observado"])
    c_serv = load_csv("tabla_3_5_cobertura_servicios", ["Primera medición", "Última medición"])
    return p, t, h, o, c_sondas, c_serv


def pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else np.nan


def fmt_num(v: float, dec: int = 1, suffix: str = "") -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:.{dec}f}{suffix}"


def estadisticas_ping(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "n": 0,
            "n_validas": 0,
            "n_con_respuesta": 0,
            "alcanzabilidad": np.nan,
            "perdida": np.nan,
            "media": np.nan,
            "mediana": np.nan,
            "p95": np.nan,
        }
    valid = df[pd.to_numeric(df["sent"], errors="coerce") > 0].copy()
    rtt = pd.to_numeric(valid["rtt_exec_ms"], errors="coerce").dropna()
    sent = pd.to_numeric(valid["sent"], errors="coerce").fillna(0).sum()
    rcvd = pd.to_numeric(valid["rcvd"], errors="coerce").fillna(0).sum()
    return {
        "n": len(df),
        "n_validas": len(valid),
        "n_con_respuesta": int((pd.to_numeric(valid["rcvd"], errors="coerce") > 0).sum()),
        "alcanzabilidad": pct((pd.to_numeric(valid["rcvd"], errors="coerce") > 0).sum(), len(valid)),
        "perdida": pct(sent - rcvd, sent),
        "media": rtt.mean() if len(rtt) else np.nan,
        "mediana": rtt.median() if len(rtt) else np.nan,
        "p95": rtt.quantile(0.95) if len(rtt) else np.nan,
    }


def resumen_sondas_ping(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(
        ["sonda_id", "sonda_nombre", "asn_origen", "tipo_sonda"], dropna=False
    ):
        s = estadisticas_ping(grp)
        rows.append(
            {
                "Probe ID": int(keys[0]) if pd.notna(keys[0]) else np.nan,
                "Sonda": keys[1],
                "ASN": keys[2],
                "Grupo": keys[3],
                "Ping válidas": int(s["n_validas"]),
                "Alcanzabilidad ICMP (%)": s["alcanzabilidad"],
                "Pérdida agregada (%)": s["perdida"],
                "RTT mediano (ms)": s["mediana"],
                "RTT P95 (ms)": s["p95"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {p: i for i, p in enumerate(PROBE_ORDER)}
    out["_orden"] = out["Sonda"].map(order).fillna(999)
    numeric = [
        "Alcanzabilidad ICMP (%)",
        "Pérdida agregada (%)",
        "RTT mediano (ms)",
        "RTT P95 (ms)",
    ]
    out[numeric] = out[numeric].round(2)
    return out.sort_values("_orden").drop(columns="_orden")


def _token_ixp(ip_value) -> str | None:
    if ip_value is None or (isinstance(ip_value, float) and np.isnan(ip_value)):
        return None
    try:
        addr = ipaddress.ip_address(str(ip_value).strip())
    except ValueError:
        return None
    for network, label in IXP_NETWORKS:
        if addr in network:
            return label
    return None


def _normalizar_asn(valor) -> str | None:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    txt = str(valor).strip()
    if not txt or txt.lower() in {"nan", "none", "n/a"}:
        return None
    m = re.fullmatch(r"AS\s*(\d+)", txt, flags=re.I)
    if m:
        return f"AS{m.group(1)}"
    if txt.isdigit():
        return f"AS{txt}"
    m = re.search(r"['\"]asn['\"]\s*:\s*(\d+)", txt, flags=re.I)
    if m:
        return f"AS{m.group(1)}"
    m = re.search(r"\bAS\s*(\d+)\b", txt, flags=re.I)
    return f"AS{m.group(1)}" if m else None


def _holder_desde_valor_asn(valor) -> str | None:
    if valor is None:
        return None
    m = re.search(r"['\"]holder['\"]\s*:\s*['\"]([^'\"]+)", str(valor), flags=re.I)
    return m.group(1).strip() if m else None


def _limpiar_nombre_as(nombre: str | None, asn: str | None = None, max_len: int = 34) -> str | None:
    if nombre is None or (isinstance(nombre, float) and np.isnan(nombre)):
        return None
    txt = re.sub(r"\s+", " ", str(nombre)).strip()
    if not txt or txt.lower() in {"nan", "none", "desconocido"}:
        return None
    if asn:
        txt = re.sub(rf"^{re.escape(asn)}\s*[-–:]?\s*", "", txt, flags=re.I)
    return txt[: max_len - 1].rstrip() + "…" if len(txt) > max_len else txt


def mapa_nombres_asn(df_hops: pd.DataFrame) -> dict[str, str]:
    if df_hops.empty or "asn_hop" not in df_hops.columns:
        return {}
    candidates: dict[str, list[str]] = {}
    cols = [c for c in ["asn_hop", "as_name", "org"] if c in df_hops.columns]
    for row in df_hops[cols].itertuples(index=False, name=None):
        raw_asn = row[0]
        asn = _normalizar_asn(raw_asn)
        if not asn:
            continue
        as_name = row[1] if len(row) > 1 else None
        org = row[2] if len(row) > 2 else None
        holder = _holder_desde_valor_asn(raw_asn)
        for val in [as_name, org, holder]:
            cleaned = _limpiar_nombre_as(val, asn)
            if cleaned:
                candidates.setdefault(asn, []).append(cleaned)
                break
    out = {}
    for asn, names in candidates.items():
        name = Counter(names).most_common(1)[0][0] if names else None
        out[asn] = f"{asn} · {name}" if name else asn
    return out


def secuencias_asn_por_trace(df_hops: pd.DataFrame) -> dict[str, list[str]]:
    """Secuencias ASN/IXP excluyendo hop=255 como TTL ordinario."""
    if df_hops.empty:
        return {}
    work = df_hops[df_hops["hop_especial_255"].astype(str).str.lower() != "true"].copy()
    if work.empty:
        return {}
    work["asn_norm"] = work["asn_hop"].map(_normalizar_asn) if "asn_hop" in work.columns else None
    work["ixp_token"] = work["ip_hop"].map(_token_ixp) if "ip_hop" in work.columns else None
    out: dict[str, list[str]] = {}
    for trace_id, grp in work.groupby("trace_id"):
        seq: list[str] = []
        for _, hg in grp.groupby("hop_num", sort=True):
            ixps = hg["ixp_token"].dropna().astype(str)
            if not ixps.empty:
                token = ixps.mode().iloc[0]
            else:
                vals = hg["asn_norm"].dropna().astype(str)
                if vals.empty:
                    continue
                token = vals.mode().iloc[0]
            if not seq or seq[-1] != token:
                seq.append(token)
        out[str(trace_id)] = seq
    return out


def _compactar_path(tokens: list[str | None]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        token = str(token)
        if not out or out[-1] != token:
            out.append(token)
    return out


def construir_paths_asn(
    df_traces: pd.DataFrame,
    df_hops: pd.DataFrame,
    servicio: str,
    sondas: list[str] | None = None,
) -> pd.DataFrame:
    seqs = secuencias_asn_por_trace(df_hops[df_hops["sitio_web"] == servicio])
    tr = df_traces[df_traces["sitio_web"] == servicio].copy()
    if sondas:
        tr = tr[tr["sonda_nombre"].isin(sondas)]
    rows = []
    for _, row in tr.iterrows():
        seq = seqs.get(str(row["trace_id"]), [])
        origin = _normalizar_asn(row.get("asn_origen")) or str(row.get("asn_origen", "Origen"))
        path = _compactar_path([origin] + seq)
        if not path:
            continue
        reached = bool(row["respondio_destino"])
        terminal = f"Destino · {servicio}" if reached else "Traza incompleta · último punto observado"
        rows.append(
            {
                "trace_id": row["trace_id"],
                "sonda_nombre": row["sonda_nombre"],
                "respondio_destino": reached,
                "estado_traza": "Destino alcanzado" if reached else "Traza incompleta",
                "path_tokens": path,
                "firma_visual": " > ".join(path + [terminal]),
            }
        )
    return pd.DataFrame(rows)


def sankey_asn_limpio(
    df_traces: pd.DataFrame,
    df_hops: pd.DataFrame,
    servicio: str,
    sondas: list[str],
    top_paths: int = 8,
):
    """Sankey que separa trazas incompletas y conserva IXPs conocidos."""
    paths = construir_paths_asn(df_traces, df_hops, servicio, sondas)
    if paths.empty:
        return None

    counts = paths.groupby("firma_visual").size().reset_index(name="trazas").sort_values("trazas", ascending=False)
    selected = counts.head(max(1, top_paths))["firma_visual"]
    paths_top = paths[paths["firma_visual"].isin(selected)].copy()

    edges: list[tuple[str, str]] = []
    for _, prow in paths_top.iterrows():
        terminal = (
            f"Destino · {servicio}"
            if bool(prow["respondio_destino"])
            else "Traza incompleta · último punto observado"
        )
        full = prow["path_tokens"] + [terminal]
        edges.extend(zip(full[:-1], full[1:]))
    if not edges:
        return None

    e = pd.DataFrame(edges, columns=["source", "target"]).value_counts().reset_index(name="trazas")
    name_map = mapa_nombres_asn(df_hops)
    nodes_raw = list(dict.fromkeys(e["source"].tolist() + e["target"].tolist()))
    labels = [name_map.get(n, n) for n in nodes_raw]
    idx = {n: i for i, n in enumerate(nodes_raw)}

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=18,
                label=labels,
                customdata=nodes_raw,
                hovertemplate="%{label}<br>%{customdata}<extra></extra>",
            ),
            link=dict(
                source=e["source"].map(idx),
                target=e["target"].map(idx),
                value=e["trazas"],
                hovertemplate="%{source.label} → %{target.label}<br>%{value} traceroutes<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=f"Rutas ASN/IXP más frecuentes hacia {servicio}",
        height=max(620, 40 * len(nodes_raw)),
        margin=dict(l=20, r=20, t=75, b=20),
        font=dict(size=11),
    )
    return fig


def _tokens_clasificacion(grp: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    if grp.empty:
        return tokens
    work = grp[grp["hop_especial_255"].astype(str).str.lower() != "true"]
    for row in work.itertuples(index=False):
        ip_value = getattr(row, "ip_hop", None)
        ixp = _token_ixp(ip_value)
        if ixp:
            tokens.add(ixp)
        asn = _normalizar_asn(getattr(row, "asn_hop", None))
        if asn:
            tokens.add(asn)
    return tokens


def _clasificar_tokens(tokens: set[str]) -> str:
    fuerte = {
        "IXP · FL-IX (Miami)",
        "IXP · NAP Colombia (Bogotá)",
        "AS174",
        "AS18678",
    }
    if tokens & fuerte:
        return "Exterior fuerte"
    if "AS52320" in tokens:
        return "Moderada"
    return "Sin marcador exterior observable"


@st.cache_data(show_spinner=False)
def clasificacion_enrutamiento_nacional(df_traces: pd.DataFrame, df_hops: pd.DataFrame) -> pd.DataFrame:
    """Clasificación conservadora usada en la metodología final de la tesis."""
    t_nat = df_traces[df_traces["tipo_sonda"] == "Nacional"].copy()
    h_nat = df_hops[df_hops["tipo_sonda"] == "Nacional"].copy()
    token_map = {str(tid): _tokens_clasificacion(grp) for tid, grp in h_nat.groupby("trace_id")}
    t_nat["Clasificación"] = t_nat["trace_id"].astype(str).map(lambda tid: _clasificar_tokens(token_map.get(tid, set())))
    tab = (
        t_nat.groupby(["sonda_nombre", "Clasificación"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Exterior fuerte", "Moderada", "Sin marcador exterior observable"], fill_value=0)
    )
    tab["Total"] = tab.sum(axis=1)
    tab = tab.reset_index().rename(columns={"sonda_nombre": "Sonda"})
    order = {p: i for i, p in enumerate(NATIONAL_PROBES)}
    tab["_orden"] = tab["Sonda"].map(order).fillna(999)
    return tab.sort_values("_orden").drop(columns="_orden")


try:
    df_p, df_t, df_h, df_o, df_cov_s, df_cov_d = cargar_datos()
except FileNotFoundError:
    st.error(
        "No se pudieron cargar los datos procesados de la campaña. Verifique que los seis CSV "
        "se encuentren en la carpeta `cap4_data`, `salidas_tesis_v3` o `salidas_tesis`."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Encabezado y navegación
# -----------------------------------------------------------------------------
st.title("Análisis de conectividad hacia servicios de la Universidad de Los Andes")
st.caption(
    "Herramienta interactiva de apoyo al Trabajo de Grado · RIPE Atlas Ping/Traceroute · "
    "OONI Web Connectivity como evidencia complementaria · campaña junio–agosto de 2026"
)
st.info(
    "Esta aplicación presenta el corpus experimental cerrado utilizado en la tesis. "
    "No realiza mediciones en tiempo real y no constituye una fuente experimental independiente."
)

servicios_disponibles = [s for s in SERVICE_ORDER if s in set(df_p["sitio_web"].dropna())]
st.sidebar.header("Consulta")
servicio = st.sidebar.selectbox("Servicio institucional", servicios_disponibles)
st.sidebar.markdown("**Período general:** 15/06/2026–14/08/2026")
st.sidebar.caption("Las coberturas efectivas varían por sonda y destino.")

p_serv = df_p[df_p["sitio_web"] == servicio].copy()
t_serv = df_t[df_t["sitio_web"] == servicio].copy()
h_serv = df_h[df_h["sitio_web"] == servicio].copy()
o_serv = df_o[df_o["sitio_web"] == servicio].copy()

nombres_tabs = [
    "Resumen",
    "Desempeño y alcanzabilidad",
    "Enrutamiento",
    "OONI",
    "Cobertura y metodología",
]
if not MODO_JURADO:
    nombres_tabs.append("Técnico")
tabs = st.tabs(nombres_tabs)
tab_resumen, tab_desempeno, tab_rutas, tab_ooni, tab_metodo = tabs[:5]
tab_tecnico = tabs[5] if not MODO_JURADO else None

# -----------------------------------------------------------------------------
# 1. RESUMEN
# -----------------------------------------------------------------------------
with tab_resumen:
    st.subheader(servicio)
    st.write(SERVICE_INSIGHT.get(servicio, ""))

    stats = estadisticas_ping(p_serv)
    trace_reach = pct(t_serv["respondio_destino"].astype(bool).sum(), len(t_serv)) if len(t_serv) else np.nan
    c = st.columns(6)
    c[0].metric("Ping válidas", f"{int(stats['n_validas']):,}")
    c[1].metric("Alcanzabilidad ICMP", fmt_num(stats["alcanzabilidad"], 2, "%"))
    c[2].metric("Pérdida agregada", fmt_num(stats["perdida"], 2, "%"))
    c[3].metric("RTT mediano", fmt_num(stats["mediana"], 2, " ms"))
    c[4].metric("RTT P95", fmt_num(stats["p95"], 2, " ms"))
    c[5].metric("Traceroute al destino", fmt_num(trace_reach, 2, "%"))

    st.caption(
        "La alcanzabilidad y la pérdida se calculan sobre ejecuciones Ping válidas (sent > 0). "
        "Los estadísticos de RTT utilizan el RTT promedio por ejecución de aquellas mediciones que obtuvieron respuesta."
    )

    resumen = resumen_sondas_ping(p_serv)
    st.markdown("### Comparación por punto de observación")
    show_cols = [
        "Probe ID", "Sonda", "ASN", "Ping válidas", "Alcanzabilidad ICMP (%)",
        "Pérdida agregada (%)", "RTT mediano (ms)", "RTT P95 (ms)",
    ]
    st.dataframe(resumen[show_cols], use_container_width=True, hide_index=True)

    if not resumen.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                resumen,
                x="Sonda",
                y="Alcanzabilidad ICMP (%)",
                text_auto=".1f",
                range_y=[0, 100],
                title="Alcanzabilidad ICMP por sonda",
            )
            fig.update_layout(height=430, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())
        with col2:
            rtt = resumen[resumen["RTT mediano (ms)"].notna()].copy()
            fig = px.bar(
                rtt,
                x="Sonda",
                y="RTT mediano (ms)",
                text_auto=".1f",
                title="RTT mediano por sonda (solo ejecuciones con respuesta)",
            )
            fig.update_layout(height=430, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

# -----------------------------------------------------------------------------
# 2. DESEMPEÑO Y ALCANZABILIDAD
# -----------------------------------------------------------------------------
with tab_desempeno:
    st.subheader("Comportamiento temporal y estados de respuesta")
    st.caption(
        "Las diferencias entre las franjas matutina y nocturna se describen estadísticamente; "
        "no se interpretan por sí solas como diferencias de demanda o congestión."
    )

    disponibles = [p for p in PROBE_ORDER if p in set(p_serv["sonda_nombre"].dropna())]
    sonda_temporal = st.selectbox(
        "Serie temporal a visualizar",
        ["Todas las sondas"] + disponibles,
        key="sonda_temporal_jurado",
    )
    valid = p_serv[p_serv["rtt_exec_ms"].notna()].copy()
    if sonda_temporal != "Todas las sondas":
        valid = valid[valid["sonda_nombre"] == sonda_temporal]
    if not valid.empty:
        fig = px.line(
            valid.sort_values("timestamp"),
            x="timestamp",
            y="rtt_exec_ms",
            color="sonda_nombre" if sonda_temporal == "Todas las sondas" else None,
            markers=True,
            labels={"timestamp": "Fecha / hora local", "rtt_exec_ms": "RTT promedio por ejecución (ms)", "sonda_nombre": "Sonda"},
            title="Serie temporal de RTT",
        )
        fig.update_traces(connectgaps=False, marker=dict(size=4))
        fig.update_layout(height=540, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())
    else:
        st.info("No hay RTT válidos para la selección indicada.")

    st.markdown("### Comparación entre franjas")
    franjas = []
    for f in ["Matutina (~10:00)", "Nocturna (~22:00)"]:
        s = estadisticas_ping(p_serv[p_serv["franja"] == f])
        franjas.append(
            {
                "Franja": f,
                "Ping válidas": s["n_validas"],
                "Alcanzabilidad ICMP (%)": s["alcanzabilidad"],
                "Pérdida agregada (%)": s["perdida"],
                "RTT mediano (ms)": s["mediana"],
                "RTT P95 (ms)": s["p95"],
            }
        )
    st.dataframe(pd.DataFrame(franjas).round(2), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        states = p_serv.groupby(["sonda_nombre", "estado_ping"]).size().reset_index(name="Ejecuciones")
        if not states.empty:
            fig = px.bar(
                states,
                x="sonda_nombre",
                y="Ejecuciones",
                color="estado_ping",
                barmode="stack",
                labels={"sonda_nombre": "Sonda", "estado_ping": "Estado"},
                title="Estados de respuesta Ping",
            )
            fig.update_layout(height=470, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())
    with col2:
        daily = p_serv[p_serv["sent"] > 0].copy()
        if not daily.empty:
            daily["fecha_dt"] = pd.to_datetime(daily["fecha"])
            heat = (
                daily.groupby(["sonda_nombre", "fecha_dt"])["alcanzable_icmp"]
                .mean().mul(100).reset_index()
            )
            pivot = heat.pivot(index="sonda_nombre", columns="fecha_dt", values="alcanzable_icmp")
            order = [p for p in PROBE_ORDER if p in pivot.index]
            pivot = pivot.reindex(order)
            fig = px.imshow(
                pivot,
                labels={"x": "Fecha", "y": "Sonda", "color": "Alcanzabilidad (%)"},
                aspect="auto",
                color_continuous_scale=HEATMAP_SCALE,
                range_color=[0, 100],
                title="Alcanzabilidad diaria por sonda",
            )
            fig.update_layout(height=470)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())
            st.caption("Las celdas sin valor representan ausencia de observación efectiva para esa combinación sonda-día.")

# -----------------------------------------------------------------------------
# 3. ENRUTAMIENTO
# -----------------------------------------------------------------------------
with tab_rutas:
    st.subheader("Enrutamiento e interconexión observable")
    st.warning(
        "Traceroute observa parcialmente el plano de datos. Una ruta exterior no demuestra por sí sola "
        "que exista una alternativa doméstica disponible, ni permite reconstruir directamente las políticas BGP."
    )

    st.markdown("### Síntesis global de evidencia exterior en sondas nacionales")
    clasif = clasificacion_enrutamiento_nacional(df_t, df_h)
    st.dataframe(clasif, use_container_width=True, hide_index=True)
    st.caption(
        "'Sin marcador exterior observable' no demuestra que toda la trayectoria haya permanecido físicamente dentro de Venezuela. "
        "La clasificación es conservadora y corresponde a los criterios metodológicos de la tesis."
    )

    disponibles_nat = [p for p in NATIONAL_PROBES if p in set(t_serv["sonda_nombre"].dropna())]
    if disponibles_nat:
        default_probe = "Airtek (Maracaibo)" if "Airtek (Maracaibo)" in disponibles_nat else disponibles_nat[0]
        idx_default = disponibles_nat.index(default_probe)
        origen_ruta = st.selectbox(
            "Origen para visualizar en detalle",
            disponibles_nat,
            index=idx_default,
            key="origen_ruta_jurado",
        )
        t_route = t_serv[t_serv["sonda_nombre"] == origen_ruta].copy()
        h_route = h_serv[h_serv["sonda_nombre"] == origen_ruta].copy()
        n_reached = int(t_route["respondio_destino"].astype(bool).sum()) if len(t_route) else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Traceroutes", f"{len(t_route):,}")
        c2.metric("Destino alcanzado", f"{n_reached:,}")
        c3.metric("Alcanzabilidad Traceroute", fmt_num(pct(n_reached, len(t_route)), 2, "%"))

        st.markdown("### Sankey interactivo de rutas ASN/IXP")
        c_sankey, c_help = st.columns([1, 2])
        with c_sankey:
            top_paths = st.selectbox(
                "Rutas principales a mostrar",
                [5, 8, 12],
                index=1,
                key="top_paths_jurado",
                help="Solo cambia cuántas familias de ruta frecuentes se muestran; no modifica los datos.",
            )
        with c_help:
            st.info(
                "El Sankey es interactivo: puede pasar el cursor sobre nodos y enlaces para ver detalle "
                "y arrastrar los nodos para reorganizar temporalmente la vista."
            )

        fig = sankey_asn_limpio(t_serv, h_serv, servicio, [origen_ruta], top_paths=top_paths)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config=sankey_config())
            st.caption(
                "El ancho representa frecuencia de traceroutes, no RTT ni volumen real de tráfico. "
                "Las trazas incompletas terminan en un nodo separado; hop=255 no se interpreta como 255 saltos. "
                "FL-IX y NAP Colombia se conservan como nodos IXP cuando son observables."
            )
        else:
            st.info("No hay secuencias ASN/IXP suficientes para construir esta visualización.")
    else:
        st.info("No hay sondas nacionales disponibles para el servicio seleccionado.")

# -----------------------------------------------------------------------------
# 4. OONI
# -----------------------------------------------------------------------------
with tab_ooni:
    st.subheader("OONI Web Connectivity — evidencia complementaria")
    st.caption(
        "OONI se utiliza como evidencia complementaria a nivel de aplicación. Las categorías anomaly y failure "
        "no se interpretan automáticamente como censura, bloqueo o indisponibilidad completa del servicio."
    )

    if o_serv.empty:
        st.info("No hay datos OONI disponibles para este servicio.")
    else:
        totals = o_serv[["ok_count", "anomaly_count", "failure_count", "confirmed_count", "measurement_count"]].sum()
        total = int(totals["measurement_count"])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mediciones", f"{total:,}")
        c2.metric("OK", f"{int(totals['ok_count']):,}")
        c3.metric("Anomaly", f"{int(totals['anomaly_count']):,}")
        c4.metric("Failure", f"{int(totals['failure_count']):,}")
        c5.metric("Confirmed", f"{int(totals['confirmed_count']):,}")

        service_bar = pd.DataFrame(
            {
                "Estado": ["OK", "Anomaly", "Failure", "Confirmed"],
                "Mediciones": [
                    int(totals["ok_count"]),
                    int(totals["anomaly_count"]),
                    int(totals["failure_count"]),
                    int(totals["confirmed_count"]),
                ],
            }
        )
        fig = px.bar(
            service_bar,
            x="Estado",
            y="Mediciones",
            color="Estado",
            text_auto=True,
            color_discrete_map=OONI_COLOR_MAP,
            title=f"Resultados agregados de OONI — {servicio}",
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    st.markdown("### Comparación de los cuatro servicios")
    oo_sum = (
        df_o.groupby("sitio_web")[["ok_count", "anomaly_count", "failure_count", "confirmed_count"]]
        .sum().reindex(SERVICE_ORDER).reset_index()
    )
    long_oo = oo_sum.melt(
        id_vars="sitio_web",
        value_vars=["ok_count", "anomaly_count", "failure_count", "confirmed_count"],
        var_name="Estado",
        value_name="Mediciones",
    )
    long_oo["Estado"] = long_oo["Estado"].map(
        {"ok_count": "OK", "anomaly_count": "Anomaly", "failure_count": "Failure", "confirmed_count": "Confirmed"}
    )
    fig = px.bar(
        long_oo,
        x="sitio_web",
        y="Mediciones",
        color="Estado",
        barmode="stack",
        color_discrete_map=OONI_COLOR_MAP,
        category_orders={"Estado": ["OK", "Anomaly", "Failure", "Confirmed"]},
        labels={"sitio_web": "Servicio"},
        title="Resultados OONI agregados por servicio",
    )
    fig.update_layout(height=460, xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())
    st.caption("Los archivos OONI disponibles están agregados por día; no se realiza emparejamiento exacto 10:00 vs. 22:00.")

# -----------------------------------------------------------------------------
# 5. COBERTURA Y METODOLOGÍA
# -----------------------------------------------------------------------------
with tab_metodo:
    st.subheader("Cobertura efectiva de la campaña")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros Ping", f"{len(df_p):,}")
    c2.metric("Registros Traceroute", f"{len(df_t):,}")
    c3.metric("Sondas consideradas", f"{df_p['sonda_id'].nunique():,}")

    st.markdown("### Cobertura por sonda")
    st.dataframe(df_cov_s, use_container_width=True, hide_index=True)
    st.markdown("### Cobertura por servicio")
    st.dataframe(df_cov_d, use_container_width=True, hide_index=True)
    st.caption(
        "La incorporación tardía de la sonda ULA y la interrupción de cobertura de CANTV se tratan como variaciones de cobertura, "
        "no como pérdida de paquetes ni como falla del destino."
    )

    st.markdown("### Criterios de interpretación mostrados en el portal")
    criterios = pd.DataFrame(
        [
            ["RTT", "Cada ejecución Ping con respuestas se representa por su RTT promedio; los agregados se calculan sobre esas ejecuciones."],
            ["Pérdida agregada", "Se calcula como (Σsent − Σrcvd) / Σsent × 100 para Σsent > 0."],
            ["Alcanzabilidad ICMP", "Proporción de ejecuciones Ping válidas en las que se recibió al menos una respuesta."],
            ["Traceroute", "La llegada al destino se determina por evidencia de respuesta de la IP destino; una traza incompleta no se dibuja como completa."],
            ["hop=255", "Registro especial de RIPE Atlas; puede conservar evidencia de llegada, pero no se cuenta como un TTL ordinario ni como 255 saltos."],
            ["ASN / geografía", "La atribución se usa como apoyo interpretativo; la geolocalización IP no demuestra ubicación física exacta."],
            ["OONI", "Complementa la evidencia a nivel de aplicación; anomaly/failure no equivalen automáticamente a censura o indisponibilidad."],
        ],
        columns=["Elemento", "Criterio"],
    )
    st.dataframe(criterios, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Modo técnico opcional, oculto por defecto
# -----------------------------------------------------------------------------
if tab_tecnico is not None:
    with tab_tecnico:
        st.subheader("Inspección técnica interna")
        st.warning("Esta sección está desactivada en el modo de presentación para tutores y jurados.")
        st.write("Ping procesado")
        st.dataframe(p_serv, use_container_width=True, hide_index=True)
        st.write("Traceroute procesado")
        st.dataframe(t_serv, use_container_width=True, hide_index=True)

        if not t_serv.empty:
            sondas_trace = sorted(t_serv["sonda_nombre"].dropna().unique())
            sonda_trace = st.selectbox("Sonda", sondas_trace, key="trace_probe_tecnico")
            subset = t_serv[t_serv["sonda_nombre"] == sonda_trace].sort_values("timestamp", ascending=False)
            trace_id = st.selectbox("Ejecución Traceroute", subset["trace_id"].astype(str).tolist(), key="trace_exec_tecnico")
            hd = h_serv[h_serv["trace_id"].astype(str) == str(trace_id)].sort_values(["hop_num", "intento"])
            st.dataframe(hd, use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar Ping filtrado",
            p_serv.to_csv(index=False).encode("utf-8-sig"),
            file_name="ping_filtrado.csv",
            mime="text/csv",
        )
        st.download_button(
            "Descargar Traceroute filtrado",
            t_serv.to_csv(index=False).encode("utf-8-sig"),
            file_name="traceroute_filtrado.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "Portal de apoyo a la tesis. Los resultados mostrados proceden del mismo corpus procesado utilizado en el análisis académico; "
    "la aplicación facilita su consulta y visualización."
)
