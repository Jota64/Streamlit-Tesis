"""Portal de revisión para tutores del Trabajo de Grado ULA.

Mantiene la narrativa principal de la tesis y añade una capa acotada de
verificación técnica sobre los datos procesados. No incorpora análisis fuera
del alcance del trabajo ni realiza mediciones en tiempo real.
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
MODO_JURADO = False

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
    page_title="Revisión técnica de conectividad ULA",
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



def _service_short(servicio: str) -> str:
    return {
        "Biblioteca Digital / Vereda": "Vereda",
        "Intranet ULA": "Intranet",
        "Sistema de Grados": "Grados",
        "Saber ULA": "Saber",
    }.get(str(servicio), str(servicio))


def construir_paths_globales(
    df_traces: pd.DataFrame,
    df_hops: pd.DataFrame,
    servicios: list[str] | None = None,
    sondas: list[str] | None = None,
) -> pd.DataFrame:
    """Construye las rutas ASN/IXP de todos los servicios sin recortar a top-N."""
    tr = df_traces.copy()
    hp = df_hops.copy()
    if servicios:
        tr = tr[tr["sitio_web"].isin(servicios)]
        hp = hp[hp["sitio_web"].isin(servicios)]
    if sondas:
        tr = tr[tr["sonda_nombre"].isin(sondas)]
        hp = hp[hp["sonda_nombre"].isin(sondas)]
    if tr.empty:
        return pd.DataFrame()

    seqs = secuencias_asn_por_trace(hp)
    rows = []
    for _, row in tr.iterrows():
        servicio = str(row["sitio_web"])
        sonda = str(row["sonda_nombre"])
        origin_asn = _normalizar_asn(row.get("asn_origen")) or str(row.get("asn_origen", "Origen"))
        red_tokens = _compactar_path([origin_asn] + seqs.get(str(row["trace_id"]), []))
        reached = bool(row["respondio_destino"])
        terminal = f"Destino · {servicio}" if reached else f"Traza incompleta · {servicio}"
        visual_tokens = _compactar_path([f"Origen · {sonda}"] + red_tokens + [terminal])
        if len(visual_tokens) < 2:
            continue
        rows.append(
            {
                "trace_id": row["trace_id"],
                "sonda_nombre": sonda,
                "sitio_web": servicio,
                "respondio_destino": reached,
                "red_tokens": red_tokens,
                "visual_tokens": visual_tokens,
            }
        )
    return pd.DataFrame(rows)


def _edge_table_with_breakdown(paths: pd.DataFrame) -> pd.DataFrame:
    """Agrega transiciones y conserva un desglose por servicio para el hover."""
    if paths.empty:
        return pd.DataFrame()
    events = []
    for _, row in paths.iterrows():
        servicio = str(row["sitio_web"])
        estado = "Destino alcanzado" if bool(row["respondio_destino"]) else "Traza incompleta"
        tokens = row["visual_tokens"]
        for source, target in zip(tokens[:-1], tokens[1:]):
            events.append((source, target, servicio, estado))
    if not events:
        return pd.DataFrame()
    ev = pd.DataFrame(events, columns=["source", "target", "servicio", "estado"])
    totals = ev.groupby(["source", "target"], as_index=False).size().rename(columns={"size": "trazas"})
    breakdown = (
        ev.groupby(["source", "target", "servicio"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    detail_map = {}
    for (source, target), grp in breakdown.groupby(["source", "target"], sort=False):
        parts = [f"{_service_short(r.servicio)}: {int(r.n)}" for r in grp.itertuples(index=False)]
        detail_map[(source, target)] = " · ".join(parts)
    totals["desglose"] = [detail_map.get((r.source, r.target), "") for r in totals.itertuples(index=False)]
    return totals.sort_values("trazas", ascending=False)


def sankey_global_servicios(
    df_traces: pd.DataFrame,
    df_hops: pd.DataFrame,
    servicios: list[str],
    sondas: list[str],
):
    """Sankey global: todos los servicios y todas las familias observadas."""
    paths = construir_paths_globales(df_traces, df_hops, servicios, sondas)
    edges = _edge_table_with_breakdown(paths)
    if edges.empty:
        return None, paths, edges

    name_map = mapa_nombres_asn(df_hops)
    nodes_raw = list(dict.fromkeys(edges["source"].tolist() + edges["target"].tolist()))
    labels = [name_map.get(n, n) for n in nodes_raw]
    idx = {n: i for i, n in enumerate(nodes_raw)}

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=17,
                label=labels,
                customdata=nodes_raw,
                hovertemplate="%{label}<br>%{customdata}<extra></extra>",
            ),
            link=dict(
                source=edges["source"].map(idx),
                target=edges["target"].map(idx),
                value=edges["trazas"],
                customdata=edges["desglose"],
                hovertemplate=(
                    "%{source.label} → %{target.label}<br>"
                    "%{value} traceroutes<br>%{customdata}<extra></extra>"
                ),
            ),
        )
    )
    fig.update_layout(
        title="Rutas ASN/IXP observadas hacia los cuatro servicios · campaña completa",
        height=max(780, 27 * len(nodes_raw)),
        margin=dict(l=15, r=15, t=80, b=20),
        font=dict(size=11),
    )
    return fig, paths, edges


def rutas_entrada_global_df(
    df_traces: pd.DataFrame,
    df_hops: pd.DataFrame,
    servicios: list[str],
    sondas: list[str],
) -> tuple[pd.DataFrame, int, int]:
    """Resume SOLO rutas completadas, para los cuatro servicios simultáneamente."""
    paths = construir_paths_globales(df_traces, df_hops, servicios, sondas)
    if paths.empty:
        return pd.DataFrame(), 0, 0
    incompletas = int((~paths["respondio_destino"].astype(bool)).sum())
    reached = paths[paths["respondio_destino"].astype(bool)].copy()
    rows = []
    sin_atribucion = 0
    for _, row in reached.iterrows():
        seq = list(row["red_tokens"])
        if len(seq) < 2:
            sin_atribucion += 1
            continue
        rows.append(
            {
                "Sonda": row["sonda_nombre"],
                "Servicio": row["sitio_web"],
                "Penúltimo nodo atribuible": seq[-2],
                "Último nodo atribuible": seq[-1],
                "Destino alcanzado": True,
            }
        )
    if not rows:
        return pd.DataFrame(), incompletas, sin_atribucion
    detail = pd.DataFrame(rows)
    grouped = (
        detail.groupby(
            ["Sonda", "Servicio", "Penúltimo nodo atribuible", "Último nodo atribuible", "Destino alcanzado"],
            dropna=False,
        )
        .size()
        .reset_index(name="Traceroutes")
        .sort_values("Traceroutes", ascending=False)
    )
    return grouped, incompletas, sin_atribucion


def sankey_rutas_entrada_global(rutas: pd.DataFrame, df_hops: pd.DataFrame):
    """Sankey global de rutas de entrada usando exclusivamente trazas completadas."""
    if rutas.empty:
        return None
    name_map = mapa_nombres_asn(df_hops)
    edge_rows = []
    for _, r in rutas.iterrows():
        sonda = f"Origen · {r['Sonda']}"
        pen_raw = r["Penúltimo nodo atribuible"]
        ult_raw = r["Último nodo atribuible"]
        pen = name_map.get(pen_raw, pen_raw)
        ult = name_map.get(ult_raw, ult_raw)
        dest = f"Destino alcanzado · {r['Servicio']}"
        val = int(r["Traceroutes"])
        edge_rows.extend(
            [
                (sonda, f"Penúltimo · {pen}", val, r["Servicio"]),
                (f"Penúltimo · {pen}", f"Último · {ult}", val, r["Servicio"]),
                (f"Último · {ult}", dest, val, r["Servicio"]),
            ]
        )
    ev = pd.DataFrame(edge_rows, columns=["source", "target", "value", "servicio"])
    if ev.empty:
        return None
    edges = ev.groupby(["source", "target"], as_index=False)["value"].sum()
    breakdown = ev.groupby(["source", "target", "servicio"], as_index=False)["value"].sum()
    detail_map = {}
    for (source, target), grp in breakdown.groupby(["source", "target"], sort=False):
        detail_map[(source, target)] = " · ".join(
            f"{_service_short(r.servicio)}: {int(r.value)}" for r in grp.itertuples(index=False)
        )
    edges["desglose"] = [detail_map.get((r.source, r.target), "") for r in edges.itertuples(index=False)]

    nodes = list(dict.fromkeys(edges["source"].tolist() + edges["target"].tolist()))
    idx = {n: i for i, n in enumerate(nodes)}
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=nodes,
                pad=16,
                thickness=17,
                customdata=nodes,
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=edges["source"].map(idx),
                target=edges["target"].map(idx),
                value=edges["value"],
                customdata=edges["desglose"],
                hovertemplate=(
                    "%{source.label} → %{target.label}<br>"
                    "%{value} traceroutes completados<br>%{customdata}<extra></extra>"
                ),
            ),
        )
    )
    fig.update_layout(
        title="Rutas de entrada completadas hacia los cuatro servicios · ancho = frecuencia",
        height=max(760, 28 * len(nodes)),
        margin=dict(l=15, r=15, t=80, b=20),
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
st.title("Revisión de conectividad hacia servicios de la Universidad de Los Andes")
st.caption(
    "Versión para tutores · resultados de la tesis + verificación técnica acotada · "
    "RIPE Atlas Ping/Traceroute · OONI Web Connectivity · campaña junio–agosto de 2026"
)
st.info(
    "Esta aplicación utiliza el mismo corpus experimental cerrado de la tesis. La pestaña de verificación "
    "permite revisar registros procesados y trazas concretas sin añadir variables, hipótesis o análisis fuera del alcance del trabajo."
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
    nombres_tabs.append("Verificación técnica")
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
        key="sonda_temporal_tutores",
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

    st.markdown("### Vista global de los cuatro servicios")
    st.caption(
        "Esta vista integra simultáneamente los cuatro destinos institucionales y todas las familias de ruta "
        "ASN/IXP observables de los puntos seleccionados; no está limitada a un número de rutas principales. "
        "El filtro de servicio de la barra lateral no modifica esta visualización global."
    )
    sondas_globales_disp = [p for p in PROBE_ORDER if p in set(df_t["sonda_nombre"].dropna())]
    sondas_globales = st.multiselect(
        "Puntos de observación incluidos en las vistas globales",
        sondas_globales_disp,
        default=sondas_globales_disp,
        key="sondas_globales_tutores",
        help="Por defecto se representa la campaña completa desde las nueve sondas. Puede desmarcar una sonda para facilitar una explicación puntual.",
    )
    servicios_globales = [s for s in SERVICE_ORDER if s in set(df_t["sitio_web"].dropna())]

    if sondas_globales:
        fig_global, paths_global, edges_global = sankey_global_servicios(
            df_t, df_h, servicios_globales, sondas_globales
        )
        if fig_global is not None:
            st.plotly_chart(fig_global, use_container_width=True, config=sankey_config())
            completas_global = int(paths_global["respondio_destino"].astype(bool).sum()) if not paths_global.empty else 0
            st.caption(
                f"Se integran {len(paths_global):,} traceroutes de los cuatro servicios: "
                f"{completas_global:,} alcanzaron el destino y {len(paths_global) - completas_global:,} quedaron incompletos. "
                "El ancho representa frecuencia de traceroutes, no RTT ni volumen real de tráfico. "
                "Las trazas incompletas terminan en nodos separados por servicio; hop=255 no se interpreta como 255 saltos. "
                "FL-IX y NAP Colombia se conservan como nodos IXP cuando son observables."
            )
        else:
            st.info("No hay secuencias ASN/IXP suficientes para construir la vista global.")

        st.markdown("### Rutas de entrada completadas — cuatro servicios")
        st.caption(
            "Esta segunda vista fue construida exclusivamente con traceroutes que sí alcanzaron el destino. "
            "Resume el origen y los dos últimos nodos atribuibles (ASN o IXP) antes de cada servicio; "
            "no representa necesariamente puntos físicos de entrada a la ULA."
        )
        rutas_globales, incompletas_global, sin_atrib_global = rutas_entrada_global_df(
            df_t, df_h, servicios_globales, sondas_globales
        )
        fig_entrada = sankey_rutas_entrada_global(rutas_globales, df_h)
        if fig_entrada is not None:
            st.plotly_chart(fig_entrada, use_container_width=True, config=sankey_config())
            incluidas = int(rutas_globales["Traceroutes"].sum()) if not rutas_globales.empty else 0
            st.caption(
                f"Traceroutes completados representados: {incluidas:,}. "
                f"Trazas incompletas excluidas por definición: {incompletas_global:,}. "
                f"Trazas completadas sin dos nodos atribuibles suficientes para esta síntesis: {sin_atrib_global:,}. "
                "El ancho representa frecuencia de traceroutes completados."
            )
            with st.expander("Ver tabla resumida de rutas de entrada completadas"):
                st.dataframe(rutas_globales, use_container_width=True, hide_index=True)
        else:
            st.info("No hay rutas completadas con atribución suficiente para construir esta visualización.")
    else:
        st.info("Seleccione al menos un punto de observación para construir las vistas globales.")

    st.markdown("---")
    st.markdown("### Síntesis global de evidencia exterior en sondas nacionales")
    clasif = clasificacion_enrutamiento_nacional(df_t, df_h)
    st.dataframe(clasif, use_container_width=True, hide_index=True)
    st.caption(
        "'Sin marcador exterior observable' no demuestra que toda la trayectoria haya permanecido físicamente dentro de Venezuela. "
        "La clasificación es conservadora y corresponde a los criterios metodológicos de la tesis."
    )

    with st.expander(f"Detalle opcional del servicio seleccionado: {servicio}"):
        st.caption(
            "Esta vista conserva el análisis por servicio para responder preguntas puntuales. "
            "A diferencia de la vista global anterior, aquí puede seleccionarse un origen y limitar la presentación a las familias más frecuentes."
        )
        disponibles_nat = [p for p in NATIONAL_PROBES if p in set(t_serv["sonda_nombre"].dropna())]
        if disponibles_nat:
            default_probe = "Airtek (Maracaibo)" if "Airtek (Maracaibo)" in disponibles_nat else disponibles_nat[0]
            idx_default = disponibles_nat.index(default_probe)
            origen_ruta = st.selectbox(
                "Origen para visualizar en detalle",
                disponibles_nat,
                index=idx_default,
                key="origen_ruta_tutores",
            )
            t_route = t_serv[t_serv["sonda_nombre"] == origen_ruta].copy()
            n_reached = int(t_route["respondio_destino"].astype(bool).sum()) if len(t_route) else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Traceroutes", f"{len(t_route):,}")
            c2.metric("Destino alcanzado", f"{n_reached:,}")
            c3.metric("Alcanzabilidad Traceroute", fmt_num(pct(n_reached, len(t_route)), 2, "%"))

            top_paths = st.selectbox(
                "Familias más frecuentes a mostrar en el detalle",
                [5, 8, 12],
                index=1,
                key="top_paths_tutores",
            )
            fig = sankey_asn_limpio(t_serv, h_serv, servicio, [origen_ruta], top_paths=top_paths)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config=sankey_config())
                st.caption(
                    "Vista de apoyo por servicio. El ancho representa frecuencia de traceroutes y las trazas incompletas "
                    "no se dibujan como si hubieran alcanzado el destino."
                )
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
# 6. VERIFICACIÓN TÉCNICA PARA TUTORES
# -----------------------------------------------------------------------------
if tab_tecnico is not None:
    with tab_tecnico:
        st.subheader("Verificación técnica de los datos procesados")
        st.caption(
            "Esta sección permite comprobar cómo se obtienen los indicadores mostrados en la tesis. "
            "Se limita a los mismos datos y variables definidos en la metodología; no incorpora análisis exploratorios adicionales."
        )

        # Resumen de integridad del corpus.
        ping_validas = p_serv[pd.to_numeric(p_serv["sent"], errors="coerce") > 0].copy()
        sent0 = int((pd.to_numeric(p_serv["sent"], errors="coerce").fillna(0) == 0).sum())
        hop255 = int(
            h_serv["hop_especial_255"].astype(str).str.lower().eq("true").sum()
        ) if "hop_especial_255" in h_serv.columns else 0
        trazas_ok = int(t_serv["respondio_destino"].astype(bool).sum()) if len(t_serv) else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ping del servicio", f"{len(p_serv):,}")
        c2.metric("Ping válidas", f"{len(ping_validas):,}")
        c3.metric("Registros sent=0", f"{sent0:,}")
        c4.metric("Traceroutes al destino", f"{trazas_ok:,} / {len(t_serv):,}")
        st.caption(
            f"Registros especiales hop=255 presentes en el detalle del servicio: {hop255:,}. "
            "Estos registros no se contabilizan como 255 saltos."
        )

        st.markdown("### Recalcular indicadores Ping del servicio")
        s_check = estadisticas_ping(p_serv)
        check = pd.DataFrame(
            [
                ["Ejecuciones Ping válidas", int(s_check["n_validas"]), "sent > 0"],
                ["Alcanzabilidad ICMP (%)", round(float(s_check["alcanzabilidad"]), 4), "ejecuciones válidas con ≥1 respuesta / válidas"],
                ["Pérdida agregada (%)", round(float(s_check["perdida"]), 4), "(Σsent − Σrcvd) / Σsent × 100"],
                ["RTT mediano (ms)", round(float(s_check["mediana"]), 4) if pd.notna(s_check["mediana"]) else np.nan, "mediana del RTT promedio por ejecución"],
                ["RTT P95 (ms)", round(float(s_check["p95"]), 4) if pd.notna(s_check["p95"]) else np.nan, "P95 del RTT promedio por ejecución"],
            ],
            columns=["Indicador", "Valor recalculado", "Criterio"],
        )
        st.dataframe(check, use_container_width=True, hide_index=True)

        st.markdown("### Vista procesada de Ping")
        ping_cols = [
            c for c in [
                "timestamp", "sonda_id", "sonda_nombre", "asn_origen", "franja",
                "sent", "rcvd", "packet_loss_pct", "estado_ping", "alcanzable_icmp",
                "rtt_exec_ms",
            ] if c in p_serv.columns
        ]
        probe_ping = st.selectbox(
            "Sonda para revisar Ping",
            ["Todas"] + [x for x in PROBE_ORDER if x in set(p_serv["sonda_nombre"].dropna())],
            key="audit_ping_probe",
        )
        ping_view = p_serv if probe_ping == "Todas" else p_serv[p_serv["sonda_nombre"] == probe_ping]
        st.dataframe(ping_view[ping_cols].sort_values("timestamp", ascending=False).head(250), use_container_width=True, hide_index=True)
        st.caption("Se muestran como máximo 250 registros en pantalla; la descarga contiene el conjunto filtrado completo.")
        st.download_button(
            "Descargar Ping procesado del servicio",
            ping_view[ping_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ping_procesado_{servicio.replace(' ', '_').replace('/', '-')}.csv",
            mime="text/csv",
        )

        st.markdown("### Inspección controlada de una traza")
        if t_serv.empty:
            st.info("No hay Traceroute para el servicio seleccionado.")
        else:
            trace_probes = [x for x in PROBE_ORDER if x in set(t_serv["sonda_nombre"].dropna())]
            sonda_trace = st.selectbox("Sonda", trace_probes, key="audit_trace_probe")
            subset = t_serv[t_serv["sonda_nombre"] == sonda_trace].sort_values("timestamp", ascending=False)
            trace_options = subset["trace_id"].astype(str).tolist()
            stamp_map = subset.set_index(subset["trace_id"].astype(str))["timestamp"].to_dict()
            trace_id = st.selectbox(
                "Ejecución Traceroute",
                trace_options,
                format_func=lambda x: f"{stamp_map.get(x)} · {x}",
                key="audit_trace_exec",
            )
            trace_row = subset[subset["trace_id"].astype(str) == str(trace_id)].head(1)
            if not trace_row.empty:
                r = trace_row.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Destino alcanzado", "Sí" if bool(r.get("respondio_destino")) else "No")
                c2.metric("Último TTL ordinario respondiente", str(r.get("ultimo_hop_respondiente", "N/D")))
                c3.metric("Firma disponible", "Sí" if str(r.get("firma_ip", "")).strip() else "No")

            hd = h_serv[h_serv["trace_id"].astype(str) == str(trace_id)].sort_values(["hop_num", "intento"])
            hop_cols = [
                c for c in [
                    "hop_num", "hop_especial_255", "intento", "ip_hop", "rtt_hop_ms",
                    "timeout", "error", "es_destino", "asn_hop", "as_name", "pais", "ciudad",
                ] if c in hd.columns
            ]
            st.dataframe(hd[hop_cols], use_container_width=True, hide_index=True)
            st.caption(
                "La atribución ASN/geográfica es auxiliar. Una ciudad o país asociado a una IP no se interpreta como prueba de ubicación física exacta."
            )

        st.markdown("### Firmas de ruta más frecuentes")
        route_cols = [c for c in ["sonda_nombre", "firma_ip", "respondio_destino"] if c in t_serv.columns]
        if "firma_ip" in t_serv.columns:
            sig = (
                t_serv[t_serv["firma_ip"].fillna("").str.len() > 0]
                .groupby(["sonda_nombre", "firma_ip", "respondio_destino"], dropna=False)
                .size().reset_index(name="Frecuencia")
                .sort_values("Frecuencia", ascending=False)
            )
            st.dataframe(sig.head(50), use_container_width=True, hide_index=True)

        st.markdown("### Descargas de verificación")
        st.caption(
            "Se ofrecen únicamente conjuntos procesados utilizados por la tesis. Los JSON originales de RIPE Atlas se conservan como fuente primaria, "
            "pero no se exponen en esta interfaz para evitar mezclar datos crudos con los criterios de procesamiento documentados."
        )
        trace_cols = [
            c for c in [
                "timestamp", "trace_id", "sonda_id", "sonda_nombre", "asn_origen", "franja",
                "respondio_destino", "ultimo_hop_respondiente", "firma_ip",
            ] if c in t_serv.columns
        ]
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Descargar Traceroute procesado",
                t_serv[trace_cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"traceroute_procesado_{servicio.replace(' ', '_').replace('/', '-')}.csv",
                mime="text/csv",
            )
        with d2:
            hop_download_cols = [
                c for c in ["timestamp", "trace_id", "sonda_nombre", "hop_num", "hop_especial_255", "intento", "ip_hop", "rtt_hop_ms", "timeout", "error", "es_destino", "asn_hop", "as_name", "pais", "ciudad"]
                if c in h_serv.columns
            ]
            st.download_button(
                "Descargar detalle hop-by-hop procesado",
                h_serv[hop_download_cols].to_csv(index=False).encode("utf-8-sig"),
                file_name=f"hops_procesados_{servicio.replace(' ', '_').replace('/', '-')}.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption(
    "Portal de apoyo a la tesis. Los resultados mostrados proceden del mismo corpus procesado utilizado en el análisis académico; "
    "la aplicación facilita su consulta y visualización."
)
