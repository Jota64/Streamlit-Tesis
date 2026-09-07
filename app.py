"""Dashboard V3.6 de telemetría para la tesis ULA.

Objetivo de V3
--------------
Combinar la claridad visual del dashboard original (V1) con el procesamiento
metodológicamente corregido del pipeline V3. La app es exploratoria: no todos
los gráficos tienen que terminar en el Capítulo IV.

Preparación de datos:
    python procesar_datos_v3.py --base-dir . --out-dir salidas_tesis_v3

Para habilitar ASN / Sankey / rutas de entrada:
    python procesar_datos_v3.py --base-dir . --out-dir salidas_tesis_v3 --enriquecer

Ejecución:
    streamlit run app_v3.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import gzip
import io
import ipaddress
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT_CANDIDATES = [Path("salidas_tesis_v3"), Path("salidas_tesis")]

# Hitos externos documentados. Se usan exclusivamente como contexto temporal;
# la app NO atribuye causalidad a estos eventos.
SISMO_VE_2026 = pd.Timestamp("2026-06-24 18:04:31")
# Cirion comunicó la reconexión total el 23-Jul; al no publicarse aquí una hora
# operativa exacta, el período posterior comienza el 24-Jul a las 00:00.
RECONEXION_CABLE_FECHA = pd.Timestamp("2026-07-23 12:00:00")
INICIO_POST_RECONEXION = pd.Timestamp("2026-07-24 00:00:00")

EVENT_SOURCE_USGS = "https://www.usgs.gov/programs/landslide-hazards/science/2026-venezuela-sequence-earthquake-triggered-landslide-hazards"
EVENT_SOURCE_CIRION = "https://press.ciriontechnologies.com/2026/07/23/venezuela-vuelve-conectada-mundo/"

# Algunos LAN de IXP no poseen un ASN atribuible por IP. Si se eliminan por
# completo del flujo ASN, se oculta información topológica relevante.
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
SERVICE_SHORT = {
    "Biblioteca Digital / Vereda": "Vereda",
    "Intranet ULA": "Intranet",
    "Sistema de Grados": "Grados",
    "Saber ULA": "Saber",
}
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
HEATMAP_SCALE_GOOD = [
    [0.0, "#d73027"],
    [0.5, "#fee08b"],
    [1.0, "#1a9850"],
]
OONI_COLOR_MAP = {
    "ok_count": "#1a9850",
    "anomaly_count": "#f1c40f",
    "failure_count": "#d73027",
    "confirmed_count": "#7f0000",
    "OK": "#1a9850",
    "Anomaly": "#f1c40f",
    "Failure": "#d73027",
    "Confirmed": "#7f0000",
}
SESSION_COLOR_MAP = {"Todas": "#1a9850", "Parcial": "#f1c40f", "Ninguna": "#d73027"}
FRANJA_COLOR_MAP = {"Matutina (~10:00)": "#1a9850", "Nocturna (~22:00)": "#d73027"}
PERIODO_COLOR_MAP = {
    "Antes del sismo": "#7f8c8d",
    "Contingencia (24-Jun a 23-Jul)": "#f39c12",
    "Después de la reconexión": "#1a9850",
}


def _slug_archivo(valor: str) -> str:
    """Nombre seguro y legible para descargas de figuras."""
    txt = str(valor)
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N",
    }
    for a, b in reemplazos.items():
        txt = txt.replace(a, b)
    txt = re.sub(r"[^A-Za-z0-9._-]+", "_", txt)
    return re.sub(r"_+", "_", txt).strip("_")


def plotly_config(nombre: str) -> dict:
    """Configura el botón cámara de Plotly con un nombre descriptivo."""
    return {
        "displaylogo": False,
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": _slug_archivo(nombre),
            "scale": 2,
        },
    }

st.set_page_config(
    page_title="Telemetría ULA — Dashboard V3.6",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Telemetría de conectividad hacia servicios ULA — V3.6")
st.caption(
    "RIPE Atlas (Ping/Traceroute) + OONI · Hora local America/Caracas · "
    "Visualización exploratoria basada en el pipeline metodológico V3"
)


def _buscar_out_dir() -> Path:
    for candidate in OUT_CANDIDATES:
        if (candidate / "pings_ejecuciones.csv").exists():
            return candidate
    return OUT_CANDIDATES[0]


OUT = _buscar_out_dir()


@st.cache_data(show_spinner=False)
def load_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Carga CSV normal o su variante comprimida .csv.gz si existe."""
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
    # Para Streamlit/GitHub se usa primero el dataset compacto. Mantiene solo
    # respuestas útiles para rutas, ASN, TTL y perfiles; excluye timeouts redundantes.
    try:
        h = load_csv("traceroutes_hops_app", ["timestamp"])
    except FileNotFoundError:
        # Compatibilidad local con salidas antiguas/completas.
        h = load_csv("traceroutes_hops", ["timestamp"])
    o = load_csv("ooni_diario")
    c_sondas = load_csv(
        "tabla_3_4_cobertura_sondas", ["Inicio observado", "Fin observado"]
    )
    c_serv = load_csv(
        "tabla_3_5_cobertura_servicios", ["Primera medición", "Última medición"]
    )
    return p, t, h, o, c_sondas, c_serv



@st.cache_data(show_spinner=False)
def paquete_capitulo_iv() -> bytes:
    """Empaqueta seis CSV normales para el análisis del Capítulo IV.

    `traceroutes_hops_app` se guarda comprimido en el repositorio para no exceder
    límites de GitHub, pero se descomprime al crear el ZIP de descarga para que
    el usuario reciba seis archivos CSV normales.
    """
    nombres = [
        "pings_ejecuciones",
        "traceroutes_resumen",
        "traceroutes_hops_app",
        "ooni_diario",
        "tabla_3_4_cobertura_sondas",
        "tabla_3_5_cobertura_servicios",
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for nombre in nombres:
            csv_path = OUT / f"{nombre}.csv"
            gz_path = OUT / f"{nombre}.csv.gz"
            if csv_path.exists():
                zf.write(csv_path, arcname=f"{nombre}.csv")
            elif gz_path.exists():
                with gzip.open(gz_path, "rb") as fh:
                    contenido = fh.read()
                zf.writestr(f"{nombre}.csv", contenido)
            else:
                raise FileNotFoundError(
                    f"No se encontró {nombre}.csv ni {nombre}.csv.gz"
                )
    return buffer.getvalue()


def pct(n: float, d: float) -> float:
    return (100.0 * n / d) if d else np.nan


def short_service_name(servicio: str) -> str:
    return SERVICE_SHORT.get(servicio, servicio)


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
            "min": np.nan,
            "max": np.nan,
            "media": np.nan,
            "mediana": np.nan,
            "p95": np.nan,
            "std": np.nan,
        }
    valid = df[df["sent"] > 0]
    rtt = pd.to_numeric(valid["rtt_exec_ms"], errors="coerce").dropna()
    sent = pd.to_numeric(valid["sent"], errors="coerce").fillna(0).sum()
    rcvd = pd.to_numeric(valid["rcvd"], errors="coerce").fillna(0).sum()
    return {
        "n": len(df),
        "n_validas": len(valid),
        "n_con_respuesta": int((valid["rcvd"] > 0).sum()),
        "alcanzabilidad": pct((valid["rcvd"] > 0).sum(), len(valid)),
        "perdida": pct(sent - rcvd, sent),
        "min": rtt.min() if len(rtt) else np.nan,
        "max": rtt.max() if len(rtt) else np.nan,
        "media": rtt.mean() if len(rtt) else np.nan,
        "mediana": rtt.median() if len(rtt) else np.nan,
        "p95": rtt.quantile(0.95) if len(rtt) else np.nan,
        "std": rtt.std(ddof=1) if len(rtt) > 1 else np.nan,
    }


def resumen_sondas_ping(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, grp in df.groupby(
        ["sonda_id", "sonda_nombre", "asn_origen", "isp", "ubicacion", "tipo_sonda"],
        dropna=False,
    ):
        s = estadisticas_ping(grp)
        rows.append(
            {
                "Probe ID": int(keys[0]) if pd.notna(keys[0]) else np.nan,
                "Sonda": keys[1],
                "ASN": keys[2],
                "Proveedor (ISP)": keys[3],
                "Ubicación": keys[4],
                "Grupo": keys[5],
                "N": int(s["n"]),
                "N válidas": int(s["n_validas"]),
                "Alcanzabilidad (%)": s["alcanzabilidad"],
                "Pérdida (%)": s["perdida"],
                "RTT medio (ms)": s["media"],
                "RTT mediano (ms)": s["mediana"],
                "P95 (ms)": s["p95"],
                "Desv. estándar (ms)": s["std"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    numeric = [
        "Alcanzabilidad (%)",
        "Pérdida (%)",
        "RTT medio (ms)",
        "RTT mediano (ms)",
        "P95 (ms)",
        "Desv. estándar (ms)",
    ]
    out[numeric] = out[numeric].round(2)
    return out.sort_values(["Grupo", "RTT mediano (ms)"], na_position="last")


def periodo_contextual_2026(timestamp) -> str:
    """Clasificación temporal descriptiva para el evento sísmico/cable de 2026."""
    ts = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(ts):
        return "Sin fecha"
    if ts < SISMO_VE_2026:
        return "Antes del sismo"
    if ts < INICIO_POST_RECONEXION:
        return "Contingencia (24-Jun a 23-Jul)"
    return "Después de la reconexión"


def resumen_eventos_ping(df: pd.DataFrame) -> pd.DataFrame:
    """RTT/alcanzabilidad antes, durante y después del período contextual."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["Periodo contextual"] = work["timestamp"].map(periodo_contextual_2026)
    rows = []
    orden = [
        "Antes del sismo",
        "Contingencia (24-Jun a 23-Jul)",
        "Después de la reconexión",
    ]
    for (serv, sonda, periodo), grp in work.groupby(
        ["sitio_web", "sonda_nombre", "Periodo contextual"], dropna=False
    ):
        valid = grp[grp["sent"] > 0].copy()
        rtt = pd.to_numeric(valid["rtt_exec_ms"], errors="coerce").dropna()
        rows.append(
            {
                "Servicio": serv,
                "Sonda": sonda,
                "Periodo": periodo,
                "N mediciones": int(len(grp)),
                "N válidas": int(len(valid)),
                "Alcanzabilidad (%)": pct(int(valid["alcanzable_icmp"].astype(bool).sum()), len(valid)),
                "Pérdida media (%)": pd.to_numeric(valid["packet_loss_pct"], errors="coerce").mean(),
                "RTT mediano (ms)": rtt.median() if not rtt.empty else np.nan,
                "RTT P95 (ms)": rtt.quantile(0.95) if not rtt.empty else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_orden"] = pd.Categorical(out["Periodo"], categories=orden, ordered=True)
    return out.sort_values(["Servicio", "Sonda", "_orden"]).drop(columns="_orden")


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


def _anotar_eventos_contextuales(fig):
    """Añade hitos documentados sin convertirlos en explicación causal."""
    # add_shape evita algunos problemas de add_vline con timestamps + annotations.
    eventos = [
        (SISMO_VE_2026, "24-Jun · Sismos Mw 7,2/7,5", "dash"),
        (RECONEXION_CABLE_FECHA, "23-Jul · Reconexión reportada", "dot"),
    ]
    for x, label, dash in eventos:
        fig.add_shape(
            type="line",
            x0=x, x1=x, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(width=2, dash=dash),
        )
        fig.add_annotation(
            x=x, y=1.0, xref="x", yref="paper",
            text=label, showarrow=False,
            xanchor="left", yanchor="bottom",
            textangle=-90, font=dict(size=10),
        )
    return fig


def _normalizar_asn(valor) -> str | None:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    txt = str(valor).strip()
    if not txt or txt.lower() in {"nan", "none", "n/a"}:
        return None
    # Caso correcto: AS14522 o 14522
    m = re.fullmatch(r"AS\s*(\d+)", txt, flags=re.I)
    if m:
        return f"AS{m.group(1)}"
    if txt.isdigit():
        return f"AS{txt}"
    # Compatibilidad con V2: AS{'asn': 14522, 'holder': '...'}
    m = re.search(r"['\"]asn['\"]\s*:\s*(\d+)", txt, flags=re.I)
    if m:
        return f"AS{m.group(1)}"
    # Último recurso: primer AS seguido de dígitos
    m = re.search(r"\bAS\s*(\d+)\b", txt, flags=re.I)
    if m:
        return f"AS{m.group(1)}"
    return None


def _holder_desde_valor_asn(valor) -> str | None:
    if valor is None:
        return None
    txt = str(valor)
    m = re.search(r"['\"]holder['\"]\s*:\s*['\"]([^'\"]+)", txt, flags=re.I)
    return m.group(1).strip() if m else None


def _limpiar_nombre_as(nombre: str | None, asn: str | None = None, max_len: int = 34) -> str | None:
    if nombre is None or (isinstance(nombre, float) and np.isnan(nombre)):
        return None
    txt = re.sub(r"\s+", " ", str(nombre)).strip()
    if not txt or txt.lower() in {"nan", "none", "desconocido"}:
        return None
    if asn:
        txt = re.sub(rf"^{re.escape(asn)}\s*[-–:]?\s*", "", txt, flags=re.I)
    if len(txt) > max_len:
        txt = txt[: max_len - 1].rstrip() + "…"
    return txt


def mapa_nombres_asn(df_hops: pd.DataFrame) -> dict[str, str]:
    """ASN -> etiqueta corta y legible."""
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
    """Secuencia de nodos atribuibles por traza.

    Prioriza un marcador IXP conocido cuando la IP pertenece a un LAN de
    intercambio, incluso si esa IP no tiene ASN atribuible. En el resto de TTL
    utiliza el ASN modal, manteniendo el detalle intento-a-intento intacto.
    """
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
                "firma_asn_limpia": " > ".join(path),
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
    """Sankey de rutas observadas sin fingir llegada al destino.

    Las trazas completas terminan en el destino. Las incompletas terminan en un
    nodo explícito de último punto observado. Los LAN de IXP conocidos se
    conservan aunque no tengan ASN atribuible.
    """
    paths = construir_paths_asn(df_traces, df_hops, servicio, sondas)
    if paths.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    counts = (
        paths.groupby(["firma_visual", "estado_traza"], dropna=False)
        .size()
        .reset_index(name="trazas")
        .sort_values("trazas", ascending=False)
    )
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
        return None, counts, pd.DataFrame()

    e = (
        pd.DataFrame(edges, columns=["source", "target"])
        .value_counts()
        .reset_index(name="trazas")
        .sort_values("trazas", ascending=False)
    )
    name_map = mapa_nombres_asn(df_hops)
    nodes_raw = list(dict.fromkeys(e["source"].tolist() + e["target"].tolist()))
    labels = [name_map.get(n, n) for n in nodes_raw]
    idx = {n: i for i, n in enumerate(nodes_raw)}

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18, thickness=18, label=labels, customdata=nodes_raw,
                hovertemplate="%{label}<br>%{customdata}<extra></extra>",
            ),
            link=dict(
                source=e["source"].map(idx),
                target=e["target"].map(idx),
                value=e["trazas"],
                customdata=e["trazas"],
                hovertemplate="%{source.label} → %{target.label}<br>%{value} traceroutes<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=(
            f"Rutas ASN/IXP más frecuentes hacia {servicio} · "
            "trazas incompletas separadas del destino"
        ),
        height=max(620, 40 * len(nodes_raw)),
        margin=dict(l=20, r=20, t=80, b=20),
        font=dict(size=11),
    )
    return fig, counts, e


def rutas_entrada_df(
    df_traces: pd.DataFrame,
    df_hops: pd.DataFrame,
    servicio: str,
    sondas: list[str],
) -> tuple[pd.DataFrame, int]:
    """Rutas de entrada calculadas SOLO sobre traceroutes que alcanzaron destino."""
    paths_all = construir_paths_asn(df_traces, df_hops, servicio, sondas)
    if paths_all.empty:
        return pd.DataFrame(), 0
    incompletas = int((~paths_all["respondio_destino"].astype(bool)).sum())
    paths = paths_all[paths_all["respondio_destino"].astype(bool)].copy()
    if paths.empty:
        return pd.DataFrame(), incompletas

    rows = []
    omitidas = incompletas
    for _, row in paths.iterrows():
        seq = row["path_tokens"]
        if len(seq) < 2:
            omitidas += 1
            continue
        ultimo = seq[-1]
        penultimo = seq[-2]
        rows.append(
            {
                "Sonda": row["sonda_nombre"],
                "ASN origen": seq[0],
                "Penúltimo nodo atribuible": penultimo,
                "Último nodo atribuible": ultimo,
                "Destino alcanzado": True,
            }
        )
    if not rows:
        return pd.DataFrame(), omitidas
    detail = pd.DataFrame(rows)
    grouped = (
        detail.groupby(
            [
                "Sonda", "ASN origen", "Penúltimo nodo atribuible",
                "Último nodo atribuible", "Destino alcanzado"
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="Traceroutes")
        .sort_values("Traceroutes", ascending=False)
    )
    return grouped, omitidas


def sankey_rutas_entrada(
    rutas: pd.DataFrame,
    df_hops: pd.DataFrame,
    servicio: str,
    top_rows: int = 20,
):
    if rutas.empty:
        return None
    # Por construcción todas estas filas corresponden a destino alcanzado.
    work = rutas[rutas["Destino alcanzado"].astype(bool)].head(top_rows).copy()
    if work.empty:
        return None
    name_map = mapa_nombres_asn(df_hops)
    edges = []
    for _, r in work.iterrows():
        sonda = f"Origen · {r['Sonda']}"
        pen_raw = r["Penúltimo nodo atribuible"]
        ult_raw = r["Último nodo atribuible"]
        pen = name_map.get(pen_raw, pen_raw)
        ult = name_map.get(ult_raw, ult_raw)
        dest = f"Destino alcanzado · {servicio}"
        val = int(r["Traceroutes"])
        edges.extend([
            (sonda, f"Penúltimo · {pen}", val),
            (f"Penúltimo · {pen}", f"Último · {ult}", val),
            (f"Último · {ult}", dest, val),
        ])
    e = pd.DataFrame(edges, columns=["source", "target", "value"])
    e = e.groupby(["source", "target"], as_index=False)["value"].sum()
    nodes = list(dict.fromkeys(e["source"].tolist() + e["target"].tolist()))
    idx = {n: i for i, n in enumerate(nodes)}
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=nodes, pad=18, thickness=18),
            link=dict(
                source=e["source"].map(idx),
                target=e["target"].map(idx),
                value=e["value"],
                hovertemplate="%{source.label} → %{target.label}<br>%{value} traceroutes<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=f"Rutas de entrada completadas hacia {servicio} · ancho = frecuencia",
        height=max(600, 35 * len(nodes)),
        margin=dict(l=20, r=20, t=75, b=20),
        font=dict(size=11),
    )
    return fig


def ttl_destino_ordinario(df_hops: pd.DataFrame) -> pd.DataFrame:
    """Un TTL de destino solo cuando la IP destino respondió en un hop ordinario."""
    if df_hops.empty:
        return pd.DataFrame(columns=["trace_id", "ttl_destino"])
    work = df_hops[
        (df_hops["es_destino"].astype(str).str.lower() == "true")
        & (df_hops["hop_especial_255"].astype(str).str.lower() != "true")
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=["trace_id", "ttl_destino"])
    return work.groupby("trace_id", as_index=False)["hop_num"].min().rename(columns={"hop_num": "ttl_destino"})


def dominante_por_sonda(df_traces: pd.DataFrame) -> pd.DataFrame:
    valid = df_traces[df_traces["firma_ip"].fillna("").str.len() > 0]
    rows = []
    for sonda, grp in valid.groupby("sonda_nombre"):
        counts = grp["firma_ip"].value_counts()
        if counts.empty:
            continue
        firma = counts.index[0]
        n = int(counts.iloc[0])
        rows.append(
            {
                "Sonda": sonda,
                "Firma IP dominante": firma,
                "Frecuencia": n,
                "% de trazas con firma": pct(n, len(grp)),
                "Total trazas con firma": len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values("Frecuencia", ascending=False) if rows else pd.DataFrame()


def perfil_ruta_dominante(df_traces: pd.DataFrame, df_hops: pd.DataFrame) -> pd.DataFrame:
    """RTT mediano por TTL restringido a la firma IP dominante de cada sonda."""
    rows = []
    valid = df_traces[df_traces["firma_ip"].fillna("").str.len() > 0]
    for sonda, grp in valid.groupby("sonda_nombre"):
        counts = grp["firma_ip"].value_counts()
        if counts.empty:
            continue
        firma = counts.index[0]
        trace_ids = set(grp.loc[grp["firma_ip"] == firma, "trace_id"].astype(str))
        hd = df_hops[
            (df_hops["trace_id"].astype(str).isin(trace_ids))
            & (df_hops["hop_especial_255"].astype(str).str.lower() != "true")
            & df_hops["rtt_hop_ms"].notna()
        ].copy()
        if hd.empty:
            continue
        agg = hd.groupby("hop_num")["rtt_hop_ms"].median().reset_index()
        agg["Sonda"] = sonda
        agg["N trazas ruta dominante"] = len(trace_ids)
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


try:
    df_p, df_t, df_h, df_o, df_cov_s, df_cov_d = cargar_datos()
except FileNotFoundError as e:
    st.error(
        f"Falta {e.filename}. Ejecuta primero `python procesar_datos_v3.py --base-dir . "
        "--out-dir salidas_tesis_v3`."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Filtros globales
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Filtros del análisis")
st.sidebar.caption(f"Directorio cargado: `{OUT}`")
servicios = sorted(df_p["sitio_web"].dropna().unique())
servicio = st.sidebar.selectbox("Servicio institucional", servicios)
tipo = st.sidebar.radio("Grupo geográfico", ["Todos", "Nacional", "Internacional"])
franja = st.sidebar.selectbox("Franja temporal", ["Todas", "Matutina (~10:00)", "Nocturna (~22:00)"])

p_base = df_p[df_p["sitio_web"] == servicio].copy()
t_base = df_t[df_t["sitio_web"] == servicio].copy()
h_base = df_h[df_h["sitio_web"] == servicio].copy()

if tipo != "Todos":
    p_base = p_base[p_base["tipo_sonda"] == tipo]
    t_base = t_base[t_base["tipo_sonda"] == tipo]
    h_base = h_base[h_base["tipo_sonda"] == tipo]
if franja != "Todas":
    p_base = p_base[p_base["franja"] == franja]
    t_base = t_base[t_base["franja"] == franja]
    h_base = h_base[h_base["franja"] == franja]

sondas_disp = sorted(set(p_base["sonda_nombre"].dropna()) | set(t_base["sonda_nombre"].dropna()))
sondas_sel = st.sidebar.multiselect("Sondas", sondas_disp, default=sondas_disp)

p = p_base[p_base["sonda_nombre"].isin(sondas_sel)].copy()
t = t_base[t_base["sonda_nombre"].isin(sondas_sel)].copy()
h = h_base[h_base["sonda_nombre"].isin(sondas_sel)].copy()

# Versiones globales para las figuras comparativas del Capítulo IV.
p_all = df_p.copy()
t_all = df_t.copy()
o_all = df_o.copy()
if tipo != "Todos":
    p_all = p_all[p_all["tipo_sonda"] == tipo]
    t_all = t_all[t_all["tipo_sonda"] == tipo]
if franja != "Todas":
    p_all = p_all[p_all["franja"] == franja]
    t_all = t_all[t_all["franja"] == franja]
p_all = p_all[p_all["sonda_nombre"].isin(sondas_sel)].copy()
t_all = t_all[t_all["sonda_nombre"].isin(sondas_sel)].copy()

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
stats = estadisticas_ping(p)
trace_reach = pct(t["respondio_destino"].astype(bool).sum(), len(t)) if len(t) else np.nan
cols = st.columns(6)
cols[0].metric("Ejecuciones Ping", f"{int(stats['n']):,}")
cols[1].metric("Alcanzabilidad ICMP", fmt_num(stats["alcanzabilidad"], 1, "%"))
cols[2].metric("Pérdida agregada", fmt_num(stats["perdida"], 1, "%"))
cols[3].metric("RTT mediano", fmt_num(stats["mediana"], 1, " ms"))
cols[4].metric("RTT P95", fmt_num(stats["p95"], 1, " ms"))
cols[5].metric("Traceroute alcanza destino", fmt_num(trace_reach, 1, "%"))

st.caption(
    "La alcanzabilidad y pérdida se calculan únicamente sobre mediciones Ping válidas. "
    "Los registros sin transmisión efectiva no se tratan como 100 % de pérdida."
)

(
    tab_resumen,
    tab_rtt,
    tab_alc,
    tab_trace,
    tab_rutas,
    tab_ooni,
    tab_comp,
    tab_cov,
    tab_raw,
) = st.tabs(
    [
        "Resumen por sonda",
        "RTT y tiempo",
        "Alcanzabilidad",
        "Traceroute",
        "Rutas / ASN",
        "OONI",
        "Comparativo Cap. IV",
        "Cobertura",
        "Auditoría",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1 - RESUMEN
# -----------------------------------------------------------------------------
with tab_resumen:
    st.subheader(f"📌 Estado por origen / probe hacia {servicio}")
    resumen = resumen_sondas_ping(p)
    if resumen.empty:
        st.info("No hay mediciones Ping para los filtros seleccionados.")
    else:
        st.dataframe(
            resumen.drop(columns=["Grupo"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Comparación de RTT por sonda")
        metric_map = {
            "Mediana": "RTT mediano (ms)",
            "Media": "RTT medio (ms)",
            "P95": "P95 (ms)",
        }
        met = st.radio(
            "Estadístico para comparar",
            list(metric_map),
            horizontal=True,
            key="resumen_stat",
        )
        col = metric_map[met]
        plot = resumen.sort_values(col)
        fig = px.bar(
            plot,
            x="Sonda",
            y=col,
            color="Grupo",
            text_auto=".1f",
            labels={col: f"RTT {met.lower()} (ms)"},
        )
        fig.update_traces(texttemplate="%{y:.1f} ms", textposition="outside")
        fig.update_layout(height=470, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_04_RTT_por_sonda"))

        st.markdown("#### Nacionales vs. internacionales")
        st.caption(
            "Se muestra la distribución de ejecuciones válidas, no solo un promedio agregado. "
            "La sonda ULA puede separarse porque su proximidad al destino altera fuertemente el grupo nacional."
        )
        include_ula = st.checkbox("Incluir ULA (Mérida) en la comparación por grupo", value=False)
        comp = p[p["rtt_exec_ms"].notna()].copy()
        if not include_ula:
            comp = comp[comp["sonda_nombre"] != "ULA (Mérida)"]
        if not comp.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.box(
                    comp,
                    x="tipo_sonda",
                    y="rtt_exec_ms",
                    points=False,
                    labels={"tipo_sonda": "Grupo", "rtt_exec_ms": "RTT por ejecución (ms)"},
                    title="Distribución por grupo",
                )
                st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_05_Nacional_vs_Internacional_distribucion"))
            with c2:
                grp = (
                    comp.groupby("tipo_sonda")["rtt_exec_ms"]
                    .agg(["median", "mean", lambda x: x.quantile(0.95)])
                    .reset_index()
                )
                grp.columns = ["Grupo", "Mediana", "Media", "P95"]
                long = grp.melt("Grupo", var_name="Estadístico", value_name="RTT (ms)")
                fig = px.bar(
                    long,
                    x="Grupo",
                    y="RTT (ms)",
                    color="Estadístico",
                    barmode="group",
                    text_auto=".1f",
                    title="Mediana, media y P95 por grupo",
                )
                st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_05_Nacional_vs_Internacional_estadisticos"))

# -----------------------------------------------------------------------------
# TAB 2 - RTT Y TIEMPO
# -----------------------------------------------------------------------------
with tab_rtt:
    valid = p[p["rtt_exec_ms"].notna()].copy()
    if valid.empty:
        st.info("No hay RTT válidos para los filtros seleccionados.")
    else:
        st.subheader(f"📈 Serie temporal de RTT hacia {servicio}")
        modo = st.selectbox(
            "Representación temporal",
            ["RTT por ejecución", "Mediana diaria", "Media diaria", "P95 diario"],
        )
        if modo == "RTT por ejecución":
            ts_df = valid.sort_values(["sonda_nombre", "timestamp"]).copy()
            y_col = "rtt_exec_ms"
            y_label = "RTT medio por ejecución (ms)"
            hover = ["hora", "franja", "asn_origen", "ubicacion", "packet_loss_pct"]
        else:
            valid["fecha_dt"] = pd.to_datetime(valid["fecha"])
            if modo == "Mediana diaria":
                ts_df = valid.groupby(["fecha_dt", "sonda_nombre"], as_index=False)["rtt_exec_ms"].median()
            elif modo == "Media diaria":
                ts_df = valid.groupby(["fecha_dt", "sonda_nombre"], as_index=False)["rtt_exec_ms"].mean()
            else:
                ts_df = valid.groupby(["fecha_dt", "sonda_nombre"], as_index=False)["rtt_exec_ms"].quantile(0.95)
            ts_df = ts_df.rename(columns={"fecha_dt": "timestamp"})
            y_col = "rtt_exec_ms"
            y_label = f"{modo} (ms)"
            hover = []

        fig = px.line(
            ts_df,
            x="timestamp",
            y=y_col,
            color="sonda_nombre",
            line_group="sonda_nombre",
            markers=True,
            hover_data=hover,
            labels={"timestamp": "Fecha / hora local", y_col: y_label, "sonda_nombre": "Sonda / ISP"},
        )
        fig.update_traces(line=dict(width=2.2), marker=dict(size=5), connectgaps=False)
        fig.update_layout(
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1S", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(step="all", label="Todo"),
                ]
            )
        )
        mostrar_hitos = st.checkbox(
            "Mostrar hitos externos documentados del 24-Jun y 23-Jul", value=True
        )
        if mostrar_hitos:
            _anotar_eventos_contextuales(fig)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_02_Serie_temporal_RTT"))
        st.caption(
            "Las líneas verticales, cuando están activas, son referencias temporales documentadas y NO implican causalidad. "
            "24-Jun: secuencia sísmica; 23-Jul: Cirion comunicó la reconexión total del cable afectado."
        )

        with st.expander("🌎 Análisis contextual: antes · contingencia · después", expanded=False):
            st.markdown(
                f"Fuentes externas: [USGS — sismos del 24-Jun]({EVENT_SOURCE_USGS}) · "
                f"[Cirion — reconexión reportada el 23-Jul]({EVENT_SOURCE_CIRION}). "
                "La comparación es descriptiva y no atribuye los cambios de RTT o alcanzabilidad al evento."
            )
            ev = resumen_eventos_ping(p)
            if ev.empty:
                st.info("No hay datos suficientes para este análisis contextual.")
            else:
                ev_serv = ev[ev["Servicio"] == servicio].copy()
                cols_show = [
                    "Sonda", "Periodo", "N mediciones", "N válidas",
                    "Alcanzabilidad (%)", "Pérdida media (%)",
                    "RTT mediano (ms)", "RTT P95 (ms)"
                ]
                st.dataframe(
                    ev_serv[cols_show].round(2), use_container_width=True, hide_index=True
                )
                ev_rtt = ev_serv[ev_serv["RTT mediano (ms)"].notna()].copy()
                if not ev_rtt.empty:
                    fig_ev = px.bar(
                        ev_rtt, x="Sonda", y="RTT mediano (ms)", color="Periodo",
                        barmode="group", text_auto=".1f",
                        title=f"RTT mediano antes, durante y después de la contingencia · {servicio}",
                    )
                    fig_ev.update_layout(height=520, xaxis_tickangle=-35)
                    st.plotly_chart(
                        fig_ev, use_container_width=True,
                        config=plotly_config(f"{servicio}_Evento_2026_RTT_antes_durante_despues")
                    )
                st.download_button(
                    "⬇️ Descargar tabla del análisis contextual de este servicio",
                    data=ev_serv[cols_show].to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{_slug_archivo(servicio)}_evento_2026_antes_contingencia_despues.csv",
                    mime="text/csv",
                )

        st.subheader("🔥 Mapa de calor diario por sonda")
        heat_stat = st.radio("Estadístico del mapa", ["Mediana", "Media", "P95"], horizontal=True)
        work = valid.copy()
        if heat_stat == "Mediana":
            heat = work.groupby(["fecha", "sonda_nombre"])["rtt_exec_ms"].median().reset_index()
        elif heat_stat == "Media":
            heat = work.groupby(["fecha", "sonda_nombre"])["rtt_exec_ms"].mean().reset_index()
        else:
            heat = work.groupby(["fecha", "sonda_nombre"])["rtt_exec_ms"].quantile(0.95).reset_index()
        pivot = heat.pivot(index="sonda_nombre", columns="fecha", values="rtt_exec_ms")
        if not pivot.empty:
            fig = px.imshow(
                pivot,
                labels={"x": "Fecha", "y": "Sonda / proveedor", "color": f"RTT {heat_stat.lower()} (ms)"},
                aspect="auto",
                color_continuous_scale="RdYlGn_r",
                height=460,
            )
            fig.update_xaxes(side="bottom")
            st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_03_Mapa_calor_RTT"))
            st.caption("Los espacios sin color representan ausencia de un RTT válido para esa combinación sonda-día.")

        st.subheader("Distribución de RTT por sonda")
        fig = px.box(
            valid,
            x="sonda_nombre",
            y="rtt_exec_ms",
            points=False,
            labels={"sonda_nombre": "Sonda", "rtt_exec_ms": "RTT por ejecución (ms)"},
        )
        fig.update_layout(height=500, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_04_Distribucion_RTT_por_sonda"))

# -----------------------------------------------------------------------------
# TAB 3 - ALCANZABILIDAD
# -----------------------------------------------------------------------------
with tab_alc:
    st.subheader("📶 Alcanzabilidad ICMP y estados de respuesta")
    if p.empty:
        st.info("No hay Ping para los filtros seleccionados.")
    else:
        states = (
            p.groupby(["sonda_nombre", "estado_ping"])
            .size()
            .reset_index(name="Mediciones")
        )
        fig = px.bar(
            states,
            x="sonda_nombre",
            y="Mediciones",
            color="estado_ping",
            barmode="stack",
            labels={"sonda_nombre": "Sonda", "estado_ping": "Estado"},
        )
        fig.update_layout(height=500, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_06_Estados_Ping"))

        st.markdown("#### Alcanzabilidad y pérdida por sonda")
        resumen = resumen_sondas_ping(p)
        if not resumen.empty:
            long = resumen.melt(
                id_vars=["Sonda"],
                value_vars=["Alcanzabilidad (%)", "Pérdida (%)"],
                var_name="Indicador",
                value_name="Porcentaje",
            )
            fig = px.bar(
                long,
                x="Sonda",
                y="Porcentaje",
                color="Indicador",
                barmode="group",
                text_auto=".1f",
            )
            fig.update_layout(height=480, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_06_Alcanzabilidad_y_perdida"))

        st.markdown("#### Comparación entre franjas")
        fr = []
        src = df_p[(df_p["sitio_web"] == servicio) & (df_p["sonda_nombre"].isin(sondas_sel))].copy()
        if tipo != "Todos":
            src = src[src["tipo_sonda"] == tipo]
        for f, grp in src.groupby("franja"):
            if f == "Fuera de franja":
                continue
            s = estadisticas_ping(grp)
            fr.append({"Franja": f, "Alcanzabilidad (%)": s["alcanzabilidad"], "Pérdida (%)": s["perdida"], "RTT mediano (ms)": s["mediana"]})
        if fr:
            st.dataframe(pd.DataFrame(fr).round(2), hide_index=True, use_container_width=True)
            st.caption("La comparación temporal es descriptiva; no implica por sí sola diferencias de carga de red.")

# -----------------------------------------------------------------------------
# TAB 4 - TRACEROUTE
# -----------------------------------------------------------------------------
with tab_trace:
    st.subheader("🗺️ Alcance del destino y saltos observados")
    if t.empty:
        st.info("No hay traceroutes para los filtros seleccionados.")
    else:
        n_reached = int(t["respondio_destino"].astype(bool).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Traceroutes", f"{len(t):,}")
        c2.metric("Destino alcanzado", f"{n_reached:,} / {len(t):,}")
        c3.metric("Porcentaje alcanzado", fmt_num(pct(n_reached, len(t)), 1, "%"))

        # TTL del destino solo cuando el destino respondió en un hop ordinario.
        ttls = ttl_destino_ordinario(h)
        if not ttls.empty:
            t_ttl = t[["trace_id", "sonda_nombre"]].merge(ttls, on="trace_id", how="inner")
            ttl_summary = t_ttl.groupby("sonda_nombre")["ttl_destino"].agg(["count", "median", "mean"]).reset_index()
            ttl_summary.columns = ["Sonda", "N con TTL destino observable", "TTL mediano destino", "TTL medio destino"]
            st.markdown("#### Saltos hasta el destino cuando el TTL final es observable")
            fig = px.bar(
                ttl_summary,
                x="Sonda",
                y="TTL mediano destino",
                text_auto=".1f",
                labels={"TTL mediano destino": "TTL mediano del destino (hops)"},
            )
            fig.update_layout(height=460, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_07_Saltos_hasta_destino"))
            st.caption(
                "Se excluyen de esta métrica los casos donde RIPE Atlas confirma el destino mediante el registro especial hop=255, porque ese valor no representa 255 saltos."
            )

        st.markdown("#### Distribución del último TTL ordinario respondiente")
        if t["ultimo_hop_respondiente"].notna().any():
            fig = px.histogram(
                t,
                x="ultimo_hop_respondiente",
                color="sonda_nombre",
                barmode="overlay",
                opacity=0.65,
                labels={"ultimo_hop_respondiente": "Último TTL ordinario respondiente", "sonda_nombre": "Sonda"},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_07_Ultimo_TTL_respondiente"))

        st.markdown("#### Rutas IP más frecuentes")
        route_counts = (
            t[t["firma_ip"].fillna("").str.len() > 0]
            .groupby(["sonda_nombre", "firma_ip"])
            .size()
            .reset_index(name="Frecuencia")
            .sort_values("Frecuencia", ascending=False)
        )
        st.dataframe(route_counts.head(40), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 5 - RUTAS / ASN
# -----------------------------------------------------------------------------
with tab_rutas:
    st.subheader("🔎 Rutas dominantes y perfil salto a salto")
    dom = dominante_por_sonda(t)
    if dom.empty:
        st.info("No hay firmas IP suficientes para construir rutas dominantes.")
    else:
        dom_display = dom.copy()
        dom_display["% de trazas con firma"] = dom_display["% de trazas con firma"].round(1)
        st.dataframe(dom_display, use_container_width=True, hide_index=True)

        perfil = perfil_ruta_dominante(t, h)
        if not perfil.empty:
            fig = px.line(
                perfil,
                x="hop_num",
                y="rtt_hop_ms",
                color="Sonda",
                markers=True,
                labels={"hop_num": "Número de salto (TTL)", "rtt_hop_ms": "RTT mediano del hop (ms)"},
                title=f"Perfil de RTT de la ruta IP dominante hacia {servicio}",
            )
            fig.update_layout(height=560, legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
            st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_09_Perfil_ruta_dominante"))
            st.caption(
                "A diferencia del gráfico V1, cada línea se restringe a la firma IP dominante de esa sonda; no mezcla indiscriminadamente hops de rutas diferentes."
            )

    st.markdown("---")
    st.subheader("🌊 Sankey de rutas ASN / IXP")
    asn_available = "asn_hop" in h.columns and h["asn_hop"].map(_normalizar_asn).notna().any()
    if not asn_available:
        st.warning(
            "No hay atribución ASN utilizable en los hops. Ejecuta `python procesar_datos_v3.py --base-dir . --out-dir salidas_tesis_v3 --enriquecer` y recarga."
        )
    else:
        top_n = st.slider("Número máximo de firmas ASN principales", 3, 15, 8)
        fig, paths_counts, edges = sankey_asn_limpio(t, h, servicio, sondas_sel, top_n)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_10_Sankey_ASN"))
            st.caption(
                "El ancho representa frecuencia de traceroutes. Las trazas incompletas terminan en un nodo separado y nunca se dibujan como si hubieran alcanzado el destino. Los LAN de IXP conocidos se conservan aunque no tengan ASN."
            )
            with st.expander("Ver firmas ASN más frecuentes"):
                st.dataframe(paths_counts.head(30), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader(f"🔀 Rutas de entrada observadas hacia {servicio}")
        rutas, omitidas = rutas_entrada_df(t, h, servicio, sondas_sel)
        if rutas.empty:
            st.info("No hay suficientes ASN atribuibles para construir rutas de entrada.")
        else:
            top_rows = st.slider("Máximo de combinaciones de entrada a mostrar", 5, 40, 20)
            fig = sankey_rutas_entrada(rutas, h, servicio, top_rows)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_11_Rutas_de_entrada"))
            st.caption(
                "'Penúltimo' y 'último' se refieren a los últimos nodos atribuibles (ASN o IXP) observados en traceroutes que SÍ alcanzaron el destino; no representan necesariamente puntos físicos de entrada. "
                f"Trazas sin secuencia ASN suficiente omitidas de esta visualización: {omitidas}."
            )
            st.dataframe(rutas.head(40), use_container_width=True, hide_index=True)

            # Resumen simple del último ASN para facilitar interpretación.
            ult = rutas.groupby("Último nodo atribuible", as_index=False)["Traceroutes"].sum().sort_values("Traceroutes", ascending=False)
            name_map = mapa_nombres_asn(h)
            ult["Nodo / organización"] = ult["Último nodo atribuible"].map(lambda x: name_map.get(x, x))
            fig = px.bar(
                ult.head(15),
                x="Nodo / organización",
                y="Traceroutes",
                text_auto=True,
                title="Últimos nodos atribuibles más frecuentes antes del destino",
            )
            fig.update_layout(height=470, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_11_Ultimos_ASN_antes_destino"))

    st.markdown("---")
    st.subheader("🧪 Inspección de una traza concreta")
    if not t.empty:
        sondas_trace = sorted(t["sonda_nombre"].dropna().unique())
        sonda_trace = st.selectbox("Sonda", sondas_trace, key="trace_probe_v3")
        subset = t[t["sonda_nombre"] == sonda_trace].sort_values("timestamp", ascending=False)
        if not subset.empty:
            options = subset["trace_id"].astype(str).tolist()
            stamp_map = subset.set_index(subset["trace_id"].astype(str))["timestamp"].to_dict()
            trace_id = st.selectbox(
                "Ejecución",
                options,
                format_func=lambda x: f"{stamp_map.get(x)} · {x}",
                key="trace_exec_v3",
            )
            hd = h[h["trace_id"].astype(str) == str(trace_id)].sort_values(["hop_num", "intento"])
            cols_show = [
                c
                for c in [
                    "hop_num",
                    "hop_especial_255",
                    "intento",
                    "ip_hop",
                    "tipo_ip",
                    "rtt_hop_ms",
                    "timeout",
                    "error",
                    "es_destino",
                    "asn_hop",
                    "as_name",
                    "pais",
                    "ciudad",
                ]
                if c in hd.columns
            ]
            st.dataframe(hd[cols_show], use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 6 - OONI
# -----------------------------------------------------------------------------
with tab_ooni:
    oo = df_o[df_o["sitio_web"] == servicio].copy()
    if oo.empty:
        st.info("No hay datos OONI para este servicio.")
    else:
        st.subheader(f"🌐 OONI Web Connectivity — {servicio}")
        totals = oo[["ok_count", "anomaly_count", "failure_count", "confirmed_count", "measurement_count"]].sum()
        total = int(totals["measurement_count"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mediciones", f"{total:,}")
        c2.metric("OK", fmt_num(pct(totals["ok_count"], total), 1, "%"))
        c3.metric("Anomalía + fallo", fmt_num(pct(totals["anomaly_count"] + totals["failure_count"], total), 1, "%"))
        c4.metric("Confirmados", f"{int(totals['confirmed_count']):,}")

        long = oo.melt(
            id_vars=["fecha", "sitio_web"],
            value_vars=["ok_count", "anomaly_count", "failure_count", "confirmed_count"],
            var_name="estado",
            value_name="Mediciones",
        )
        long["estado"] = long["estado"].map(
            {
                "ok_count": "OK",
                "anomaly_count": "Anomaly",
                "failure_count": "Failure",
                "confirmed_count": "Confirmed",
            }
        )
        fig = px.bar(
            long,
            x="fecha",
            y="Mediciones",
            color="estado",
            barmode="stack",
            labels={"fecha": "Fecha", "estado": "Resultado"},
            color_discrete_map=OONI_COLOR_MAP,
            category_orders={"estado": ["OK", "Anomaly", "Failure", "Confirmed"]},
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_12_OONI"))
        st.caption(
            "OONI se usa como evidencia complementaria. 'Anomalía' o 'fallo' no se interpretan automáticamente como bloqueo, censura o indisponibilidad total."
        )

        st.markdown("#### Contraste diario descriptivo: OONI vs. RIPE Atlas")
        daily_p = df_p[df_p["sitio_web"] == servicio].copy()
        if not daily_p.empty:
            daily_p["fecha"] = pd.to_datetime(daily_p["fecha"])
            oo2 = oo.copy()
            oo2["fecha"] = pd.to_datetime(oo2["fecha"])
            ping_daily = (
                daily_p[daily_p["sent"] > 0]
                .groupby("fecha")
                .apply(lambda g: pd.Series({"RIPE alcanzabilidad (%)": pct((g["rcvd"] > 0).sum(), len(g))}), include_groups=False)
                .reset_index()
            )
            oo2["OONI OK (%)"] = oo2.apply(lambda r: pct(r["ok_count"], r["measurement_count"]), axis=1)
            merged = ping_daily.merge(oo2[["fecha", "OONI OK (%)"]], on="fecha", how="inner")
            if not merged.empty:
                long2 = merged.melt("fecha", var_name="Indicador", value_name="Porcentaje")
                fig = px.line(long2, x="fecha", y="Porcentaje", color="Indicador", markers=True)
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True, config=plotly_config(f"{servicio}_12_OONI_vs_RIPE"))
                st.caption("Comparación agregada a escala diaria; los CSV OONI no permiten emparejamiento exacto 10:00 vs. 22:00.")

# -----------------------------------------------------------------------------
# TAB 7 - COMPARATIVO CAPÍTULO IV
# -----------------------------------------------------------------------------
with tab_comp:
    st.subheader("📚 Figuras comparativas candidatas para el Capítulo IV")
    st.caption(
        "Estas figuras comparan simultáneamente los cuatro servicios institucionales. "
        "Respetan los filtros de grupo geográfico, franja y sondas seleccionadas. "
        "La selección lateral de un servicio individual no afecta este panel."
    )

    pv_all = p_all[pd.to_numeric(p_all["sent"], errors="coerce").fillna(0) > 0].copy()
    if pv_all.empty:
        st.info("No hay mediciones Ping suficientes con los filtros actuales para construir las figuras comparativas.")
    else:
        st.markdown("#### 1) Alcanzabilidad ICMP por sonda y servicio")
        reach = (
            pv_all.groupby(["sonda_nombre", "sitio_web"])["alcanzable_icmp"]
            .mean()
            .mul(100)
            .unstack()
            .reindex(index=PROBE_ORDER, columns=SERVICE_ORDER)
        )
        reach.columns = [short_service_name(c) for c in reach.columns]
        fig = px.imshow(
            reach,
            text_auto=".1f",
            aspect="auto",
            labels={"x": "Servicio", "y": "Sonda / proveedor", "color": "Alcanzabilidad (%)"},
            color_continuous_scale=HEATMAP_SCALE_GOOD,
            zmin=0,
            zmax=100,
            height=500,
        )
        fig.update_traces(texttemplate="%{z:.1f}%%")
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_01_Alcanzabilidad_Sonda_Servicio"))
        st.caption("Escala semántica: verde = bien / mejor alcanzabilidad; rojo = peor alcanzabilidad.")

        st.markdown("#### 2) Alcanzabilidad ICMP por franja")
        fr = (
            pv_all.groupby(["sitio_web", "franja"])["alcanzable_icmp"]
            .mean()
            .mul(100)
            .unstack()
            .reindex(SERVICE_ORDER)
        )
        fr.index = [short_service_name(i) for i in fr.index]
        fr = fr[[c for c in ["Matutina (~10:00)", "Nocturna (~22:00)"] if c in fr.columns]]
        fr_long = fr.reset_index().rename(columns={"index": "Servicio"}).melt(id_vars="Servicio", var_name="Franja", value_name="Alcanzabilidad (%)")
        fig = px.bar(
            fr_long,
            x="Servicio",
            y="Alcanzabilidad (%)",
            color="Franja",
            barmode="group",
            text_auto=".1f",
            color_discrete_map=FRANJA_COLOR_MAP,
            title="Alcanzabilidad por servicio y franja de observación",
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_02_Alcanzabilidad_Matutina_Nocturna"))

        st.markdown("#### 3) Estado conjunto de las sesiones")
        sess = (
            pv_all.groupby(["sitio_web", "fecha", "franja"])
            .agg(sondas_con_medicion=("sonda_id", "nunique"),
                 sondas_con_respuesta=("alcanzable_icmp", "sum"))
            .reset_index()
        )
        sess["estado"] = np.select(
            [
                sess["sondas_con_respuesta"].eq(0),
                sess["sondas_con_respuesta"].eq(sess["sondas_con_medicion"]),
            ],
            ["Ninguna", "Todas"],
            default="Parcial",
        )
        sess_counts = (
            sess.groupby(["sitio_web", "estado"]).size().unstack(fill_value=0)
            .reindex(index=SERVICE_ORDER, columns=["Todas", "Parcial", "Ninguna"], fill_value=0)
        )
        sess_counts.index = [short_service_name(i) for i in sess_counts.index]
        sess_long = sess_counts.reset_index().rename(columns={"index": "Servicio"}).melt(id_vars="Servicio", var_name="Estado", value_name="Sesiones")
        fig = px.bar(
            sess_long,
            x="Servicio",
            y="Sesiones",
            color="Estado",
            barmode="stack",
            text_auto=True,
            color_discrete_map=SESSION_COLOR_MAP,
            category_orders={"Estado": ["Todas", "Parcial", "Ninguna"]},
            title="Sesiones fecha-franja con respuesta total, parcial o nula",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_03_Estado_Conjunto_Sesiones"))
        st.caption("“Todas” significa que respondieron todas las sondas con medición efectiva en esa sesión.")

        st.markdown("#### 4) RTT mediano por sonda y servicio")
        rtt = (
            pv_all.dropna(subset=["rtt_exec_ms"])
            .groupby(["sonda_nombre", "sitio_web"])["rtt_exec_ms"]
            .median()
            .reset_index()
        )
        rtt["Servicio"] = rtt["sitio_web"].map(short_service_name)
        fig = px.scatter(
            rtt,
            x="rtt_exec_ms",
            y="sonda_nombre",
            color="Servicio",
            symbol="Servicio",
            labels={"rtt_exec_ms": "RTT mediano (ms)", "sonda_nombre": "Sonda / proveedor"},
            category_orders={"sonda_nombre": PROBE_ORDER},
            title="RTT mediano de ejecuciones con respuesta por sonda y servicio",
        )
        fig.update_layout(height=540)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_04_RTT_Mediano_Sonda_Servicio"))

        st.markdown("#### 5) RTT mediano: nacional vs internacional")
        pi = pv_all[(pv_all["sonda_nombre"] != "ULA (Mérida)") & pv_all["rtt_exec_ms"].notna()].copy()
        if pi.empty:
            st.info("No hay datos suficientes para comparar sondas nacionales e internacionales.")
        else:
            ni = pi.groupby(["sitio_web", "tipo_sonda"])["rtt_exec_ms"].median().reset_index()
            ni["Servicio"] = ni["sitio_web"].map(short_service_name)
            fig = px.bar(
                ni,
                x="Servicio",
                y="rtt_exec_ms",
                color="tipo_sonda",
                barmode="group",
                text_auto=".1f",
                labels={"rtt_exec_ms": "RTT mediano (ms)", "tipo_sonda": "Grupo"},
                title="Comparación del RTT mediano entre sondas nacionales externas e internacionales",
            )
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_05_RTT_Nacional_Internacional"))
            st.caption("La sonda ULA (Mérida) se excluye de este agregado por actuar como referencia local institucional.")

        st.markdown("#### 6) Contexto 2026: antes, contingencia y después")
        ev = resumen_eventos_ping(df_p)
        ev_mov = ev[(ev["Sonda"] == "Movistar (Maracay)") & (ev["RTT mediano (ms)"].notna())].copy()
        if not ev_mov.empty:
            fig = px.bar(
                ev_mov,
                x="Servicio",
                y="RTT mediano (ms)",
                color="Periodo",
                barmode="group",
                text_auto=".1f",
                color_discrete_map=PERIODO_COLOR_MAP,
                title="Movistar → ULA: RTT mediano antes, durante y después de la contingencia de 2026",
            )
            fig.update_xaxes(ticktext=[short_service_name(s) for s in SERVICE_ORDER], tickvals=SERVICE_ORDER)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_06_Evento_2026_Movistar"))
            st.caption("Estas etapas son contextuales. La figura resume asociación temporal, no causalidad demostrada.")

        st.markdown("#### 7) Resumen comparativo de OONI")
        if not o_all.empty:
            oo_sum = (
                o_all.groupby("sitio_web")[["ok_count", "anomaly_count", "failure_count", "confirmed_count"]]
                .sum().reindex(SERVICE_ORDER).reset_index()
            )
            oo_sum["Servicio"] = oo_sum["sitio_web"].map(short_service_name)
            long_oo = oo_sum.melt(
                id_vars="Servicio",
                value_vars=["ok_count", "anomaly_count", "failure_count", "confirmed_count"],
                var_name="Estado",
                value_name="Mediciones",
            )
            long_oo["Estado"] = long_oo["Estado"].map({
                "ok_count": "OK",
                "anomaly_count": "Anomaly",
                "failure_count": "Failure",
                "confirmed_count": "Confirmed",
            })
            fig = px.bar(
                long_oo,
                x="Servicio",
                y="Mediciones",
                color="Estado",
                barmode="stack",
                text_auto=True,
                color_discrete_map=OONI_COLOR_MAP,
                category_orders={"Estado": ["OK", "Anomaly", "Failure", "Confirmed"]},
                title="Resultados agregados de OONI Web Connectivity por servicio",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_07_OONI_Comparativo"))
            st.caption("Semántica visual: OK en verde; Failure en rojo; Confirmed en rojo oscuro.")

        st.markdown("#### 8) Alcanzabilidad del destino mediante Traceroute")
        if not t_all.empty:
            tr = (
                t_all.groupby("sitio_web")["respondio_destino"].mean().mul(100).reindex(SERVICE_ORDER).reset_index()
            )
            tr["Servicio"] = tr["sitio_web"].map(short_service_name)
            fig = px.bar(
                tr,
                x="Servicio",
                y="respondio_destino",
                text_auto=".2f",
                labels={"respondio_destino": "Traceroutes que alcanzaron la IP destino (%)"},
                color="respondio_destino",
                color_continuous_scale=HEATMAP_SCALE_GOOD,
                range_color=[0, 100],
                title="Alcanzabilidad del destino mediante Traceroute",
            )
            fig.update_layout(height=460, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config("Global_08_Alcanzabilidad_Traceroute"))

    st.info(
        "Estas figuras se incorporaron para apoyar la selección del paquete visual definitivo del Capítulo IV. "
        "Los gráficos individuales por servicio se mantienen en las demás pestañas para exploración detallada."
    )


# -----------------------------------------------------------------------------
# TAB 7 - COBERTURA
# -----------------------------------------------------------------------------
with tab_cov:
    st.subheader("Tabla 3.4 — Cobertura efectiva de sondas")
    st.dataframe(df_cov_s, use_container_width=True, hide_index=True)
    st.subheader("Tabla 3.5 — Cobertura temporal de servicios")
    st.dataframe(df_cov_d, use_container_width=True, hide_index=True)
    st.caption("Estas tablas se generan directamente desde los registros procesados y no se rellenan manualmente.")

# -----------------------------------------------------------------------------
# TAB 8 - AUDITORÍA
# -----------------------------------------------------------------------------
with tab_raw:
    st.subheader("🛠️ Auditoría e inspección de datos procesados")

    st.markdown("### 📸 Guía de exportación por servicio — 12 capturas")
    st.write(
        "Para trabajar el Capítulo IV de forma ordenada, selecciona un servicio y "
        "guarda estas doce evidencias. Los PNG descargados desde el icono de cámara "
        "ya reciben un nombre descriptivo con el servicio y el tipo de gráfico."
    )
    guia_capturas = pd.DataFrame(
        [
            ["01", "Resumen por sonda", "Tabla principal de la pestaña Resumen"],
            ["02", "Serie temporal RTT", "RTT y tiempo"],
            ["03", "Mapa de calor RTT", "RTT y tiempo"],
            ["04", "RTT por sonda", "Resumen / RTT y tiempo"],
            ["05", "Nacional vs. internacional", "Resumen por sonda"],
            ["06", "Alcanzabilidad / pérdida / estados Ping", "Alcanzabilidad"],
            ["07", "Destino alcanzado y saltos", "Traceroute"],
            ["08", "Rutas IP dominantes", "Rutas / ASN"],
            ["09", "Perfil salto-a-salto de ruta dominante", "Rutas / ASN"],
            ["10", "Sankey ASN / IXP", "Rutas / ASN"],
            ["11", "Rutas de entrada", "Rutas / ASN"],
            ["12", "OONI", "OONI"],
            ["13", "Comparativo Cap. IV", "Comparativo Cap. IV"],
        ],
        columns=["N.º", "Evidencia", "Pestaña"],
    )
    st.dataframe(guia_capturas, use_container_width=True, hide_index=True)
    st.caption(
        "Las tablas 01 y 08 pueden capturarse con la herramienta de captura del sistema "
        "o descargarse como datos. Los gráficos 02–12 usan el icono de cámara de Plotly."
    )

    st.markdown("### 🌎 Tabla global de hitos 2026")
    ev_global = resumen_eventos_ping(df_p)
    if not ev_global.empty:
        st.download_button(
            "⬇️ Descargar análisis global antes · contingencia · después",
            data=ev_global.to_csv(index=False).encode("utf-8-sig"),
            file_name="analisis_eventos_2026_todos_los_servicios.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Períodos: antes del 24-Jun 18:04; contingencia desde ese momento hasta el 23-Jul; "
            "posterior desde el 24-Jul. Los resultados son descriptivos, no causales."
        )

    st.markdown("### 📦 Paquete de datos para el Capítulo IV")
    st.write(
        "Descarga en un solo ZIP los seis archivos procesados necesarios para "
        "revisar los resultados, comparar las figuras y desarrollar el Capítulo IV."
    )
    try:
        st.download_button(
            "⬇️ Descargar paquete completo del Capítulo IV (6 CSV)",
            data=paquete_capitulo_iv(),
            file_name="datos_capitulo_IV_ULA.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.caption(
            "Incluye seis CSV normales: Ping, resumen de Traceroute, hops compactos, OONI y las tablas "
            "de cobertura 3.4 y 3.5."
        )
    except FileNotFoundError as exc:
        st.warning(f"No se pudo preparar el paquete: {exc}")

    st.info(
        "📷 Para guardar cualquier gráfico como PNG, coloca el cursor sobre la figura "
        "y pulsa el icono de cámara de la barra superior derecha. La captura se genera "
        "en alta resolución (escala 2×)."
    )

    st.write("Ping filtrado")
    st.dataframe(p, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar Ping filtrado (CSV)",
        p.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ping_{servicio}.csv",
        mime="text/csv",
    )
    st.write("Traceroute filtrado")
    st.dataframe(t, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar Traceroute filtrado (CSV)",
        t.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"traceroute_{servicio}.csv",
        mime="text/csv",
    )
    with st.expander("Detalle hop-by-hop"):
        st.dataframe(h, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "V3 conserva la intención visual del dashboard original, pero evita usar RTT como 'flujo' de Sankey, "
    "no trata hop=255 como 255 saltos y separa visualización exploratoria de inferencias causales."
)
