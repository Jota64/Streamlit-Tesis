import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dashboard Telemetría ULA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Telemetría de Red - Servicios Web ULA")
st.caption(
    "Auditoría de Latencia, Pérdidas y Rutas de Transporte (Hora Local Venezuela UTC-4)"
)


@st.cache_data
def cargar_datos():
    from procesar_datos import (
        cargar_json,
        procesar_pings,
        procesar_traceroutes,
    )

    pings_json = cargar_json("ripe_pings_consolidado.json")
    traceroutes_json = cargar_json("ripe_traceroutes_consolidado.json")

    df_p = procesar_pings(pings_json)
    df_t, df_saltos = procesar_traceroutes(traceroutes_json)
    return df_p, df_t, df_saltos


try:
    df_pings, df_traceroutes, df_saltos = cargar_datos()

    # --- BARRA LATERAL (FILTROS) ---
    st.sidebar.header("🎯 Filtros del Análisis")

    lista_sitios = sorted(df_pings["sitio_web"].unique().tolist())
    sitio_sel = st.sidebar.selectbox("Seleccionar Servicio Web:", lista_sitios)

    filtro_tipo = st.sidebar.radio(
        "Grupo Geográfico:",
        ["Todas", "Solo Nacionales", "Solo Internacionales"],
    )

    filtro_bloque = st.sidebar.selectbox(
        "Ventana Horaria de Medición:",
        ["Todas las Horas", "Solo 10:00 AM", "Solo 10:00 PM"],
    )

    # Aplicación de filtros a dataframes base
    df_p_base = df_pings.copy()
    df_t_base = df_traceroutes.copy()

    if filtro_tipo == "Solo Nacionales":
        df_p_base = df_p_base[df_p_base["tipo_sonda"] == "Nacional"]
        df_t_base = df_t_base[df_t_base["tipo_sonda"] == "Nacional"]
    elif filtro_tipo == "Solo Internacionales":
        df_p_base = df_p_base[df_p_base["tipo_sonda"] == "Internacional"]
        df_t_base = df_t_base[df_t_base["tipo_sonda"] == "Internacional"]

    if filtro_bloque == "Solo 10:00 AM":
        df_p_base = df_p_base[
            df_p_base["bloque_horario"] == "10:00 AM (Mañana)"
        ]
    elif filtro_bloque == "Solo 10:00 PM":
        df_p_base = df_p_base[df_p_base["bloque_horario"] == "10:00 PM (Noche)"]

    lista_sondas_disp = sorted(df_p_base["sonda_nombre"].unique().tolist())
    sondas_sel = st.sidebar.multiselect(
        "Sondas Específicas:", lista_sondas_disp, default=lista_sondas_disp
    )

    # Filtrado final
    df_p_filt = df_p_base[
        (df_p_base["sitio_web"] == sitio_sel)
        & (df_p_base["sonda_nombre"].isin(sondas_sel))
    ]
    df_t_filt = df_t_base[
        (df_t_base["sitio_web"] == sitio_sel)
        & (df_t_base["sonda_nombre"].isin(sondas_sel))
    ]

    # --- TARJETAS DE MÉTRICAS GENERALES ---
    col1, col2, col3, col4, col5 = st.columns(5)

    total_pings = len(df_p_filt)
    exitosos = (
        len(df_p_filt[df_p_filt["alcanzable"] == True]) if total_pings > 0 else 0
    )
    pct_alcanzabilidad = (
        round((exitosos / total_pings) * 100, 1) if total_pings > 0 else 0
    )
    rtt_prom = (
        round(df_p_filt["rtt_avg_ms"].mean(), 2)
        if not df_p_filt.empty and exitosos > 0
        else 0
    )
    loss_prom = (
        round(df_p_filt["packet_loss_pct"].mean(), 1)
        if not df_p_filt.empty
        else 0
    )
    saltos_prom = (
        round(df_t_filt["total_saltos"].mean(), 1)
        if not df_t_filt.empty
        else 0
    )

    with col1:
        st.metric("Total Muestras", total_pings)
    with col2:
        st.metric("Alcanzabilidad", f"{pct_alcanzabilidad}%")
    with col3:
        st.metric("Pérdida Paquetes", f"{loss_prom}%")
    with col4:
        st.metric("Latencia Promedio", f"{rtt_prom} ms")
    with col5:
        st.metric("Saltos Promedio", f"{saltos_prom}")

    st.markdown("---")

    # --- TABLA DETALLADA POR PROBE / ASN ---
    st.subheader(f"📌 Estado por Origen / Probe hacia {sitio_sel}")

    resumen_sondas = (
        df_p_filt.groupby(
            ["sonda_nombre", "asn", "isp", "ubicacion", "ip_origen"]
        )
        .agg(
            muestras=("alcanzable", "count"),
            exitosas=("alcanzable", lambda x: sum(x)),
            perdida_promedio_pct=("packet_loss_pct", "mean"),
            rtt_promedio_ms=("rtt_avg_ms", "mean"),
        )
        .reset_index()
    )

    if not resumen_sondas.empty:
        resumen_sondas["% Alcanzabilidad"] = (
            (resumen_sondas["exitosas"] / resumen_sondas["muestras"]) * 100
        ).round(1)
        resumen_sondas["Pérdida Prom. (%)"] = resumen_sondas[
            "perdida_promedio_pct"
        ].round(1)
        resumen_sondas["rtt_promedio_ms"] = resumen_sondas[
            "rtt_promedio_ms"
        ].round(2)

        resumen_sondas = resumen_sondas.rename(
            columns={
                "sonda_nombre": "Sonda",
                "asn": "ASN",
                "isp": "Proveedor (ISP)",
                "ubicacion": "Ubicación (Ciudad, País)",
                "ip_origen": "IP Origen",
                "rtt_promedio_ms": "RTT Prom. (ms)",
            }
        )

        st.dataframe(
            resumen_sondas[
                [
                    "Sonda",
                    "ASN",
                    "Proveedor (ISP)",
                    "Ubicación (Ciudad, País)",
                    "IP Origen",
                    "% Alcanzabilidad",
                    "Pérdida Prom. (%)",
                    "RTT Prom. (ms)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # --- GRÁFICO 1: SERIES TEMPORALES DE LATENCIA ---
    st.subheader(
        f"📈 Series de Tiempo de Latencia (Hora Local VLA UTC-4) hacia {sitio_sel}"
    )

    if not df_p_filt.empty:
        df_exitosos = df_p_filt[df_p_filt["alcanzable"] == True].copy()
        df_exitosos["timestamp"] = pd.to_datetime(df_exitosos["timestamp"])

        df_exitosos = (
            df_exitosos.groupby(
                [
                    "timestamp",
                    "sonda_nombre",
                    "hora_corta",
                    "asn",
                    "ubicacion",
                ],
                as_index=False,
            )
            .agg({"rtt_avg_ms": "mean"})
            .reset_index(drop=True)
        )

        df_exitosos = df_exitosos.sort_values(
            by=["sonda_nombre", "timestamp"]
        ).reset_index(drop=True)

        if not df_exitosos.empty:
            fig_line = px.line(
                df_exitosos,
                x="timestamp",
                y="rtt_avg_ms",
                color="sonda_nombre",
                line_group="sonda_nombre",
                markers=True,
                hover_data=["hora_corta", "asn", "ubicacion"],
                labels={
                    "timestamp": "Fecha y Hora (UTC-4)",
                    "rtt_avg_ms": "RTT Promedio (ms)",
                    "sonda_nombre": "Sonda / ISP",
                },
                height=600,
            )

            fig_line.update_traces(
                line=dict(width=2.5), marker=dict(size=6), connectgaps=False
            )

            fecha_sismo = pd.to_datetime("2026-06-24 18:00:00")
            fecha_cable = pd.to_datetime("2026-07-22 00:00:00")

            fig_line.add_vline(
                x=fecha_sismo.timestamp() * 1000,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text="Sismo (24/Jun 6:00 PM)",
                annotation_position="top left",
                annotation_font_color="red",
            )

            fig_line.add_vline(
                x=fecha_cable.timestamp() * 1000,
                line_width=2,
                line_dash="dash",
                line_color="green",
                annotation_text="Reparación Cable Submarino (22/Jul)",
                annotation_position="top right",
                annotation_font_color="green",
            )

            fig_line.update_xaxes(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1,
                                label="1D",
                                step="day",
                                stepmode="backward",
                            ),
                            dict(
                                count=7,
                                label="1S",
                                step="day",
                                stepmode="backward",
                            ),
                            dict(
                                count=1,
                                label="1M",
                                step="month",
                                stepmode="backward",
                            ),
                            dict(step="all", label="Todo"),
                        ]
                    )
                ),
                title_font=dict(size=14),
                tickfont=dict(size=12),
            )

            fig_line.update_yaxes(
                title_font=dict(size=14), tickfont=dict(size=12)
            )
            fig_line.update_layout(
                hovermode="x unified",
                legend=dict(
                    font=dict(size=13),
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # --- MAPA DE CALOR ---
    st.subheader("🔥 Mapa de Calor: Latencia Promedio Diaria por Sonda")
    st.caption(
        "Identifica patrones globales de congestión y el impacto antes/después del sismo y la reparación del cable."
    )

    if not df_p_filt.empty:
        df_heat = (
            df_p_filt.groupby(["fecha", "sonda_nombre"])["rtt_avg_ms"]
            .mean()
            .reset_index()
        )
        if not df_heat.empty:
            df_pivot = df_heat.pivot(
                index="sonda_nombre", columns="fecha", values="rtt_avg_ms"
            )

            fig_heat = px.imshow(
                df_pivot,
                labels=dict(
                    x="Fecha", y="Sonda / Proveedor", color="RTT Promedio (ms)"
                ),
                color_continuous_scale="Viridis_r",
                aspect="auto",
                height=420,
            )
            fig_heat.update_xaxes(side="bottom")
            fig_heat.update_layout(font=dict(size=13))
            st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # --- GRÁFICO 2: COMPARATIVO NACIONAL VS INTERNACIONAL ---
    st.subheader("⚖️ Comparativo de Latencia: Nacionales vs. Internacionales")

    df_p_sitio = df_pings[
        (df_pings["sitio_web"] == sitio_sel)
        & (df_pings["alcanzable"] == True)
    ]

    if not df_p_sitio.empty:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            df_grupo = (
                df_p_sitio.groupby("tipo_sonda")["rtt_avg_ms"]
                .mean()
                .reset_index()
            )
            fig_grupo = px.bar(
                df_grupo,
                x="tipo_sonda",
                y="rtt_avg_ms",
                color="tipo_sonda",
                text_auto=".1f",
                title="Promedio General por Grupo (ms)",
                labels={
                    "tipo_sonda": "Grupo",
                    "rtt_avg_ms": "RTT Promedio (ms)",
                },
                color_discrete_map={
                    "Nacional": "#1f77b4",
                    "Internacional": "#ff7f0e",
                },
                height=450,
            )
            fig_grupo.update_traces(
                texttemplate="%{y:.1f} ms",
                textposition="outside",
                textfont_size=14,
            )
            fig_grupo.update_layout(font=dict(size=13))
            st.plotly_chart(fig_grupo, use_container_width=True)

        with col_c2:
            df_sondas_prom = (
                df_p_sitio.groupby(["sonda_nombre", "tipo_sonda"])["rtt_avg_ms"]
                .mean()
                .reset_index()
            )
            fig_sondas = px.bar(
                df_sondas_prom,
                x="sonda_nombre",
                y="rtt_avg_ms",
                color="tipo_sonda",
                text_auto=".1f",
                title="Promedio por Sonda Individual (ms)",
                labels={
                    "sonda_nombre": "Sonda",
                    "rtt_avg_ms": "RTT Promedio (ms)",
                    "tipo_sonda": "Tipo",
                },
                color_discrete_map={
                    "Nacional": "#1f77b4",
                    "Internacional": "#ff7f0e",
                },
                height=450,
            )
            fig_sondas.update_layout(
                xaxis_tickangle=-45, font=dict(size=13)
            )
            fig_sondas.update_traces(
                texttemplate="%{y:.1f} ms",
                textposition="outside",
                textfont_size=12,
            )
            st.plotly_chart(fig_sondas, use_container_width=True)

    st.markdown("---")

    # --- SECCIÓN TRACEROUTES 1: ESTABILIDAD DE SALTOS ---
    st.subheader("🗺️ Estabilidad de Rutas de Transporte (Saltos / Hops)")
    st.caption(
        "Muestra el promedio acumulado de saltos requeridos por cada sonda para llegar al destino."
    )

    if not df_t_filt.empty:
        df_hops_stat = (
            df_t_filt.groupby("sonda_nombre")["total_saltos"]
            .agg(Mínimo="min", Promedio="mean", Máximo="max")
            .reset_index()
        )
        df_hops_stat["Promedio"] = df_hops_stat["Promedio"].round(1)

        fig_hops = px.bar(
            df_hops_stat,
            x="sonda_nombre",
            y="Promedio",
            hover_data=["Mínimo", "Máximo"],
            text="Promedio",
            title="Saltos Promedio por Sonda",
            labels={
                "sonda_nombre": "Sonda / Proveedor",
                "Promedio": "Saltos Promedio",
            },
            height=450,
        )
        fig_hops.update_traces(
            texttemplate="%{y:.1f} hops",
            textposition="outside",
            textfont_size=13,
        )
        fig_hops.update_layout(
            xaxis_tickangle=-45, font=dict(size=13)
        )
        st.plotly_chart(fig_hops, use_container_width=True)

    st.markdown("---")

    # --- SECCIÓN TRACEROUTES 2: INSPECCIÓN SALTO POR SALTO ---
    st.subheader(f"🔎 Inspección de Ruta Salto por Salto hacia {sitio_sel}")
    st.caption(
        "Desglose secuencial de la ruta IP tomada por una sonda específica y la latencia (RTT) acumulada en cada salto."
    )

    sonda_inspeccion = st.selectbox(
        "Seleccionar Sonda para analizar su traza completa:", lista_sondas_disp
    )

    df_saltos_filt = df_saltos[
        (df_saltos["sitio_web"] == sitio_sel)
        & (df_saltos["sonda_nombre"] == sonda_inspeccion)
    ].copy()

    df_saltos_filt["hop_num"] = pd.to_numeric(
        df_saltos_filt["hop_num"], errors="coerce"
    )

    df_saltos_validos = df_saltos_filt[
        (df_saltos_filt["hop_num"].notnull())
        & (df_saltos_filt["hop_num"] > 0)
        & (df_saltos_filt["hop_num"] <= 30)
        & (df_saltos_filt["rtt_hop_ms"].notnull())
        & (df_saltos_filt["rtt_hop_ms"] >= 0)
    ].copy()

    if not df_saltos_validos.empty:
        df_hop_profile = (
            df_saltos_validos.groupby("hop_num", as_index=False)
            .agg(
                ip_intermedia=(
                    "ip_intermedia",
                    lambda x: x.mode()[0] if not x.empty else "*",
                ),
                pais=("pais", lambda x: x.mode()[0] if not x.empty else "N/A"),
                ciudad=(
                    "ciudad",
                    lambda x: x.mode()[0] if not x.empty else "N/A",
                ),
                proveedor_salto=(
                    "proveedor_salto",
                    lambda x: x.mode()[0] if not x.empty else "N/A",
                ),
                rtt_promedio_ms=("rtt_hop_ms", "mean"),
            )
            .sort_values(by="hop_num")
            .reset_index(drop=True)
        )

        df_hop_profile["rtt_promedio_ms"] = df_hop_profile[
            "rtt_promedio_ms"
        ].round(2)

        fig_path = px.line(
            df_hop_profile,
            x="hop_num",
            y="rtt_promedio_ms",
            markers=True,
            hover_data=["pais", "ciudad", "proveedor_salto"],
            text="ip_intermedia",
            title=f"Perfil de Latencia Salto a Salto — {sonda_inspeccion}",
            labels={
                "hop_num": "Número de Salto (Hop)",
                "rtt_promedio_ms": "Latencia Acumulada (ms)",
            },
            height=500,
        )

        fig_path.update_traces(
            textposition="top center",
            line=dict(width=2.5),
            marker=dict(size=8),
            textfont_size=11,
            connectgaps=False,
        )

        max_hop = int(df_hop_profile["hop_num"].max())
        fig_path.update_xaxes(
            dtick=1, range=[0, max_hop + 1], title="Número de Salto (Hop)"
        )

        fig_path.update_yaxes(title="RTT Acumulado (ms)")
        fig_path.update_layout(font=dict(size=13))

        st.plotly_chart(fig_path, use_container_width=True)

        st.dataframe(
            df_hop_profile[
                [
                    "hop_num",
                    "ip_intermedia",
                    "pais",
                    "ciudad",
                    "proveedor_salto",
                    "rtt_promedio_ms",
                ]
            ].rename(
                columns={
                    "hop_num": "Salto #",
                    "ip_intermedia": "IP Intermedia",
                    "pais": "País",
                    "ciudad": "Ciudad",
                    "proveedor_salto": "Proveedor / Operador",
                    "rtt_promedio_ms": "RTT Acumulado (ms)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No hay trazas de saltos válidas registradas para esta sonda.")

    st.markdown("---")

    # --- SECCIÓN TRACEROUTES 3: COMPARATIVO MULTI-SONDA SALTO POR SALTO ---
    st.subheader(
        f"📊 Comparativa de Rutas Salto a Salto entre Sondas hacia {sitio_sel}"
    )
    st.caption(
        "Superpone la progresión de latencia acumulada (ms) de todas las sondas"
        " seleccionadas para identificar en qué salto/tramo se generan las"
        " divergencias de latencia."
    )

    df_saltos_comp = df_saltos[
        (df_saltos["sitio_web"] == sitio_sel)
        & (df_saltos["sonda_nombre"].isin(sondas_sel))
    ].copy()

    df_saltos_comp["hop_num"] = pd.to_numeric(
        df_saltos_comp["hop_num"], errors="coerce"
    )

    df_saltos_comp_validos = df_saltos_comp[
        (df_saltos_comp["hop_num"].notnull())
        & (df_saltos_comp["hop_num"] > 0)
        & (df_saltos_comp["hop_num"] <= 30)
        & (df_saltos_comp["rtt_hop_ms"].notnull())
        & (df_saltos_comp["rtt_hop_ms"] >= 0)
    ].copy()

    if not df_saltos_comp_validos.empty:
        df_comp_profile = (
            df_saltos_comp_validos.groupby(
                ["hop_num", "sonda_nombre"], as_index=False
            )
            .agg(rtt_promedio_ms=("rtt_hop_ms", "mean"))
            .sort_values(by=["sonda_nombre", "hop_num"])
            .reset_index(drop=True)
        )

        df_comp_profile["rtt_promedio_ms"] = df_comp_profile[
            "rtt_promedio_ms"
        ].round(2)

        fig_comp = px.line(
            df_comp_profile,
            x="hop_num",
            y="rtt_promedio_ms",
            color="sonda_nombre",
            line_group="sonda_nombre",
            markers=True,
            title=f"Comparativa de Perfil de Latencia Salto por Salto — {sitio_sel}",
            labels={
                "hop_num": "Número de Salto (Hop #)",
                "rtt_promedio_ms": "Latencia Acumulada (ms)",
                "sonda_nombre": "Sonda / Proveedor",
            },
            height=550,
        )

        fig_comp.update_traces(
            line=dict(width=2.5), marker=dict(size=7), connectgaps=False
        )

        max_hop_comp = int(df_comp_profile["hop_num"].max())
        fig_comp.update_xaxes(
            dtick=1, range=[0, max_hop_comp + 1], title="Número de Salto (Hop)"
        )
        fig_comp.update_yaxes(title="RTT Acumulado (ms)")

        fig_comp.update_layout(
            font=dict(size=13),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info(
            "No hay datos de saltos suficientes para comparar las sondas"
            " seleccionadas."
        )

    st.markdown("---")

    # ==============================================================================
    # 🌊 SANKEY 1: LATENCIA RTT ORIGEN (ISPs / PROBES) → DESTINO (IMAGEN 1 DE LA PROF.)
    # ==============================================================================
    st.subheader(f"🌊 Diagrama Sankey: Latencia RTT Origen (ISPs) → {sitio_sel}")
    st.caption(
        "Mapea el volumen de latencia (RTT Promedio ms) consumido por cada ISP/Sonda hacia el destino final."
    )

    if not df_p_filt.empty:
        df_sankey_rtt = (
            df_p_filt[df_p_filt["alcanzable"] == True]
            .groupby(["sonda_nombre", "sitio_web"])["rtt_avg_ms"]
            .mean()
            .reset_index()
        )

        if not df_sankey_rtt.empty:
            nodos_rtt = pd.unique(df_sankey_rtt[["sonda_nombre", "sitio_web"]].values.ravel())
            idx_rtt = {n: i for i, n in enumerate(nodos_rtt)}

            df_sankey_rtt["src_idx"] = df_sankey_rtt["sonda_nombre"].map(idx_rtt)
            df_sankey_rtt["dst_idx"] = df_sankey_rtt["sitio_web"].map(idx_rtt)

            fig_s1 = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=list(nodos_rtt),
                        ),
                        link=dict(
                            source=df_sankey_rtt["src_idx"],
                            target=df_sankey_rtt["dst_idx"],
                            value=df_sankey_rtt["rtt_avg_ms"],
                            label=df_sankey_rtt["rtt_avg_ms"].round(2).astype(str) + " ms",
                        ),
                    )
                ]
            )

            fig_s1.update_layout(
                title_text=f"Sankey RTT Origen (ISPs) → Destino ({sitio_sel})",
                font_size=12,
                height=500,
            )
            st.plotly_chart(fig_s1, use_container_width=True)

    st.markdown("---")

    # ==============================================================================
    # 🔀 SANKEY 2: RUTAS DE ENTRADA POR ASN (IMAGEN 2 DE LA PROF.)
    # (Origen → Penúltimo ASN → Último ASN → Destino)
    # ==============================================================================
    st.subheader(f"🔀 Rutas de Entrada: Origen → Penúltimo ASN → Último ASN → {sitio_sel}")
    st.caption(
        "Visualiza el camino de los Sistemas Autónomos (ASN) transitados desde las sondas/ISPs hasta llegar al ASN del destino."
    )

    if not df_saltos_filt.empty:
        # Extraer el primer ASN (Origen), el penúltimo ASN y el último ASN por cada traza
        df_s_valid = df_saltos[
            (df_saltos["sitio_web"] == sitio_sel)
            & (df_saltos["sonda_nombre"].isin(sondas_sel))
            & (df_saltos["proveedor_salto"].notnull())
            & (df_saltos["proveedor_salto"] != "N/A")
            & (df_saltos["proveedor_salto"] != "*")
        ].copy()

        if not df_s_valid.empty:
            # Agrupar trazas para obtener Origen, Penúltimo y Último ASN por cada flujo
            trazas_asn = (
                df_s_valid.groupby(["sonda_nombre", "timestamp"])["proveedor_salto"]
                .apply(list)
                .reset_index()
            )

            enlaces_list = []
            for _, row in trazas_asn.iterrows():
                path = row["proveedor_salto"]
                sonda = row["sonda_nombre"]
                if len(path) >= 2:
                    penultimo = path[-2]
                    ultimo = path[-1]
                    enlaces_list.append(
                        {
                            "Origen": sonda,
                            "Penultimo_ASN": f"Penúltimo: {penultimo}",
                            "Ultimo_ASN": f"Último: {ultimo}",
                            "Destino": sitio_sel,
                        }
                    )
                elif len(path) == 1:
                    ultimo = path[0]
                    enlaces_list.append(
                        {
                            "Origen": sonda,
                            "Penultimo_ASN": f"Penúltimo: {ultimo}",
                            "Ultimo_ASN": f"Último: {ultimo}",
                            "Destino": sitio_sel,
                        }
                    )

            df_paths = pd.DataFrame(enlaces_list)

            if not df_paths.empty:
                # 1. Origen -> Penúltimo
                l1 = (
                    df_paths.groupby(["Origen", "Penultimo_ASN"])
                    .size()
                    .reset_index(name="count")
                    .rename(columns={"Origen": "source", "Penultimo_ASN": "target"})
                )

                # 2. Penúltimo -> Último
                l2 = (
                    df_paths.groupby(["Penultimo_ASN", "Ultimo_ASN"])
                    .size()
                    .reset_index(name="count")
                    .rename(columns={"Penultimo_ASN": "source", "Ultimo_ASN": "target"})
                )

                # 3. Último -> Destino
                l3 = (
                    df_paths.groupby(["Ultimo_ASN", "Destino"])
                    .size()
                    .reset_index(name="count")
                    .rename(columns={"Ultimo_ASN": "source", "Destino": "target"})
                )

                df_links = pd.concat([l1, l2, l3], ignore_index=True)

                nodos_asn = pd.unique(df_links[["source", "target"]].values.ravel())
                idx_asn = {n: i for i, n in enumerate(nodos_asn)}

                df_links["source_idx"] = df_links["source"].map(idx_asn)
                df_links["target_idx"] = df_links["target"].map(idx_asn)

                fig_s2 = go.Figure(
                    data=[
                        go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=list(nodos_asn),
                            ),
                            link=dict(
                                source=df_links["source_idx"],
                                target=df_links["target_idx"],
                                value=df_links["count"],
                            ),
                        )
                    ]
                )

                fig_s2.update_layout(
                    title_text=f"Rutas de entrada: origen → penúltimo ASN → último ASN → {sitio_sel}",
                    font_size=12,
                    height=550,
                )
                st.plotly_chart(fig_s2, use_container_width=True)
            else:
                st.info("No hay suficientes saltos ASN para armar el flujo de entrada.")
        else:
            st.info("No hay trazas de proveedores/ASN registradas para los filtros seleccionados.")

except FileNotFoundError:
    st.error(
        "Ejecuta `procesar_datos.py` para sincronizar los timestamps con UTC-4 y generar la estructura de saltos."
    )