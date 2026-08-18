import json
import time
import pandas as pd
import requests

MAPA_SITIOS = {
    "150.185.169.46": "GRADOS",
    "190.168.5.17": "Saber ULA",
    "190.168.5.91": "Vereda (Red de Arte)",
    "150.185.169.243": "Intranet",
}

MAPA_SONDAS = {
    22510: {
        "nombre": "ULA (Mérida)",
        "isp": "ULA",
        "asn": "AS23007",
        "ubicacion": "Mérida, Venezuela",
        "tipo": "Nacional",
        "ip": "Local/Interna",
    },
    1010818: {
        "nombre": "CANTV (Caracas)",
        "isp": "CANTV",
        "asn": "AS8048",
        "ubicacion": "Caracas, Venezuela",
        "tipo": "Nacional",
        "ip": "190.77.177.120",
    },
    7594: {
        "nombre": "Inter (Barquisimeto)",
        "isp": "Corporación Telemic",
        "asn": "AS21826",
        "ubicacion": "Barquisimeto, Venezuela",
        "tipo": "Nacional",
        "ip": "200.82.230.154",
    },
    1003784: {
        "nombre": "Movistar (Maracay)",
        "isp": "Telefónica Venezolana",
        "asn": "AS6306",
        "ubicacion": "Maracay, Venezuela",
        "tipo": "Nacional",
        "ip": "186.24.57.243",
    },
    32877: {
        "nombre": "Airtek (Maracaibo)",
        "isp": "Airtek Solutions",
        "asn": "AS61461",
        "ubicacion": "Maracaibo, Venezuela",
        "tipo": "Nacional",
        "ip": "38.25.230.77",
    },
    1014400: {
        "nombre": "NetUno (Caracas)",
        "isp": "Net Uno",
        "asn": "AS11562",
        "ubicacion": "Caracas, Venezuela",
        "tipo": "Nacional",
        "ip": "190.153.16.65",
    },
    6978: {
        "nombre": "Tiggee (Bogotá)",
        "isp": "Tiggee LLC",
        "asn": "AS16552",
        "ubicacion": "Bogotá, Colombia",
        "tipo": "Internacional",
        "ip": "156.154.123.254",
    },
    1011292: {
        "nombre": "Satnet (Quito)",
        "isp": "Satnet",
        "asn": "AS14522",
        "ubicacion": "Quito, Ecuador",
        "tipo": "Internacional",
        "ip": "186.69.158.99",
    },
    1010206: {
        "nombre": "AT&T (Miami)",
        "isp": "AT&T Enterprises",
        "asn": "AS7018",
        "ubicacion": "Miami, USA",
        "tipo": "Internacional",
        "ip": "99.26.108.135",
    },
}

# Diccionario para guardar el caché de IPs y no hacer consultas repetidas a la API
CACHE_GEO_IP = {}


def obtener_geolocalizacion_ip(ip):
    """Consulta país, ciudad y ISP de una IP intermedia pública.

    Ignora IPs privadas (10.x.x.x, 192.168.x.x, 172.16-31.x.x) y timeouts.
    """
    if ip == "*" or not ip:
        return {"pais": "N/A", "ciudad": "N/A", "org": "Sin Respuesta (*)"}

    # Verificar si es una IP privada (red interna/reservada)
    if (
        ip.startswith("10.")
        or ip.startswith("192.168.")
        or ip.startswith("172.16.")
    ):
        return {
            "pais": "Red Privada",
            "ciudad": "Interna",
            "org": "LAN / Enlace Privado",
        }

    # Si ya la consultamos anteriormente, la tomamos del caché
    if ip in CACHE_GEO_IP:
        return CACHE_GEO_IP[ip]

    try:
        # Consulta a API pública de Geolocalización de IP
        url = (
            f"http://ip-api.com/json/{ip}?fields=status,country,city,isp,org,as"
        )
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                info = {
                    "pais": data.get("country", "Desconocido"),
                    "ciudad": data.get("city", "Desconocida"),
                    "org": data.get("org")
                    or data.get("isp")
                    or data.get("as", "Desconocido"),
                }
                CACHE_GEO_IP[ip] = info
                time.sleep(0.05)  # Respetar rate-limit liviano de la API
                return info
    except Exception:
        pass

    info_fallback = {
        "pais": "Desconocido",
        "ciudad": "Desconocida",
        "org": "Desconocido",
    }
    CACHE_GEO_IP[ip] = info_fallback
    return info_fallback


def cargar_json(nombre_archivo):
    try:
        with open(nombre_archivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error al cargar {nombre_archivo}: {e}")
        return []


def obtener_info_sonda(prb_id):
    return MAPA_SONDAS.get(
        prb_id,
        {
            "nombre": f"Probe {prb_id}",
            "isp": "Desconocido",
            "asn": "N/A",
            "ubicacion": "Desconocida",
            "tipo": "Desconocido",
            "ip": "N/A",
        },
    )


def procesar_pings(data_pings):
    registros = []
    for item in data_pings:
        prb_id = item.get("prb_id")
        dst_addr = item.get("dst_addr")
        timestamp = item.get("timestamp")
        avg_rtt = item.get("avg", -1)

        if timestamp:
            fecha_hora_utc = pd.to_datetime(timestamp, unit="s", utc=True)
            fecha_hora_local = fecha_hora_utc.tz_convert(
                "America/Caracas"
            ).tz_localize(None)

            sitio = MAPA_SITIOS.get(dst_addr, f"IP Desconocida ({dst_addr})")
            s_info = obtener_info_sonda(prb_id)
            alcanzable = True if avg_rtt != -1 else False

            sent = item.get("sent", 0)
            rcvd = item.get("rcvd", 0)
            packet_loss = (
                round(((sent - rcvd) / sent) * 100, 1) if sent > 0 else 100.0
            )

            hora_int = fecha_hora_local.hour
            bloque_horario = "Otros"
            if 9 <= hora_int <= 11:
                bloque_horario = "10:00 AM (Mañana)"
            elif 21 <= hora_int <= 23:
                bloque_horario = "10:00 PM (Noche)"

            registros.append(
                {
                    "timestamp": fecha_hora_local,
                    "fecha": fecha_hora_local.date(),
                    "hora": fecha_hora_local.strftime("%H:%M:%S"),
                    "hora_corta": fecha_hora_local.strftime("%I:%M %p"),
                    "bloque_horario": bloque_horario,
                    "sonda_id": prb_id,
                    "sonda_nombre": s_info["nombre"],
                    "isp": s_info["isp"],
                    "asn": s_info["asn"],
                    "ubicacion": s_info["ubicacion"],
                    "tipo_sonda": s_info["tipo"],
                    "ip_origen": s_info["ip"],
                    "dst_ip": dst_addr,
                    "sitio_web": sitio,
                    "alcanzable": alcanzable,
                    "rtt_avg_ms": avg_rtt if alcanzable else None,
                    "min_ms": item.get("min") if alcanzable else None,
                    "max_ms": item.get("max") if alcanzable else None,
                    "paquetes_enviados": sent,
                    "paquetes_recibidos": rcvd,
                    "packet_loss_pct": packet_loss,
                }
            )
    return pd.DataFrame(registros)


def procesar_traceroutes(data_traceroutes):
    registros = []
    saltos_detalle = []

    print("Geolocalizando saltos de traceroute...")

    for item in data_traceroutes:
        prb_id = item.get("prb_id")
        dst_addr = item.get("dst_addr")
        timestamp = item.get("timestamp")
        result = item.get("result", [])

        if timestamp:
            fecha_hora_utc = pd.to_datetime(timestamp, unit="s", utc=True)
            fecha_hora_local = fecha_hora_utc.tz_convert(
                "America/Caracas"
            ).tz_localize(None)

            sitio = MAPA_SITIOS.get(dst_addr, f"IP Desconocida ({dst_addr})")
            s_info = obtener_info_sonda(prb_id)

            total_saltos = len(result)
            respondio_destino = False
            last_rtt = None

            ips_ruta = []
            ubicaciones_ruta = []

            for hop in result:
                hop_num = hop.get("hop")
                hop_res = hop.get("result", [])
                ip_hop = "*"
                rtt_hop = None

                for r in hop_res:
                    if "from" in r:
                        ip_hop = r["from"]
                        rtt_hop = r.get("rtt")
                        break

                # Geolocalizar la IP intermedia
                geo_info = obtener_geolocalizacion_ip(ip_hop)

                ips_ruta.append(ip_hop)
                if geo_info["pais"] != "N/A":
                    ubicaciones_ruta.append(
                        f"{geo_info['ciudad']}, {geo_info['pais']}"
                    )

                saltos_detalle.append(
                    {
                        "timestamp": fecha_hora_local,
                        "sonda_nombre": s_info["nombre"],
                        "sitio_web": sitio,
                        "hop_num": hop_num,
                        "ip_intermedia": ip_hop,
                        "pais": geo_info["pais"],
                        "ciudad": geo_info["ciudad"],
                        "proveedor_salto": geo_info["org"],
                        "rtt_hop_ms": rtt_hop,
                    }
                )

            if result:
                ultimo_salto = result[-1].get("result", [])
                for r in ultimo_salto:
                    if "from" in r and r["from"] == dst_addr:
                        respondio_destino = True
                        if "rtt" in r:
                            last_rtt = r["rtt"]
                        break

            registros.append(
                {
                    "timestamp": fecha_hora_local,
                    "fecha": fecha_hora_local.date(),
                    "hora": fecha_hora_local.strftime("%H:%M:%S"),
                    "hora_corta": fecha_hora_local.strftime("%I:%M %p"),
                    "sonda_id": prb_id,
                    "sonda_nombre": s_info["nombre"],
                    "isp": s_info["isp"],
                    "asn": s_info["asn"],
                    "ubicacion": s_info["ubicacion"],
                    "tipo_sonda": s_info["tipo"],
                    "dst_ip": dst_addr,
                    "sitio_web": sitio,
                    "respondio_destino": respondio_destino,
                    "total_saltos": total_saltos,
                    "rtt_final_ms": last_rtt,
                    "camino_ips": " ➔ ".join(ips_ruta),
                }
            )

    return pd.DataFrame(registros), pd.DataFrame(saltos_detalle)


if __name__ == "__main__":
    print("Cargando y procesando telemetría en hora local VLA (UTC-4)...")
    pings_raw = cargar_json("ripe_pings_consolidado.json")
    traceroutes_raw = cargar_json("ripe_traceroutes_consolidado.json")

    df_pings = procesar_pings(pings_raw)
    df_traceroutes, df_saltos_detalle = procesar_traceroutes(traceroutes_raw)

    excel_salida = "reporte_telemetria_consolidado.xlsx"
    with pd.ExcelWriter(excel_salida, engine="openpyxl") as writer:
        df_pings.to_excel(writer, sheet_name="Pings_Latencia", index=False)
        df_traceroutes.to_excel(
            writer, sheet_name="Traceroutes_Resumen", index=False
        )
        df_saltos_detalle.to_excel(
            writer, sheet_name="Detalle_Salto_por_Salto", index=False
        )

    print(
        f"[¡LISTO!] Archivo '{excel_salida}' generado con País, Ciudad y Proveedor por salto."
    )