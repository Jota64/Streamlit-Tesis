# Portal ULA — versión para jurado

Versión de presentación del portal asociado al Trabajo de Grado sobre conectividad hacia servicios digitales de la Universidad de Los Andes.

## Objetivo de esta versión

Mostrar únicamente resultados y visualizaciones directamente alineados con los objetivos y conclusiones de la tesis, evitando herramientas internas de exploración que puedan distraer durante la defensa.

## Secciones visibles

1. Resumen.
2. Desempeño y alcanzabilidad.
3. Enrutamiento.
4. OONI.
5. Cobertura y metodología.

La sección de enrutamiento incluye un **Sankey interactivo ASN/IXP**. Se puede pasar el cursor sobre nodos/enlaces y arrastrar nodos para reorganizar la vista. También puede seleccionarse si mostrar 5, 8 o 12 familias de ruta frecuentes. El ancho representa frecuencia de traceroutes, no RTT ni volumen de tráfico.

## Criterios importantes

- El corpus es cerrado y corresponde al utilizado en la tesis.
- No se realizan mediciones en tiempo real.
- `hop=255` no se interpreta como 255 saltos.
- Las trazas incompletas no se conectan artificialmente al destino.
- FL-IX y NAP Colombia se preservan como nodos IXP cuando son observables.
- La geolocalización IP es auxiliar.
- OONI es evidencia complementaria; anomaly/failure no equivalen automáticamente a censura o indisponibilidad.

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

Los conjuntos procesados requeridos ya están incluidos en `cap4_data/`.


## Visualizaciones de enrutamiento solicitadas en tutoría

La pestaña **Enrutamiento** abre con dos Sankey interactivos globales: (1) todas las rutas ASN/IXP observadas hacia los cuatro servicios simultáneamente, sin recorte a top-N; y (2) rutas de entrada completadas hacia los cuatro servicios, construidas exclusivamente con traceroutes que alcanzaron el destino. Ambas vistas permiten hover y reorganización de nodos.
