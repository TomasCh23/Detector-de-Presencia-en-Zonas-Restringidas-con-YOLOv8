# Detector-de-Presencia-en-Zonas-Restringidas-con-YOLOv8

👁️ Detector de Presencia en Zonas Restringidas con YOLOv8

Sistema de visión artificial diseñado para transformar una cámara de vigilancia pasiva en un sensor activo inteligente. Este proyecto permite definir zonas de interés (ROI) interactivamente y detectar intrusiones humanas en tiempo real, generando un registro detallado de eventos para auditoría.

🚀 Características Principales

Definición Interactiva de Zonas: Dibuja polígonos personalizados de cualquier forma sobre el video utilizando el mouse.

Detección de Personas: Implementa YOLOv8 Nano para una detección de humanos rápida y precisa, optimizada para CPU.

Tracking de Objetos (ID): Uso del algoritmo ByteTrack para asignar identificadores únicos y persistentes a cada persona.

Lógica de Intersección Inteligente:

Calcula el porcentaje del área del objeto que entra en la zona.

Umbral de Sensibilidad: Requiere un 20% de superposición para activar la alerta, evitando falsos positivos por sombras o cruces tangenciales.

Sistema Anti-Rebote (Histéresis): Memoria de 15 frames para estabilizar las detecciones y evitar registros duplicados por parpadeos del modelo.

Reportes Automáticos: Exportación de logs en formato .csv con Timestamp, Tipo de Evento (Entrada/Salida), ID y FPS promedio.

Sincronización de Video: Ajuste automático de la velocidad de reproducción para analizar videos pre-grabados sin distorsión temporal.

📁 Recursos de Prueba

El repositorio incluye un video de demostración para probar el sistema inmediatamente sin necesidad de webcam.

Ubicación: assets/video_prueba.mp4.

Descripción: Video de ejemplo mostrando entrada y salida de personas en un pasillo.

🛠️ Requisitos e Instalación

Prerrequisitos

Python 3.8 o superior.

Cámara Web o archivo de video .mp4.
