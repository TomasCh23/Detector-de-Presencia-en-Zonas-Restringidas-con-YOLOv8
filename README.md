# 👁️ Detector de Presencia en Zonas Restringidas con YOLOv8

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red)

Sistema de visión artificial diseñado para transformar una cámara de vigilancia pasiva en un sensor activo inteligente. Este proyecto permite definir zonas de interés (ROI) interactivamente y detectar intrusiones humanas en tiempo real, generando un registro detallado de eventos para auditoría.

## 🚀 Características Principales

* **Definición Interactiva de Zonas:** Dibuja polígonos personalizados de cualquier forma sobre el video utilizando el mouse.
* **Detección de Personas:** Implementa **YOLOv8 Nano** para una detección de humanos rápida y precisa.
* **Tracking de Objetos (ID):** Uso del algoritmo **ByteTrack** para asignar identificadores únicos y persistentes a cada persona.
* **Lógica de Intersección Inteligente:**
    * Calcula el porcentaje del área del objeto que entra en la zona.
    * **Umbral de Sensibilidad:** Requiere un **20% de superposición** para activar la alerta.
* **Sistema Anti-Rebote (Histéresis):** Memoria de **15 frames** para estabilizar las detecciones y evitar registros duplicados.
* **Reportes Automáticos:** Exportación de logs en formato `.csv` con Timestamp, Tipo de Evento, ID y FPS promedio.
* **Sincronización de Video:** Ajuste automático de la velocidad de reproducción para analizar videos pre-grabados.

## 📁 Estructura del Proyecto

```text
├── assets/
│   └── video_prueba.mp4   # Video de demostración (Opcional)
├── main.py                # Código fuente principal
├── requirements.txt       # Lista de dependencias necesarias
└── README.md              # Documentación del proyecto
```

🛠️ **Instalación y Dependencias**

Este proyecto utiliza librerías externas para el procesamiento de imágenes e inteligencia artificial.

**1. Clonar el repositorio** 
```
git clone [https://github.com/TomasCh23/Detector-de-Presencia-en-Zonas-Restringidas-con-YOLOv8.git](https://github.com/TomasCh23/Detector-de-Presencia-en-Zonas-Restringidas-con-YOLOv8.git)
cd Detector-de-Presencia-en-Zonas-Restringidas-con-YOLOv8
```
**2. Instalar Dependencias**

Para garantizar la compatibilidad y el correcto funcionamiento, se proporciona el archivo `requirements.txt` con las versiones exactas de las librerías probadas.

Ejecuta el siguiente comando en tu terminal:
```
pip install -r requirements.txt
```
**Librerías principales utilizadas:**

* `ultralytics`: Modelo de detección y tracking (YOLOv8).

* `opencv-python`: Procesamiento de imágenes, GUI y manejo de video.

* `numpy`: Operaciones matemáticas y manejo de matrices para el cálculo de áreas (ROI).

* `lap`: Necesario para el algoritmo de tracking lineal.

💻 Uso

**1. Ejecutar el programa:**
```
python main.py
```
**2. Configuración Inicial (Menú en Consola):**

* **Opción 1:** Usar Webcam en tiempo real.

* **Opción 2:** Cargar archivo de video (se abrirá una ventana para seleccionarlo).

* **Opcional:** Ajustar la velocidad de reproducción (1.0 = normal).

**3. Definir la Zona (ROI):**

* El video se pausará en el primer fotograma.

* Click Izquierdo: Añadir punto al polígono.

* Click Derecho: Eliminar el último punto.

* ENTER: Confirmar zona y comenzar la vigilancia.

**4. Durante la Vigilancia:**

* 🟦 **Cuadro Azul:** Persona detectada fuera de la zona segura.

* 🟥 **Cuadro Rojo:** Intruso detectado (Entrada confirmada).

* **Teclas:**

  * `S`: Guardar reporte CSV inmediatamente.

  * `Q`: Salir del programa.

🧠 **Lógica del Sistema**

**1. Intersección sobre Unión (IoU) Modificada**

El sistema calcula matemáticamente qué porcentaje del "Bounding Box" de la persona está dentro del polígono dibujado:

$$ Porcentaje = \frac{Area_{Intersección}}{Area_{Persona}} $$

Si $Porcentaje \ge 0.20$ (20%), se considera una intrusión válida.

**2. Máquina de Estados (Tracking)**

Para evitar múltiples alertas por el mismo evento, cada ID detectado mantiene un estado interno:

* **Estado:** `Dentro` o `Fuera`.

* **Contador de Olvido:** Si un intruso sale de la zona, el sistema espera **15 frames consecutivos** (aprox. 0.5 segundos) antes de registrar oficialmente la "SALIDA". Esto filtra errores momentáneos de detección.

📊 **Reportes**

El sistema genera automáticamente un archivo `reporte_vigilancia_[FECHA].csv` con la siguiente estructura:

```
| Timestamp | Evento       | ID_Objeto | FPS_Sesion |
|           |              |           |            |
| 14:30:05  | ENTRADA ZONA |    ID_1   |        32.5|
| 14:30:12  | SALIDA ZONA  |    ID_1   |        31.8|
| 14:32:01  | ENTRADA ZONA |    ID_5   |        33.0|
```

👥 **Autores**
Proyecto realizado para la asignatura de **Visión Artificial**.

* Tomás Chaigneau
* Carlos Downing
* Juan Reyes
* Andres Ibarra




















