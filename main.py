import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import csv
import os
import sys

# ==========================================
# 1. CONFIGURACIÓN GENERAL DEL SISTEMA
# ==========================================
# Cargamos el modelo YOLOv8 Nano. Es la versión más ligera y rápida, ideal para CPU.
# La primera vez que se ejecuta, descargará el archivo 'yolov8n.pt' automáticamente.
model = YOLO('yolov8n.pt') 

# Definimos qué clases del dataset COCO queremos detectar.
# El ID 0 corresponde a 'person'. Si quisieras autos, agregarías el 2.
TARGET_CLASSES = [0] 

# ==========================================
# 2. VARIABLES GLOBALES (ESTADO DEL SISTEMA)
# ==========================================
roi_points = []          # Almacena las coordenadas [(x,y), ...] del polígono dibujado por el usuario
event_buffer = []        # Memoria temporal (RAM) para guardar eventos antes de escribirlos en el CSV
total_frames_processed = 0   # Contador para calcular el promedio de FPS al final

# Diccionario para el Tracking e Histéresis
# Estructura: { ID_OBJETO: { 'ya_entro': Bool, 'frames_fuera': Int } }
# Nos permite recordar si una persona específica ya fue contada o si está saliendo.
track_history = {} 

# --- PARÁMETROS DE SENSIBILIDAD ---
# Porcentaje del cuerpo que debe estar dentro de la zona para activar la alerta.
# 0.20 significa el 20%. Esto evita falsas alarmas si alguien solo pisa el borde.
UMBRAL_ENTRADA = 0.20    

# Cantidad de frames consecutivos que un objeto debe estar FUERA para considerar que salió.
# Esto es el sistema "Anti-Rebote": evita que si la detección parpadea, se cuente doble.
FRAMES_OLVIDO = 15       

# ==========================================
# 3. FUNCIONES DE UTILIDAD
# ==========================================

def get_user_configuration():
    """
    Despliega el menú inicial. Intenta abrir una ventana gráfica para seleccionar archivo.
    Si falla (común en algunos entornos), permite escribir la ruta en la terminal.
    Retorna: La fuente de video (0 o ruta) y el factor de velocidad.
    """
    print("\n" + "="*40)
    print("   SISTEMA DE VIGILANCIA - CONFIGURACION")
    print("="*40)
    print("1. Usar Webcam en vivo (Cámara por defecto)")
    print("2. Analizar un archivo de Video")
    print("-" * 40)
    
    opcion = input(">> Seleccione una opcion (1 o 2): ").strip()
    
    source = 0      # Por defecto webcam (ID 0)
    speed = 1.0     # Velocidad normal
    
    if opcion == '2':
        file_path = ""
        use_gui = True
        
        # Intentamos importar Tkinter para la ventana de selección
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            print("[AVISO] La librería 'tkinter' no está instalada.")
            use_gui = False

        # Si Tkinter funciona, abrimos el explorador de archivos
        if use_gui:
            print("\n[INFO] Abriendo ventana de selección de archivo...")
            try:
                root = tk.Tk()
                root.attributes('-topmost', True) # Forzamos la ventana al frente
                root.withdraw() # Ocultamos la ventana principal fea de Tkinter
                
                file_path = filedialog.askopenfilename(
                    title="Seleccionar Video de Vigilancia",
                    filetypes=[("Archivos de Video", "*.mp4 *.avi *.mov *.mkv")]
                )
                root.destroy()
            except Exception as e:
                print(f"[ERROR GUI] Falló la ventana emergente: {e}")
        
        # Si no se seleccionó nada (o falló la GUI), pedimos ruta manual
        if not file_path:
            print("\n" + "!"*50)
            print("[MODO MANUAL ACTIVADO]")
            print("Por favor, ARRASTRA el archivo de video a esta terminal o escribe la ruta.")
            file_path = input(">> Ruta del archivo: ").strip()
            # Limpiamos comillas si el sistema operativo las agregó
            if file_path.startswith('"') and file_path.endswith('"'):
                file_path = file_path[1:-1]
                
        # Verificamos que el archivo realmente exista
        if not file_path or not os.path.exists(file_path):
            print(f"\n[ERROR] El archivo no existe: '{file_path}'")
            input("Presiona Enter para salir...")
            exit()
            
        source = file_path
        
        # Opción extra para controlar la velocidad de análisis
        try:
            print("\n--- Velocidad de Reproducción ---")
            print("1.0 = Normal | 0.5 = Lento | 2.0 = Rápido")
            speed_input = input(">> Ingrese valor (Enter para 1.0): ").strip()
            speed = float(speed_input) if speed_input else 1.0
        except ValueError:
            speed = 1.0
            
    elif opcion != '1':
        print("[INFO] Opción no reconocida. Usando Webcam por defecto.")
        
    return source, speed

def mouse_callback(event, x, y, flags, param):
    """
    Función 'callback' que escucha los clics del mouse.
    Se ejecuta automáticamente cada vez que tocas la ventana de OpenCV.
    """
    global roi_points
    # Click Izquierdo: Agregar punto al polígono
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
    # Click Derecho: Borrar último punto (deshacer)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if roi_points:
            roi_points.pop()

def calculate_iou_percentage(bbox, polygon_points, frame_shape):
    """
    CALCULO MATEMÁTICO DEL PROYECTO.
    Determina qué porcentaje del área del objeto (caja) cae dentro de la zona (polígono).
    """
    # 1. Creamos una imagen vacía (negra) y dibujamos la ZONA en blanco (255)
    mask_roi = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [np.array(polygon_points, np.int32)], 255)

    # 2. Creamos otra imagen vacía y dibujamos la CAJA DEL OBJETO en blanco
    mask_bbox = np.zeros(frame_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(mask_bbox, (x1, y1), (x2, y2), 255, -1)

    # 3. Calculamos la INTERSECCIÓN (AND lógico). 
    # Solo quedan píxeles blancos donde AMBAS figuras se superponen.
    intersection = cv2.bitwise_and(mask_roi, mask_bbox)
    intersection_area = cv2.countNonZero(intersection)
    
    # 4. Calculamos el área total de la caja del objeto
    box_area = (x2 - x1) * (y2 - y1)

    if box_area == 0: return 0.0

    # 5. Retornamos la proporción (0.0 a 1.0)
    return intersection_area / box_area

def save_csv_log(avg_fps):
    """
    Guarda los eventos acumulados en un archivo CSV.
    Abre una ventana de diálogo para elegir dónde guardar.
    """
    if not event_buffer:
        print("\n[AVISO] No hay eventos registrados para guardar.")
        return False

    print("\n[SISTEMA] Preparando para guardar reporte...")
    
    file_path = ""
    use_gui = True
    
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        use_gui = False

    # Nombre sugerido por defecto con fecha y hora
    default_filename = f"reporte_vigilancia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    if use_gui:
        try:
            root = tk.Tk()
            root.attributes('-topmost', True) 
            root.withdraw()
            
            # Abrir ventana "Guardar Como"
            file_path = filedialog.asksaveasfilename(
                title="Guardar Reporte CSV",
                defaultextension=".csv",
                initialfile=default_filename,
                filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
            )
            root.destroy()
        except Exception:
            file_path = ""

    # Si falla la ventana, preguntar en consola
    if not file_path:
        print("\n[INFO] No se seleccionó archivo en la ventana.")
        opcion = input(">> ¿Deseas escribir la ruta manualmente? (s/n): ").lower()
        if opcion == 's':
            file_path = input(f">> Nombre del archivo (Enter para '{default_filename}'): ").strip()
            if not file_path: file_path = default_filename
            if not file_path.endswith('.csv'): file_path += ".csv"
        else:
            return False

    # Escribir el archivo
    try:
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Si el archivo es nuevo, escribimos los encabezados
            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writerow(["Timestamp", "Evento", "ID_Objeto", "FPS_Sesion"])
            
            for event in event_buffer:
                row = event[:] 
                row.append(f"{avg_fps:.2f}") # Agregamos FPS al final
                writer.writerow(row)
                
        print(f"\n[ÉXITO] Reporte guardado en:\n -> {os.path.abspath(file_path)}")
        event_buffer.clear() # Limpiamos la memoria
        return True
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] No se pudo guardar el archivo: {e}")
        return False

# ==========================================
# 4. PROGRAMA PRINCIPAL (MAIN LOOP)
# ==========================================

# Paso 1: Obtener configuración del usuario
VIDEO_SOURCE, SPEED_FACTOR = get_user_configuration()

print(f"\n[SISTEMA] Iniciando vigilancia con fuente: {VIDEO_SOURCE}")
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Verificación de seguridad
if not cap.isOpened():
    print("XXX ERROR CRÍTICO: No se pudo abrir la fuente de video XXX")
    exit()

# Leemos el primer frame solo para configurar la zona
ret, first_frame = cap.read()
if not ret:
    print("Error: No se pudo leer el primer frame.")
    exit()

cv2.namedWindow("Sistema de Vigilancia")
cv2.setMouseCallback("Sistema de Vigilancia", mouse_callback)

# -------------------------------------------------
# FASE 1: DIBUJO DE ZONA (Video Pausado)
# -------------------------------------------------
print("\n--- FASE 1: DEFINICIÓN DE ZONA ---")
print("1. Usa el mouse para dibujar los puntos de la zona en la ventana.")
print("2. Presiona ENTER cuando termines.")

is_configuring = True
while is_configuring:
    # Usamos una copia para no ensuciar la imagen original con líneas
    display_img = first_frame.copy()
    
    cv2.putText(display_img, "MODO CONFIGURACION", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_img, "Dibuja puntos y presiona ENTER", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Dibujamos el polígono en tiempo real
    if len(roi_points) > 0:
        pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(display_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for pt in roi_points:
            cv2.circle(display_img, pt, 4, (0, 0, 255), -1)

    cv2.imshow("Sistema de Vigilancia", display_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 13: # Tecla ENTER
        if len(roi_points) > 2:
            print("Zona definida. Iniciando Tracking...")
            is_configuring = False
        else:
            print(">> [AVISO] Marca al menos 3 puntos para definir la zona.")
    elif key == ord('q'): # Tecla Q para salir antes de empezar
        cap.release()
        cv2.destroyAllWindows()
        exit()

# -------------------------------------------------
# CALCULO DE VELOCIDAD DE REPRODUCCIÓN (Sincronización)
# -------------------------------------------------
fps_original = cap.get(cv2.CAP_PROP_FPS)
if fps_original <= 0 or fps_original > 120: fps_original = 30 # Corrección por si el video tiene metadatos rotos

# Calculamos cuánto esperar entre frames (en ms) para mantener la velocidad real
delay_time = int((1000 / fps_original) / SPEED_FACTOR)
if delay_time < 1: delay_time = 1 

print(f"[INFO] Velocidad x{SPEED_FACTOR}. Delay calculado: {delay_time}ms")
print(">> Presiona 'Q' para salir y guardar.")
print(">> Presiona 'S' para guardar en cualquier momento.")

# -------------------------------------------------
# FASE 2: BUCLE DE VIGILANCIA (Tiempo Real)
# -------------------------------------------------
start_time_monitoring = time.time()
prev_frame_time = 0

while True:
    ret, frame = cap.read()
    
    # === DETECCIÓN DE FIN DE VIDEO ===
    if not ret: 
        print("\n[INFO] Fin del video detectado.")
        # Calculamos FPS finales
        elapsed_time = time.time() - start_time_monitoring
        avg_fps_final = total_frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Guardado automático al terminar
        print("[INFO] Generando reporte final...")
        save_csv_log(avg_fps_final)
        
        print("[INFO] Listo. Puedes cerrar el programa.")
        print("Presiona cualquier tecla en la ventana para cerrar.")
        cv2.waitKey(0) 
        break

    # === MÉTRICAS DE RENDIMIENTO ===
    new_frame_time = time.time()
    fps_real = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    total_frames_processed += 1
    
    # === INTELIGENCIA ARTIFICIAL (YOLO) ===
    # persist=True es VITAL para el tracking (mantener el mismo ID entre frames)
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
    
    # Dibujamos la zona fija en verde
    roi_poly = np.array(roi_points, np.int32)
    cv2.polylines(frame, [roi_poly], True, (0, 255, 0), 2)
    
    # Procesamos las detecciones
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, cls in zip(boxes, ids, classes):
            # Filtramos para que solo procese Personas (clase 0)
            if cls not in TARGET_CLASSES: continue

            x1, y1, x2, y2 = box
            
            # Calculamos el porcentaje de intrusión
            overlap_percent = calculate_iou_percentage((x1, y1, x2, y2), roi_points, frame.shape)
            
            # --- MÁQUINA DE ESTADOS (Tracking) ---
            # Inicializamos el ID en el historial si es nuevo
            if track_id not in track_history:
                track_history[track_id] = {'ya_entro': False, 'frames_fuera': 0}

            estado = track_history[track_id]
            is_inside_now = overlap_percent >= UMBRAL_ENTRADA

            if is_inside_now:
                # === CASO: PERSONA DENTRO ===
                estado['frames_fuera'] = 0 # Reseteamos contador de olvido
                
                # Si es la primera vez que entra, registramos el evento
                if not estado['ya_entro']:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    msg = "ENTRADA ZONA"
                    print(f"[{timestamp}] ID: {track_id} -> {msg}")
                    event_buffer.append([timestamp, msg, f"ID_{track_id}"])
                    estado['ya_entro'] = True # Marcamos como "Adentro"
                
                # Visualización ROJA
                color = (0, 0, 255) 
                label = f"ID:{track_id} INTRUSO {overlap_percent:.0%}"

            else:
                # === CASO: PERSONA FUERA ===
                # Si no toca nada (0%), aumentamos contador de frames fuera
                if overlap_percent == 0:
                    estado['frames_fuera'] += 1
                else:
                    estado['frames_fuera'] = 0 # Si toca el borde, reseteamos (no está totalmente fuera)
                
                # --- LOGICA DE SALIDA (HISTÉRESIS) ---
                # Solo registramos la salida si lleva más de 15 frames fuera
                if estado['frames_fuera'] > FRAMES_OLVIDO:
                    if estado['ya_entro']:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        msg = "SALIDA ZONA"
                        print(f"[{timestamp}] ID: {track_id} -> {msg}")
                        event_buffer.append([timestamp, msg, f"ID_{track_id}"])
                        estado['ya_entro'] = False # Reseteamos estado, puede volver a entrar
                
                # Visualización AZUL
                color = (255, 0, 0) 
                label = f"ID:{track_id}"

            # Dibujamos caja y texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # === INTERFAZ DE USUARIO (HUD) ===
    elapsed_time = time.time() - start_time_monitoring
    avg_fps_session = total_frames_processed / elapsed_time if elapsed_time > 0 else 0
    
    cv2.putText(frame, f"FPS: {int(fps_real)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Eventos: {len(event_buffer)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Sistema de Vigilancia", frame)

    # === CONTROL DE TECLADO ===
    key = cv2.waitKey(delay_time) & 0xFF
    if key == ord('q'):
        # Salida manual con guardado
        print("\n[INFO] Salida manual detectada...")
        elapsed_time = time.time() - start_time_monitoring
        avg_fps_final = total_frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Si hay datos pendientes, preguntamos dónde guardar antes de cerrar
        if event_buffer:
            print("[INFO] Eventos pendientes. Abriendo ventana de guardado...")
            save_csv_log(avg_fps_final)
        
        break
    elif key == ord('s'):
        # Guardado manual sin salir
        save_csv_log(avg_fps_session)
        # Mensaje visual en pantalla
        cv2.putText(frame, "GUARDADO", (frame.shape[1]//2, frame.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sistema de Vigilancia", frame)
        cv2.waitKey(500) # Pausa breve para leer

# Limpieza final
cap.release()
cv2.destroyAllWindows()
