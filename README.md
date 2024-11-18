# Fancine 2024 - Málaga: Vision por Computadora en Tiempo Real
Código utilizado en la charla "Cómo enseñamos a las máquinas: introducción al aprendizaje de la IA" dentro de Fancine 2024

## Descripción
Este proyecto contiene dos scripts de visión por computadora presentados en el marco del **Fancine 2024 de Málaga**, diseñados para tareas en tiempo real:
1. **`yolo_detector.py`**: Utiliza YOLO11 para clasificación, detección, segmentación y detección de poses humanas.
2. **`mapa_profundidad.py`**: Utiliza MiDaS para obtener mapas de profundidad en tiempo real.

Ambos scripts están diseñados para trabajar con cámaras y mostrar resultados en tiempo real.

---

## Scripts y Funcionalidades

### 1. YOLO Detector (`yolo_detector.py`)
Realiza diferentes tareas basadas en el modelo YOLO11:
- **Clasificación** de objetos.
- **Detección** de objetos.
- **Segmentación** semántica.
- **Detección de poses humanas**.

#### Uso
Ejecuta el script especificando la tarea deseada:
```bash
python yolo_detector.py <tarea> [opciones]
```

![detectar](https://github.com/user-attachments/assets/2b8e7bcb-00f8-4d1e-b2d2-324398ea94b9)

![poses](https://github.com/user-attachments/assets/95eedcfd-37eb-49b6-b2ac-56245c19c130)

Parámetros principales:
- `<tarea>`:
  - `clasificar` → Clasificación de objetos.
  - `detectar` → Detección de objetos.
  - `segmentar` → Segmentación semántica.
  - `pose` → Detección de poses humanas.

- `[opciones]`:
  - `--no-flip` → No invierte el video como espejo.

Ejemplo:
```bash
python yolo_detector.py pose
```

---

### 2. Mapa de Profundidad (`mapa_profundidad.py`)
Utiliza MiDaS para generar mapas de profundidad en tiempo real desde la cámara.

![mapa-profundidad](https://github.com/user-attachments/assets/2c3c81b2-a9c4-47cb-813d-46d5383a4861)

#### Uso
Ejecuta el script seleccionando el tamaño del modelo:
```bash
python mapa_profundidad.py <size> [opciones]
```

Parámetros principales:
- `<size>`:
  - `large` → Modelo grande para mayor precisión.
  - `hybrid` → Compromiso entre tamaño y precisión.
  - `small` → Modelo ligero para mayor velocidad.

- `[opciones]`:
  - `--no-flip` → No invierte el video como espejo.
  - `--use-gpu` → Usa GPU para acelerar el procesamiento (si está disponible).

Ejemplo:
```bash
python mapa_profundidad.py large --use-gpu
```

---

## Instalación

### Requisitos previos
- Python 3.8 o superior.
- `pip` para la instalación de paquetes.

### Pasos para la instalación

1. **Clonar o descargar el proyecto.**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <CARPETA_DEL_PROYECTO>
   ```

2. **Crear un entorno virtual.**
   ```bash
   python -m venv venv
   source venv/bin/activate    # En Windows: venv\Scripts\activate
   ```

3. **Instalar pytorch con soporte CUDA (versión según sistema). YOLO da error con pytorch >= 2.4**
   ```bash
   pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Instalar las dependencias.**
   ```bash
   pip install -r requirements.txt
   ```

---

## Notas importantes
- El script de YOLO espera los archivos de pesos:
  - `yolo11n-pose.pt` → Para detección de poses.
  - `yolo11n-seg.pt` → Para segmentación.

- MiDaS descarga automáticamente los modelos necesarios.

- Usa la tecla `q` para salir del modo de video.

---

## Créditos
- **YOLO**: Basado en la librería [Ultralytics](https://docs.ultralytics.com/).
- **MiDaS**: Proyecto de mapas de profundidad de [Intel ISL](https://github.com/isl-org/MiDaS).
