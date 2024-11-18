import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import time
import argparse

def main(size, flip_video, use_gpu):
    # Seleccionar el modelo según el tamaño especificado
    if size == "large":
        model_type = "DPT_Large"
    elif size == "hybrid":
        model_type = "DPT_Hybrid"
    elif size == "small":
        model_type = "MiDaS_small"
    else:
        raise ValueError("El tamaño debe ser 'large', 'hybrid' o 'small'.")

    # Cargar el modelo
    model = torch.hub.load("isl-org/MiDaS", model_type)
    model.eval()

    # Configurar transformaciones para la entrada
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform
    else:
        transform = torch.hub.load("isl-org/MiDaS", "transforms").small_transform

    # Determinar el dispositivo a usar
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model = model.to(device)

    # Inicializar webcam
    cap = cv2.VideoCapture(0)  # Cambia el índice si tienes más de una cámara

    if not cap.isOpened():
        print("No se pudo acceder a la webcam")
        exit()

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame")
            break

        # Invertir el video si está habilitado
        if flip_video:
            frame = cv2.flip(frame, 1)

        # Convertir imagen al formato RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Aplicar transformaciones
        input_tensor = transform(img_rgb).unsqueeze(0)

        # Eliminar la dimensión extra si está presente
        if input_tensor.shape[1] == 1:
            input_tensor = input_tensor.squeeze(1)

        # Mover el tensor al dispositivo seleccionado
        input_tensor = input_tensor.to(device)

        # Inferencia con el modelo
        with torch.no_grad():
            depth_map = model(input_tensor).squeeze().cpu().numpy()

        # Normalizar para visualizar
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map_visual = (depth_map * 255).astype(np.uint8)
        depth_map_visual = cv2.cvtColor(depth_map_visual, cv2.COLOR_GRAY2BGR)  # Convertir a 3 canales para combinar

        # Redimensionar el mapa de profundidad para que coincida con la imagen de la webcam
        depth_map_visual = cv2.resize(depth_map_visual, (frame.shape[1], frame.shape[0]))

        # Combinar ambas imágenes horizontalmente
        combined_image = np.hstack((frame, depth_map_visual))

        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Añadir texto de FPS a la imagen combinada
        cv2.putText(combined_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar la imagen combinada
        cv2.imshow("Webcam + Depth Map", combined_image)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selecciona el tamaño del modelo y otras configuraciones.")
    parser.add_argument(
        "size",
        choices=["large", "hybrid", "small"],
        help="Tamaño del modelo a usar: 'large', 'hybrid', 'small'."
    )
    parser.add_argument(
        "--no-flip",
        action="store_false",
        dest="flip_video",
        help="No invertir el video como un espejo."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Usar GPU para acelerar el procesamiento (si está disponible)."
    )
    args = parser.parse_args()

    main(args.size, args.flip_video, args.use_gpu)
