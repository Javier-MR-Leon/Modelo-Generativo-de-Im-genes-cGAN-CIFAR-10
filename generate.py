import os
import torch

from models.generator import Generator
from utils.visualizer import generate_and_save_images

def main():
    # 1. Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    num_classes = 10
    weights_path = "generator_weights.pth"

    print(f"[*] Generando imágenes con dispositivo: {device}")

    # 2. Inicializar arquitectura del generador
    generator = Generator(in_channels=latent_dim + num_classes).to(device)
    
    # 3. Comprobar que existen los pesos y cargarlos
    if not os.path.exists(weights_path):
        print(f"[!] ERROR: No se encontró '{weights_path}'. Ejecuta train.py primero.")
        return
        
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 4. Generar y guardar la imagen
    generate_and_save_images(
        generator=generator, 
        latent_dim=latent_dim, 
        num_classes=num_classes, 
        device=device, 
        save_path="generated_by_class.png"
    )
    
    print("[*] Imagen guardada como 'generated_by_class.png'")

if __name__ == "__main__":
    main()
