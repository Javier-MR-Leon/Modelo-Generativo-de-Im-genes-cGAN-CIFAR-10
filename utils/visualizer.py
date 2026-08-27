import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(g_losses, d_losses, save_path="training_losses.png"):
    """
    Genera y guarda el gráfico de la evolución de las pérdidas del Generador y Discriminador.
    """
    plt.figure(figsize=(10, 5))
    epochs_range = range(1, len(g_losses) + 1)
    
    plt.plot(epochs_range, g_losses, label="Generador", color="blue", linewidth=2)
    plt.plot(epochs_range, d_losses, label="Discriminador", color="red", linewidth=2)
    
    plt.title("Evolución de la pérdida durante el entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.grid(True)
    
    # Guardar en archivo y cerrar figura para liberar memoria
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de pérdidas guardado en: {save_path}")


def generate_and_save_images(generator, latent_dim, num_classes, device, epoch=None, save_path="generated_by_class.png"):
    """
    Genera una imagen por cada clase (0 a 9) y guarda la visualización en una cuadrícula.
    """
    generator.eval() # Cambiar modelo a modo evaluación
    
    with torch.no_grad():
        # Generar ruido aleatorio para cada clase
        noise = torch.randn(num_classes, latent_dim).to(device)
        labels = torch.arange(num_classes).to(device)
        
        # Codificación one-hot
        one_hot = torch.zeros(num_classes, num_classes).to(device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Concatenar ruido y etiquetas para el generador
        noise_with_labels = torch.cat([noise, one_hot], dim=1)
        generated = generator(noise_with_labels)

        # Desnormalizar tensores de [-1, 1] a rango [0, 1] para visualizar
        generated = (generated + 1) / 2

        # Crear figura
        fig, axes = plt.subplots(1, num_classes, figsize=(15, 3))
        for i in range(num_classes):
            img = generated[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1) # Asegurar límites de color
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Clase {i}')
        
        if epoch is not None:
            plt.suptitle(f'Epoch {epoch}')
            
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    generator.train() # Devolver el modelo a modo entrenamiento
