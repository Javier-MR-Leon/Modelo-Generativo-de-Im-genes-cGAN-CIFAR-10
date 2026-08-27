import torch
import torch.nn as nn
import torch.optim as optim

from models.discriminator import Discriminator
from models.generator import Generator
from models.cgan import ConditionalGAN
from utils.dataset import get_dataloader
from utils.visualizer import plot_losses

def set_seed(seed=42):
    """Establece la semilla para asegurar la reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    latent_dim = 100
    num_classes = 10

    print(f"[*] Iniciando con dispositivo: {device}")

    # 2. Cargar datos
    train_loader = get_dataloader(batch_size=128)

    # 3. Inicializar modelos
    discriminator = Discriminator(in_channels=3 + num_classes)
    generator = Generator(in_channels=latent_dim + num_classes)

    # 4. Crear y compilar la cGAN
    cgan_model = ConditionalGAN(discriminator, generator, latent_dim, device)
    cgan_model.compile(
        d_optimizer=optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        g_optimizer=optim.Adam(generator.parameters(), lr=0.00025, betas=(0.5, 0.999)),
        loss_fn=nn.BCELoss()
    )

    # 5. Entrenar el modelo
    cgan_model.fit(train_loader, epochs=epochs)

    # 6. Guardar la gráfica y los pesos
    plot_losses(cgan_model.g_losses, cgan_model.d_losses, save_path="training_losses.png")
    torch.save(generator.state_dict(), "generator_weights.pth")
    
    print("[*] ¡Entrenamiento finalizado y modelo guardado!")

if __name__ == "__main__":
    main()
