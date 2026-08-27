import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def weights_init(m):
    """Inicialización de pesos según el paper de DCGAN"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ConditionalGAN:
    """Clase que implementa la lógica de entrenamiento de una CGAN"""
    
    def __init__(self, discriminator, generator, latent_dim, device, num_classes=10):
        self.discriminator = discriminator.to(device)
        self.generator = generator.to(device)
        self.latent_dim = latent_dim
        self.device = device
        self.num_classes = num_classes

        # Inicializar pesos según DCGAN
        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

        # Métricas
        self.g_losses = []
        self.d_losses = []

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def add_instance_noise(self, data, std=0.1):
        if std == 0: return data
        noise = torch.randn_like(data) * std
        return data + noise

    def train_step(self, real_images, labels):
        batch_size = real_images.size(0)

        # ETIQUETAS DINÁMICAS (Label Smoothing Aleatorio)
        real_label_smooth = torch.FloatTensor(batch_size, 1).uniform_(0.8, 1.0).to(self.device)
        fake_label_smooth = torch.FloatTensor(batch_size, 1).uniform_(0.0, 0.1).to(self.device)

        # Convertir etiquetas a one-hot
        one_hot_labels = torch.zeros(batch_size, self.num_classes).to(self.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        # 1. Expandir one-hot a mapas espaciales
        image_one_hot_labels = one_hot_labels.view(batch_size, self.num_classes, 1, 1)
        image_one_hot_labels = image_one_hot_labels.expand(-1, -1, 32, 32)

        # 2. Generar imágenes falsas
        random_latent_vectors = torch.randn(batch_size, self.latent_dim).to(self.device)
        random_vector_labels = torch.cat([random_latent_vectors, one_hot_labels], dim=1)
        fake_images = self.generator(random_vector_labels)

        # 3. Concatenaciones para el discriminador
        fake_input_noisy = torch.cat([self.add_instance_noise(fake_images.detach(), std=0.05), image_one_hot_labels], dim=1)
        real_input_noisy = torch.cat([self.add_instance_noise(real_images, std=0.05), image_one_hot_labels], dim=1)

        # 4. Entrenar el Discriminador
        self.discriminator.zero_grad()
        output_real = self.discriminator(real_input_noisy)
        loss_real = self.loss_fn(output_real, real_label_smooth)
        loss_real.backward()

        output_fake = self.discriminator(fake_input_noisy)
        loss_fake = self.loss_fn(output_fake, fake_label_smooth)
        loss_fake.backward()
        
        d_loss = loss_real + loss_fake
        self.d_optimizer.step()

        # 5. Entrenar el Generador
        random_latent_vectors = torch.randn(batch_size, self.latent_dim).to(self.device)
        random_vector_labels = torch.cat([random_latent_vectors, one_hot_labels], dim=1)
        
        self.generator.zero_grad()
        fake_images_g = self.generator(random_vector_labels)
        fake_image_and_labels_g = torch.cat([fake_images_g, image_one_hot_labels], dim=1)

        output = self.discriminator(fake_image_and_labels_g)
        target_ones = torch.full((batch_size, 1), 1.0, device=self.device)
        g_loss = self.loss_fn(output, target_ones)
        
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item()

    def fit(self, dataloader, epochs):
        print("....INICIANDO ENTRENAMIENTO....")
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            g_loss_epoch = 0
            d_loss_epoch = 0

            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                g_loss, d_loss = self.train_step(images, labels)
                g_loss_epoch += g_loss
                d_loss_epoch += d_loss

                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}")

            self.g_losses.append(g_loss_epoch / len(dataloader))
            self.d_losses.append(d_loss_epoch / len(dataloader))
            print(f"Epoch {epoch+1}/{epochs} >>> g_loss: {self.g_losses[-1]:.4f} - d_loss: {self.d_losses[-1]:.4f}")

            if (epoch + 1) % 5 == 0:
                self.save_sample_images(epoch + 1)
                
        print("....ENTRENAMIENTO COMPLETADO....")
        return self

    def save_sample_images(self, epoch):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(10, self.latent_dim).to(self.device)
            labels = torch.arange(10).to(self.device)
            one_hot = torch.zeros(10, self.num_classes).to(self.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            noise_with_labels = torch.cat([noise, one_hot], dim=1)
            fake_images = self.generator(noise_with_labels)
            fake_images = (fake_images + 1) / 2 # Desnormalizar

            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            for i in range(10):
                img = fake_images[i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'{i}')
            plt.suptitle(f'Epoch {epoch}')
            plt.savefig(f'samples_epoch_{epoch}.png', bbox_inches='tight')
            plt.close()
        self.generator.train()
