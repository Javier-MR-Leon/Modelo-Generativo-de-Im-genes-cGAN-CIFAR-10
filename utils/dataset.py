import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=128, data_dir="./data", num_workers=2):
    """
    Descarga (si es necesario) y prepara el DataLoader para CIFAR-10.
    Aplica la conversión a Tensor y normaliza las imágenes al rango [-1, 1].
    """
    # Transformaciones para normalizar las imágenes al rango [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Cargar CIFAR-10 desde torchvision.datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # DataLoader para iterar por batches
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(), 
        drop_last=True
    )
    
    print(f"\nDataset cargado:")
    print(f"- Número de imágenes: {len(train_dataset)}")
    print(f"- Shape de una imagen: {train_dataset[0][0].shape}")
    print(f"- Número de batches: {len(train_loader)}\n")
    
    return train_loader
