import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=110, ngf=64):
        super(Generator, self).__init__()
        
        self.ngf = ngf
        
        self.fc = nn.Linear(in_channels, 4 * 4 * ngf * 8)
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.ngf * 8, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv4(x)
        return x
