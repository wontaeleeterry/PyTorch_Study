import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from base.base_net import BaseNet
from networks.mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Decoder
from base.base_dataset import BaseADDataset
from base.base_trainer import BaseTrainer

class MNIST_Hybrid_Model(BaseNet):
    def __init__(self, rep_dim=32):
        super().__init__()
        
        self.rep_dim = rep_dim
        # Autoencoder components
        self.encoder = MNIST_LeNet(rep_dim=rep_dim)
        self.decoder = MNIST_LeNet_Decoder(rep_dim=rep_dim)
        
        # Classifier component
        self.classifier = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # 10 classes for MNIST
        )
        
    def forward(self, x):
        # Get encoded features
        code = self.encoder(x)
        
        # Reconstruct original image
        reconstructed = self.decoder(code)
        
        # Classify the encoded features
        class_output = self.classifier(code)
        
        return reconstructed, class_output


class MNIST_Dataset(BaseADDataset):
    def __init__(self, root):
        super().__init__(root)
        self.normal_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # All digits are normal
        self.outlier_classes = ()

    def loaders(self, batch_size, shuffle_train=True, shuffle_test=False, num_workers=0):
        # Define transforms for the data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])

        # Load the datasets
        train_dataset = datasets.MNIST(self.root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(self.root, train=False, download=True, transform=transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

        return train_loader, test_loader

class MNIST_Hybrid_Trainer(BaseTrainer):
    def __init__(self, optimizer_name='adam', lr=0.001, n_epochs=100, lr_milestones=(50,), batch_size=128,
                 weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, classification_loss_weight=0.5):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device, n_jobs_dataloader)
        self.classification_loss_weight = classification_loss_weight

    def train(self, dataset, net):
        # Set up the optimizer
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError('Unsupported optimizer')

        # Load training data
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, shuffle_train=True, num_workers=self.n_jobs_dataloader)

        # Training loop
        for epoch in range(self.n_epochs):
            net.train()
            total_recon_loss = 0.0
            total_class_loss = 0.0
            
            for data in train_loader:
                inputs, labels = data  # MNIST dataset returns (data, label)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                reconstructed, class_output = net(inputs)
                
                # Reconstruction loss (MSE)
                recon_loss = nn.MSELoss()(reconstructed, inputs)    
                # 디코더로 생성한 이미지와 원본 비교 (260713)
                # recon_loss가 크게 개선되지 않음 (260713)

                # Classification loss (Cross Entropy)
                class_loss = nn.CrossEntropyLoss()(class_output, labels)
                
                # Combined loss
                total_loss = recon_loss + self.classification_loss_weight * class_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                total_recon_loss += recon_loss.item()
                total_class_loss += class_loss.item()
            
            if epoch in self.lr_milestones:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            
            print(f'Epoch [{epoch+1}/{self.n_epochs}], '
                  f'Recon Loss: {total_recon_loss/len(train_loader):.4f}, '
                  f'Class Loss: {total_class_loss/len(train_loader):.4f}')
        
        return net

    def test(self, dataset, net):
        # Load test data
        _, test_loader = dataset.loaders(batch_size=self.batch_size, shuffle_test=False, num_workers=self.n_jobs_dataloader)

        # Testing loop
        net.eval()
        total_recon_loss = 0.0
        total_class_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data  # MNIST dataset returns (data, label)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                reconstructed, class_output = net(inputs)
                
                # Reconstruction loss
                recon_loss = nn.MSELoss()(reconstructed, inputs)
                
                # Classification loss
                class_loss = nn.CrossEntropyLoss()(class_output, labels)
                
                # Accuracy calculation for classification
                _, predicted = torch.max(class_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_recon_loss += recon_loss.item()
                total_class_loss += class_loss.item()
        
        avg_recon_loss = total_recon_loss / len(test_loader)
        avg_class_loss = total_class_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        print(f'Test Results - Recon Loss: {avg_recon_loss:.4f}, '
              f'Class Loss: {avg_class_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_recon_loss, avg_class_loss, accuracy


# Main training function
def main():
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')

    # 맥 전용 코드 (MPS GPU용)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # Create dataset and trainer
    dataset = MNIST_Dataset(root='./datasets')
    trainer = MNIST_Hybrid_Trainer(device=device, n_epochs=50, classification_loss_weight=0.3)  # Reduced epochs for demonstration

    # Create the hybrid model (autoencoder + classifier)
    hybrid_model = MNIST_Hybrid_Model(rep_dim=32).to(device)

    print("Training the MNIST hybrid model (autoencoder + classifier)...")
    trained_net = trainer.train(dataset, hybrid_model)
    
    print("Testing the MNIST hybrid model...")
    recon_loss, class_loss, accuracy = trainer.test(dataset, trained_net)
    
    print("Training and testing completed.")

    # Save the trained model
    model_save_path = 'mnist_hybrid_model_260710.pt'
    torch.save({
        'model_state_dict': trained_net.state_dict(),
        'rep_dim': trained_net.rep_dim,
        'test_recon_loss': recon_loss,
        'test_class_loss': class_loss,
        'test_accuracy': accuracy,
    }, model_save_path)
    print(f"모델이 저장되었습니다: {model_save_path}")

if __name__ == '__main__':
    main()


