import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 데이터 로더 설정 (예: MNIST 데이터셋)
def create_client_dataloaders(num_clients, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    client_loaders = []
    split_sizes = [len(trainset) // num_clients for _ in range(num_clients)]
    # 만약 나누어떨어지지 않으면 남는 데이터를 마지막 클라이언트에 추가
    split_sizes[-1] += len(trainset) - sum(split_sizes)

    client_data_split = torch.utils.data.random_split(trainset, split_sizes)

    for subset in client_data_split:
        loader = data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append({'trainloader': loader})

    for i, subset in enumerate(client_data_split):
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append({'trainloader': loader})
        print(f"Client {i} has {len(subset)} samples.")

    return client_loaders

def create_test_dataloader(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    print(f"Test dataset has {len(testset)} samples.")
    return testloader