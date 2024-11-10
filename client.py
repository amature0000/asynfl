import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import copy

from server import Server
from model import SimpleModel, MLP
from utils import get_loss_function
# client
def client_loop(client_id, server:Server, client_loader, max_rounds, epochs=1, lr=0.01, device='cpu',
               download_delay_range=(1, 2), upload_delay_range=(1, 2)):
    # 클라이언트 모델 설정
    model = copy.deepcopy(server.global_model)
    model.to(device)
    update_result = True
    while update_result:
        # 글로벌 모델과 현재 라운드 가져오기
        model_weight, starting_round = server.get_model_and_round()
        if model_weight == -1 and starting_round == -1:
            break
        model.load_state_dict(model_weight)

        # 다운로드 지연
        download_delay = random.uniform(*download_delay_range)
        time.sleep(download_delay)  

        # loss function 및 optimizer
        criterion = get_loss_function(model)
        optimizer = optim.SGD(model.parameters(), lr=lr)

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 학습 수행
        for epoch in range(epochs):
            for data, target in client_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                if isinstance(criterion, nn.CrossEntropyLoss): 
                    pass  
                elif isinstance(criterion, nn.NLLLoss):
                    output = torch.log(output + 1e-10)
                else:
                    raise ValueError("Unsupported loss function.")
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                loss.backward()
                optimizer.step()

                # 정확도 계산
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(client_loader.dataset)
        accuracy = 100 * correct / total

        # 네트워크 업로드 지연
        upload_delay = random.uniform(*upload_delay_range)
        time.sleep(upload_delay)  

        # 서버에 업데이트 전송
        update_result = server.update_model(model.state_dict(), client_id, starting_round, avg_loss, accuracy)
    print(f"{client_id=} stop")

