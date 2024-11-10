import torch
import torch.nn as nn
import copy
import threading

from dataloader import create_test_dataloader
from utils import aggregate_models
from model import SimpleModel, MLP
from utils import get_loss_function

# server
class Server:
    def __init__(self, num_clients, max_rounds, device='cpu'):
        dim_in = 28 * 28
        dim_hidden = 128
        dim_out = 10
        self.global_model = MLP(dim_in=dim_in, dim_hidden=dim_hidden, dim_out=dim_out)
        self.global_model.to(device)
        self.global_round_counter = 0
        self.lock = threading.Lock()
        self.client_update_counts = {client_id: 0 for client_id in range(num_clients)}
        self.test_loader = create_test_dataloader(32)
        self.device = device
        self.max_rounds = max_rounds

    def get_model_and_round(self):
        with self.lock:
            return self.global_model.state_dict(), self.global_round_counter

    def update_model(self, client_state_dict, client_id, starting_round, loss, accuracy):
        with self.lock:
            if self.global_round_counter >= self.max_rounds:
                return False
            staleness = self.global_round_counter - starting_round
            # model aggregation
            global_weight = copy.deepcopy(aggregate_models(self.global_model.state_dict(), client_state_dict, alpha=0.2))
            self.global_model.load_state_dict(global_weight)
            # model evaluation
            global_loss, global_acc = self.test_model()
            print(f"|R: {self.global_round_counter}\t"
                  f"|id: {client_id}\t|st: {staleness}({starting_round})\t"
                  f"|acc: {accuracy:.2f}%/{global_acc:.2f}%\t"
                  f"|loss: {loss:.4f}/{global_loss:.4f}")

            # 글로벌 변수 제어
            self.global_round_counter += 1
            self.client_update_counts[client_id] += 1
        return True

    def test_model(self):
        self.global_model.eval()
        criterion = get_loss_function(self.global_model)
        test_loss = 0.0
        correct = 0
        total = 0  

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                if isinstance(criterion, nn.CrossEntropyLoss):
                    a=0 # do nothing
                elif isinstance(criterion, nn.NLLLoss):
                    output = torch.log(output + 1e-10)
                else:
                    raise ValueError("Unsupported loss function.")
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                total += len(target)
                correct += (predicted == target).sum().item()

        avg_loss = test_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy