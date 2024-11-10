import threading

from dataloader import create_client_dataloaders
from server import Server
from client import client_loop

def main():
    # 설정
    num_clients = 4          # 클라이언트 수
    max_rounds = 20          # 총 글로벌 라운드 수
    epochs_per_client = 1    # 클라이언트 당 에폭 수
    batch_size = 32          # 배치 사이즈
    device = 'cpu'           # 학습 디바이스 ('cpu' 또는 'cuda')
    # 다운로드 지연 범위 (초)
    download_delay_range = [(1, 5), (5,10), (2,4),(6,10)]  
    # 업로드 지연 범위 (초)
    upload_delay_range = [(1, 5), (5,10), (2,4),(6,10)]      

    # 데이터 로더 생성
    clients = create_client_dataloaders(num_clients, batch_size=batch_size)

    # 서버 초기화
    server = Server(num_clients= num_clients, max_rounds= max_rounds)

    # 클라이언트 스레드 생성 및 시작
    threads = []
    for client_id in range(num_clients):
        t = threading.Thread(target=client_loop, args=(
            client_id, server, clients[client_id]['trainloader'], max_rounds,
            epochs_per_client, 0.01, device, download_delay_range[client_id], upload_delay_range[client_id]))
        t.start()
        threads.append(t)

    # 모든 클라이언트 스레드가 종료될 때까지 기다림
    for t in threads:
        t.join()

    print("\nclient call count")
    for client_id, count in server.client_update_counts.items():
        print(f"Client {client_id}: {count} updates")

    # 최종 글로벌 모델 저장
    #torch.save(server.global_model.state_dict(), 'async_federated_model.pth')
    #print("Training complete. Global model saved as 'async_federated_model.pth'.")

if __name__ == "__main__":
    main()

# TODO : 모델을 선택할 수 있도록 인자로 받는 동작을 추가해야 할까?