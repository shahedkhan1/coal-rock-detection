from src.train import TrainConfig, run_training


def main():
    cfg = TrainConfig(weights='', epochs=1, batch_size=4, device='cpu', img_size=640, name='smoke_test')
    run_training(cfg)


if __name__ == '__main__':
    main()
