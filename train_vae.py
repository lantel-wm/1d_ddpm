from vae.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(batch_size=64, 
        epochs=20, 
        device="cuda",
        data_path="./datasets",
        z_dim=64,
    )
    trainer.train()