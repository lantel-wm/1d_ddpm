from CLIP.trainer import Trainer

trainer = Trainer(batch_size=1024, epochs=100)
trainer.train()