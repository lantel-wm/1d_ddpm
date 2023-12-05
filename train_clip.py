from CLIP.trainer import Trainer

trainer = Trainer(batch_size=512, epochs=100)
trainer.train()