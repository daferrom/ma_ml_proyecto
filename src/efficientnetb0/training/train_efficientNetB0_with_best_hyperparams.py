import os
import json
from ..models import train_efficientnetB0_with_transfer_learning
from scripts.eda.split_data import train_ds, val_ds

# Configurar callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_efficientnet_model_bhp.keras', monitor='val_accuracy', mode='max', save_best_only=True)

# Entrenar el modelo
history_effNetB0_bhp = efficientNetB0_with_best_hp.fit(
    train_ds_resized,
    validation_data=val_ds_resized,
    epochs=20,
    callbacks=[early_stopping, checkpoint],
)