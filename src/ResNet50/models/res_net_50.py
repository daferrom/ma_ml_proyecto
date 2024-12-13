from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def train_resnet50_with_transfer_learning(input_shape , num_classes, train_dataset, val_dataset):
    
    input_layer = Input(shape=(256, 256, 3))

    base_model = ResNet50(
        weights='imagenet', 
        include_top=False,
        input_tensor=input_layer
    )

    base_model.trainable = True
    for layer in base_model.layers[:-40]: 
        layer.trainable = False

    x = base_model(input_layer) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    resNet50_model = models.Model(inputs=input_layer, outputs=output_layer)

    optimizer = Adam(learning_rate=0.00005)
    resNet50_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.2,
        patience=3,
        min_lr=0.000005
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=7,
        restore_best_weights=True
    )

    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3, 
        horizontal_flip=True,
        vertical_flip=True, 
        fill_mode='nearest'
    )

    resNet50_model.summary()

    history_resnet50 = resNet50_model.fit(
        train_dataset,
        validation_data=val_dataset, 
        epochs=60, 
        callbacks=[reduce_lr, early_stopping]
    )


    base_model.trainable = True
    

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    resNet50_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history_resNet50_tranfer_learning = resNet50_model.fit(train_dataset, epochs=60, validation_data=val_dataset)

    return resNet50_model , history_resNet50_tranfer_learning