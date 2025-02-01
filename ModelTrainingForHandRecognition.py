import tensorflow as tf  # Libreria para redes neuronales
from tensorflow.keras.applications import MobileNetV2  # Importamos el modelo preentrenado
from tensorflow.keras.models import Sequential  # Para apilar capas secuenciales
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Para cargar imagenes con aumentacion
from tensorflow.keras.callbacks import ModelCheckpoint  # Para guardar el mejor modelo
from tensorflow.keras.callbacks import EarlyStopping # Para controlar overfitting

# Preprocesamiento y Aumentacion de Datos
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza los valores de los pixeles entre 0 y 1
    rotation_range=20,  # Rotacion aleatoria de hasta 20 grados
    width_shift_range=0.2,  # Mueve la imagen hasta un 20% en horizontal
    height_shift_range=0.2,  # Mueve la imagen hasta un 20% en vertical
    shear_range=0.2,  # Aplica inclinacion a la imagen
    zoom_range=0.2,  # Aplica zoom aleatorio
    horizontal_flip=False,  # No volteamos horizontalmente (porque las letras en ASL no son simetricas)
    validation_split=0.2  # Reservamos el 20% de los datos para validacion
)

# Carga de imagenes de entrenamiento
train_generator = train_datagen.flow_from_directory(
    "C:/Users/juaki/OneDrive/Escritorio/ASLHandRecognition/asl_dataset",  # Ruta donde estan las imagenes organizadas en carpetas (A, B, C... Z)
    target_size=(224, 224),  # MobileNetV2 espera imagenes de 224x224 pixeles
    batch_size=32,  # Numero de imagenes procesadas por lote
    class_mode='categorical',  # 26 clases (A-Z) en formato one-hot encoding
    subset='training'  # Solo carga el 80% para entrenamiento
)

# Carga de imagenes de validacion
val_generator = train_datagen.flow_from_directory(
    "C:/Users/juaki/OneDrive/Escritorio/ASLHandRecognition/asl_dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Solo carga el 20% para validacion
)

# Cargar MobileNetV2 preentrenado
base_model = MobileNetV2(weights='imagenet',  # Cargamos pesos preentrenados en ImageNet
                         input_shape=(224, 224, 3),  # Tamano de imagen esperado
                         include_top=False)  # Quitamos la ultima capa para personalizarlo

# Congelar las capas preentrenadas
base_model.trainable = False  # No entrenamos las capas de MobileNetV2 (solo usaremos sus caracteristicas)

# Construir el modelo con nuestras capas personalizadas
model = Sequential([
    base_model,  # Usamos MobileNetV2 como base
    GlobalAveragePooling2D(),  # Reduce las caracteristicas extraidas a un solo vector
    Dense(256, activation='relu'),  # Capa densa con 256 neuronas para aprender patrones especificos
    Dropout(0.5),  # Desactiva el 50% de las neuronas para evitar sobreajuste
    Dense(26, activation='softmax')  # Capa de salida con 26 neuronas (una por cada letra de ASL)
])

# Compilar el modelo
model.compile(optimizer='adam',  # Optimizador eficiente para deep learning
              loss='categorical_crossentropy',  # Funcion de perdida para clasificacion multiclase
              metrics=['accuracy'])  # Medimos precision (accuracy)

# Guardar el mejor modelo basado en validacion
checkpoint = ModelCheckpoint('modelo_mobilenet_asl.keras',  
                             save_best_only=True,  # Guarda solo si el modelo mejora
                             monitor='val_accuracy',  # Basado en precision de validacion
                             mode='max')  # Queremos el modelo con mayor precision

early_stopping = EarlyStopping(monitor='val_loss',  # Monitorear la perdida en validacion
                               patience=5,  # Numero de epocas sin mejora antes de detener el entrenamiento
                               restore_best_weights=True)  # Restaurar los mejores pesos del modelo

# Entrenar el modelo
model.fit(
    train_generator,  # Datos de entrenamiento
    validation_data=val_generator,  # Datos de validacion
    epochs=30,  # Numero de veces que pasamos por todo el dataset
    callbacks=[checkpoint]  # Guarda el mejor modelo automaticamente
)

# Guardar el modelo final
model.save("modelo_mobilenet_asl_final.keras")  # Guardamos el modelo completo para usarlo despues
