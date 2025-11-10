# Clasificador de Perros y Gatos

Este proyecto implementa un modelo de aprendizaje profundo para clasificar imágenes de perros y gatos utilizando una red neuronal convolucional (CNN) con TensorFlow/Keras.

## Características

- Arquitectura CNN con capas convolucionales, de pooling y fully connected
- Preprocesamiento de imágenes con aumento de datos
- Dataset de perros y gatos de FreeCodeCamp
- Entrenamiento con early stopping y regularización

## Estructura del Proyecto

```
fcc_cat_dog.ipynb - Notebook principal con el código completo
cats_and_dogs/    - Dataset descargado automáticamente
  ├── train/      - Imágenes de entrenamiento
  ├── validation/ - Imágenes de validación
  └── test/       - Imágenes de prueba
```

## Requisitos

- TensorFlow 2.x
- NumPy
- Matplotlib
- Google Colab (recomendado) o entorno con GPU

## Uso

1. Ejecuta el notebook en Google Colab o localmente
2. El notebook descargará automáticamente el dataset
3. Preprocesa las imágenes y entrena el modelo
4. Evalúa el modelo con el conjunto de prueba

## Resultados

El modelo alcanza una precisión de validación del 97% (valor específico depende del entrenamiento) después de 20 épocas. El conjunto de prueba contiene 50 imágenes para evaluación final.

## Licencia

Este proyecto utiliza el dataset de FreeCodeCamp que tiene fines educativos. El código es de libre uso para proyectos de aprendizaje.
