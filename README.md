# 🏗️ Detección de Grietas e Inclinación Estructural para Evaluación de Riesgo

> **Proyecto del Curso:** Algoritmos y Programación (2026-2)
> **Estudiante:** Samuel David Botett Garrido
> **Programa:** Ingeniería en Inteligencia Artificial  
> **Institución:** Universidad Industrial de Santander (UIS)  
> **Docente:** Prof. Jheyston Omar Serrano  

---

## 📌 1. Formulación del Problema y Contexto

En países con alta actividad sísmica como Colombia y en municipios con un porcentaje significativo de autoconstrucción o edificación informal (como Barrancabermeja o el área metropolitana de Bucaramanga), el monitoreo del estado de la infraestructura es crítico. De acuerdo con el **Reglamento Colombiano de Construcción Sismo Resistente (NSR-10)**, la aparición de grietas en elementos estructurales clave (muros de carga, columnas, vigas y placas) puede ser el primer indicador de patologías graves, sobrecargas o asentamientos diferenciales.

Este proyecto aborda el reto de construir un sistema asistido por Computación e Inteligencia Artificial que combine dos capacidades de evaluación:
1. **Clasificación Automática de Grietas:** Uso de Redes Neuronales Convolucionales (CNN) para determinar la presencia de daño superficial/estructural en imágenes.
2. **Estimación Geométrica de Inclinación:** Medición del ángulo de desaplome respecto a la vertical u horizontal en elementos estructurales mediante técnicas de Visión por Computador (detección de bordes y transformada de Hough con OpenCV).

El objetivo final es consolidar un prototipo *End-to-End* que apoye la estimación preliminar del nivel de riesgo estructural de manera accesible.

---

## 📊 2. Descripción del Dataset y Análisis Exploratorio (EDA)


Para garantizar la robustez, generalización y mitigación del cambio de dominio (domain shift) en la detección de patologías, el proyecto evolucionó desde un dataset base hacia un esquema de datos unificados e integrados:

* Fuentes de Datos:
  1. Concrete Crack Images for Classification (Özgenel / METU - Middle East 
     Technical University): Base de referencia con imágenes de 227x227 píxeles.
     
  2. SDNET2018: Conjunto de datos complementario que aporta una gran variedad 
     de entornos reales en muros, puentes y pavimentos con texturas e 
     iluminaciones complejas.

* Volumen Consolidado:
  Cerca de 100,000 imágenes a color (RGB) unificadas en directorios estandarizados.

* Estructura de Clases:
  - Positive (Con Grieta / Fisura)
  - Negative (Sin Grieta / Superficie Sana)

* Preprocesamiento y Aumentación (Data Augmentation):
  - Reescalado de muestras a dimensiones compatibles con arquitecturas 
    profundas (128x128 o 224x224 píxeles).
    
  - Normalización de píxeles en el rango [0, 1].
    
  - Aplicación de transformaciones dinámicas (rotaciones, zoom y volteos) en el 
    cuaderno de fusión (05_Fusion_Datasets.ipynb) y de entrenamiento avanzado 
    (06_transfer_learning.ipynb) para enriquecer la variabilidad visual ante 
    escenarios de campo real.

---

## 📚 3. Revisión del Estado del Arte

La inspección de obras e infraestructura mediante procesamiento de imágenes ha evolucionado significativamente en la última década:

1. **Métodos Tradicionales de Visión por Computador:**
   Históricamente, la detección de grietas se realizaba aplicando filtros de detección de bordes (Canny, Sobel) y operadores morfológicos. Aunque eficientes computacionalmente, estos métodos presentan alta sensibilidad al ruido, textura rugosa del concreto y variaciones de iluminación.

2. **Redes Neuronales Convolucionales (CNN):**
   Con la llegada del *Deep Learning*, clasificadores convolucionales simples (como la arquitectura de línea base implementada en esta entrega) lograron extraer patrones complejos de textura, superando el $98\%$ de precisión en conjuntos de datos controlados como METU o SDNET2018.

3. **Arquitecturas Ligeras y Transfer Learning (Tendencia Actual):**
   Para entornos reales y dispositivos móviles (Edge AI), el estado del arte utiliza modelos preconvalidados en *ImageNet* adaptados mediante *Transfer Learning* (como **MobileNetV2** o **EfficientNet-Lite**). Estas arquitecturas optimizan la cantidad de parámetros y reducen el tiempo de inferencia, haciéndolas viables para aplicaciones en teléfonos inteligentes mediante *TensorFlow Lite / LiteRT*.

---

## 🛠️ 4. Prototipo de Línea Base e Implementación Funcional (Entrega 1)

El prototipo se estructuró de manera modular de extremo a extremo (*End-to-End*):

* **Módulo A (Clasificador CNN):** Se diseñó e implementó una arquitectura CNN secuencial de 3 capas convolucionales + pooling en Keras/TensorFlow.
  * **Accuracy en Validación:** $99.73\%$
  * **Validation Loss:** $0.0106$
  * **Guardado:** `Models/modelo_linea_base.keras`
* **Módulo B (Estimador de Inclinación):** Implementación en OpenCV con suavizado Gaussiano, detección de bordes Canny y la Transformada Probabilística de Hough (`HoughLinesP`) para identificar los vectores de inclinación/desaplome.
* **Módulo C (Demo Integrada):** Cuaderno `04_demo_entrega1.ipynb` que recibe una fotografía, evalúa la probabilidad de grieta y calcula los grados de desviación angular en una sola interfaz visual.

---

## 📋 5. Roles de trabajo

## 👥 Plan de Trabajo y Roles del Equipo

| Integrante | Rol | Responsabilidades Principales |
| :--- | :--- | :--- |
| **Samuel** | **Líder de Proyecto & Integrador** | Gestión del repositorio en GitHub, arquitectura modular de carpetas, control de versiones y desarrollo del cuaderno de Demo Integrada End-to-End. |
| **Samuel** | **Ingeniero de Machine Learning** | Exploración de datos (EDA), preprocesamiento, diseño, entrenamiento y validación del modelo CNN de línea base en Keras/TensorFlow. |
| **Samuel** | **Desarrollador de Visión por Computador** | Desarrollo e implementación del módulo geométrico con OpenCV (filtros Canny y Transformada de Hough Probabilística) para la medición de desaplomes. |
| **Samuel** | **Documentador & Analista Técnico** | Redacción del informe técnico, formulación del problema, investigación del estado del arte y análisis de complejidad para la arquitectura del proyecto. |
---

## 📁 6. Estructura del Repositorio

```text
Proyecto_Grietas/
│
├── README.md                                   <-- Documento principal e informe del proyecto
│
├── 01_Exploracion/
│   └── 01_exploracion.ipynb                    <-- Carga, descompresión y análisis exploratorio del dataset base (METU)
│
├── 02_Modelo/
│   └── 02_entrenamiento_baseline.ipynb         <-- Entrenamiento y exportación de la CNN de línea base
│
├── 03_Inclinacion/
│   └── 03_inclinacion.ipynb                    <-- Módulo OpenCV (Canny + Transformada de Hough) para desaplome
│
├── 04_Demo/
│   └── 04_demo_entrega1.ipynb                  <-- Pipeline completo End-to-End
│
├── 05_Preprocesamiento_Augmentation/
│   └── 05_Fusion_Datasets.ipynb                <-- Script de unificación y estructuración del dataset combinado (METU + SDNET2018)
│
├── 06_Transfer_Learning/
│   └── 06_transfer_learning.ipynb              <-- Entrenamiento avanzado con MobileNetV2 y Transfer Learning
│
├── data/                                       <-- Directorio contenedor de clases (Positive / Negative) y datos procesados
└── Models/
    ├── modelo_linea_base.keras                 <-- Pesos y arquitectura de la CNN inicial
    └── mobilenetv2_grietas_v2.keras            <-- Pesos y arquitectura optimizada con Transfer Learning
                             

