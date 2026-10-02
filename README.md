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

Para la etapa de entrenamiento y validación se seleccionó una base de datos libre y ampliamente reconocida en la literatura científica:

* **Nombre del Dataset:** *Concrete Crack Images for Classification* (Özgenel / METU - Middle East Technical University).
* **Volumen Total:** 40,000 imágenes a color (RGB).
* **Resolución Original:** $227 \times 227$ píxeles.
* **Estructura de Clases:**
  * `Positive` (Con Grieta): 20,000 imágenes.
  * `Negative` (Sin Grieta): 20,000 imágenes.
* **Balance de Datos:** Perfectamente balanceado ($50\%$ / $50\%$).
* **Preprocesamiento:** Reescalado de imágenes a $128 \times 128$ píxeles y normalización de píxeles en el rango $[0, 1]$ para optimizar el tiempo de cómputo.

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
├── README.md                          <-- Documento principal e informe del proyecto
│
├── 01_Exploracion/
│   └── 01_exploracion.ipynb           <-- Carga, descompresión y análisis del dataset
│
├── 02_Modelo/
│   └── 02_entrenamiento_baseline.ipynb <-- Entrenamiento y exportación de la CNN
│
├── 03_Inclinacion/
│   └── 03_inclinacion.ipynb           <-- Módulo OpenCV (Canny + Hough) para desaplome
│
├── 04_Demo/
│   └── 04_demo_entrega1.ipynb         <-- Pipeline completo End-to-End
│
├── data/                              <-- Carpeta contenedora de datos y fotos de prueba
└── Models/
    └── modelo_linea_base.keras        <-- Archivo binario con los pesos del modelo
                              ```text

