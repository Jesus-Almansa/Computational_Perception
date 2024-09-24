# Procesamiento de Imágenes: Transformaciones de Intensidad

Este proyecto se enfoca en aplicar diferentes **transformaciones** a imágenes con el objetivo de mejorar su calidad o destacar ciertas características visuales. Entre las transformaciones implementadas se encuentran ajustes de **brillo**, **contraste**, **ecualización de histograma**, y transformaciones más avanzadas como la **transformación logarítmica** y **corrección gamma**.

## Estructura del Proyecto

El proyecto está organizado en las siguientes carpetas y archivos:

Computational_Perception/
├── data/                 # Imágenes de entrada
├── results/              # Imágenes resultantes después de aplicar las transformaciones
├── doc/                  # Documentación del proyecto
├── Notebooks/            # Cuadernos Jupyter para pruebas
├── src/                  # Código fuente del proyecto
│   └── procesamiento.py  # Funciones principales de procesamiento de imágenes
├── tests/                # Pruebas unitarias
├── README.md             # Descripción del proyecto
└── setup.py              # Configuración del proyecto


# Proyecto de Procesamiento de Imágenes: Transformaciones y Filtros

Este proyecto está enfocado en la aplicación de diferentes **transformaciones** y **filtros** a imágenes, utilizando la biblioteca OpenCV, NumPy y Matplotlib. Entre las técnicas aplicadas se incluyen ajustes de brillo y contraste, ecualización de histograma, transformaciones logarítmicas y gamma, así como operaciones aritméticas entre imágenes y varios tipos de filtros.

## Funciones Implementadas

### Transformaciones

1. **Ecualización de Histograma (YCrCb y escala de grises)**
   - Mejora el contraste de la imagen distribuyendo uniformemente los niveles de intensidad.
   - Implementado tanto para el canal de luminancia en el espacio de color YCrCb, como para imágenes en escala de grises.

2. **Transformación Logarítmica**
   - Se aplica para expandir los valores de intensidad en imágenes con detalles oscuros.

3. **Corrección Gamma**
   - Ajusta la luminosidad de la imagen mediante la fórmula:
     \[
     I_{\text{transformada}} = I_{\text{original}}^{\gamma}
     \]

4. **Transformaciones Lineales (Brillo y Contraste)**
   - Ajustes del brillo y el contraste de una imagen usando los parámetros `alpha` y `beta`.

5. **Operaciones Aritméticas**
   - Permite realizar operaciones de suma, resta, multiplicación y división entre dos imágenes.

### Filtros Aplicados

1. **Filtro Gaussiano**
   - Aplicación de un desenfoque gaussiano para suavizar la imagen y reducir el ruido.

2. **Filtro de la Media**
   - Reduce el ruido en la imagen calculando el valor promedio de los píxeles en un área definida.

3. **Filtro de la Mediana**
   - Filtra el ruido, especialmente el ruido de sal y pimienta, utilizando la mediana de los píxeles vecinos.

4. **Filtro Bilateral**
   - Reduce el ruido mientras preserva los bordes, aplicando un filtro bilateral.

## Requisitos

Este proyecto está desarrollado con **Python 3.12** y hace uso de las siguientes bibliotecas:

- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-image`

Puedes instalar las dependencias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt


## Requisitos del Sistema

Este proyecto utiliza **Python 3.12** y las siguientes bibliotecas:

- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-image`

Estas dependencias se instalan automáticamente al configurar el proyecto.

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu_usuario/Computational_Perception.git
   cd Computational_Perception



### Detalles de las secciones:

1. **Descripción general**: Explica el propósito del proyecto y qué transformaciones y filtros se aplican.
2. **Funciones implementadas**: Da una breve descripción de cada función que has implementado.
3. **Requisitos**: Explica cómo instalar las dependencias.
4. **Estructura del proyecto**: Describe la organización de los archivos del proyecto.
5. **Uso**: Muestra ejemplos de cómo usar las funciones para aplicar transformaciones y filtros a las imágenes.
6. **Cuadernos Jupyter**: Indica cómo acceder a los cuadernos para explorar ejemplos interactivos.
7. **Contribuciones**: Indica cómo colaborar en el proyecto.
8. **Licencia**: Proporciona detalles de la licencia.

Este `README.md` debería proporcionar toda la información necesaria para comprender y usar el proyecto. Puedes modificar o ampliar las secciones según sea necesario.

Si tienes más preguntas o necesitas ayuda adicional, ¡no dudes en avisarme!
