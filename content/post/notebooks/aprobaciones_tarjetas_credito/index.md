---
title: "Predecir aprobaciones de tarjetas de crédito"
date: 2021-05-21
summary: Introducción al machine learning y los modelos de clasificación. Aquí construimos un modelo que es capaz de clasificar solicitudes de tarjetas de crédito prediciendo si las mismas serán aprobadas o rechazadas.
authors: [admin]
categories: [Notebook]
tags:
  [machine learning, feature scaling, imputing missing values, label encoding]
image:
  placement: 1
  preview_only: false
---

## Solicitudes de tarjetas de crédito

Los bancos comerciales reciben infinidad de solicitudes de tarjetas de crédito. Muchos de ellos son rechazados por diversas razones, como saldos elevados de préstamos, bajos niveles de ingresos o demasiadas consultas sobre el informe crediticio de una persona, por ejemplo. El análisis manual de estas aplicaciones es tedioso, propenso a errores y requiere mucho tiempo. Afortunadamente, esta tarea se puede automatizar con el poder del machine learning o aprendizaje automático y casi todos los bancos comerciales lo hacen hoy en día. En este post, crearemos un predictor automático de aprobación de tarjetas de crédito utilizando técnicas de aprendizaje automático, tal como lo hacen los bancos reales.

Usaremos el conjunto de datos [Credit Card Approval](http://archive.ics.uci.edu/ml/datasets/credit+approval) del Repositorio de Machine Learning de UCI. La estructura de este trabajo será la siguiente:

- Primero, comenzaremos cargando y viendo el conjunto de datos.
- Veremos que el dataset tiene una mezcla de _features_ (características o predictores) numéricas y no numéricas, valores en diferentes rangos, y además una cantidad considerable de datos faltantes.
- Tendremos que preprocesar el dataset para asegurarnos de que el modelo de machine learning que elijamos pueda hacer buenas predicciones.
- Una vez que nuestros datos estén en buena forma, haremos un análisis de datos exploratorio para formar nuestras intuiciones.
- Finalmente, crearemos un modelo de machine learning que pueda predecir si se aceptará o no la solicitud de una persona para una tarjeta de crédito.

## Carguemos y observemos los datos

Dado que estos datos son confidenciales, el contribuyente de este dataset ha anonimizado los nombres de las funciones.

```python
import pandas as pd

# Cargamos dataset
cc_apps = pd.read_csv("datasets/cc_approvals.data", header=None)

# Un vistazo a los datos
cc_apps.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.25</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00202</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>f</td>
      <td>g</td>
      <td>00043</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>t</td>
      <td>g</td>
      <td>00100</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>00120</td>
      <td>0</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>

El resultado puede parecer un poco confuso a primera vista, pero intentemos descubrir las características más importantes de una aplicación de tarjeta de crédito.

Como dijimos, los predictores de este conjunto de datos se han anonimizado para proteger la privacidad, pero [este blog](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) nos brinda una descripción general bastante buena de cuales pueden ser los probables predictores. Las características probables en una solicitud de tarjeta de crédito típica podrían ser `Género`, `Edad`, `Deuda`, `Estado Civil`, `Cliente bancario`, `Nivel de educación`, `Etnia`, `Años de empleo`, `Incumplimiento previo`, `Empleado`, `Puntuación de crédito`, `Licencia de conductor`, `Ciudadano`, `Código postal`, `Ingresos` y finalmente el `Estado de aprobación`. Esto nos da un buen punto de partida y podemos mapear estas características con respecto a las columnas en nuestro dataset.

Como podemos ver, el dataset tiene una combinación de predictores numéricos y no numéricos. Esto se puede solucionar con un poco de preprocesamiento, pero antes de hacerlo, investiguemos un poco más para ver si hay otros problemas del conjunto de datos que deban solucionarse.

```python
# Resumen estadístico
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

# Características del dataset
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")

# Un vistazo a los últimos registros
print(cc_apps.tail(17))
```

                   2           7          10             14
    count  690.000000  690.000000  690.00000     690.000000
    mean     4.758725    2.223406    2.40000    1017.385507
    std      4.978163    3.346513    4.86294    5210.102598
    min      0.000000    0.000000    0.00000       0.000000
    25%      1.000000    0.165000    0.00000       0.000000
    50%      2.750000    1.000000    0.00000       5.000000
    75%      7.207500    2.625000    3.00000     395.500000
    max     28.000000   28.500000   67.00000  100000.000000


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 690 entries, 0 to 689
    Data columns (total 16 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   0       690 non-null    object
     1   1       690 non-null    object
     2   2       690 non-null    float64
     3   3       690 non-null    object
     4   4       690 non-null    object
     5   5       690 non-null    object
     6   6       690 non-null    object
     7   7       690 non-null    float64
     8   8       690 non-null    object
     9   9       690 non-null    object
     10  10      690 non-null    int64
     11  11      690 non-null    object
     12  12      690 non-null    object
     13  13      690 non-null    object
     14  14      690 non-null    int64
     15  15      690 non-null    object
    dtypes: float64(2), int64(2), object(12)
    memory usage: 86.4+ KB
    None


        0      1       2  3  4   5   6      7  8  9   10 11 12     13   14 15
    673  ?  29.50   2.000  y  p   e   h  2.000  f  f   0  f  g  00256   17  -
    674  a  37.33   2.500  u  g   i   h  0.210  f  f   0  f  g  00260  246  -
    675  a  41.58   1.040  u  g  aa   v  0.665  f  f   0  f  g  00240  237  -
    676  a  30.58  10.665  u  g   q   h  0.085  f  t  12  t  g  00129    3  -
    677  b  19.42   7.250  u  g   m   v  0.040  f  t   1  f  g  00100    1  -
    678  a  17.92  10.210  u  g  ff  ff  0.000  f  f   0  f  g  00000   50  -
    679  a  20.08   1.250  u  g   c   v  0.000  f  f   0  f  g  00000    0  -
    680  b  19.50   0.290  u  g   k   v  0.290  f  f   0  f  g  00280  364  -
    681  b  27.83   1.000  y  p   d   h  3.000  f  f   0  f  g  00176  537  -
    682  b  17.08   3.290  u  g   i   v  0.335  f  f   0  t  g  00140    2  -
    683  b  36.42   0.750  y  p   d   v  0.585  f  f   0  f  g  00240    3  -
    684  b  40.58   3.290  u  g   m   v  3.500  f  f   0  t  s  00400    0  -
    685  b  21.08  10.085  y  p   e   h  1.250  f  f   0  f  g  00260    0  -
    686  a  22.67   0.750  u  g   c   v  2.000  f  t   2  t  g  00200  394  -
    687  a  25.25  13.500  y  p  ff  ff  2.000  f  t   1  t  g  00200    1  -
    688  b  17.92   0.205  u  g  aa   v  0.040  f  f   0  f  g  00280  750  -
    689  b  35.00   3.375  u  g   c   h  8.290  f  f   0  t  g  00000    0  -

## Manejo de los valores faltantes

Mediante las observaciones anteriores, hemos descubierto algunos problemas en el dataset que afectarán el rendimiento de nuestros modelos de machine learning si no se modifican:

- Contiene datos numéricos, puntualmente las _features_ 2, 7, 10 y 14, de tipo `float64` o `int64`, y categóricos o no numéricos, de tipo `object` para las características restantes.
- Posee valores con rangos disímiles. Algunas características tienen un rango de valores que va de 0 a 28, mientras otras tienen máximos que alcanzan los 100000.
- Presenta valores faltantes. Los mismos están etiquetados con el caracter '?', que se puede ver, por ejemplo, en el valor de la feature 0 de la fila 673 en la muestra anterior.

Vamos a ocuparnos ahora de estos valores faltantes. Comencemos reemplazando temporalmente estos signos de interrogación con valores nulos `NaN`.

```python
import numpy as np

# Reemplazamos los '?'s con NaN
cc_apps = cc_apps.replace('?', np.nan)

# Observemos nuevamente el valor de la feature 0 para la fila 673
print(cc_apps.tail(17))
```

          0      1       2  3  4   5   6      7  8  9   10 11 12     13   14 15
    673  NaN  29.50   2.000  y  p   e   h  2.000  f  f   0  f  g  00256   17  -
    674    a  37.33   2.500  u  g   i   h  0.210  f  f   0  f  g  00260  246  -
    675    a  41.58   1.040  u  g  aa   v  0.665  f  f   0  f  g  00240  237  -
    676    a  30.58  10.665  u  g   q   h  0.085  f  t  12  t  g  00129    3  -
    677    b  19.42   7.250  u  g   m   v  0.040  f  t   1  f  g  00100    1  -
    678    a  17.92  10.210  u  g  ff  ff  0.000  f  f   0  f  g  00000   50  -
    679    a  20.08   1.250  u  g   c   v  0.000  f  f   0  f  g  00000    0  -
    680    b  19.50   0.290  u  g   k   v  0.290  f  f   0  f  g  00280  364  -
    681    b  27.83   1.000  y  p   d   h  3.000  f  f   0  f  g  00176  537  -
    682    b  17.08   3.290  u  g   i   v  0.335  f  f   0  t  g  00140    2  -
    683    b  36.42   0.750  y  p   d   v  0.585  f  f   0  f  g  00240    3  -
    684    b  40.58   3.290  u  g   m   v  3.500  f  f   0  t  s  00400    0  -
    685    b  21.08  10.085  y  p   e   h  1.250  f  f   0  f  g  00260    0  -
    686    a  22.67   0.750  u  g   c   v  2.000  f  t   2  t  g  00200  394  -
    687    a  25.25  13.500  y  p  ff  ff  2.000  f  t   1  t  g  00200    1  -
    688    b  17.92   0.205  u  g  aa   v  0.040  f  f   0  f  g  00280  750  -
    689    b  35.00   3.375  u  g   c   h  8.290  f  f   0  t  g  00000    0  -

Si deseamos observarlo más claro, hay 12 valores faltantes para la feature 0.

```python
# Filtramos los valores NaN para la columna 0
cc_apps[cc_apps[0].isna()]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>248</th>
      <td>NaN</td>
      <td>24.50</td>
      <td>12.750</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>bb</td>
      <td>4.750</td>
      <td>t</td>
      <td>t</td>
      <td>2</td>
      <td>f</td>
      <td>g</td>
      <td>00073</td>
      <td>444</td>
      <td>+</td>
    </tr>
    <tr>
      <th>327</th>
      <td>NaN</td>
      <td>40.83</td>
      <td>3.500</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>bb</td>
      <td>0.500</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>01160</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>346</th>
      <td>NaN</td>
      <td>32.25</td>
      <td>1.500</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>0.250</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00372</td>
      <td>122</td>
      <td>-</td>
    </tr>
    <tr>
      <th>374</th>
      <td>NaN</td>
      <td>28.17</td>
      <td>0.585</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.040</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00260</td>
      <td>1004</td>
      <td>-</td>
    </tr>
    <tr>
      <th>453</th>
      <td>NaN</td>
      <td>29.75</td>
      <td>0.665</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>0.250</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00300</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>479</th>
      <td>NaN</td>
      <td>26.50</td>
      <td>2.710</td>
      <td>y</td>
      <td>p</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.085</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>00080</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>489</th>
      <td>NaN</td>
      <td>45.33</td>
      <td>1.000</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>v</td>
      <td>0.125</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00263</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>520</th>
      <td>NaN</td>
      <td>20.42</td>
      <td>7.500</td>
      <td>u</td>
      <td>g</td>
      <td>k</td>
      <td>v</td>
      <td>1.500</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00160</td>
      <td>234</td>
      <td>+</td>
    </tr>
    <tr>
      <th>598</th>
      <td>NaN</td>
      <td>20.08</td>
      <td>0.125</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>v</td>
      <td>1.000</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00240</td>
      <td>768</td>
      <td>+</td>
    </tr>
    <tr>
      <th>601</th>
      <td>NaN</td>
      <td>42.25</td>
      <td>1.750</td>
      <td>y</td>
      <td>p</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00150</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>641</th>
      <td>NaN</td>
      <td>33.17</td>
      <td>2.250</td>
      <td>y</td>
      <td>p</td>
      <td>cc</td>
      <td>v</td>
      <td>3.500</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00200</td>
      <td>141</td>
      <td>-</td>
    </tr>
    <tr>
      <th>673</th>
      <td>NaN</td>
      <td>29.50</td>
      <td>2.000</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>2.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00256</td>
      <td>17</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>

Hemos reemplazado todos los signos de interrogación con NaN. Esto nos ayudará al momento de aplicar la estrategia de valores faltantes que vamos a realizar.

Una pregunta importante que surge aquí es _¿por qué le damos tanta importancia a los valores perdidos? ¿No se pueden simplemente ignorar?_

Ignorar los valores perdidos puede afectar en gran medida el rendimiento del modelo de machine learning. Si bien podría ignorar los valores faltantes, nuestro modelo también perdería información potencialmente útil del dataset para su entrenamiento. Debido a esto, hay muchos modelos que no pueden manejar valores faltantes implícitamente.

Entonces, para evitar este problema, vamos a imputar o "llenar" los valores faltantes con una estrategia llamada _mean imputation_. Esta estrategia lo que hace es reemplazar los valores faltantes con el valor de la media para todos los valores de esa característica en el dataset. Obviamente, esto aplica solo para las features de tipo numéricas.

```python
# Imputamos los valores faltantes con la media
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Contamos el número de NaNs para verificar
cc_apps.isnull().sum()
```

    0     12
    1     12
    2      0
    3      6
    4      6
    5      9
    6      9
    7      0
    8      0
    9      0
    10     0
    11     0
    12     0
    13    13
    14     0
    15     0
    dtype: int64

Como vemos, nos hemos ocupado con éxito de los valores faltantes presentes en las columnas numéricas. Todavía hay algunos valores faltantes que imputar para las columnas 0, 1, 3, 4, 5, 6 y 13. Todas estas columnas contienen datos categóricos (no numéricos) y por eso la estrategia de imputación media no funcionaría aquí. Esto necesita un tratamiento diferente.

Vamos a imputar estos valores faltantes con los valores más frecuentes presentes en sus respectivas columnas. Esta es una buena práctica cuando se trata de imputar valores faltantes para datos categóricos en general.

Para hacerlo, recorreremos cada una de las columnas del DataFrame y sólo en aquellas con valores categóricos imputaremos el valor que mayor recuento tenga para dicha columna.

```python
# Recorremos cada columna de cc_apps
for col in cc_apps.columns:
    # Chequeamos si la columna es de tipo 'object'
    if cc_apps[col].dtypes == 'object':
        # Imputamos con el valor más frecuente
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Volvemos a contar el número de NaNs en el dataset para verificar
cc_apps.isnull().sum()
```

    0     0
    1     0
    2     0
    3     0
    4     0
    5     0
    6     0
    7     0
    8     0
    9     0
    10    0
    11    0
    12    0
    13    0
    14    0
    15    0
    dtype: int64

## Preprocesamiento y división del dataset

Hemos solucionado el problema de los valores faltantes.

Todavía se necesita un preprocesamiento de datos menor pero esencial antes de continuar con la construcción de nuestro modelo. Vamos a dividir estos pasos de preprocesamiento restantes en tres tareas principales:

1.  Convertir los datos categóricos en numéricos.
2.  Dividir los datos en conjuntos de entrenamiento y pruebas (_train and test sets_).
3.  Escalar los valores de las características a un rango uniforme.

Primero, convertiremos todos los valores no numéricos en valores numéricos. Hacemos esto porque no solo da como resultado un cálculo más rápido, sino que también muchos modelos de machine learning (especialmente los desarrollados con scikit-learn) requieren que los datos estén en un formato estrictamente numérico. Haremos esto utilizando una técnica llamada _label encoding_.

```python
from sklearn.preprocessing import LabelEncoder

# Instanciamos LabelEncoder
le = LabelEncoder()

# Recorremos todos los valores de cada columna y extraemos su tipo de dato
for col in cc_apps.columns:
    # Chequeamos si la columna es de tipo 'object'
    if cc_apps[col].dtypes == 'object':
    # Usamos LabelEncoder para realizar la transformación numérica
        cc_apps[col]=le.fit_transform(cc_apps[col])
```

Hemos convertido todos los valores categóricos en valores numéricos.

Ahora, dividiremos nuestro dataset en un conjunto de entrenamiento y otro de pruebas que utilizaremos en esas dos fases diferentes del modelado respectivamente.

Idealmente, no se debe utilizar ninguna información de los datos del conjunto de pruebas para escalar los datos de entrenamiento ni mucho menos se deben utilizar durante el proceso de entrenamiento del modelo de machine learning. Por lo tanto, primero dividiremos los datos y luego aplicaremos el proceso de reescalamiento.

Además, podemos intuír que algunos datos como la `Licencia de conductor` y el `Código Postal` no son tan significativos al momento a predecir las aprobaciones de tarjetas de crédito como sí lo son otros datos de este dataset. Por lo tanto, deberíamos descartarlos para diseñar nuestro modelo de machine learning con el mejor conjunto de características. En la literatura sobre ciencia de datos, esto a menudo se denomina _**selección de características** (feature selection)_.

```python
from sklearn.model_selection import train_test_split

# Eliminamos las features 11 y 13 y convertimos el DataFrame en un NumPy array
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.to_numpy()

# Separamos características y etiquetas en variables distintas
X, y = cc_apps[:,0:12] , cc_apps[:,13]

# Dividimos el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

Los datos ya fueron divididos en dos conjuntos separados. Sólo nos queda un paso final antes de que podamos entrenar nuesto modelo, el escalado de las variables.

Cuando un dataset tiene rangos variables, como en este caso, es posible que un pequeño cambio en una característica en particular no genere un efecto significativo en otra, lo que puede causar muchos problemas en el modelado predictivo. De aquí la necesidad de llevar todas las características a una escala similar.

Intentemos comprender qué significan estos valores escalados en el mundo real. Usemos `Puntuación de Crédito` como ejemplo. El puntaje crediticio de una persona es su solvencia basada en su historial crediticio. Cuanto mayor sea este número, se considera que una persona es más confiable desde el punto de vista financiero. Por lo tanto, un puntaje crediticio de 1 será el más alto, ya que estamos escalando todos los valores al rango entre 0 y 1.

```python
from sklearn.preprocessing import MinMaxScaler

# Instanciamos MinMaxScaler y lo utilizamos para escalar X_train y X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
```

## Entrenando el modelo

Esencialmente, predecir si una solicitud de tarjeta de crédito será aprobada o no es una tarea de [clasificación](https://en.wikipedia.org/wiki/Statistical_classification). [Según UCI](http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names), nuestro conjunto de datos contiene más instancias que corresponden al estado "Denegado" que las instancias correspondientes al estado "Aprobado". Específicamente, de 690 casos, hay 383 (55,5%) aplicaciones que fueron denegadas y 307 (44,5%) aplicaciones que fueron aprobadas.

Esto nos da un punto de referencia. Un buen modelo de aprendizaje automático debería poder predecir con precisión el estado de las aplicaciones con respecto a estas estadísticas.

¿Qué modelo deberíamos elegir?

Una pregunta que debe hacerse es: ¿las características que afectan el proceso de decisión de aprobación de la tarjeta de crédito están correlacionadas entre sí? Aunque podemos medir la correlación, en este caso nos limitaremos a confiar en nuestra intuición de que, de hecho, están correlacionados por ahora. Debido a esta correlación, aprovecharemos el hecho de que los modelos lineales generalizados funcionan bien en estos casos. Comencemos nuestro modelado de machine learning con un modelo de **logistic regression** (modelo lineal generalizado).

```python
from sklearn.linear_model import LogisticRegression

# Instanciamos el clasificador LogisticRegression con sus parámetros por defecto
logreg = LogisticRegression()

# Entrenamos logreg con los datos escalados
logreg.fit(rescaledX_train, y_train)
```

    LogisticRegression()

## Evaluemos la performance

Ya tenemos nuestro modelo entrenado, pero ¿qué tan bien funciona?

Evaluaremos nuestro modelo con el conjunto de prueba respecto a la exactitud de la clasificación, es decir, evaluando la métrica `accuracy`, pero también echaremos un vistazo a la matriz de confusión del modelo.

En nuestro caso de estudio, es igualmente importante ver si nuestro modelo es capaz de predecir como aprobadas aquellas solicitudes realmente aprobadas tanto como predecir como denegadas aquellas originalmente rechazadas. Si nuestro modelo no está funcionando bien en este aspecto, entonces podría terminar aprobando solicitudes que deberían haber sido rechazadas. La matriz de confusión nos ayuda a ver el desempeño de nuestro modelo desde estos aspectos.

```python
from sklearn.metrics import confusion_matrix

# Utilizamos el estimador logreg para predecir instancias sobre el test set y las almacenamos
y_pred = logreg.predict(rescaledX_test)

# Obtenemos la puntuación "accuracy score"
print("Accuracy: ", logreg.score(rescaledX_test, y_test))

# Mostramos la matriz de confusión del modelo
print(confusion_matrix(y_test, y_pred))
```

    Accuracy:  0.8377192982456141
    [[93 10]
     [27 98]]

## Ajustando el modelo

¡Nuestro modelo fue bastante bueno! Pudo producir un _accuracy_ de casi el 84%.

En la matriz de confusión, el primer elemento de la primera fila representa los verdaderos negativos, es decir, el número de instancias negativas (solicitudes denegadas) predichas correctamente por el modelo. El último elemento de la segunda fila representa los verdaderos positivos, es decir, el número de instancias positivas (solicitudes aprobadas) predichas correctamente por el modelo.

Veamos si podemos mejorarlo. Podemos realizar una búsqueda en cuadrícula -_grid search_- de los parámetros del modelo para mejorar su capacidad para predecir las solicitudes de tarjetas de crédito.

La implementación de scikit-learn de logistic regression consta de diferentes hiperparámetros, pero en este caso buscaremos en la cuadrícula sólo los siguientes:

- `tol`
- `max_iter`

```python
from sklearn.model_selection import GridSearchCV

# Definimos la grilla de valores para 'tol' y 'max_iter'
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Creamos un diccionario con 'tol' y 'max_iter' como claves y las listas anteriores como sus valores
param_grid = dict(tol=tol, max_iter=max_iter)
```

Hemos definido la cuadrícula de valores de hiperparámetros y los hemos convertido en un formato de diccionario único que `GridSearchCV()` espera como uno de sus parámetros. Ahora, comenzaremos la búsqueda en la cuadrícula para ver qué valores funcionan mejor.

Crearemos una instancia de `GridSearchCV()` con nuestro modelo **logreg** anterior y todos los datos que tenemos. En lugar de pasar `X_train` y `X_test` por separado, proporcionaremos `X` (versión escalada) e `y`. También indicaremos a `GridSearchCV()` que realice [cross-validation](https://es.wikipedia.org/wiki/Validaci%C3%B3n_cruzada) de cinco pliegues.

Finalizaremos este proyecto almacenando la puntuación mejor lograda y los mejores parámetros respectivos.

```python
# Instanciamos GridSearchCV con los parámetros requeridos
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Utilizamos nuevamente 'scaler' para escalar X
rescaledX = scaler.fit_transform(X)

# Entrenamos el modelo
grid_model_result = grid_model.fit(rescaledX, y)

# Obtenemos los valores de los hiperparámetros que mejores resultados arrojan
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Mejor puntuación: %f , utilizando %s" % (best_score, best_params))
```

    Mejor puntuación: 0.850725 , utilizando {'max_iter': 100, 'tol': 0.01}

## Conclusiones

Al crear este predictor de solicitudes de tarjetas de crédito, abordamos algunos de los pasos de preprocesamiento más conocidos, como el **escalado**, la **codificación de etiquetas** y la **imputación de valores faltantes**. Terminamos con algo de **machine learning** para predecir si la solicitud de una persona para una tarjeta de crédito se aprobaría o no, dada cierta información sobre esa persona.

La idea es que fuera algo introductorio. Más adelante podríamos retomarlo para evaluar el modelo con otras métricas más significativas e incluso probar y comparar con otros modelos de clasificación diferentes.

---

Espero que este post te haya resultado interesante y si tenés alguna consulta o sugerencia no dudes en contactarme por mail o seguirme en redes sociales.

¡Hasta la próxima!

---

<small>Este trabajo forma parte de los proyectos propuestos en la carrera [Data Scientist with Python](https://www.datacamp.com/tracks/data-scientist-with-python?version=3) en Datacamp.</small>

[![View on nbviewer](https://img.shields.io/badge/View_on-nbviewer-orange?logo=jupyter&style=flat-square)](https://nbviewer.jupyter.org/github/martinmuelas/data-science-portfolio/blob/master/datacamp_projects/aprobaciones_tarjetas_credito/notebook.ipynb)
