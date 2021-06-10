import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm
import streamlit as st
from stqdm import stqdm
import pandas as pd
import plotly.express as px

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

cols = 50
rows = 50

## función para encontrar la diferencia entre dos listas

def diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

## función para ajustar las dimensiones para que sean compatibles
## con el modelo de CNN

def reshapeImage(img):
    img = np.array(img).astype('float32')/255
    img = img.reshape((1,50,50,1))
    return img

## función que dado un archivo con una imagen "completa" devuelve un
## arreglo con cada uno de los fragmentos de 50x50 pixeles

def splitImage(raw):

    #raw = Image.open(fileName,mode='r')

    width, height = raw.size

    ncols = width//cols
    nrows = height//rows

    img = raw.crop((0,0,ncols*cols,nrows*rows))

    img2 = np.array(img.convert('L'))

    tiles = [reshapeImage(img2[x:x+cols,y:y+rows])\
             for x in range(0,img2.shape[0],cols)\
             for y in range(0,img2.shape[1],rows)]

    corners = [[x,y]\
               for x in range(0,img2.shape[0],cols)\
               for y in range(0,img2.shape[1],rows)]

    return img,tiles,corners



## función para hacer la clasificación de una imagen

def predictImage(img):
    predictions = model.predict(img)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)]

## fijar el ancho de la aplicación

st.set_page_config(layout="wide")

## cargar el modelo de TF

model = keras.models.load_model('modelo138056.h5')

## fijar lós códigos de cada una de las especies

class_names = ["001","002","003","004","006","007","008","009","013","014","015"]

## fijar el título del la aplicación

st.markdown("<H2>Clasificador de Microfotografías de Epidermis de Hojas <BR> utilizando  Redes Neuronales e Inteligencia Artificial</H2>",unsafe_allow_html=True)

## romper aplicación en dos columnas

col1, col2 = st.beta_columns(2)

#### contenido del SIDEBAR ######

# st.sidebar.markdown("<b>Clasificador de Microfotografías de Epidermis de Hojas Utilizando  Redes Neuronales e Inteligencia Artificial</b>",unsafe_allow_html=True)


with st.sidebar.beta_expander("Información adicional"):
    st.write("Este programa se entrenó...")

st.sidebar.markdown("<hr>",unsafe_allow_html=True)

st.sidebar.image("planta2.png")


#st.sidebar.markdown("Este programa se entrenó utilizandoi muestras de hojas de plantas colectadas
#                        en el Bosque de la Universidad de Puerto Rico en Humacao")


# El clasificador fragmenta la imagen en pedazos de 50x50 pixeles y clasifica cada pedazo dentro de once
# categorias de especies. El histograma el porciento de pedazos de que fueron clasificados dentro de una #clase (especie). En general, la barra de más altura corresponde la clasificación de más alta probabilidad

st.sidebar.markdown("<hr>",unsafe_allow_html=True)

#st.sidebar.markdown("Universidad de Puerto Rico -Humacao")

### SIDEBAR ######

## CARGAR imagen #############

img_file_buffer = col1.file_uploader("Subir imagen", type=["png", "jpg", "jpeg"])


my_bar = col1.progress(0)

tiles = None

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    #draw = ImageDraw.Draw(image)
    #draw.rectangle((0,0,50,50))
    #img_array = np.array(image) # if you want to pass it to OpenCV
    #col1.image(image, caption=img_file_buffer, use_column_width=True)
    #fileName = img_file_buffer
    image2,tiles,corners = splitImage(image)
    #st.write(len(tiles))
    draw = ImageDraw.Draw(image2)
    nn = len(tiles)
    nSample = st.sidebar.slider('Cantidad de submuestras en imagen:',
                                min_value=100,
                                max_value=nn,
                                step=10)
    st.sidebar.markdown("Para la cantidad máxima de submuestras (p.e. 672) la\
                        clasificación puede demorar aproximadamente 1 minuto")
    if nSample == len(tiles):
        randIndex = np.arange(len(tiles))
    else:
        randIndex = np.random.choice(len(tiles),int(nSample))
    temp = np.array(corners)[randIndex]
    for i in range(len(temp)):
        tempX = temp[i][1]
        tempY = temp[i][0]
        draw.rectangle([(tempX,tempY),(tempX+cols,tempY+rows)],
                       outline=(255,0,0),width=3)
    col1.image(image2, caption=img_file_buffer, use_column_width=True)

    tiles = np.array(tiles)[randIndex]

my_bar.progress(100)

classPredict = [predictImage(tiles[inx]) for inx in stqdm(range(len(tiles)),st_container=col1)]


with st.spinner("Construyendo gráfica..."):

    myClass, myCount = np.unique(classPredict, return_counts=True)

    frecuencia = {}

    temp = np.array([myClass,myCount]).transpose()


    for inx in range(len(temp)):
         frecuencia[temp[inx][0]] = 100*(int(temp[inx][1]))/myCount.sum()

    frecuencia_df = pd.DataFrame.from_dict(frecuencia,orient="index")

    frecuencia_df.columns = ["porciento"]

## pedazo de código para añadir al dataframe de frecuencia
## los casos con frecuencia cero de tal manera que la gráfica
## de barras siempre tenga las mismas "clases" en el eje de x

    diffSet = diff(list(frecuencia_df.index),list(class_names))

    zeroFrec = pd.DataFrame(list([0]*len(diffSet)))

    zeroFrec.index = diffSet

    zeroFrec.columns = ["porciento"]

    frecuencia_df = frecuencia_df.append(zeroFrec)

    frecuencia_df = frecuencia_df.rename_axis('clase').reset_index()

    frecuencia_df = frecuencia_df.sort_values('clase', ascending=True)

## gráfico de barras en plotly

    fig = px.bar(frecuencia_df,
                 x='clase',
                 y='porciento')

    fig.update_layout(
        autosize=True,
        height=720,
        margin=dict(l=20,r=20,t=30,b=10))

    col2.write(fig)

#col1.success("Completado...")

#st.balloons()


#my_bar.progress(100)

#col2.bar_chart(frecuencia_df,height=500)




