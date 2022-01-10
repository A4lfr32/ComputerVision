# ImageProcessing

Una observación inicial es que `cv2.HoughLines()` detecta puntos blancos que estén contenidos en una misma línea con un límite inferior del número de puntos. Así que el primer paso es que solo los puntos de interés sean blancos. Por lo que he aplicado el siguiente filtro:

``` python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst=cv2.Laplacian(gray, cv2.CV_8UC1)
```
[![Imagen filtrada 1][1]][1]

Ahora detectamos las líneas con (segmento de código modificado de [documentación: Hough Line Transform](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)):
``` python
# %% HoughLines

# Matriz de ceros con mismas dimenciones que dst
out=np.zeros_like(dst)

# Input:
#   imagen, 
#   pixeles de resolución,
#   np.pi/180 -> 1 grado de resolución angular
#   mínimo 50 puntos por línea
#  **Nota importante** los puntos de interés son los blancos,
#   ojo con usar puntos negros en fondo blanco, no funcionará
lines = cv2.HoughLines(dst, 1, np.pi / 180, 50, None, 0, 0)
# Salida:
#   radio rho, y angulo theta
#  ver: https://www.geogebra.org/graphing/vmwe4kcb

# Dibujar rectas [fuente:https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html]
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(out, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)

```

``` python
# %% Considerar solo los puntos de las líneas
result=cv2.bitwise_and(dst,dst,mask = out)
show(result)
```
[![Líneas detectadas][2]][2]

Filtrando o eliminando los puntos que está por fuera de las líneas de interés se obtiene:
``` python
# %% Considerar solo los puntos de las líneas
result=cv2.bitwise_and(dst,dst,mask = out)
show(result)
```
[![introducir la descripción de la imagen aquí][3]][3]

Para conectar la línea y no sean puntos separados, usamos:
``` python
# %% Poner más gruesa y conectar lineas débiles

# opción 1
e_im = cv2.dilate(result, kernel, iterations=5) 
d_im = cv2.erode(e_im, kernel, iterations=4)
# opción 2
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# e_im = cv2.dilate(result, kernel, iterations=3) 
# d_im = cv2.erode(e_im, kernel, iterations=2)

show(d_im)
```
[![introducir la descripción de la imagen aquí][4]][4]

Ya terminando filtramos para obtener los contornos con un área superior a los puntos de ruido, además como el rectángulo tiene contorno interno un filtro adicional es mostrar solo los componentes que tienen hijos:
``` python
# %% (Alternativa 1) connectedComponentsWithStats con área mínima
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(d_im, connectivity=4)
# Output:
# nb_components -> número de etiquetas o grupos detectados
# output-> matriz 2D (shape) multi-etiqueta
# stats -> contiene (x top left, y top left, width, height, area de la sombra)
# centroids-> ubicación (x,y) del centroide

sizes = stats[1:, -1]; nb_components = nb_components - 1

min_size = 1100 # Este tamaño es variable, yo he puesto 1000 en mi caso

img2 = np.zeros((output.shape))

for i in range(0, nb_components):
    if sizes[i] >= min_size and stats[i+1][0]>0:
        # matriz booleana para cada etiqueta, activa una agrupacion completa
        img2[output == i + 1] = 255
        # cv2.rectangle(img2,(stats[i+1][0],stats[i+1][1]),(stats[i+1][0]+stats[i+1][2],stats[i+1][1]+stats[i+1][3]),(155,155,155),3)
Image.fromarray(np.uint8(img2) , 'L')

```

``` python
# %% Dibujar solo el borde que tenga hijos: https://stackoverflow.com/questions/52397592/only-find-contour-without-child
ChildContour = hierarchy [0, :,2]
indices=(ChildContour!=-1).nonzero()[0]

ncontours=tuple(contours[i] for i in indices)
imgOut=img.copy()
cv2.drawContours(imgOut, ncontours, -1, (0,255,0), 3)
show(imgOut)

```

#### Resultado final
[![introducir la descripción de la imagen aquí][5]][5]

**Nota:** 
En la última etapa agregué para ajustar `and stats[i+1][0]>0`, la idea fue por eliminar también los contornos que están conectados a los bordes, en este caso específico solo se usó el borde izquierdo, si es necesario se puede ignorar los contornos que estén conectados a todos los bordes de la imagen.

En todo caso al no tener contornos internos sería filtrado en la última parte por no tener contornos hijos.

  [1]: https://i.stack.imgur.com/NhoUp.png
  [2]: https://i.stack.imgur.com/fTKH2.png
  [3]: https://i.stack.imgur.com/SjpCX.png
  [4]: https://i.stack.imgur.com/HoI3A.png
  [5]: https://i.stack.imgur.com/Na7HJ.png
