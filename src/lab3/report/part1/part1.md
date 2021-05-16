**Типы шумов**

Цифровые изображения, полученные различными оптикоэлектронными приборами, могут содержать в себе разнообразные
искажения, обусловленные разного рода помехами, которые принято называть шумом. Шум на изображении затрудняет его 
обработку автоматическими средствами и, поскольку шум может иметь различную природу, для его успешного подавления 
необходимо определить адекватную математическую модель. Рассмотрим наиболее распространенные модели шумов. 


```python
import cv2
import skimage
import skimage.filters
import skimage.restoration
import scipy
import scipy.ndimage
import scipy.signal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from numba import njit, jit
from math import *
import warnings
warnings.filterwarnings('ignore')
export_parallel = True
use_graph = True
```


```python
# Исходное изображение до наложения шума
image = cv2.cvtColor(cv2.imread("data/night_city.jpg"), cv2.COLOR_BGR2GRAY)
print(image.shape)

images_noised = {}

# Общая для всех зашумлённость
amount = 0.2
var = 0.1
mean = 0.0
lam = 12.0
```

    (2074, 3782)
    


```python
# Исходное изображение до наложения шума
if use_graph:
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```


    
![png](output_3_0.png)
    


**Импульсный шум**

При импульсном шуме сигнал искажается выбросами с очень
большими отрицательными или положительными значениями малой длительностью и может возникать, например, из-за ошибок
декодирования. Такой шум приводит к появлению на изображении белых («соль») или черных («перец») точек, поэтому зачастую
называется точечным шумом. Для его описания следует принять
во внимание тот факт, что появление шумового выброса в каждом
пикселе 𝐼(𝑥,𝑦) не зависит ни от качества исходного изображения, ни
от наличия шума в других точках и имеет вероятность появления
𝑝, причем значение интенсивности пикселя 𝐼(𝑥,𝑦) будет изменено
на значение 𝑑 ∈ [0,255]


```python
# Наложение импульсного шума
image_noised_s_and_p = skimage.util.random_noise(image, mode="s&p", amount=amount, salt_vs_pepper=0.3)
image_noised_s_and_p = np.clip(image_noised_s_and_p, 0.0, 1.0)
image_noised_s_and_p = skimage.img_as_ubyte(image_noised_s_and_p)
images_noised["salt and pepper"] = image_noised_s_and_p.copy()

image_noised_all = image_noised_s_and_p

# Изображение после наложения импульсного шума
if use_graph:
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_noised_s_and_p, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```


    
![png](output_5_0.png)
    


**Аддитивный шум**

Аддитивный шум описывается следующим выражением:

𝐼𝑛𝑒𝑤(𝑥,𝑦) = 𝐼(𝑥,𝑦) + 𝜂(𝑥,𝑦),

где 𝐼𝑛𝑒𝑤 — зашумленное изображение, 
𝐼 — исходное изображение,
𝜂 — не зависящий от сигнала аддитивный шум с гауссовым или
любым другим распределением функции плотности вероятности.

**Мультипликативный шум**

Мультипликативный шум описывается следующим выражением:

𝐼𝑛𝑒𝑤(𝑥,𝑦) = 𝐼(𝑥,𝑦) · 𝜂(𝑥,𝑦),

где 𝐼𝑛𝑒𝑤 — зашумленное изображение, 𝐼 — исходное изображение, 𝜂
— не зависящий от сигнала мультипликативный шум, умножающий
зарегистрированный сигнал. В качестве примера можно привести 
зернистость фотопленки, ультразвуковые изображения и т.д. Частным случаем мультипликативного шума является спекл-шум, 
который появляется на изображениях, полученных устройствами с когерентным формированием изображений, например, 
медицинскими
сканерами или радарами. На таких изображениях можно отчетливо
наблюдать светлые пятна, крапинки (спеклы), которые разделены
темными участками изображения.


```python
# Наложение мультипликативного шума
image_noised_multi = skimage.util.random_noise(image, mode="speckle", var=var, mean=mean)
image_noised_multi = np.clip(image_noised_multi, 0.0, 1.0)
image_noised_multi = skimage.img_as_ubyte(image_noised_multi)
images_noised["speckle"] = image_noised_multi.copy()

image_noised_all = skimage.util.random_noise(image_noised_all, mode="speckle", var=var, mean=mean)
image_noised_all = np.clip(image_noised_all, 0.0, 1.0)
image_noised_all = skimage.img_as_ubyte(image_noised_all)

# Изображение после наложения мультипликативного шума
if use_graph:
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_noised_multi, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```


    
![png](output_8_0.png)
    


**Гауссов (нормальный) шум**

Гауссов шум на изображении может возникать в следствие недостатка освещенности сцены, высокой температуры и т.д. 
Модель шума широко распространена в задачах низкочастотной фильтрации изображений. 


```python
# Наложение Гауссова шума
image_noised_gaussian = skimage.util.random_noise(image, mode="gaussian", var=var, mean=mean)
image_noised_gaussian = np.clip(image_noised_gaussian, 0.0, 1.0)
image_noised_gaussian = skimage.img_as_ubyte(image_noised_gaussian)
images_noised["gaussian"] = image_noised_gaussian.copy()

image_noised_all = skimage.util.random_noise(image_noised_all, mode="gaussian", var=var, mean=mean)
image_noised_all = np.clip(image_noised_all, 0.0, 1.0)
image_noised_all = skimage.img_as_ubyte(image_noised_all)

# Изображение после наложения Гауссова шума
if use_graph:
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_noised_gaussian, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```


    
![png](output_10_0.png)
    


**Шум квантования**

Зависит от выбранного шага квантования и самого сигнала.
Шум квантования может приводить, например, к появлению ложных контуров вокруг объектов или убирать слабо 
контрастные детали на изображении. Такой шум не устраняется. Приближенно
шум квантования можно описать распределением Пуассона


```python
# Создание шума квантования с заданными параметрами распределения Пуассона
noise_poisson = np.random.poisson(lam=lam, size=image.shape) * (1.0 / 255.0)
noise_poisson = np.clip(noise_poisson, 1.0 / 255.0, 1.0)

# Наложение шума квантования
image_noised_poisson = skimage.util.random_noise(image, mode="localvar", local_vars=noise_poisson)
image_noised_poisson = np.clip(image_noised_poisson, 0.0, 1.0)
image_noised_poisson = skimage.img_as_ubyte(image_noised_poisson)
images_noised["poisson"] = image_noised_poisson.copy()

image_noised_all = skimage.util.random_noise(image_noised_all, mode="localvar", local_vars=noise_poisson)
image_noised_all = np.clip(image_noised_all, 0.0, 1.0)
image_noised_all = skimage.img_as_ubyte(image_noised_all)

# Изображение после наложения шума квантования
if use_graph:
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_noised_poisson, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```


    
![png](output_12_0.png)
    



```python
# Изображение после наложения всех шумов
if use_graph:
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_noised_all, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```


    
![png](output_13_0.png)
    


### 2. Низкочастотная фильтрация

Низкочастотные пространственные фильтры ослабляют высокочастотные компоненты (области с сильным изменением интенсивностей) и оставляют низкочастотные компоненты изображения
без изменений. Используются для снижения уровня шума и удаления высокочастотных компонент, что позволяет повысить точность
исследования содержания низкочастотных компонент. В результате применения низкочастотных фильтров получим сглаженное или
размытое изображение. Главными отличительными особенностями
являются:

1. неотрицательные коэффициенты маски;

2. сумма всех коэффициентов равна единице.

### 2.1 Фильтр Гаусса


```python
# Применеие фильтра Гаусса ко всем зашумлённым изображениям
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name)
        image_filtered = skimage.filters.gaussian(images_noised[image_name], sigma=4.0)
        image_filtered = np.clip(image_filtered, 0.0, 1.0)
        image_filtered = skimage.img_as_ubyte(image_filtered)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    salt and pepper
    speckle
    gaussian
    poisson
    


    
![png](output_15_1.png)
    


### 2.2 Контргармонический усредняющий фильтр с различными значениями параметра 𝑄


```python
def counter_harmonic_mean_filter(image, kernel_size=3, Q=0):
    rows, cols = image.shape
    kernel_mid = kernel_size // 2
    kernel_k = 1 / pow(kernel_size, 2.0)
    
    image_bordered = cv2.copyMakeBorder(
        image,
        top=kernel_size // 2,
        bottom=kernel_size // 2 + kernel_size % 2,
        left=kernel_size // 2,
        right=kernel_size // 2 + kernel_size % 2,
        borderType=cv2.BORDER_REPLICATE,
        value=[mean, mean, mean]
    )
    
    image_float = np.clip(image / 255.0, 0.0, 1.0)
    image_bordered_float = np.zeros_like(image_bordered, dtype=np.float64)
    
    image_float_power = []
    for q in range(2):
        image_float_power.append(np.power(image_float, Q + q))
        image_float_power[q] = np.clip(image_float_power[q], 0.0, 10)
    for y in range(-kernel_mid, kernel_mid + kernel_size % 2):
        for x in range(-kernel_mid, kernel_mid + kernel_size % 2):
            image_bordered_float[kernel_mid + y:kernel_mid + rows + y,
            kernel_mid + x:kernel_mid + cols + x] += kernel_k * image_float_power[1] / image_float_power[0]
    image_bordered_float = np.clip(image_bordered_float, 0.0, 1.0)
    image = image_bordered_float[kernel_mid:rows+kernel_mid, kernel_mid:cols+kernel_mid] * 255.0
    return image.astype(np.uint8)
```


```python
kernel_size = 7
Q = -1.5
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = -1.5
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_18_1.png)
    



```python
Q = -1.25
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = -1.25
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_19_1.png)
    



```python
Q = -1.0
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = -1.0
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_20_1.png)
    



```python
Q = -0.75
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = -0.75
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_21_1.png)
    



```python
Q = -0.5
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = -0.5
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_22_1.png)
    



```python
Q = -0.25
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = -0.25
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_23_1.png)
    



```python
Q = 0.0
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = 0.0
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_24_1.png)
    



```python
Q = 0.05
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = 0.05
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_25_1.png)
    



```python
Q = 0.1
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = 0.1
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_26_1.png)
    



```python
Q = 0.25
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = 0.25
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_27_1.png)
    



```python
Q = 0.5
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = 0.5
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_28_1.png)
    



```python
Q = 1.0
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = 1.0
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_29_1.png)
    



```python
Q = 1.5
# Применеие Контргармонического усредняющего фильтра с различными значениями параметра 𝑄 
# ко всем зашумлённым изображениям
print(f"Q = {Q}")
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = counter_harmonic_mean_filter(images_noised[image_name], kernel_size=kernel_size, Q=Q)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    Q = 1.5
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_30_1.png)
    


### 3 Нелинейная фильтрация.

Низкочастотные фильтры линейны и оптимальны в случае, когда имеет место нормальное распределение помех на цифровом
изображении. Линейные фильтры локально усредняют импульсные
помехи, сглаживая изображения. Для устранения импульсных помех лучше использовать нелинейные, например, медианные фильтры.

### 3.1 Медианный фильтр


```python
kernel = np.ones((11,11),np.float64)
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = scipy.ndimage.median_filter(images_noised[image_name], footprint=kernel)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_32_1.png)
    


### 3.2 Взвешенный медианный фильтр


```python
kernel = np.random.random(size=(11,11))
kernel /= np.max(kernel)
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = scipy.ndimage.median_filter(images_noised[image_name], footprint=kernel)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    [[0.97382674 0.9325392  0.69862872 0.20333828 0.65707435 0.71193683
      0.69558845 0.96717136 0.26337122 0.49805412 0.83990152]
     [0.83949176 0.41462445 0.0203347  0.42803996 0.61177122 0.70597494
      0.70216593 0.23009838 0.41190495 0.17795031 0.81642714]
     [0.37015619 0.5693179  0.54098948 0.21896704 0.20677685 0.05300312
      0.16465316 0.66420005 0.09758817 0.11497937 0.10925915]
     [0.83855696 0.1056581  0.72585274 0.82298379 0.27819324 0.52393089
      0.07536032 0.48582723 0.42953149 0.0974819  0.68617407]
     [0.52009085 0.02191055 0.61292397 0.36342135 0.15223126 0.24635073
      0.94553247 0.55403716 0.16701514 0.76662335 0.18754365]
     [0.68101818 0.46062369 0.11193957 0.14526321 0.03651154 0.6470802
      0.67324948 0.48439868 0.5697191  0.43531954 0.26352682]
     [0.00333751 1.         0.7964391  0.83933214 0.62613382 0.73691401
      0.77733884 0.46723474 0.76696047 0.01369346 0.64514322]
     [0.22824263 0.90100615 0.15198291 0.26384624 0.93457356 0.11348153
      0.15091364 0.85149485 0.31596884 0.92317719 0.65010302]
     [0.25804225 0.78592857 0.56605026 0.32663542 0.35688438 0.8055483
      0.96981129 0.00751225 0.84062165 0.91282315 0.41904401]
     [0.58364686 0.25909439 0.44186336 0.74383907 0.93230125 0.32701005
      0.51025255 0.18717439 0.8114184  0.05918906 0.96425375]
     [0.23335366 0.55343483 0.47993814 0.36551515 0.77411159 0.8783723
      0.10587177 0.31634728 0.54205638 0.31337927 0.76984312]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_34_1.png)
    


### 3.3 Ранговый фильтр


```python
rank = -25
kernel = np.ones((5,5),np.float64)
print(f"rank = {rank + 25}")
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = scipy.ndimage.rank_filter(images_noised[image_name], footprint=kernel, rank=rank)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    rank = 0
    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_36_1.png)
    



```python
rank = -18
kernel = np.ones((5,5),np.float64)
print(f"rank = {rank + 25}")
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = scipy.ndimage.rank_filter(images_noised[image_name], footprint=kernel, rank=rank)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    rank = 7
    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_37_1.png)
    



```python
rank = -12
kernel = np.ones((5,5),np.float64)
print(f"rank = {rank + 25}")
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = scipy.ndimage.rank_filter(images_noised[image_name], footprint=kernel, rank=rank)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    rank = 13
    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_38_1.png)
    



```python
rank = -6
kernel = np.ones((5,5),np.float64)
print(f"rank = {rank + 25}")
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = scipy.ndimage.rank_filter(images_noised[image_name], footprint=kernel, rank=rank)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    rank = 19
    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_39_1.png)
    



```python
rank = -1
kernel = np.ones((5,5),np.float64)
print(f"rank = {rank + 25}")
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = scipy.ndimage.rank_filter(images_noised[image_name], footprint=kernel, rank=rank)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    rank = 24
    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_40_1.png)
    


### 3.4 Фильтр Винера

Также добавлено адаптивное автоисправление гистограммы CLAHE


```python
kernel = np.ones((15,15),np.float64)
print(kernel)
if use_graph:
    image_combined = None
    for image_name in images_noised.keys():
        print(image_name, end=', ')
        image_filtered = skimage.img_as_float64(images_noised[image_name])
        image_filtered = scipy.signal.convolve2d(image_filtered, kernel, 'same')
        image_filtered = skimage.restoration.wiener(image_filtered, kernel, 5.1e4)
        image_filtered = skimage.img_as_ubyte(image_filtered)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        image_filtered = clahe.apply(image_filtered)
        image_filtered = cv2.hconcat([images_noised[image_name], image_filtered])
        if image_combined is None:
            image_combined = image_filtered
        else:
            image_combined = cv2.vconcat([image_combined, image_filtered])
    print()
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
    salt and pepper, speckle, gaussian, poisson, 
    


    
![png](output_42_1.png)
    



```python
kernel = np.ones((15,15),np.float64)
print(kernel)
if use_graph:
    image_filtered = skimage.img_as_float64(image_noised_all)
    image_filtered = scipy.signal.convolve2d(image_filtered, kernel, 'same')
    image_filtered = skimage.restoration.wiener(image_filtered, kernel, 1.3e6)
    image_filtered = skimage.img_as_ubyte(image_filtered)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    image_filtered = clahe.apply(image_filtered)
    image_filtered = cv2.vconcat([image_noised_all, image_filtered])
    figure(figsize=(32, 16), dpi=80)
    plt.imshow(cv2.cvtColor(image_filtered, cv2.COLOR_GRAY2RGB))
    plt.xticks([]),plt.yticks([])
    plt.show()
```

    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
    


    
![png](output_43_1.png)
    



```python

```
