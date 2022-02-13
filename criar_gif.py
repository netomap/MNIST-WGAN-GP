import pathlib, re
from pickletools import optimize
from PIL import Image, ImageDraw
from numpy import dtype
import numpy as np

def listar_imagens(img_dir):
    lista = [str(l) for l in list(pathlib.Path(img_dir).glob('*.png'))]
    aux = []
    for img_path in lista:
        epoch = int(re.findall(r'[0-9]{1,}', img_path)[0])
        aux.append([img_path, epoch])
    
    # Coloca as imagens em ordem crescente de época
    aux = sorted([[imgp, epo] for imgp, epo in aux], key=lambda item: item[1], reverse=False)
    aux = np.array(aux, dtype=np.object)
    return aux

if (__name__ == '__main__'):
    lista = listar_imagens('./imgs')

    imgs_pil = [Image.open(img) for img in lista[:,0]]
    imgs_pil[0].save('video.gif', save_all=True, append_images=imgs_pil[1:], optimize=False, duration=120, loop=0)
    print ('ok')