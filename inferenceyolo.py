import torch
from torch import nn
import torch.nn.functional as F
import sys
from PIL import Image
from tool.utils import *
import cv2
from scipy.spatial import distance
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression
from math import sqrt


produccion = False

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        assert (x.data.dim() == 4)
        _, _, H, W = target_size
        return F.interpolate(x, size=(H, W), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


def agregar_hbox(hboxes, boxes):
    if(len(hboxes) >= 10):
        hboxes.remove(hboxes[0])
    hboxes.append(boxes)
    return hboxes

class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size())
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size())
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo1 = YoloLayer(anchor_mask=[0, 1, 2], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo2 = YoloLayer(anchor_mask=[3, 4, 5], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo3 = YoloLayer(anchor_mask=[6, 7, 8], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        # y1 = self.yolo1(x2)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        # y2 = self.yolo2(x10)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        return [x2, x10, x18]
        # y3 = self.yolo3(x18)
        # return [y1, y2, y3]
        # return y3


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neck = Neck()
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        # head
        self.head = Yolov4Head(output_ch)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output



def cuadro(box):
    return [(box[0] - (box[2] / 2) ) , (box[1] - (box[3] / 2) ) , (box[0] + (box[2] / 2) ) , (box[1] + (box[3] / 2) )]

def centroide(box):
    x = int(box[0] + ((box[2] - box[0]) / 2))
    y = int(box[1] + ((box[3] - box[1]) / 2))
    return[x , y]

def dentro(centroide, box):
    return ((box[0] <= centroide[0] <= box[2]) and (box[1] <= centroide[1] <= box[3]))

def ancho(box):
    return int( (box[2] - box[0]))

def alto(box):
    return int( (box[3] - box[1]) )  

def cruz(box):
    return ancho(box) + alto(box)  

def carea(box):
    return (box[2] - box[0]) * ( box[3] - box[1])      

def proporcion(box):
    x = box[2] - box[0]
    y = box[3] - box[1]
    p = y / x
    return ( 0.7 < p < 3)     

def superficie(elemento ,  maximo = 1.2 , minimo = 0.6):
    area = carea(elemento)
    return ((area < (modelo_elem.predict([[elemento[7] ]]) * maximo)) and (area > (modelo_elem.predict([[elemento[7] ]]) * minimo)) and proporcion(elemento))
"""
def puesto(centroide, cuadro , altura = 0 , delta = 1.3):
    global cmodelo_elem
    global modelo_elem
    alto = int( (cuadro[3] - cuadro[1]) * altura)
    ancho = int( (cuadro[2] - cuadro[0]) * 0.15 )
    posicion = (((cuadro[0] + ancho) <= centroide[0] <= (cuadro[2] - ancho)) and (cuadro[1] <= centroide[1] <= (cuadro[3] - alto)))
    if(cmodelo_elem):
        area = carea(cuadro)
        geometria = (area < (modelo_elem.predict([[cuadro[7] ]]) * delta))
        return (posicion and geometria)
    else: 
        return posicion
"""

def puesto(clavija, elemento , altura = 0 , delta = 1.3):
    global cmodelo_elem
    global modelo_elem
    ccuadro = np.zeros([608,608])
    ccuadro[int(clavija[0]):int(clavija[2]) , int(clavija[1]):int(clavija[3])] = ccuadro[int(clavija[0]):int(clavija[2]) , int(clavija[1]):int(clavija[3])] + 1
    ccuadro[int(elemento[0]):int(elemento[2]) , int(elemento[1]):int(elemento[3])] = ccuadro[int(elemento[0]):int(elemento[2]) , int(elemento[1]):int(elemento[3])] + 1
    
    posicion = (ccuadro[ccuadro == 2].size > 10) and (elemento[0] <= centroide(clavija)[0] <= elemento[2])
    if(cmodelo_elem):
        area = carea(elemento)
        geometria = (area < (modelo_elem.predict([[elemento[7] ]]) * delta))
        return (posicion and geometria)
    else: 
        return posicion

def enfocado(elemento):
    return ((carea(elemento) < 1000) and (elemento[0] > 30) and (elemento[1] > 30) and (elemento[2] < 578) and (elemento[3] < 578))


def distancia(c1, c2):
    return distance.euclidean([c1[6] , c1[7]],[c2[6] , c2[7]])

def delta_alt():
    return prom_cla
    

def do_inference(frame):
    global model
    global n_classes
    global class_names
    global medio
    global modelo_elem
    global modelo_cla
    global cmodelo_elem
    global prom_cla
    
    def bbox_iou(box1, box2):
        x1 = max(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        y1 = max(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        x2 = min(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        y2 = min(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        iou = intersection / (area1 + area2 - intersection)
        return iou

    cboxes = {
            "0" : []
        }
    
    frame = cv2.resize(frame , (608, 608))

    sized = Image.fromarray(frame)
    boxes = do_detect(model, sized, 0.5, n_classes,0.4, 1)
    
    i = 0
    while i < len(boxes):
        j = i + 1
        while j < len(boxes):
            if bbox_iou(boxes[i], boxes[j]) > 0.2:
                if boxes[i][2] * boxes[i][3] < boxes[j][2] * boxes[j][3]:
                    del boxes[i]
                    i -= 1  
                    break
                else:
                    del boxes[j]
                    continue
            j += 1
        i += 1


    bbox_xcycwh = []
    cls_conf = []
    xyboxes = []
    
    
    for i , box in enumerate(boxes):
        if(box[6]==1) or True:
            boxes[i].append(class_names[boxes[i][6]])
            xyr = [int(i * 608) for i in box[0:4]]
    
    
            cbox = cuadro(xyr)
            cbox[0] = 0 if cbox[0] < 0 else cbox[0] 
            cbox[1] = 0 if cbox[1] < 0 else cbox[1] 
            cbox[2] = 608 if cbox[2] > 608 else cbox[2] 
            cbox[3] = 608 if cbox[3] > 608 else cbox[3] 
    
            xyboxes = [cbox[0] , cbox[1] , cbox[2] , cbox[3] , 0 , -1 ]
            cx = centroide(xyboxes)
            xyboxes.extend( cx  )


            cboxes[str(box[6])].append(xyboxes)

    return boxes , cboxes

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def shortsort(indice1 , points1 , indice2 , points2):
    try:
        if(indice1.size <= indice2.size):
            C = cdist(points1, points2)
            _, assigment = linear_sum_assignment(C)
            Z = indice2[list(assigment)] 
            return( list(indice1) ,  list(Z) , [], list(set(indice2) - set(Z)))
        else:
            C = cdist(points2, points1)
            _, assigment = linear_sum_assignment(C)
            Z = indice1[list(assigment)]
            return( list(Z) ,  list(indice2) , list(set(indice1) - set(Z)) , [])
    except:
        return([],[], list(indice1) , list(indice2))

def clean_and_dilate(mask, dilate_px=0, return_ring=False):
    """
    Salida: máscara binaria uint8 con valores {0,1}

    - Elimina objetos secundarios (mantiene el mayor)
    - Dilata el objeto principal
    - Opcional: retorna solo el contorno dilatado (anillo)

    mask: 0/1 o 0/255
    """
    if mask is None:
        return None

    mask = mask.astype(np.uint8)

    # Normalizar a binaria 0/255
    mask_255 = (mask > 0).astype(np.uint8) * 255

    # Si la máscara está vacía -> devolver vacío (0/1)
    if cv2.countNonZero(mask_255) == 0:
        return np.zeros_like(mask_255, dtype=np.uint8)  # ya es 0/1 (todo 0)

    # --- Componentes conectados ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_255, connectivity=8)

    # Solo fondo + 1 objeto -> no hay secundarios
    if num_labels <= 2:
        main_255 = mask_255.copy()
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(areas))

        main_255 = np.zeros_like(mask_255)
        main_255[labels == largest_label] = 255

    # --- Dilatación ---
    k = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated_255 = cv2.dilate(main_255, kernel, iterations=1)

    # --- Salida ---
    if return_ring:
        out_255 = cv2.subtract(dilated_255, main_255)  # 0/255
    else:
        out_255 = dilated_255

    # Convertir a {0,1}
    out_01 = (out_255 > 0).astype(np.uint8)
    return out_01

def numerar_anclas( boxes , hanclas , gradiente_x_y, gradiente_y_x ):
    if(hanclas is None ):
        hanclas = boxes["anclas"].copy()
    else:
        bancla , biancla  = [] , [] 
        for i , elemento in enumerate(boxes["anclas"]):
            
            bancla.append([elemento[6] , elemento[7]])
            biancla.append(i)


        hancla , hiancla  = [] , []
        for i , elemento in enumerate(hanclas):

            hancla.append([elemento[6] , elemento[7]])
            hiancla.append(i)

        iha, iba, rha, rba = shortsort(np.array(hiancla) , np.array(hancla) , np.array(biancla) , np.array(bancla))
        
        for i in range(len(iha)):
            if (distancia(hanclas[iha[i]], boxes["anclas"][iba[i]]) < 100):
                
                gradiente_x_y[0].append(hanclas[iha[i]][6])
                gradiente_x_y[1].append(hanclas[iha[i]][7] -  boxes["anclas"][iba[i]][7])
                gradiente_x_y[2].append( "ancla-"+  str(boxes["anclas"][iba[i]][5]))
                gradiente_y_x[0].append(hanclas[iha[i]][7])
                gradiente_y_x[1].append(hanclas[iha[i]][6] -  boxes["anclas"][iba[i]][6])
                gradiente_y_x[2].append("ancla-"+ str(boxes["anclas"][iba[i]][5]))
                
                boxes["anclas"][iba[i]][5] = iha[i]
                hanclas[iha[i]] = boxes["anclas"][iba[i]].copy()
                
        for i in range(len(rba)):
            boxes["anclas"][rba[i]][5] = rba[i]
            hanclas.append(boxes["anclas"][rba[i]].copy())
            

            
        boxes["anclas"] = hanclas.copy()
    return  (boxes , hanclas , gradiente_x_y, gradiente_y_x)


def numerar_anclas2( boxes , hanclas):
    if(hanclas is None ):
        hanclas = boxes["anclas"].copy()
    else:
        bancla , biancla  = [] , [] 
        for i , elemento in enumerate(boxes["anclas"]):
            
            bancla.append([elemento[6] , elemento[7]])
            biancla.append(i)


        hancla , hiancla  = [] , []
        for i , elemento in enumerate(hanclas):

            hancla.append([elemento[6] , elemento[7]])
            hiancla.append(i)

        iha, iba, rha, rba = shortsort(np.array(hiancla) , np.array(hancla) , np.array(biancla) , np.array(bancla))
        
        for i in range(len(iha)):
            if (distancia(hanclas[iha[i]], boxes["anclas"][iba[i]]) < 100):
                boxes["anclas"][iba[i]][5] = iha[i]
                hanclas[iha[i]] = boxes["anclas"][iba[i]].copy()
                
        for i in range(len(rba)):
            boxes["anclas"][rba[i]][5] = rba[i]
            hanclas.append(boxes["anclas"][rba[i]].copy())
            
        boxes["anclas"] = hanclas.copy()
    return  (boxes , hanclas)



def graficar(frame, boxes ):
    img = Image.fromarray(frame)
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)


        

      

    
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0]) 
        y1 = (box[1])
        x2 = (box[2]) 
        y2 = (box[3]) 
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0))


        
        
    return np.array(img)



def graficar_sin_elementos(frame, boxes):
    img = Image.fromarray(frame)
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes["anclas"])):
        box = boxes["anclas"][i]
        x1 = (box[0]) 
        y1 = (box[1])
        x2 = (box[2]) 
        y2 = (box[3]) 
        cls_id = box[4]

        draw.rectangle([x1, y1, x2, y2], outline=(0,127,255))
        draw.text((x1, y1), ( str(box[5])), fill=(0,0,0))



    for i in range(len(boxes["clavijas_izquierda"])):
        box = boxes["clavijas_izquierda"][i]
        x1 = (box[0]) 
        y1 = (box[1])
        x2 = (box[2]) 
        y2 = (box[3]) 
        cls_id = box[4]

        
        if("uso" in box[8]):
            draw.rectangle([x1, y1, x2, y2], outline=(0,255,0))
            draw.text((x1, y1), ( str(box[5]) + " izquierda"), fill=(255,255,255))
        else:
            draw.rectangle([x1, y1, x2, y2], outline=(0,0,255))
            draw.text((x1, y1), ( str(box[5]) + " izquierda"), fill=(255,255,255))


    for i in range(len(boxes["clavijas_derecha"])):
        box = boxes["clavijas_derecha"][i]
        x1 = (box[0]) 
        y1 = (box[1])
        x2 = (box[2]) 
        y2 = (box[3]) 
        cls_id = box[4]

        if("uso" in box[8]):
            draw.rectangle([x1, y1, x2, y2], outline=(0,255,0))
            draw.text((x1, y1), (str(box[5]) + " derecha"), fill=(255,255,255))
        else:
            draw.rectangle([x1, y1, x2, y2], outline=(0,0,255))
            draw.text((x1, y1), (str(box[5]) + " derecha"), fill=(255,255,255))


    if (not boxes["recipiente_inicio"] is None ):
        box = boxes["recipiente_inicio"]
        x1 = (box[0]) 
        y1 = (box[1])
        x2 = (box[2]) 
        y2 = (box[3]) 
        cls_id = box[4]
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0))
        draw.text((x1, y1), ("inicio"), fill=(0,0,0))
    if (not boxes["recipiente_final"] is None ):
        box = boxes["recipiente_final"]
        x1 = (box[0]) 
        y1 = (box[1])
        x2 = (box[2]) 
        y2 = (box[3]) 
        cls_id = box[4]
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0))
        draw.text((x1, y1), ("final"), fill=(0,0,0))
        
        
        
    return np.array(img)

def transform_coordinate(M, coordinate):
    homogeneous_coordinate = np.array([coordinate[0], coordinate[1], 1])
    transformed_homogeneous_coordinate = np.dot(M, homogeneous_coordinate)
    transformed_coordinate = transformed_homogeneous_coordinate[:2] / transformed_homogeneous_coordinate[2]
    return transformed_coordinate

def calcular_distancia_perspectiva(M , hpinzas, pinzas):
    x1 = transform_coordinate(M, [hpinzas[6] , hpinzas[7]])
    x2 = transform_coordinate(M, [pinzas[6] , pinzas[7]])
    return ((distance.euclidean(x1, x2) / 608) * 20)


n_classes = 1
medio = -1
modelo_elem = LinearRegression()
cmodelo_elem = False
modelo_cla = LinearRegression()
prom_cla = 50


"""

imgfile = "test/21.png"

img = Image.open(imgfile).convert('RGB')

model = Yolov4(n_classes=n_classes)

if(produccion):
    pretrained_dict = torch.load("/home/administrator/servicio/cp/modelos/detector.pth", map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.cuda()
    class_names = load_class_names("/home/administrator/servicio/cp/modelos/_classes.txt")
else:
    pretrained_dict = torch.load("modelos/detector.pth", map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.cuda()
    class_names = load_class_names("modelos/_classes.txt")


salida = do_inference( np.array(sized))

"""
# del model
# torch.cuda.empty_cache()