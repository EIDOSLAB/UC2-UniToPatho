# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pyeddl.eddl as eddl
import numpy as np


def Resnet18_onnx( num_classes = 6, pretreined = True, path = 'onnx_models'):

    #! models already have last softmax layer.
    if pretreined:
        model = eddl.import_net_from_onnx_file("{}/resnet18_{}c_pretrained.onnx".format(path,str(num_classes)))
    else:
        model = eddl.import_net_from_onnx_file("{}/resnet18_{}c.onnx".format(path,str(num_classes)))
 
    out = model.lout[0] 
    
    return model, out
    
def Resnet50_onnx( num_classes = 6, pretreined = True, path = 'onnx_models'):

    #! models already have last softmax layer.
    if pretreined:
        model = eddl.import_net_from_onnx_file("{}/resnet50_{}c_pretrained.onnx".format(path,str(num_classes)))
    else:
        model = eddl.import_net_from_onnx_file("{}/resnet50_{}c.onnx".format(path,str(num_classes)))
 
    out = model.lout[0] 
    
    return model, out
    

def resnet18(x, num_classes):
    l = [2, 2, 2, 2]
    inplanes = 64
    
    x = eddl.Conv(x, inplanes, [7, 7], [2, 2], "same", use_bias=False)
    x = eddl.BatchNormalization(x, True, momentum=0.1, epsilon=1e-05, name='')
    x = eddl.ReLu(x)
    x = eddl.MaxPool2D(x, pool_size=[3, 3], strides=[2, 2], padding='same', name='')


    def downsample(x,planes,stride):
       x = eddl.Conv(x, planes, [1, 1], [1, 1], "same", use_bias=False)
       x = eddl.BatchNormalization(x, True, momentum=0.1, epsilon=1e-05, name='')
       return x

    def block(x,planes,stride):
        identity = x

        x = eddl.Conv(x, planes, [3, 3], [2, 2], "same", use_bias=False, dilation_rate=[1, 1])
        x = eddl.BatchNormalization(x, True, momentum=0.1, epsilon=1e-05, name='')
        x = eddl.ReLu(x)

        x = eddl.Conv(x, planes, [3, 3], [1, 1], "same", use_bias=False, dilation_rate=[1, 1])
        x = eddl.BatchNormalization(x, True, momentum=0.1, epsilon=1e-05, name='')

        identity = downsample(x,planes,stride)
        x = eddl.Add(x,identity)

        x = eddl.ReLu(x)
        return x

    def make_layer(x, planes, blocks, stride):

        for i in range(blocks):
            x = block(x, planes, stride)
    
        return x


    x = make_layer(x,64,l[0],2)
    x = make_layer(x,128,l[1],2)
    x = make_layer(x,256,l[2],2)
    x = make_layer(x,512,l[3],2)

    x = eddl.GlobalAveragePool2D(x)
    x = eddl.Flatten(x)
    x = eddl.Softmax(eddl.Dense(x, num_classes))

    return x
    
   
def load_onnx_models(name, path = 'onnx_models'):
    model = eddl.import_net_from_onnx_file(f"{path}/{name}.onnx")
    out = model.lout[0]

    return model, out




