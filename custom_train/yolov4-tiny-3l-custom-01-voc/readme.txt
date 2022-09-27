customizing yolov4-tiny-3l for novatek embedded device 

leaky -> relu
maxpool -> 삭제 및 다음 conv_layer를 convolutional(size=3, stride=2, pad=1)로 변경(stride=2로 바꿈)
route with groups -> just route