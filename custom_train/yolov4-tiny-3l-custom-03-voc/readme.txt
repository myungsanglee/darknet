customizing yolov4-tiny-3l for novatek embedded device 

leaky -> relu
maxpool -> convolutional(size=2, stride=2, pad=0)
route with groups -> just route and add convolutional(size=1, stride=1, pad=1)
