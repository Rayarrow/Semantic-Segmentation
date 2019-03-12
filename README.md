Encoder:
	VGG16
	Res50

Decoder:
	FCN32s
	FCN16s
	FCN8s
	PSPNet
	Deeplab v2

## TODO

1. get the edge as the soft gate.
2. get the stage-wise heatmap as the soft gate.


detection great thanks to https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe


ResNet101: 49*3*64+3*(256*64+9*64*64+64*256)+4*(512*128+9*128*128+128*512)+23*(1024*256+9*256*256+256*1024)+3*(2048*512+9*512*512+512*2048)=40326336
VGG16: 9*(3*64+64*64+64+128+128*128+128*256+2*256*256+256*512+5*512*512)=14638464

Simple=49*(3*64+7*64*64)=1414336
Simple_Conv_Part = 49*(3*64+3*64*64)=611520
Deconv = 49*4*64*64=802816
Simple_Stack_Conv_Part = 27*(3*64+3*64*64)=336960