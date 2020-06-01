import timm

m = timm.create_model('mobilenetv3_100', pretrained=True)
print(m.eval())