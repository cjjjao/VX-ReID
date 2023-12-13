# import cv2
# import numpy as np
# path = './0003.jpg'
# src = cv2.imread(path,1)
# src = src.astype(np.float32) / 255
# src[:,:,0] *= 0.5
# src[:,:,1] *= 0.8
# src = (src * 255).astype('uint8')
# cv2.imwrite('./000r.jpg',src)
# #image = np.array(src / 255., dtype=float)

class CLanguage:
    def __init__ (self):
        self.name = "C语言中文网"
        self.add = "http://c.biancheng.net"
    def say(self):
        print("我正在学Python")
clangs = CLanguage()
x = getattr(clangs,'name')
x = '123'
print(clangs.name)