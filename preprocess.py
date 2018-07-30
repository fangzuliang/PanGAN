import cv2
import os

data_path='./data/image/source_ir'
save_path='./data/image/ir'

if not os.path.exists(save_path):
    os.makedirs(save_path)

img_list=os.listdir(data_path)
kernel_size=(5,5)
sigma=1.5

for img_name in img_list:
    img_path=os.path.join(data_path, img_name)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    size=(int(width*0.25), int(height*0.25))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(save_path, img_name), img)
    
    