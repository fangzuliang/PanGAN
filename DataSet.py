import os
import numpy as np
import h5py
import cv2

class DataSet(object):

    def __init__(self, pan_size, ms_size, img_path, data_save_path, batch_size, num_spectrum, stride, norm):
        self.pan_size=pan_size
        self.ms_size=ms_size
        if not os.path.exists(data_save_path):
            self.make_data(img_path, data_save_path, stride, norm)
        self.all_pan_patch, self.all_ms_patch = self.read_data(data_save_path)
        self.data_generator=self.generator(batch_size, num_spectrum)

    def generator(self, batch_size, num_spectrum):

        num_data=self.all_pan_patch.shape[0]
        while True:
            batch_pan_img=np.zeros((batch_size, self.pan_size, self.pan_size, 1))
            batch_ms_img=np.zeros((batch_size, self.ms_size, self.ms_size, num_spectrum))
            for i in range(batch_size):
                random_index=np.random.randint(0, num_data)
                batch_pan_img[i]=self.all_pan_patch[random_index]
                batch_ms_img[i]=self.all_ms_patch[random_index]
            yield batch_pan_img, batch_ms_img

    def make_data(self,img_path, save_path,stride, norm ):
        ms_img_path=img_path + '/source_ir'
        pan_img_path=img_path + '/vi'
        all_pan_img=[]
        all_ms_img=[]
        img_list=os.listdir(ms_img_path)
        for img_name in img_list:
            ms_img=os.path.join(ms_img_path,img_name)
            pan_img=os.path.join(pan_img_path,img_name)
            ms_img_list=self.crop_to_patch(ms_img, stride, norm, name='ms')
            print('ir_'+img_name + ': ' + str(len(ms_img_list)))
            pan_img_list=self.crop_to_patch(pan_img, stride, norm, name='pan')
            print('vi_'+img_name + ': ' + str(len(pan_img_list)))
            all_pan_img.extend(pan_img_list)
            all_ms_img.extend(ms_img_list)
        print('The number of ms patch is: ' + str(len(all_ms_img)))
        print('The number of pan patch is: ' + str(len(all_pan_img)))
        all_pan_img=np.array(all_pan_img)
        all_ms_img=np.array(all_ms_img)
        f=h5py.File(save_path, 'w')
        f.create_dataset('pan', data=all_pan_img)
        f.create_dataset('ms', data=all_ms_img)

    def read_data(self,path):
        f=h5py.File(path, 'r')
        all_pan_patch=np.array(f['pan'])
        all_ms_patch=np.array(f['ms'])
        return all_pan_patch, all_ms_patch

    def crop_to_patch(self,img_path,stride, norm=True, name='pan'):
        kernel_size=(5,5)
        sigma=1.5
        img_patch_list=[]
        patch_size=self.pan_size
        '''ms_img should read in another way'''
        img=self.read_img(img_path, norm)
        h,w = img.shape
        for i in range(0,h-patch_size,stride):
            for j in range(0, w-patch_size, stride):
                img_patch=(img[i:i+patch_size, j:j+patch_size]).reshape( patch_size, patch_size,1)
                if name == 'ms':
                    size=(self.ms_size, self.ms_size)
                    img_patch=cv2.GaussianBlur(img_patch, kernel_size, sigma)
                    img_patch=cv2.resize(img_patch, size, interpolation=cv2.INTER_AREA).reshape(self.ms_size,self.ms_size,1)
                img_patch_list.append(img_patch)
                if i + patch_size >= h:
                    img_patch=img[h-patch_size:,j:j+patch_size].reshape( patch_size, patch_size,1)
                    if name == 'ms':
                        size=(self.ms_size, self.ms_size)
                        img_patch=cv2.GaussianBlur(img_patch, kernel_size, sigma)
                        img_patch=cv2.resize(img_patch, size, interpolation=cv2.INTER_AREA).reshape(self.ms_size,self.ms_size,1)
                        
                    img_patch_list.append(img_patch)
            
            img_patch=img[i:i+patch_size,w-patch_size:].reshape(patch_size,patch_size,1)
            if name == 'ms':
                size=(self.ms_size, self.ms_size)
                img_patch=cv2.GaussianBlur(img_patch, kernel_size, sigma)
                img_patch=cv2.resize(img_patch, size, interpolation=cv2.INTER_AREA).reshape(self.ms_size,self.ms_size,1)
            img_patch_list.append(img_patch)
        return img_patch_list

    def read_img(self, img_path, norm):
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if norm:
            img= img/255.0 -0.5
        return img