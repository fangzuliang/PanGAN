class FLAGES(object):

    pan_size= 64
    
    ms_size=16
    
    
    num_spectrum=1
    
    ratio=4
    stride=16
    norm=True
    
    
    batch_size=64
    lr=0.001
    decay_rate=0.99
    decay_step=10000
    
    img_path='./data/image'
    data_path='./data/train.h5'
    log_dir='./log'
    model_save_dir='./model'
    
    is_pretrained=False
    
    iters=50000
    model_save_iters = 100
