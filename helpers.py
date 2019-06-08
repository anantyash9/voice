import os
import siamese_comparator
registered ={}
PATH = os.path.dirname(os.path.realpath(__file__))
default_wav_path=PATH+'/'+'default.wav'
threshold=0.15

def to_digits(text):
    list1=[s for s in text if s.isdigit()]
    str1 = ''.join(list1)
    return str1
    

def load_all_reg():
    global registered
    root_folder = PATH+'/reg_emps'
    content = {}
    
    for root, dirs, files in os.walk(root_folder):
        for subdir in dirs:
            if subdir not in content.keys():
                content[subdir] = []
        if root.split('/')[-1] in content.keys():
            content[root.split('/')[-1]]=[root+'/'+f for f in files]
    registered=content    

def is_reg(id):
    return id in registered.keys()

def compare(id):
    list1=registered[id]
    list2=[default_wav_path for i in range(len(list1))]
    distance=siamese_comparator.average_distance(list1,list2)
    print(distance)
    return distance<=threshold

def validate(text):
    digits=to_digits(text)
    if is_reg(digits):
        return compare(digits), digits
    else:
        print('id not registered')
        return False, None


load_all_reg()
#print(validate('738608'))
