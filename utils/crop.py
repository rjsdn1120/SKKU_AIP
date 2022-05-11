import json
from PIL import Image
import os
import torch
from torch import nn
from glob import glob


def run():
    path='/media/data2/rjsdn/AIP/data/img2/real/'
    data2 =sorted([y for x in os.walk(path) for y in glob(os.path.join(x[0],'*.jpg'))])
    # data2 = data2[0]
    save_path='/media/data2/rjsdn/AIP/data/pp'

    path_dict={}
    for x in data2:
        path_dict[x.split('/')[-1]] = x


    json_path = '/media/data2/rjsdn/AIP/data/annotation/2D/'
    jsons =[y for x in os.walk(json_path) for y in glob(os.path.join(x[0],'*.json'))]


    cnt=0
    for x in jsons:
    #     each = path_dict.get(x.split('/')[-1].split('.json')[0][:-3]+'.jpg',-1)
    #     print(x.split('/')[-1].split('.json')[0][:-3]+'.jpg')
    #     if each == -1 : continue
    #     print(each,x,x.split('/')[-1].split('.json')[0][:-3])
        cls = int(x.split('/')[-2].split('-')[0])
        if cls > 25 or cls<=23 : continue
        with open(x) as json_file:
            json_data = json.load(json_file)
            json_annot=json_data["annotations"]
            json_image=json_data["images"]
            img_path=[]
            bbox=[]

            for image in json_image:        
                img_path.append(image["img_path"].split('/')[-1])


    #         if len(json_annot) <= 1 or json_annot[1]["person_no"]==1:
    #             for annot in json_annot:
    #                 bbox.append(annot['bbox'])

    #         elif json_annot[1]["person_no"]==2:
    #             for i in range(len(img_path)):
    #                 person1=json_annot[2*i]['bbox']
    #                 person2=json_annot[2*i+1]['bbox']
    #                 box=[min(person1[0],person2[0]),min(person1[1],person2[1]),max(person1[2],person2[2]),max(person1[3],person2[3])]
    #                 bbox.append(box)


            for an in json_annot:
                box=[1e9,1e9,-1,-1]
                for k in range(4):
                    f = max
                    if k<2:
                        f = min
                    try:
                        box[k]=f(box[k],an['bbox'][k])
                    except:
                        print('box error')
                        box[k]=None
                bbox.append(box)

            try:
                for i,paths in enumerate(img_path):
                    if bbox[i][0]==None: continue
                    img = path_dict.get(paths,-1)
                    if img==-1:
                        continue
                    if cnt%1000==0:
                        print(img)
                    cnt+=1
                    save = os.path.join(save_path,'/'.join(img.split('/')[-3:-1]))
                    if not os.path.exists(save):
                        os.makedirs(save)
                    save = os.path.join(save,img.split('/')[-1])
                    image=Image.open(img)
                    cropped=image.crop(bbox[i])
                    cropped.save(save)
            except:
                print('error',img)

if __name__=='__main__':
    run()