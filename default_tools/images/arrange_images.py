import os
import json
import shutil
import cv2

CLASS_DICT = {'短袖Top': 'duanxiushangyi', '长袖Top': 'changxiushangyi', '短袖衬衫': 'duanxiuchenshan',
              '长袖衬衫': 'changxiuchenshan', '背心上衣': 'beixinshangyi', '吊带上衣': 'diaodaishangyi',
              '无袖上衣': 'wuxiushangyi', '短外套': 'duanwaitao', '短马甲': 'duanmajia',
              '长袖连衣裙': 'changxiulianyiqun', '短袖连衣裙': 'duanxiulianyiqun', '无袖连衣裙': 'wuxiulianyiqun',
              '长马甲': 'changmajia', '长款外套': 'changkuanwaitao', '连体衣': 'liantiyi',
              '古风': 'gufeng', '古装': 'gufeng', '短裙': 'duanqun',
              '中等半身裙（及膝）': 'zhongdengbanshenqun', '长半身裙（到脚）': 'changbanshenqun', '短裤': 'duanku',
              '中裤': 'zhongku', '长裤': 'changku', '背带裤': 'beidaiku'}

def arrage_image(img_dir, ann_dir, save_dir):
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith('jpg'):
                img_dir_name = root.replace('\\', '/').split('/')[-1]
                filename = img_dir_name + '_' + file
                # shutil.copy(os.path.join(root, file), os.path.join(save_dir, filename))
                img = cv2.imread(os.path.join(root, file))
                h, w, c = img.shape

                ann_name = file.replace('jpg', 'json')
                ann_path = os.path.join(ann_dir, img_dir_name, ann_name)
                if os.path.exists(ann_path):
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                else:
                    continue
                ann['img_name'] = ann['item_id'] + '_' + ann['img_name']
                ann['height'] = h
                ann['width'] = w
                if ann['annotations'] != []:
                    for ann_ in ann['annotations']:
                        if ann_['label'] in CLASS_DICT.keys():
                            ann_['label'] = CLASS_DICT[ann_['label']]
                        else:
                            print(ann_['label'])
                    shutil.copy(os.path.join(root, file), os.path.join(save_dir, filename))
                    with open(os.path.join(save_dir, filename.replace('jpg', 'json')), 'w') as f:
                        json.dump(ann, f, indent=4)


if __name__ == '__main__':
    img_dir = r'G:\Tianchi\Live_demo_20200117\image'
    ann_dir = r'G:\Tianchi\Live_demo_20200117\image_annotation'
    save_dir = r'G:\Tianchi\Live_demo_20200117\demo_images'
    arrage_image(img_dir, ann_dir, save_dir)
