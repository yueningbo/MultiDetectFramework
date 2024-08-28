import xml.etree.ElementTree as ET
import os
import json

xml_path = 'data/datasets/VOCdevkit/VOC2007/Annotations'
image_path = 'data/dataset/VOCdevkit/voc2007/JPEGImages'

train_txt = 'trainval.txt'
val_txt = 'test.txt'
annot = "./data/dataset/annotation"

coco = {
    'images': [],
    'type': 'instances',
    'annotations': [],
    'categories': []
}

category_set = {}
image_set = set()

category_item_id = 0
image_id = 0
annotation_id = 0


def addCatItem(name):
    global category_item_id
    category_item_id += 1
    category_item = {
        'supercategory': 'none',
        'id': category_item_id,
        'name': name
    }
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if None in (file_name, size['width'], size['height']):
        raise ValueError('Missing filename or image size information in XML.')

    image_id += 1
    image_item = {
        'id': image_id,
        'file_name': file_name,
        'width': size['width'],
        'height': size['height']
    }
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    seg = [
        bbox[0], bbox[1],
        bbox[0], bbox[1] + bbox[3],
                 bbox[0] + bbox[2], bbox[1] + bbox[3],
                 bbox[0] + bbox[2], bbox[1]
    ]

    annotation_id += 1
    annotation_item = {
        'segmentation': [seg],
        'area': bbox[2] * bbox[3],
        'iscrowd': 0,
        'ignore': 0,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id,
        'id': annotation_id
    }
    coco['annotations'].append(annotation_item)


def parseXmlFiles(xml_path, is_train_data=True):
    txt_file_name = train_txt if is_train_data else val_txt
    img_ids_set = set()

    txt_path = r'D:\git_project\MultiDetectFramework\data\datasets\VOCdevkit\VOC2007\ImageSets\Main'
    with open(os.path.join(txt_path, txt_file_name), 'r') as f:
        img_ids_set.update(line.split()[0] for line in f.readlines())

    for f in os.listdir(xml_path):
        if not f.endswith('.xml') or f.split('.')[0] not in img_ids_set:
            continue

        xml_file = os.path.join(xml_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise ValueError(f'Pascal VOC XML root element should be "annotation", found {root.tag}.')

        size = {'width': None, 'height': None, 'depth': None}
        file_name = None
        current_image_id = None

        for elem in root:
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in image_set:
                    raise ValueError(f'Duplicated image: {file_name}')

            elif elem.tag == 'size':
                for subelem in elem:
                    size[subelem.tag] = int(subelem.text)

            elif elem.tag == 'object':
                bndbox = {'xmin': None, 'ymin': None, 'xmax': None, 'ymax': None}
                object_name = None
                current_category_id = None

                for subelem in elem:
                    if subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name not in category_set:
                            current_category_id = addCatItem(object_name)
                        else:
                            current_category_id = category_set[object_name]

                    elif subelem.tag == 'bndbox':
                        for option in subelem:
                            bndbox[option.tag] = int(option.text)

                if None not in bndbox.values() and object_name:
                    if current_image_id is None:
                        current_image_id = addImgItem(file_name, size)
                        print(f'Added image: {file_name} with size {size}')

                    bbox = [
                        bndbox['xmin'],
                        bndbox['ymin'],
                        bndbox['xmax'] - bndbox['xmin'],
                        bndbox['ymax'] - bndbox['ymin']
                    ]
                    print(f'Adding annotation: {object_name}, {current_image_id}, {current_category_id}, {bbox}')
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)


def build_coco_annot(xml_path, output_path, is_train_data=True):
    parseXmlFiles(xml_path, is_train_data)
    with open(output_path, 'w') as f:
        json.dump(coco, f)


# Example usage:
# build_coco_annot(xml_path, "./train.json")
build_coco_annot(xml_path, "./test.json", is_train_data=False)
# build_coco_annot(xml_path, "/val.json")
