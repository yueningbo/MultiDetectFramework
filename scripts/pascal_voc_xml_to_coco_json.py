import xml.etree.ElementTree as ET
import os
import json


def add_category(coco, category_set, name):
    category_id = len(category_set) + 1
    category_item = {
        'supercategory': 'none',
        'id': category_id,
        'name': name
    }
    coco['categories'].append(category_item)
    category_set[name] = category_id
    return category_id


def add_image(coco, image_set, file_name, size):
    image_id = len(coco['images']) + 1
    if None in (file_name, size['width'], size['height']):
        raise ValueError('Missing filename or image size information in XML.')

    image_item = {
        'id': image_id,
        'file_name': file_name,
        'width': size['width'],
        'height': size['height']
    }
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def add_annotation(coco, annotation_id, image_id, category_id, bbox):
    seg = [
        bbox[0], bbox[1],
        bbox[0], bbox[1] + bbox[3],
                 bbox[0] + bbox[2], bbox[1] + bbox[3],
                 bbox[0] + bbox[2], bbox[1]
    ]

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


def parse_xml_files(xml_path, txt_file_name, coco, category_set, image_set, txt_path):
    with open(os.path.join(txt_path, txt_file_name), 'r') as f:
        img_ids_set = {line.split()[0] for line in f.readlines()}

    annotation_id = 0

    for xml_file in os.listdir(xml_path):
        if not xml_file.endswith('.xml') or xml_file.split('.')[0] not in img_ids_set:
            continue

        xml_full_path = os.path.join(xml_path, xml_file)
        print(f'Processing file: {xml_full_path}')

        tree = ET.parse(xml_full_path)
        root = tree.getroot()

        if root.tag != 'annotation':
            raise ValueError(f'Invalid XML root element, expected "annotation", got "{root.tag}".')

        size = {'width': None, 'height': None, 'depth': None}
        file_name = None
        current_image_id = None

        for elem in root:
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in image_set:
                    raise ValueError(f'Duplicate image: {file_name}')

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
                        current_category_id = category_set.get(object_name) or add_category(coco, category_set,
                                                                                            object_name)

                    elif subelem.tag == 'bndbox':
                        for option in subelem:
                            bndbox[option.tag] = int(option.text)

                if None not in bndbox.values() and object_name:
                    if current_image_id is None:
                        current_image_id = add_image(coco, image_set, file_name, size)
                        print(f'Added image: {file_name} with size {size}')

                    bbox = [
                        bndbox['xmin'],
                        bndbox['ymin'],
                        bndbox['xmax'] - bndbox['xmin'],
                        bndbox['ymax'] - bndbox['ymin']
                    ]
                    annotation_id += 1
                    print(
                        f'Adding annotation: {object_name}, Image ID: {current_image_id}, Category ID: {current_category_id}, BBox: {bbox}')
                    add_annotation(coco, annotation_id, current_image_id, current_category_id, bbox)


def build_coco_annotations(xml_path, output_path, txt_file_name, txt_path):
    coco = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }
    category_set = {}
    image_set = set()

    parse_xml_files(xml_path, txt_file_name, coco, category_set, image_set, txt_path)

    with open(output_path, 'w') as f:
        json.dump(coco, f)


if __name__ == "__main__":
    xml_path = 'data/datasets/VOCdevkit/VOC2007/Annotations'
    image_path = 'data/dataset/VOCdevkit/voc2007/JPEGImages'
    txt_path = 'data/datasets/VOCdevkit/VOC2007/ImageSets/Main'
    txt_file = ['train.txt', 'test.txt', 'val.txt']
    output_path = "data/datasets/VOCdevkit/VOC2007"

    for file in txt_file:
        output_file_path = os.path.join(output_path, file.replace('txt', 'json'))
        build_coco_annotations(xml_path, output_file_path, file, txt_path)
