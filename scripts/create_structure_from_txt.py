import os

# 定义目录结构
directories = {
    'Detection_Framework': {
        'data': {
            'datasets': {},
            'transforms': {'data_augmentation.py': None},
            'loaders': {'dataset_loader.py': None}
        },
        'models': {
            'base': {'base_model.py': None},
            'yolov1': {'yolov1_model.py': None},
            'utils': {'parser.py': None}
        },
        'utils': {
            'losses.py': None,
            'metrics.py': None,
            'visualization.py': None,
            'utils.py': None
        },
        'configs': {
            'yolov1.json': None,
            'train_config.json': None
        },
        'scripts': {
            'train.py': None,
            'test.py': None,
            'evaluate.py': None
        },
        'logs': {
            'train': {},
            'test': {}
        },
        'outputs': {
            'yolov1': {},
            'faster_rcnn': {}
        },
        'docs': {
            'model_architecture.md': None,
            'training_guidelines.md': None
        }
    }
}


# 递归函数创建目录和文件
def create_dir_structure(base_path, structure):
    for name, value in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(value, dict):
            os.makedirs(path, exist_ok=True)
            create_dir_structure(path, value)
        else:
            with open(path, 'w') as file:
                file.write('')  # 创建空文件


# 创建根目录
root_path = 'Detection_Framework'
os.makedirs(root_path, exist_ok=True)

# 创建目录结构
create_dir_structure(root_path, directories)

print("目录和文件已创建。")
