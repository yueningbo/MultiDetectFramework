import torch
from models.yolov1.yolov1_model import YOLOv1
from data.loaders.dataset_loader import get_loader  # 假设这个函数可以加载数据
from utils.utils import load_config
from utils.visualization import visualize_prediction  # 假设你有一个保存结果的函数


def inference(config_path, pretrained_weights_path):
    # 配置
    conf_threshold = 0.4
    nms_threshold = 0.5

    config = load_config(config_path)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = YOLOv1(pretrained_weights_path=pretrained_weights_path)

    model.load_state_dict(torch.load(config))
    model.to(device)

    # 加载数据
    dataloader = get_loader(config, device)

    # 执行推理
    results = model.inference(dataloader, device, conf_threshold, nms_threshold)

    # 保存结果
    visualize_prediction(results, 'path/to/save/results')


if __name__ == "__main__":
    config_path = '../configs/yolov1.json'
    pretrained_weights_path = ''
    inference(config_path, pretrained_weights_path)
