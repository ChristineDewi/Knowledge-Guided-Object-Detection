import sys
import os

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'D:\\PROJECT\\rescience-ijcai2017-230')

import json
import torch
import numpy as np
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from Datasets.datasets import CocoDetection
from Utils.testing import *
from Utils.metrics import *

# CONFIG
model_type = 'coco-FRCNN-resnet50'  # choose between 'coco-FRCNN-resnet50' or 'coco-FRCNN-vgg16'
matrix_type = 'KF-All-COCO'         # choose between 'KF-All-COCO' or 'KG-CNet-55-COCO'
detections_per_image = 500          # the maximum number of detected objects per image
num_iterations = 10                 # number of iterations to calculate p_hat
box_score_threshold = 1e-5          # minimum score for a bounding box to be kept as detection (default = 0.05)
bk = 5                              # number of neighbouring bounding boxes to consider for p_hat
lk = 5                              # number of largest semantic consistent classes to consider for p_hat
epsilon = 0.75                      # trade-off parameter for traditional detections and knowledge aware detections
topk = 100                          # maximum number of detections to be considered for metrics (recall@k / mAP@k)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # Load in test set
    batch_size_test = 1
    workers = 2
    path_project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path_project = 'D:\\PROJECT\\rescience-ijcai2017-230'  # My change
    coco_train_imgFile = os.path.join(path_project, "Data raw/COCO2014/train2014")
    coco_train_annFile = os.path.join(path_project, "Data raw/COCO2014/annotations/instances_train2014.json")
    coco_train_dataset = CocoDetection(root=coco_train_imgFile, annFile=coco_train_annFile)
    coco_validation_imgFile = os.path.join(path_project, "Data raw/COCO2014/val2014")
    coco_validation_annFile = os.path.join(path_project, "Data raw/COCO2014/annotations/instances_val2014.json")
    coco_validation_dataset = CocoDetection(root=coco_validation_imgFile, annFile=coco_validation_annFile)
    coco_combo_dataset = coco_validation_dataset + coco_train_dataset

    coco_minival_1k, coco_minival_4k, coco_trainset = torch.utils.data.random_split(
        coco_combo_dataset,
        [1000, 4000, 118287],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = torch.utils.data.DataLoader(
        coco_minival_4k,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CocoDetection.collate_fn
    )

    settype = 'coco'
    print("COCO test set: \t\t\t", coco_minival_4k.__len__(), " Images")

    # Load frequency-based and knowledge-based consistency matrices
    S_KF_All_COCO, S_KG_55_COCO = None, None

    file_path_freq = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_freq_info_scaled.json")
    if os.path.isfile(file_path_freq):
        print("Loading frequency-based consistency matrix")
        with open(file_path_freq, 'r') as j:
            info = json.load(j)
        S_KF_All_COCO = np.asarray(info['KF_All_COCO_info']['S'])  # Replace key if necessary

    file_path_kg = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_kg_55_info_scaled.json")
    if os.path.isfile(file_path_kg):
        print("Loading knowledge-based consistency matrix")
        with open(file_path_kg, 'r') as j:
            info = json.load(j)
        S_KG_55_COCO = np.asarray(info['KG_COCO_info']['S'])  # Replace key if necessary

    # Define the model based on the configuration
    num_classes = 92
    if model_type == 'coco-FRCNN-resnet50':
        backbone = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=5)
        backbone.out_channels = 256
        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                           aspect_ratios=((0.5, 1.0, 2.0),) * 5)

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           box_detections_per_img=detections_per_image,
                           box_score_thresh=box_score_threshold)
        
    elif model_type == 'coco-FRCNN-vgg16':
        backbone_vgg = torchvision.models.vgg16(pretrained=False).features
        out_channels = 512
        in_channels_list = [128, 256, 512, 512]
        return_layers = {'9': '0', '16': '1', '23': '2', '30': '3'}
        backbone = BackboneWithFPN(backbone_vgg, return_layers, in_channels_list, out_channels)
        backbone.out_channels = 512

        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                           aspect_ratios=(
                                               (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0),
                                               (0.5, 1.0, 2.0)))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           box_detections_per_img=detections_per_image,
                           box_score_thresh=box_score_threshold
                           )    
    else:
        print("Please select a valid model type.")
        sys.exit(1)

    model.to(device)

    model_path = "Model Training/Trained models/" + model_type + ".pth"
    file_path = os.path.join(path_project, model_path)
    checkpoint = torch.load(file_path,  map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Currently testing: ")
    print("threshold = ", box_score_threshold, "bk = ", bk, "lk = ", lk, "epsilon = ", epsilon, "S = ",
          matrix_type, "top k = ", topk)

    # Loop through weights and perform testing for combined matrix only if both matrices are loaded
    if matrix_type == 'KF-All-COCO' and S_KF_All_COCO is not None and S_KG_55_COCO is not None:
        weights = [0, 0.25, 0.5, 0.75]
        weights = [0]
        for e in weights:
            # Combine the matrices based on weight `e`
            S_combined = ( e * S_KF_All_COCO) + (e * S_KG_55_COCO)
            S = torch.from_numpy(S_combined).to(device)

            print(f"\nTesting with combined matrix (e={e}):")

            det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, true_areas = test_function_kg(
                 test_loader, model, model_type, settype, S, lk, bk, num_iterations, epsilon, num_classes - 1, topk, detections_per_image, box_score_threshold
            )

            average_precisions, mean_average_precision, classwise_recall, all_recall, recall_S, recall_M, recall_L = coco_metrics(
                 det_boxes, det_labels, det_scores, true_boxes, true_labels, true_areas
             )

            print(f"Results for weight of KF is = {e} and KG is = {e}:")
            print("AP @", topk, " per class: ", average_precisions)
            print("mAP @", topk, " : ", mean_average_precision)
            print("Recall @", topk, " per class: ", classwise_recall)
            print("Recall @", topk, " all classes (averaged): ", all_recall)
            print("Recall @", topk, " small: ", recall_S)
            print("Recall @", topk, " medium: ", recall_M)
            print("Recall @", topk, " large: ", recall_L)
            print("----------")
    elif matrix_type == 'KF-All-COCO' and S_KF_All_COCO is not None:
        S = torch.from_numpy(S_KF_All_COCO).to(device)
        print("Loaded only frequency-based matrix for testing.")
    elif matrix_type == 'KG-CNet-55-COCO' and S_KG_55_COCO is not None:
        S = torch.from_numpy(S_KG_55_COCO).to(device)
        print("Loaded only knowledge-based matrix for testing.")
    else:
        print("Error: Required matrices not loaded correctly or invalid matrix type selected.")
