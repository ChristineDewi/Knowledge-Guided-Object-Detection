import sys
import os

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'D:/PROJECT/rescience-ijcai2017-230')

import torchvision
from Datasets.datasets import CocoDetection
from Utils.testing import *
from Utils.metrics import *
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torch.utils.data import random_split

# CONFIG
model_type = 'coco-FRCNN-resnet101'  # choose between 'coco-FRCNN-resnet50' or 'coco-FRCNN-vgg16'
matrix_type = 'KG-CNet-55-COCO'     # choose between 'KF-All-COCO', 'KF-500-COCO', 'KG-CNet-57-COCO' or 'KG-CNet-55-COCO'
detections_per_image = 500          # the maximum number of detected objects per image
num_iterations = 10                 # number of iterations to calculate p_hat
box_score_threshold = 1e-5          # minimum score for a bounding box to be kept as detection (default = 0.05)
bk = 5                              # number of neighbouring bounding boxes to consider for p_hat
lk = 5                              # number of largest semantic consistent classes to consider for p_hat
epsilon = 0.75                      # trade-off parameter for traditional detections and knowledge aware detections
topk = 100                          # maximum number of detections to be considered for metrics (recall@k / mAP@k)

"""
Running this file will output the mAP@k and recall@k per class and averaged for the test split (4k images, taken from
training and validation sets) on MS COCO 2014. Above configurations will output the results as mentioned in the paper.
A different model backbone and semantic consistency matrix (matrix_type) can be selected to generate the results for
different approaches.  
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    path_project = 'D:/PROJECT/rescience-ijcai2017-230'  # My change

    # Load in the COCO dataset
    batch_size_test = 1
    workers = 2

    coco_train_imgFile = os.path.join(path_project, "Data raw/COCO2014/train2014")
    coco_train_annFile = os.path.join(path_project, "Data raw/COCO2014/annotations/instances_train2014.json")
    coco_train_dataset = CocoDetection(root=coco_train_imgFile, annFile=coco_train_annFile)
    coco_validation_imgFile = os.path.join(path_project, "Data raw/COCO2014/val2014")
    coco_validation_annFile = os.path.join(path_project, "Data raw/COCO2014/annotations/instances_val2014.json")
    coco_validation_dataset = CocoDetection(root=coco_validation_imgFile, annFile=coco_validation_annFile)
    coco_combo_dataset = coco_validation_dataset + coco_train_dataset

    # Define the percentage of the dataset to use
    subset_size = int(0.0001 * len(coco_combo_dataset))  # For example, 1% of the dataset
    _, coco_test_subset = random_split(coco_combo_dataset, [len(coco_combo_dataset) - subset_size, subset_size])

    print("Creating DataLoader...")

    coco_test_dataloader = torch.utils.data.DataLoader(
        coco_test_subset,  # Use the subset here
        batch_size=batch_size_test,
        shuffle=False,
        collate_fn=CocoDetection.collate_fn,
        num_workers=workers,
        pin_memory=True
    )

    print("COCO test set: \t\t", coco_test_subset.__len__(), " Images")

    # Load in consistency matrices
    file_path = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_freq_info.json")
    if os.path.isfile(file_path):
        print("Loading frequency based consistency matrix")
        with open(file_path, 'r') as j:
            info = json.load(j)
        KF_All_COCO_info = info['KF_All_COCO_info']
        KF_500_COCO_info = info['KF_500_COCO_info']
        S_KF_All_COCO = np.asarray(KF_All_COCO_info['S'])
        S_KF_500_COCO = np.asarray(KF_500_COCO_info['S'])
    else:
        print("No matrix available")

    file_path = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_kg_55_info.json")
    if os.path.isfile(file_path):
        print("Loading knowledge based consistency matrix")
        with open(file_path, 'r') as j:
            info = json.load(j)
        KG_COCO_info = info['KG_COCO_info']
        S_KG_55_COCO = np.asarray(KG_COCO_info['S'])
    else:
        print("No matrix available")

    file_path = os.path.join(path_project, "Semantic Consistency/Stored matrices/CM_kg_57_info.json")
    if os.path.isfile(file_path):
        print("Loading knowledge based consistency matrix")
        with open(file_path, 'r') as j:
            info = json.load(j)
        KG_COCO_info = info['KG_COCO_info']
        S_KG_57_COCO = np.asarray(KG_COCO_info['S'])
    else:
        print("No matrix available")

    if matrix_type == 'KF-All-COCO':
        S = torch.from_numpy(S_KF_All_COCO).to(device)
    elif matrix_type == 'KF-500-COCO':
        S = torch.from_numpy(S_KF_500_COCO).to(device)
    elif matrix_type == 'KG-CNet-57-COCO':
        S = torch.from_numpy(S_KG_57_COCO).to(device)
    elif matrix_type == 'KG-CNet-55-COCO':
        S = torch.from_numpy(S_KG_55_COCO).to(device)
    else:
        print("Wrong matrix type selected")

    settype = 'coco'
    num_classes = 92  # Adjust based on the dataset and task
    test_loader = coco_test_dataloader

    if model_type == 'coco-FRCNN-resnet50':

        backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=5)

        backbone.out_channels = 256

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
                           box_score_thresh=box_score_threshold)
        
    elif model_type == 'coco-FRCNN-resnet101':

        backbone = resnet_fpn_backbone('resnet101', pretrained=False, trainable_layers=5)
        backbone.out_channels = 256

        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            box_detections_per_img=detections_per_image,
            box_score_thresh=box_score_threshold
        )
    

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
        print("please select a valid model")

    model.to(device)

    model_path = "Model Training/Trained models/" + model_type + ".pth"
    file_path = os.path.join(path_project, model_path)
    checkpoint = torch.load(file_path, map_location=device)  
  
    print(checkpoint.keys())  
    
    ######################################
    #this is only for restnet101
    
    # Load only the parts of the state dict that match the model
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()

   # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
   
   # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
   
   # Load the new state dict
    model.load_state_dict(model_dict)
    
    #till here for restnet101
    ######################################

    # model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print("Currently testing: ")
    print("threshold = ", box_score_threshold, "bk = ", bk, "lk = ", lk, "epsilon = ", epsilon, "S = ", matrix_type, "top k = ", topk)

    det_boxes, det_labels, det_scores, \
    true_boxes, true_labels, true_difficulties, true_areas = test_function_kg(test_loader, model, settype,
                                                                              S, lk, bk, num_iterations, epsilon,
                                                                              num_classes - 1, topk)

    average_precisions, mean_average_precision, classwise_recall, all_recall, \
    recall_S, recall_M, recall_L = coco_metrics(det_boxes, det_labels, det_scores,
                                                true_boxes, true_labels, true_areas
                                                )

    print("AP @", topk, " per class: ", average_precisions)
    print("mAP @", topk, " : ", mean_average_precision)
    print("Recall @", topk, " per class: ", classwise_recall)
    print("Recall @", topk, " all classes (averaged): ", all_recall)
    print("Recall @", topk, " small: ", recall_S)
    print("Recall @", topk, " medium: ", recall_M)
    print("Recall @", topk, " large: ", recall_L)
