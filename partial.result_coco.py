import sys
sys.path.insert(0, 'D:\\PROJECT\\rescience-ijcai2017-230')
import torchvision
import torch.nn as nn
from Datasets.datasets import CocoDetection
from Utils.testing import *
from Utils.metrics import *
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from yolox.models import CSPDarknet
from yolox.models.network_blocks import BaseConv, CSPLayer, DWConv
from yolox.exp import get_exp, Exp
from torch.utils.data import random_split
# CONFIG
# choose model type between:
# 'coco-FRCNN-resnet50', 'coco-FRCNN-vgg16', 'coco-FRCNN-yolox_darknet53'
# 'coco-YOLOX-darknet53'
model_type = 'coco-YOLOX-darknet53'
yolox_exp = None                    # for yolox only, None if model_type not YOLOX-darknet53
matrix_type = 'KG-CNet-55-COCO'     # choose between 'None', 'KF-All-COCO', 'KF-500-COCO', 'KG-CNet-57-COCO' or 'KG-CNet-55-COCO'
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

    print(f'Device: {device}')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    print('\n')

    path_project = 'D:\\PROJECT\\rescience-ijcai2017-230'
 
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

    coco_minival_1k, coco_minival_4k, coco_trainset = torch.utils.data.random_split(
        coco_combo_dataset,
        [1000, 4000, 118287],
        generator=torch.Generator().manual_seed(42)
    )

    generator = torch.Generator().initial_seed()  # reset generator to random instead of seed 42

    # Define the percentage of the dataset to use
    subset_size = int(0.001 * len(coco_minival_4k))  # For example, 1.0% of the dataset
    _, coco_test_subset = random_split(coco_minival_4k, [len(coco_minival_4k) - subset_size, subset_size])

    print("\nCreating DataLoader...")
 
    coco_test_dataloader = torch.utils.data.DataLoader(
        coco_minival_4k,  # Use the subset here
        batch_size=batch_size_test,
        shuffle=False,
        collate_fn=CocoDetection.collate_fn,
        num_workers=workers,
        pin_memory=True
    )
 
    print("COCO test set: \t\t", coco_minival_4k.__len__(), " Images\n")

 
    # Load in consistency matrices
    if not matrix_type is 'None':
        matrices = {
            'KF-All-COCO': "Semantic Consistency/Stored matrices/CM_freq_info.json",
            'KF-500-COCO': "Semantic Consistency/Stored matrices/CM_freq_info.json",
            'KG-CNet-55-COCO': "Semantic Consistency/Stored matrices/CM_kg_55_info.json",
            'KG-CNet-57-COCO': "Semantic Consistency/Stored matrices/CM_kg_57_info.json",
        }

        file_path = os.path.join(path_project, matrices.get(matrix_type, ""))
        if os.path.isfile(file_path):
            print(f"Loading matrix for {matrix_type} from {file_path}")
            with open(file_path, 'r') as j:
                info = json.load(j)
            S = torch.from_numpy(np.asarray(info['KG_COCO_info']['S'])).to(device)
        else:
            raise FileNotFoundError(f"Matrix file for {matrix_type} not found at {file_path}")
        
    else:
        S = None
 
    settype = 'coco'
    num_classes = 92  # Adjust based on the dataset and task
    test_loader = coco_test_dataloader
 
    if model_type == 'coco-FRCNN-resnet50':
 
        backbone = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=5)
 
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
                           box_score_thresh=box_score_threshold)
        
    elif model_type == 'coco-FRCNN-yolox_darknet53':

        # Custom module to combine backbone and YOLOPAFPN with spesific out_channels attribute
        class Custom_YOLOPAFPN(nn.Module):
            """
            YOLOv3 model. Darknet 53 is the default backbone of this model.
            """

            def __init__(
                self,
                backbone: nn.Module,
                depth=1.0,
                width=1.0,
                in_features=("dark3", "dark4", "dark5"),
                in_channels=[256, 512, 1024],
                depthwise=False,
                act="silu",
                out_channels=256,
            ):
                super().__init__()
                self.backbone = backbone
                self.in_features = in_features
                self.in_channels = in_channels
                self.out_channels = out_channels
                Conv = DWConv if depthwise else BaseConv

                self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
                self.lateral_conv0 = BaseConv(
                    int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
                )
                self.C3_p4 = CSPLayer(
                    int(2 * in_channels[1] * width),
                    int(in_channels[1] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act,
                )  # cat

                self.reduce_conv1 = BaseConv(
                    int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
                )
                self.C3_p3 = CSPLayer(
                    int(2 * in_channels[0] * width),
                    int(in_channels[0] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act,
                )

                # bottom-up conv
                self.bu_conv2 = Conv(
                    int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
                )
                self.C3_n3 = CSPLayer(
                    int(2 * in_channels[0] * width),
                    int(in_channels[1] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act,
                )

                # bottom-up conv
                self.bu_conv1 = Conv(
                    int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
                )
                self.C3_n4 = CSPLayer(
                    int(2 * in_channels[1] * width),
                    int(in_channels[2] * width),
                    round(3 * depth),
                    False,
                    depthwise=depthwise,
                    act=act,
                )

                # 1x1 conv layers to map to out_channels (used for FPN)
                self.out_conv2 = nn.Conv2d(int(in_channels[0] * width), out_channels, kernel_size=1)
                self.out_conv1 = nn.Conv2d(int(in_channels[1] * width), out_channels, kernel_size=1)
                self.out_conv0 = nn.Conv2d(int(in_channels[2] * width), out_channels, kernel_size=1)

            def forward(self, input):
                """
                Args:
                    inputs: input images.

                Returns:
                    Tuple[Tensor]: FPN feature.
                """

                #  backbone
                out_features = self.backbone(input)
                features = [out_features[f] for f in self.in_features]
                [x2, x1, x0] = features

                fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
                f_out0 = self.upsample(fpn_out0)  # 512/16
                f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
                f_out0 = self.C3_p4(f_out0)  # 1024->512/16

                fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
                f_out1 = self.upsample(fpn_out1)  # 256/8
                f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
                pan_out2 = self.C3_p3(f_out1)  # 512->256/8

                p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
                p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
                pan_out1 = self.C3_n3(p_out1)  # 512->512/16

                p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
                p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
                pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

                # Apply 1x1 convolutions to match FPN out_channels
                out_pan_out2 = self.out_conv2(pan_out2)  # 256/8 -> out_channels
                out_pan_out1 = self.out_conv1(pan_out1)  # 256/16 -> out_channels
                out_pan_out0 = self.out_conv0(pan_out0)  # 256/32 -> out_channels

                outputs = {
                    'dark3': out_pan_out2, 
                    'dark4': out_pan_out1, 
                    'dark5': out_pan_out0
                }
                
                return outputs

        depth = 1.0
        width = 1.0
        out_features = ("dark3", "dark4", "dark5")
        backbone_darknet53 = CSPDarknet(dep_mul=depth, wid_mul=width,
                                        out_features=out_features)
        
        out_channels = 512
        in_channels_list = [256, 512, 1024]
        backbone = Custom_YOLOPAFPN(backbone_darknet53, depth=depth,width=width,
                                    in_features=out_features, in_channels=in_channels_list,
                                    out_channels=out_channels)
        backbone.out_channels = out_channels
        
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128, 256), (32, 64, 128, 256, 512), (64, 128, 256, 512, 1024)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3
        )
 
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['dark3', 'dark4', 'dark5'],
                                                        output_size=7,
                                                        sampling_ratio=2)

        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler,
                           box_detections_per_img=detections_per_image,
                           box_score_thresh=box_score_threshold)
    
    elif model_type == 'coco-YOLOX-darknet53':
        
        yolox_exp: Exp = get_exp(exp_name='yolov3') # YOLOX-Darknet

        # ---------------- model config ---------------- #
        yolox_exp.num_classes = 80 # default coco2014 num_classes

        # -----------------  testing config ------------------ #
        yolox_exp.test_conf = box_score_threshold
        yolox_exp.nmsthre = 0.45

        model = yolox_exp.get_model()
 
    else:
        print("please select a valid model")
 
    model.to(device)

    model_path = "Model Training/Trained models/" + model_type + ".pth"
    # model_path = "Model Training/Trained models/" + "coco-FRCNN-vgg16_83k" + ".pth"
    file_path = os.path.join(path_project, model_path)
    checkpoint = torch.load(file_path, map_location=device)
    
    if '-YOLOX-' in model_type: model.load_state_dict(checkpoint['model'])
    else: model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.eval()

    print("\nCurrently testing: ")
    print("threshold = ", box_score_threshold, "bk = ", bk, "lk = ", lk, "epsilon = ", epsilon, "S = ", matrix_type, "top k = ", topk)
 
    det_boxes, det_labels, det_scores, \
    true_boxes, true_labels, true_difficulties, true_areas = test_function_kg(test_loader, model, yolox_exp, model_type,
                                                                              settype, S, lk, bk, num_iterations, epsilon,
                                                                              num_classes - 1, topk, detections_per_image)
 
    average_precisions, mean_average_precision, classwise_recall, all_recall, \
    recall_S, recall_M, recall_L = coco_metrics(det_boxes, det_labels, det_scores,
                                                true_boxes, true_labels, true_areas)
 
    print("AP @", topk, " per class: ", average_precisions)
    print("mAP @", topk, " : ", mean_average_precision)
    print("Recall @", topk, " per class: ", classwise_recall)
    print("Recall @", topk, " all classes (averaged): ", all_recall)
    print("Recall @", topk, " small: ", recall_S)
    print("Recall @", topk, " medium: ", recall_M)
    print("Recall @", topk, " large: ", recall_L)
    print(file_path)