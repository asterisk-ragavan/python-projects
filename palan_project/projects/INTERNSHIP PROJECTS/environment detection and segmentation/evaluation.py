import tarfile
import numpy as np
from PIL import Image
import cv2 as cv
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import warnings
import dlm

warnings.simplefilter("ignore", DeprecationWarning)


class DriveSeg(object):
    def __init__(self, tarball_path):
        self.tar_file = tarfile.open(tarball_path)
        self.tar_info = self.tar_file.getmembers()

    def fetch(self, index):
        tar_info = self.tar_info[index + 1]  # exclude index 0 which is the parent directory
        file_handle = self.tar_file.extractfile(tar_info)
        gt = np.fromstring(file_handle.read(), np.uint8)
        gt = cv.imdecode(gt, cv.IMREAD_COLOR)
        gt = gt[:, :, 0]  # select a single channel from the 3-channel image
        gt[gt == 255] = 19  # void class, does not count for accuracy
        return gt


SAMPLE_IMAGE = r'C:\Users\sakth\PycharmProjects\python-projects\palan_project\projects\INTERNSHIP ' \
               r'PROJECTS\environment detection and segmentation\environment_detection_and_segmentation.png'
SAMPLE_GT = r'C:\Users\sakth\PycharmProjects\python-projects\palan_project\projects\INTERNSHIP PROJECTS\environment ' \
            r'detection and segmentation\environment_detection_and_segmentation_gt.tar.gz'
dataset = DriveSeg(SAMPLE_GT)
print('visualizing ground truth annotation on the sample image...')

original_im = Image.open(SAMPLE_IMAGE)
gt = dataset.fetch(0)  # sample image is frame 0
dlm.vis_segmentation(original_im, gt)

def evaluate_single(seg_map, ground_truth):
    """Evaluate a single frame with the MODEL loaded."""
    # merge label due to different annotation scheme
    seg_map[np.logical_or(seg_map == 14, seg_map == 15)] = 13
    seg_map[np.logical_or(seg_map == 3, seg_map == 4)] = 2
    seg_map[seg_map == 12] = 11
    # calculate accuracy on valid area
    acc = np.sum(seg_map[ground_truth != 19] == ground_truth[ground_truth != 19]) / np.sum(ground_truth != 19)
    # select valid labels for evaluation
    cm = confusion_matrix(ground_truth[ground_truth != 19], seg_map[ground_truth != 19],
                          labels=np.array([0, 1, 2, 5, 6, 7, 8, 9, 11, 13]))
    intersection = np.diag(cm)
    union = np.sum(cm, 0) + np.sum(cm, 1) - np.diag(cm)
    return acc, intersection, union


original_im = Image.open(SAMPLE_IMAGE)
seg_map = dlm.MODEL.run(original_im)
gt = dataset.fetch(0)  # sample image is frame 0
acc, intersection, union = evaluate_single(seg_map, gt)
class_iou = np.round(intersection / union, 5)
print('pixel accuracy: %.5f' % acc)
print('mean class IoU:', np.mean(class_iou))
print('class IoU:')
print(tabulate([class_iou], headers=dlm.LABEL_NAMES[[0, 1, 2, 5, 6, 7, 8, 9, 11, 13]]))