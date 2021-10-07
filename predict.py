''' Predict and visualize.
'''
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from data import PartDataset
from models import PointNetCls
from utils import classes_dict, read_pointnet_colors


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTIONS][PC_INDEX]...",
        description="Classification of a shape and visualization.",
    )
    parser.add_argument(
        "-m", "--model_path", default="cls_solutions/cls_model_24.pth"
    )
    parser.add_argument(
        "-d", "--data_path", default="shapenetcore_partanno_segmentation_benchmark_v0"
    )
    parser.add_argument("index", type=int)
    return parser

def inference(model : PointNetCls, to_cuda : bool=False):
    '''Create a prediction function.
    
    The prediction includes the forward and post-processing.

    Args:
        model: The model from which .forward is called.
        to_cuda: Tells if the model is on the GPU.
    Returns:
        The prediction function.
    '''

    def predict_fn(inputs):
        '''Make prediction and post-processing

        Args:
            inputs: torch.tensor[#points, 3]
        Returns:
            A torch.tensor[1, #classes], probabilities of each class as predicted 
        '''
        
        # perform inference on GPU
        points = inputs.unsqueeze(0).transpose(2, 1)
        if to_cuda:
            points = points.cuda()
            pred_logsoft, _ = model(points)
            # post-processing
            pred_prob = torch.exp(pred_logsoft)

            return pred_prob.cpu()
        else:
            pred_logsoft, _ = model(points)
            # post-processing
            pred_prob = torch.exp(pred_logsoft)

            return pred_prob

    return predict_fn

def main():
    
    parser = init_argparse()
    args = parser.parse_args()

    # model
    NUM_POINTS = 2500
    MODEL_PATH = args.model_path

    classifier = PointNetCls(num_points=NUM_POINTS, k=len(classes_dict))
    if torch.cuda.is_available(): #TODO: create a device
        classifier.cuda()
        classifier.load_state_dict(torch.load(MODEL_PATH))
        to_cuda = True
    else:
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    classifier.eval()
    classifier_fn = inference(classifier, to_cuda=to_cuda)

    # Data
    DATA_FOLDER = args.data_path
    test_dataset_seg = PartDataset(root=DATA_FOLDER, train=False, classification=False, npoints=NUM_POINTS)

    # Inference and visualization
    index = args.index
    print('[Sample {} / {}]'.format(index, len(test_dataset_seg)))
    point_set, seg = test_dataset_seg[index]        
    
    pred_prob = classifier_fn(point_set)
    pred_prob = pred_prob.squeeze()
    pred_class = pred_prob.argmax().item()
    print('Your object is pred_prob [{}] with probability {:0.3}.'
        .format(list(classes_dict.keys())[pred_class], pred_prob[pred_class]))

    # There's an error when uing draw_geometries and plt (https://github.com/isl-org/Open3D/issues/1715)
    plt.xticks(list(classes_dict.values()), list(classes_dict.keys()),rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('Probabilities')
    plt.plot(list(classes_dict.values()), pred_prob.detach().numpy())
    plt.show()

    # Colorized point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_set)
    cloud.colors = o3d.utility.Vector3dVector(read_pointnet_colors(seg.numpy()))
    o3d.visualization.draw_geometries([cloud])

if __name__ == "__main__":
    main()