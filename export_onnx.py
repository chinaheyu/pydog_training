import pickle
import cv2
from dqn_model import OdorDQN
import torch


policy = OdorDQN('cuda').to('cuda')
policy.load_state_dict(torch.load('output.pth'))
policy.eval()


dummy_input = torch.randn(1, 5, 11, 11, device="cuda")


input_names = ["input"]
output_names = ["output"]

torch.onnx.export(policy, dummy_input, "dqn.onnx", verbose=True, input_names=input_names, output_names=output_names)
