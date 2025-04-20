from collections import OrderedDict
import torch
import nnsight
from nnsight import NNsight

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
).requires_grad_(False)
tiny_model = NNsight(net)

input = torch.rand((1, input_size))
with tiny_model.trace(input):

    l1_output = tiny_model.layer1.output.save()
    l1_amax = torch.argmax(l1_output, dim=1).save()
    l2_output = tiny_model.layer2.output.save()
    l2_amax = torch.argmax(l2_output, dim=1).save()

print(l1_output)
print(l2_output)
print(l1_amax[0])
print(l2_amax[0])
