
# from models.model import FLRSegNet
# x = torch.randn((2, 3, 256, 128))

# model_args = dict(
#     n_classes=5,
#     num_fuzzy = [16, 32, 64],
#     num_node = [16, 32, 64],
# )
# net = FLRSegNet(**model_args)

# res = net(x)

import hashlib

md5 = hashlib.md5()

file_path = "weights/params-b5b19c.pth"

with open(file_path, 'rb') as f:
    while True:
        b = f.read(8192)
        if not b:
            break	
        md5.update(b)

file_hash = md5.hexdigest()
print(file_hash)