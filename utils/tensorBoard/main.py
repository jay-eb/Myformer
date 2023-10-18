import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='tensorboard test')

for x in range(100):
    writer.add_scalar('y=2x', x*2, x)
    writer.add_scalar('y=pow(2,x)', pow(2, x), x)
    writer.add_scalars('data/scaler_group', {'xsinx': x * np.sin(x),
                                             'xcosx': x * np.cos(x),
                                             'tanhx': np.tanh(x)}, x)

writer.close()
