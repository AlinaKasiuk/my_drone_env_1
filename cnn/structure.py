from torch import nn


class DroneQNet(nn.Module):

    def __init__(self, in_channels, img_w, img_h, num_actions):
        super(DroneQNet, self).__init__()

        self.in_channels = in_channels
        self.img_w = img_w
        self.img_h = img_h
        self.num_actions = num_actions

        self.block1 = self.conv_block(c_in=in_channels, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3)

        h = self.get_last_hw(img_h, 0)
        w = self.get_last_hw(img_w, 1)

        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=num_actions, kernel_size=(h, w), stride=1, padding=0)

    def params_dict(self):
        return {"in_channels": self.in_channels, "img_w": self.img_w,
                "img_h": self.img_h, "num_actions": self.num_actions}

    def get_last_hw(self, img_d, index):
        # inside the sequential block, the conv layer is at index 1 (index 0 is for the sequential itself)
        d = self.get_out_size(img_d, list(self.block1.modules())[1], index)
        d = self.get_out_size(d, self.maxpool, index)
        d = self.get_out_size(d, list(self.block2.modules())[1], index)
        d = self.get_out_size(d, list(self.block3.modules())[1], index)
        return self.get_out_size(d, self.maxpool, index)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        # devolver como un vector de dos dimensiones
        return x.view(x.shape[0], -1)

    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block

    @staticmethod
    def get_out_size(num_features, layer, index):
        # extracted from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # se calcula igual en las capas convolucionales que con maxpool
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

        def inner_check(possible_list):
            try:
                return possible_list[index]
            except TypeError:
                return possible_list

        return 1 + (num_features + 2 * inner_check(layer.padding) - inner_check(layer.dilation) * (
                inner_check(layer.kernel_size) - 1) - 1) // inner_check(layer.stride)


if __name__ == '__main__':
    m = DroneQNet(2, 64, 64, 10)
    print(m.get_out_size())
