#定义的deep_sort特征提取器的original_model，resnet50

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class BasicBlock(nn.Layer):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2D(c_in, c_out, 3, stride=2, padding=1, bias_attr=False)
        else:
            self.conv1 = nn.Conv2D(c_in, c_out, 3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(c_out)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(c_out,c_out,3,stride=1,padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2D(c_in, c_out, 1, stride=2, bias_attr=False),
                nn.BatchNorm2D(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2D(c_in, c_out, 1, stride=1, bias_attr=False),
                nn.BatchNorm2D(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y))

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class Net(nn.Layer):
    def __init__(self, num_classes=625 ,reid=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2D(3,32,3,stride=1,padding=1),
            nn.BatchNorm2D(32),
            nn.ELU(),
            nn.Conv2D(32,32,3,stride=1,padding=1),
            nn.BatchNorm2D(32),
            nn.ELU(),
            nn.MaxPool2D(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(32,32,2,False)
        # 32 64 32
        self.layer2 = make_layers(32,64,2,True)
        # 64 32 16
        self.layer3 = make_layers(64,128,2,True)
        # 128 16 8
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*8, 128),
            nn.BatchNorm1D(128),
            nn.ELU()
        )
        # 256 1 1 
        self.reid = reid
        self.batch_norm = nn.BatchNorm1D(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = paddle.reshape(x, [x.shape[0],-1])
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = paddle.divide(x, paddle.norm(x, p=2, axis=1,keepdim=True))
            return x
        x = self.dense(x)
        # B x 128
        # classifier
        x = self.classifier(x)
        return x