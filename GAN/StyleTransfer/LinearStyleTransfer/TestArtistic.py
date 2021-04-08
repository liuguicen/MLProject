import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from libs.Loader import Dataset
from libs.MatrixForMible import MulLayer
from libs.modelsForMobile import decoder3, decoder4
from libs.modelsForMobile import encoder3, encoder4
from libs.utils import print_options

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/r41.pth',
                    help='pre-trained model path')
parser.add_argument("--stylePath", default="data/style/",
                    help='path to style image')
parser.add_argument("--contentPath", default="data/content/",
                    help='path to frames')
parser.add_argument("--outf", default="Artistic/",
                    help='path to transferred images')
parser.add_argument("--batchSize", type=int, default=1,
                    help='batch size')
parser.add_argument('--loadSize', type=int, default=512,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='crop image size')
parser.add_argument("--layer", default="r41",
                    help='which features to transfer, either r31 or r41')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf, exist_ok=True)
cudnn.benchmark = True

################# DATA #################
content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize, test=True)
content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=False)
style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize, test=True)
style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=False)

################# MODEL #################
if (opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif (opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
matrix = MulLayer(opt.layer)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
styleV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

import time
from ml_base.mobile import model_export
################# GPU  #################
if (opt.cuda):
    vgg.cuda()
    dec.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()

# for ci, (content, contentName) in enumerate(content_loader):
#     contentName = contentName[0]
#     contentV.resize_(content.size()).copy_(content)
#     for sj, (style, styleName) in enumerate(style_loader):
#         styleName = styleName[0]
#         styleV.resize_(style.size()).copy_(style)
#
#         # forward
#         with torch.no_grad():
#             start_time = time.time()
#             sF = vgg(styleV)
#             cF = vgg(contentV)
#             feature, transmatrix = matrix(cF, sF)
#             transfer = dec(feature)
#
#         transfer = transfer.clamp(0, 1)
#         print('content size = ', contentV.size(), 'sytle size = ', styleV.size())
#         print('耗时', time.time() - start_time)
#         vutils.save_image(transfer, '%s/%s_%s.png' % (opt.outf, contentName, styleName), normalize=True,
#                           scale_each=True, nrow=opt.batchSize)
#         print('Transferred image saved at %s%s_%s.png' % (opt.outf, contentName, styleName))

model_export.exportModule(vgg, 'linear_style_encoder.pt')
model_export.exportModule(matrix, 'linear_style_middle.pt')
model_export.exportModule(dec, 'linear_style_decoder.pt')