import FileUtil
from RunRecord import RunRecord

content = r'D:\MLProject\GAN\StyleTransfer\Pytorch_AdaIN\content\neko.jpg'
style = r'D:\MLProject\GAN\StyleTransfer\Pytorch_AdaIN\style\antimonocromatismo.jpg'
output_name = r'res'
alpha = 1
model_state_path = r'E:\重要_dataset_model\预训练模型\pure_pyroch_adain_style.pth',

batch_size = 12  #
epoch = 20
gpu = 0
learning_rate = 5e-5
check_point_interval = 400
train_content_dir = r'E:\重要_dataset_model\COCO\train2014'
train_style_dir = r'E:\重要_dataset_model\wikiart\train'
test_content_dir = train_content_dir
test_style_dir = train_style_dir
check_point_dir = 'result'
record = FileUtil.readRunRecord('runRecord.pkl')
if record is None:
    record = RunRecord()
record.createDir()
debugMode = False
use_check_point_state = True
use_mobile_based = False

if debugMode:
    use_check_point_state = False
    check_point_interval = 10