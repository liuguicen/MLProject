modelName = 'styleganinv_ffhq256'

encoder_path = path.join(common_dataset.dataset_dir, r'预训练模型\InDomainGanInversion\styleganinv_ffhq256_encoder.pth')

generator_path = path.join(common_dataset.dataset_dir, r'预训练模型\InDomainGanInversion\styleganinv_ffhq256_generator.pth')

vgg_path = path.join(common_dataset.dataset_dir, r'预训练模型\InDomainGanInversion\vgg16.pth')

model_name = 'styleganinv_ffhq256'

invert_list = 'dst/image.list'
