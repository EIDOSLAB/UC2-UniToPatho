from torch.autograd import Variable
import torch.onnx
import torchvision
import gdown
import os

num_classes = 6
train = True
## Resnet18

# Export model to onnx format
dummy_input = Variable(torch.randn(1, 3, 224, 224))
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, num_classes),torch.nn.Softmax(dim=1))
model.train(mode=train)
torch.onnx.export( model , dummy_input , "onnx_models/resnet18_6c_pretrained.onnx" , export_params=True, training = torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL , opset_version=12, do_constant_folding=False)

# Export model to onnx format
dummy_input = Variable(torch.randn(1, 3, 224, 224))
model = torchvision.models.resnet18(pretrained=False, num_classes = num_classes)
model.fc = torch.nn.Sequential(model.fc,torch.nn.Softmax(dim=1))
model.train(mode=train)
torch.onnx.export(model, dummy_input, "onnx_models/resnet18_6c.onnx", export_params=True, training = torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL , opset_version=12, do_constant_folding=False)

## Resnet18

# Export model to onnx format
dummy_input = Variable(torch.randn(1, 3, 224, 224))
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, num_classes),torch.nn.Softmax(dim=1))
model.train(mode=train)
torch.onnx.export( model , dummy_input , "onnx_models/resnet50_6c_pretrained.onnx" , export_params=True, training = torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL , opset_version=12, do_constant_folding=False)

# Export model to onnx format
dummy_input = Variable(torch.randn(1, 3, 224, 224))
model = torchvision.models.resnet50(pretrained=False, num_classes = num_classes)
model.fc = torch.nn.Sequential(model.fc,torch.nn.Softmax(dim=1))
model.train(mode=train)
torch.onnx.export(model, dummy_input, "onnx_models/resnet50_6c.onnx", export_params=True, training = torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL , opset_version=12, do_constant_folding=False)

# Export model used to UniToPatho inference to onnx format
def onnx_icip_models():
    dummy_input = Variable(torch.randn(1, 3, 224, 224))
    path_mean = [0.8153967261314392, 0.7560872435569763, 0.7853971719741821]
    path_std = [0.0768752247095108, 0.13408100605010986, 0.06851005554199219]
    im_mean, im_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    models = {

        'adenoma_classifier.3c.0': {
            'id': '12nJcsw4upHodfgd6-pvJovb6Vb6mDPEb',
            'n': 3,
            'size': 7000,
            'subsample': 224, #7000_224
            'mean': path_mean,
            'std': path_std,
            'sftmax': True
        },

        'adenoma_classifier.3c.1': {
            'id': '19Of6mITftUc12XNfAR-7tmuL9873aqZp',
            'n': 3,
            'size': 7000,
            'subsample': 224, #7000_224
            'mean': path_mean,
            'std': path_std,
            'sftmax': True
        },

        'adenoma_classifier.3c.2': {
            'id': '1j8YKUkGCo7uzBbfRKb9a_xhB2Gys1lkz',
            'n': 3,
            'size': 7000,
            'subsample': 224, #7000_224
            'mean': path_mean,
            'std': path_std,
            'sftmax': True
        },

        'hp_classifier.800.0': {
            'id': '1-nAIGfc_ExEACa93ZLPJsIGNRW3-VDJt',
            'n': 6,
            'size': 800,
            'subsample': -1,
            'mean': im_mean,
            'std': im_std,
            'sftmax': True
        },

        'hp_classifier.800.1': {
            'id': '1gOT3LnhgQQssPuVQH7renm3TYmUDTy1I',
            'n': 6,
            'size': 800,
            'subsample': -1,
            'mean': im_mean,
            'std': im_std,
            'sftmax': True
        },

        'hp_classifier.800.2': {
            'id': '1-c8eWYYg5n_zHzLKIlEiXN6CimB6BB3y',
            'n': 6,
            'size': 800,
            'subsample': -1,
            'mean': im_mean,
            'std': im_std,
            'sftmax': True
        },

        'grade_classifier.800.0': {
            'id': '1H5qM_PLfg52P32WBev1zG_5x6M6QHUaV',
            'n': 2,
            'size': 800,
            'subsample': -1,
            'mean': im_mean,
            'std': im_std,
            'sftmax': True
        },

        'grade_classifier.800.1': {
            'id': '1nQ8ogZhJMo7g8oBsfFQBNM4EeHiW-kur',
            'n': 2,
            'size': 800,
            'subsample': -1,
            'mean': im_mean,
            'std': im_std,
            'sftmax': True
        },

        'grade_classifier.800.2': {
            'id': '11dexNFf4NelfsRQDgNKtE_dud485EDgi',
            'n': 2,
            'size': 800,
            'subsample': -1,
            'mean': im_mean,
            'std': im_std,
            'sftmax': True
        },

    }

    os.makedirs('tmp_models',exist_ok=True)
    for name in models.keys():

        print(f'=> Loading {name}')
        model = torchvision.models.resnet18(num_classes=models[name]['n'])

        
        if not os.path.isfile(f'tmp_models/{name}.pt'):
            url = f"https://drive.google.com/uc?id={models[name]['id']}"
            output = os.path.join('tmp_models', f'{name}.pt')
            print(f'=> Downloading {name} from {url}..')
            gdown.download(url, output, quiet=False)

        checkpoint = torch.load(f'tmp_models/{name}.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        if models[name]['sftmax']:
            model.fc = torch.nn.Sequential(model.fc,torch.nn.Softmax(dim=1))

        
        if models[name]['subsample'] == -1:
             size = int(models[name]['size']/0.4415)
             dummy_in = Variable(torch.randn(1, 3, size, size))
             
        else:
             dummy_in = Variable(torch.randn(1, 3, models[name]['subsample'], models[name]['subsample']))

        
        model.train(False)
        torch.onnx.export( model , dummy_in , f"onnx_models/{name}.onnx" , training = torch.onnx.TrainingMode.EVAL , export_params=True, opset_version=12, do_constant_folding=False)
    
onnx_icip_models()

