import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import models
import utils
from torchvision.utils import save_image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, save_path=None):
    model.eval()
    
    data_norm = {
        'inp': {'sub': [0.485, 0.456, 0.406], 'div': [0.229, 0.224, 0.225]},
        'gt': {'sub': [0], 'div': [1]}
    }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'kvasir':
        metric_fn = utils.calc_kvasir
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    if save_path is not None:
        path_img = os.path.join(save_path, 'image')
        path_gt = os.path.join(save_path, 'gt')
        path_pred = os.path.join(save_path, 'pred')
        os.makedirs(path_img, exist_ok=True)
        os.makedirs(path_gt, exist_ok=True)
        os.makedirs(path_pred, exist_ok=True)

    cnt = 0
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        inp = batch['inp']
        gt = batch['gt']

        with torch.no_grad():
            pred_logits = model.infer(inp)
            pred_prob = torch.sigmoid(pred_logits)

        result1, result2, result3, result4 = metric_fn(pred_prob, gt)
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        try:
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])
        except:
            pass

        if save_path is not None:
            batch_size = inp.shape[0]
            
            sub = torch.tensor(data_norm['inp']['sub']).cuda().view(-1, 1, 1)
            div = torch.tensor(data_norm['inp']['div']).cuda().view(-1, 1, 1)

            for i in range(batch_size):
                if 'name' in batch:
                    name = batch['name'][i]
                    name = os.path.splitext(name)[0]
                else:
                    name = f"{cnt}_{i}"
                
                target_h, target_w = inp.shape[2], inp.shape[3]
                
                if 'shape' in batch:
                     if isinstance(batch['shape'], list):
                         target_h = int(batch['shape'][0][i])
                         target_w = int(batch['shape'][1][i])
                     else:
                         target_h = int(batch['shape'][i][0])
                         target_w = int(batch['shape'][i][1])

                img_vis = inp[i].clone()
                img_vis = img_vis * div + sub
                img_vis = torch.clamp(img_vis, 0, 1)

                pred_vis = pred_prob[i:i+1].clone() # 保持 [1, H, W]
                gt_vis = gt[i:i+1].clone()          # 保持 [1, H, W]

                if target_h != img_vis.shape[1] or target_w != img_vis.shape[2]:
                    img_vis = F.interpolate(img_vis.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)

                    pred_vis = F.interpolate(pred_vis, size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
                    
                    gt_vis = F.interpolate(gt_vis, size=(target_h, target_w), mode='nearest').squeeze(0)
                
                else:
                     pred_vis = pred_vis.squeeze(0)
                     gt_vis = gt_vis.squeeze(0)

                     
                save_image(img_vis, os.path.join(path_img, f"{name}.png"))
                save_image(pred_vis, os.path.join(path_pred, f"{name}.png"))
                save_image(gt_vis, os.path.join(path_gt, f"{name}.png"))

            cnt += 1

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/cod-sam-vit-l.yaml", help="配置文件路径")
    parser.add_argument('--model', required=True, help="训练好的 .pth 模型路径")
    parser.add_argument('--save_path', default=None, help="如果不为空，则将预测结果图片保存到该文件夹")
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, shuffle=False)

    model = models.make(config['model']).cuda()
    
    print(f"Loading checkpoint from {args.model}...")
    checkpoint = torch.load(args.model, map_location='cuda:0')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys (randomly initialised): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys (ignored): {len(unexpected)}")
    print("Model loaded successfully.")

    eval_type = config.get('eval_type')
    metric_names = {
        'f1':        ['f1',      'auc',        'none', 'none'],
        'fmeasure':  ['f_mea',   'mae',         'none', 'none'],
        'ber':       ['shadow',  'non_shadow',   'ber',  'none'],
        'cod':       ['sm',      'em',           'wfm',  'mae'],
        'kvasir':    ['dice',    'iou',          'none', 'none'],
    }
    names = metric_names.get(eval_type, ['metric1', 'metric2', 'metric3', 'metric4'])

    v1, v2, v3, v4 = eval_psnr(loader, model,
                                data_norm=config.get('data_norm'),
                                eval_type=eval_type,
                                eval_bsize=config.get('eval_bsize'),
                                verbose=True,
                                save_path=args.save_path)

    print('################ Results ################')
    print(f'{names[0]}: {v1:.4f}')
    print(f'{names[1]}: {v2:.4f}')
    if names[2] != 'none':
        print(f'{names[2]}: {v3:.4f}')
    if names[3] != 'none':
        print(f'{names[3]}: {v4:.4f}')
    print('#########################################')
