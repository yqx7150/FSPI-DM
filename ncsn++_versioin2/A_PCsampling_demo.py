import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
##################################################################
import A_sampling as sampling
from A_sampling import ReverseDiffusionPredictor,LangevinCorrector,AnnealedLangevinDynamics ,EulerMaruyamaPredictor,AncestralSamplingPredictor
import aapm_sin_ncsnpp_gb as configs  # 修改config
##################################################################

sys.path.append('..')
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np

from utils import restore_checkpoint

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from sde_lib import VESDE, VPSDE, subVPSDE
import os.path as osp
if len(sys.argv) > 1:
  start = int(sys.argv[1])
  end = int(sys.argv[2])

checkpoint_num = [23]

def get_predict(num):
  if num == 0:
    return None
  elif num == 1:
    return EulerMaruyamaPredictor
  elif num == 2:
    return ReverseDiffusionPredictor

def get_correct(num):
  if num == 0:
    return None
  elif num == 1:
    return LangevinCorrector
  elif num == 2:
    return AnnealedLangevinDynamics

predicts = [2]
corrects = [1]
for predict in predicts:
  for correct in corrects:
    for check_num in checkpoint_num:
      sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
      if sde.lower() == 'vesde':
        ckpt_filename = "/home/b110/LX/dog/1w2狗check/checkpoint_27.pth"      # path of checkpoint
        config = configs.get_config()  
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5


      batch_size = 1 #@param {"type":"integer"}
      config.training.batch_size = batch_size
      config.eval.batch_size = batch_size

      random_seed = 0 #@param {"type": "integer"}

      sigmas = mutils.get_sigmas(config)
      score_model = mutils.create_model(config)

      optimizer = get_optimizer(config, score_model.parameters())
      ema = ExponentialMovingAverage(score_model.parameters(),
                                    decay=config.model.ema_rate)
      state = dict(step=0, optimizer=optimizer,
                  model=score_model, ema=ema)

      state = restore_checkpoint(ckpt_filename, state, config.device)
      ema.copy_to(score_model.parameters())

      #@title PC sampling
      img_size = config.data.image_size
      channels = config.data.num_channels
      shape = (batch_size, channels, img_size, img_size)
      # predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      # corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
      predictor = get_predict(predict) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
      corrector = get_correct(correct) #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}

      snr = 0.075#0.16 #@param {"type": "number"}
      n_steps =  1#@param {"type": "integer"}
      probability_flow = False #@param {"type": "boolean"}
      sampling_fn = sampling.get_pc_sampler(sde, predictor, corrector,
                                            None, snr, n_steps=n_steps,
                                            probability_flow=probability_flow,
                                            continuous=config.training.continuous,
                                            eps=sampling_eps, device=config.device)

      sampling_fn(score_model)

