require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'image'
require 'pl'
require 'paths'

require 'layers.LinearWP'
require 'layers.LinearGaussian'
require 'layers.LinearMix'
require 'layers.LinearMix2'
require 'layers.LinearMix3'
require 'layers.Reparametrize'
require 'layers.GaussianCriterion'
require 'layers.KLDCriterion'
require 'utils.scaled_lfw'
optim_utils = require 'utils.adam_v2'

-- parse command-line options
opts = lapp[[
  --saveFreq      (default 20)        save every saveFreq epochs
  --modelString   (default 'arch_stacked_VAE')        reload pretrained network
  -p,--plot                         plot while training
  -t,--threads    (default 4)         number of threads
  -g,--gpu        (default 0)        gpu to run on (default cpu)
  --scale         (default 64)        scale of images to train on
  -z,--zdim       (default 256)
  -y,--ydim       (default 3)
  --maxEpoch      (default 120)
  --batchSize     (default 32)
  --weightDecay   (default 0.004)
  --adam          (default 1)         lr mode
  --modelDir      (default 'models/')
  --attr_ix       (default 1)
]]

opts.modelName = string.format('LFW_%s_adam%d_bs%d_zdim%d_wd%g_stacked_ydim3', 
  opts.modelString, opts.adam, opts.batchSize, opts.zdim, opts.weightDecay)

attr_ixs = {1,9,16,17,18}  -- male, senior, wear sunglasses, mustache, smiling
opts.ydim = #attr_ixs

if opts.gpu < 0 or opts.gpu > 3 then opts.gpu = -1 end
print(opts)
--torch.randomseed(1)

-- threads
torch.setnumthreads(opts.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opts.gpu >= 0 then
  cutorch.setDevice(opts.gpu + 1)
  print('<gpu> using device ' .. opts.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

if opts.modelString == '' then
  error('empty modelString')
else
  print('scripts/' .. opts.modelString .. '.lua')

  lfwcvae_module = dofile('scripts/' .. opts.modelString .. '.lua')
  encoder, decoder = lfwcvae_module.create(opts)
end

-- retrieve parameters and gradients
epoch = 0
opts.modelPath = opts.modelDir .. opts.modelName
print(opts.modelPath)
if not paths.dirp(opts.modelPath) then
  paths.mkdir(opts.modelPath)
end

for i = opts.maxEpoch,1,-opts.saveFreq do
  if paths.filep(opts.modelPath .. string.format('/net-epoch-%d.t7', i)) then
    epoch = i
    loader = torch.load(opts.modelPath .. string.format('/net-epoch-%d.t7', i))
    encoder = loader.encoder
    decoder = loader.decoder
    print(string.format('resuming from epoch %d', i))
    break
  end
end

cvae = nn.Sequential()
enc_sampling = nn.Sequential()
enc_sampling:add(encoder:get(1):get(1)):add(nn.Reparametrize(opts.zdim))

cvae:add(nn.ParallelTable()):add(enc_sampling):add(encoder:get(1):get(2)) -- z, y
enc_sampling2 = nn.Sequential()
enc_sampling2:add(nn.LinearMix2(256, opts.ydim, opts.zdim)):add(nn.Reparametrize(opts.zdim))
cvae:add(enc_sampling2)
cvae:add(decoder)
parameters, gradients = cvae:getParameters()


-- print networks
if opts.gpu >= 0 then
  print('Copying model to gpu')
  encoder:cuda()
  decoder:cuda()
end

-- get/create dataset
ntrain = math.floor(9464 * 0.9)
nval = 9464 - ntrain

print('data preprocessing')
lfw.setScale(opts.scale)

valData = lfw.loadTrainSet(ntrain + 1, ntrain + nval)
--image_utils.normalizeGlobal(valData.data, mean, std)
valData:scaleData()
nval = valData:size()
print(nval)
  -- val set
cvae:evaluate()
-- change the attr value
-- load the attr file
local file = io.open('scripts/attr.txt')
attr = {}
if file then
    for line in file:lines() do
        local ix, name, min, max = unpack(line:split(',')) 
        attr[#attr+1] = torch.Tensor({min,max})
    end
end

attr_ix = opts.attr_ix
local y_change = torch.Tensor({-3, -1, 0, 1, 2})
local z_change = torch.Tensor({-1,-0.5,0,0.5,1})
--gen = torch.Generator()
math.randomseed(os.time())
to_plot = {}
print(valData:size())
for crr=1, 5 do
-- original image
  print('Generating images')
  --local current_z = torch.Tensor(1, opts.zdim):normal(0,1)
  idx = math.random(nval)
  --local idx = 1
  local cur_im = torch.Tensor(1,3,64,64)
  cur_im[1] = valData[idx][1]:clone()
  local cur_attr = torch.Tensor(1,opts.ydim)
  --cur_attr[1] = valData[idx][3]:clone()

  cur_im[1]:mul(2):add(-1)
  local res = cur_im[1]:clone()
  res = torch.squeeze(res)
  res:add(1):mul(0.5)
  to_plot[#to_plot+1] = res:clone()  

  -- output using same attr:
  local out1 = enc_sampling:forward(cur_im)
  cur_attr[1] = encoder:get(1):get(2):forward(cur_im)
  --print(out1:size())
 -- print(cur_attr:size())
  local outz = enc_sampling2:forward({out1, cur_attr})
  local f = decoder:forward(outz) 
  local res = f[1][1]:clone()
  res = torch.squeeze(res)
  res:add(1):mul(0.5) 
  to_plot[#to_plot+1] = res:clone()  
 --print(cur_attr[1][16])
  -- output changing the value of y
  for i = 1, z_change:size()[1] do
    local temp_z = outz:clone()
    temp_attr[1][attr_ix] = y_change[i]
    local outz = enc_sampling2:forward({out1, temp_attr})
    local f = decoder:forward(outz) 
    local res = f[1][1]:clone()
    res = torch.squeeze(res)
    res:add(1):mul(0.5) 
    to_plot[#to_plot+1] = res:clone()  -- generaged image
  end
end
print(#to_plot)

local formatted = image.toDisplayTensor({input=to_plot, nrow=7})
formatted = formatted:double()
formatted:mul(255)
formatted = formatted:byte()
print(formatted:size())
image.save(opts.modelPath .. string.format('/sample-stacked-attr-%d.jpg', attr_ix), formatted)


collectgarbage()


