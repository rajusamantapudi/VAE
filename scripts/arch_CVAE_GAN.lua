local LFWaegan = {}
-- LFWaegan12

function LFWaegan.create(opts)
  local encoder = LFWaegan.create_encoder(opts)
  local generator = LFWaegan.create_generator(opts)
  local discriminator = LFWaegan.create_discriminator(opts)
  return encoder, generator, discriminator
end

function LFWaegan.create_encoder(opts)
  local encoderX = nn.Sequential()
  -- 64 x 64 --> 32 x 32
  encoderX:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2))
  encoderX:add(cudnn.ReLU())
  encoderX:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 32 x 32 --> 16 x 16
  encoderX:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
  encoderX:add(nn.SpatialBatchNormalization(128))
  encoderX:add(cudnn.ReLU())
  encoderX:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 16 x 16 --> 8 x 8
  encoderX:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  encoderX:add(nn.SpatialBatchNormalization(256))
  encoderX:add(cudnn.ReLU())
  encoderX:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 8 x 8 --> 4 x 4
  encoderX:add(cudnn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1))
  encoderX:add(nn.SpatialBatchNormalization(256))
  encoderX:add(cudnn.ReLU())

  -- 4 x 4 --> 1 x 1
  encoderX:add(cudnn.SpatialConvolution(256, 1024, 4, 4, 1, 1, 0, 0))
  encoderX:add(cudnn.ReLU())

  -- fc 1024 --> fc 1024
  encoderX:add(nn.Reshape(1024))
  encoderX:add(nn.Linear(1024, 1024))
  encoderX:add(cudnn.ReLU())
  encoderX:add(nn.Dropout(0.5))

  local encoderY = nn.Sequential()
  encoderY:add(nn.Linear(opts.ydim, opts.zdim))
  encoderY:add(cudnn.ReLU())

  local encoder = nn.Sequential()
  encoder:add(nn.ParallelTable():add(encoderX):add(encoderY))
  encoder:add(nn.LinearMix2(1024, opts.zdim, opts.zdim))

  return encoder
end

function LFWaegan.create_generator(opts)

  local generatorY = nn.Sequential()
  generatorY:add(nn.Linear(opts.ydim, opts.zdim*2))
  generatorY:add(cudnn.ReLU())

  local generator = nn.Sequential()
  generator:add(nn.ParallelTable():add(nn.Copy()):add(generatorY))
  generator:add(nn.LinearMix(opts.zdim, opts.zdim*2, 256))
  generator:add(cudnn.ReLU())

  generator:add(nn.Linear(256, 256*8*8))
  --generator:add(nn.BatchNormalization(256*8*8))
  generator:add(cudnn.ReLU())
  generator:add(nn.Reshape(256, 8, 8))
  
  -- 8 x 8 --> 8 x 8
  generator:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
  generator:add(nn.SpatialBatchNormalization(256))
  generator:add(cudnn.ReLU())

  -- 8 x 8 --> 16 x 16
  generator:add(nn.SpatialUpSamplingNearest(2))
  generator:add(nn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2))
  generator:add(nn.SpatialBatchNormalization(256))
  generator:add(cudnn.ReLU())

  -- 16 x 16 --> 32 x 32
  generator:add(nn.SpatialUpSamplingNearest(2))
  generator:add(nn.SpatialFullConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  generator:add(nn.SpatialBatchNormalization(128))
  generator:add(cudnn.ReLU())
  
  -- 32 x 32 --> 64 x 64
  generator:add(nn.SpatialUpSamplingNearest(2))
  generator:add(nn.SpatialFullConvolution(128, 64, 5, 5, 1, 1, 2, 2))
  generator:add(nn.SpatialBatchNormalization(64))
  generator:add(cudnn.ReLU())

  -- 64 x 64 --> 64 x 64
  local cc_layer = nn.ConcatTable()
  local sp_mean_layer = nn.Sequential() 
  sp_mean_layer:add(nn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2)):add(nn.Tanh())
  cc_layer:add(sp_mean_layer)
  cc_layer:add(nn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))
  generator:add(cc_layer)
  
  return generator
end

function LFWaegan.create_discriminator(opts)

  -- got a image , conv + classifier
  -- image size 64*63 *3

  -- local discriminatorY = nn.Sequential()
  -- discriminatorY:add(nn.Linear(opts.ydim, opts.zdim*2))
  -- discriminatorY:add(cudnn.ReLU())

  local discriminator = nn.Sequential()
  discriminator:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2))
  discriminator:add(cudnn.ReLU())
  discriminator:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 32 x 32 --> 16 x 16
  discriminator:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
  discriminator:add(nn.SpatialBatchNormalization(128))
  discriminator:add(cudnn.ReLU())
  discriminator:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 16 x 16 --> 8 x 8
  discriminator:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  discriminator:add(nn.SpatialBatchNormalization(256))
  discriminator:add(cudnn.ReLU())
  discriminator:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 8 x 8 --> 4 x 4
  discriminator:add(cudnn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1))
  discriminator:add(nn.SpatialBatchNormalization(256))
  discriminator:add(cudnn.ReLU())

  -- 4 x 4 --> 1 x 1
  discriminator:add(cudnn.SpatialConvolution(256, 1024, 4, 4, 1, 1, 0, 0))
  discriminator:add(cudnn.ReLU())

  -- fc 1024 --> fc 1024
  discriminator:add(nn.Reshape(1024))

  discriminator:add(nn.View(1024))
  discriminator:add(nn.Linear(1024, 128))
  discriminator:add(nn.PReLU())
  discriminator:add(nn.Dropout(0.5))
  discriminator:add(nn.Linear(128, 1))
  discriminator:add(nn.Sigmoid())

  
  return discriminator
end

return LFWaegan


