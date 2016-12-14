local LFWcvae = {}
-- LFWcvae12

function LFWcvae.create(opts)
  local encoder = LFWcvae.create_encoder(opts)
  local decoder = LFWcvae.create_decoder(opts)
  return encoder, decoder
end

function LFWcvae.create_encoder(opts)
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

  encoderX:add(nn.LinearMix3(1024, opts.zdim))

   local encoderY = nn.Sequential()
  -- 64 x 64 --> 32 x 32
  encoderY:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 2, 2))
  encoderY:add(cudnn.ReLU())
  encoderY:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 32 x 32 --> 16 x 16
  encoderY:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
  encoderY:add(nn.SpatialBatchNormalization(128))
  encoderY:add(cudnn.ReLU())
  encoderY:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 16 x 16 --> 8 x 8
  encoderY:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  encoderY:add(nn.SpatialBatchNormalization(256))
  encoderY:add(cudnn.ReLU())
  encoderY:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  
  -- 8 x 8 --> 4 x 4
  encoderY:add(cudnn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1))
  encoderY:add(nn.SpatialBatchNormalization(256))
  encoderY:add(cudnn.ReLU())

  -- 4 x 4 --> 1 x 1
  encoderY:add(cudnn.SpatialConvolution(256, 1024, 4, 4, 1, 1, 0, 0))
  encoderY:add(cudnn.ReLU())

  -- fc 1024 --> fc 1024
  encoderY:add(nn.Reshape(1024))
  encoderY:add(nn.Linear(1024, opts.ydim))
  --encoderY:add(cudnn.ReLU())
  encoderY:add(nn.Dropout(0.5))

  local encoder = nn.Sequential()
  encoder:add(nn.ParallelTable():add(encoderX):add(encoderY))  -- z, y

  return encoder
end


function LFWcvae.create_decoder(opts)

  -- input is y
  local decoder = nn.Sequential()

  decoder:add(nn.Linear(256, 256*8*8))
  --decoder:add(nn.BatchNormalization(256*8*8))
  decoder:add(cudnn.ReLU())
  decoder:add(nn.Reshape(256, 8, 8))
  
  -- 8 x 8 --> 8 x 8
  decoder:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
  decoder:add(nn.SpatialBatchNormalization(256))
  decoder:add(cudnn.ReLU())

  -- 8 x 8 --> 16 x 16
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(256))
  decoder:add(cudnn.ReLU())

  -- 16 x 16 --> 32 x 32
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialFullConvolution(256, 128, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(128))
  decoder:add(cudnn.ReLU())
  
  -- 32 x 32 --> 64 x 64
  decoder:add(nn.SpatialUpSamplingNearest(2))
  decoder:add(nn.SpatialFullConvolution(128, 64, 5, 5, 1, 1, 2, 2))
  decoder:add(nn.SpatialBatchNormalization(64))
  decoder:add(cudnn.ReLU())

  -- 64 x 64 --> 64 x 64
  local cc_layer = nn.ConcatTable()
  local sp_mean_layer = nn.Sequential() 
  sp_mean_layer:add(nn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2)):add(nn.Tanh())
  cc_layer:add(sp_mean_layer)
  cc_layer:add(nn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))
  decoder:add(cc_layer)
  
  return decoder
end

return LFWcvae


