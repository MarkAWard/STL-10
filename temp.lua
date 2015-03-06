--randperm does not support cuda
torch.setdefaulttensortype('torch.FloatTensor')
shuffle = torch.randperm(trainData:size())
shuffle=shuffle:cuda()
torch.setdefaulttensortype('torch.CudaTensor')
-- end of manuaver



 for t = 1,100,10 do
   print(t)
end

trainData.data[{{}}]

t, math.min(t+conf.batchSize-1, trsize)


shuffle=shuffle:float()
trainData.labels=trainData.labels:float()



targets=trainData.labels[{shuffle}]

   
   