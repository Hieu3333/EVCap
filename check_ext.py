import pickle
with open('ext_data/caption_ext_memory.pkl','rb') as f:
  img_feature = pickle.load(f)
  print('loaded data')
  
  print(img_feature.shape)
  print(img_feature[:5])
