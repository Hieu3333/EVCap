import pickle
with open('ext_data/caption_ext_memory.pkl','rb') as f:
  data = pickle.load(f)
  print('loaded data')
  img_feature = data["image_features"]
  captions = data["captions"]
  print(img_feature.shape,len(captions))
  print(img_feature[:5])
  print(captions[:5])