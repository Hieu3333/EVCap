import pickle
with open('ext_data/caption_ext_memory.pkl','rb') as f:
  data = pickle.load(f)
  img_features = data["image_features"]
  captions = data["captions"]
  print('loaded data')
  
  print(img_features.shape)
  print(img_features[:5])
  print(len(captions))
  print(captions[:5])
