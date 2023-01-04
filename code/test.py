import pickle
import matplotlib.pyplot as plt

f = open("../models/bird_AttnGAN2.pth", 'rb')
u = pickle._Unpickler(f)
model = u.load()
print(model)
cap = "this bird has wings that are black and has a white belly"
genimg = model.predict(cap)
plt.imsave('output.png', genimg)

