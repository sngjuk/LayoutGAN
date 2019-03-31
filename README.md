# LayoutGAN
LayoutGAN implementation, (under construction)
<br><br>
README
- my implementation of max pooling was totally wrong.
- the self-attention should be checked if its back-prop gradient flows well.
- the hidden layer size was my arbitrary choice.
- check it out. [PointNet](https://www.youtube.com/watch?v=Cge-hot0Oc0)
TODO <br>
- Logic bugfix on random z generation on training. <br>
- loss.backward and optimizer.step should be added to genenrator traininig.
- fix number in datasets.py : np.float32(2*id +1)/48 -> np.float32(2*id +1)/56
- Wireframe method implementation. <br>
- Network optimization. <br>
- Test with Clipart, Tangram dataset.<br>
