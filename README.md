# LayoutGAN
LayoutGAN implementation, (under construction)
<br><br>
<b>README :</b> Failed. <br>
- my implementation of max pooling is totally wrong.<br>
- the self-attention(relation_module) should be checked if its back-prop gradient flows well.<br>
- the hidden layer size is my arbitrary choice.<br>
- check it out for implementation reference. [PointNet](https://www.youtube.com/watch?v=Cge-hot0Oc0)<br>
<br>
TODO <br>
- Logic bugfix on random z generation on training. <br>
- loss.backward and optimizer.step should be added to genenrator traininig.
- fix number in datasets.py : np.float32(2*id +1)/48 -> np.float32(2*id +1)/56
- Wireframe method implementation. <br>
- Network optimization. <br>
- Test with Clipart, Tangram dataset.<br>
