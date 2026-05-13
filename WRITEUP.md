2.4 vit_pooling
The class token compresses a lot of information into a smaller latent dimensional space. It is therefore ideal for global classificaion, but not for tasks like counting, OCR, or visual question answering. For these more complex tasks, we would ideally assimilate all of the information. The vision transformer will likely store various information about the image in different tokens, so an attention-pooling head would best handle this.

2.4 vit_patch_size

(1)
N = (img_size // P)^2 = (224 / P)^2

P =  8 --> N = 784
P = 16 --> N = 196
P = 32 --> N =  49

The attention compute cost is O(N^2 d_model), which is
O(d_model / P^4), so if you halve P, the compute cost goes
up by a factor of 16.

(2)
On my MacBook M2 Max on MPS,
Patch size 8: 0.11446342468261719 ± 0.013505539328942637 seconds
Patch size 16: 0.022238004207611083 ± 0.0036221620536692274 seconds
Patch size 32: 0.01563140153884888 ± 0.00030120027145953823 seconds

(3)
We might accept this tradeoff if our desired downstream tasks rely on high fidelity spatial detail in our representation and we do not need particularly fast inference.

3.2 infonce

The loss is symmetric because the image-text matching problem has two directions: given some image, which is the best-matching caption and given some caption, which is the best-matching image?