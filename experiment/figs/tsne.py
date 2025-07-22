import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = mpimg.imread('tsne_slices.png')
img2 = mpimg.imread('tsne_ns_slicer_like.png')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

axes[0].imshow(img1)
axes[0].axis('off')

axes[1].imshow(img2)
axes[1].axis('off')

plt.tight_layout()
plt.savefig('tsne_comparison_vertical.png', dpi=300)
plt.close()
