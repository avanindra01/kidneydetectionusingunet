
## Unet for kidney detection-semantic segmentation


# Goal

The goal of this project was take expert labeled images of kidney glomeruli and generate a semantic segmentation strategy to label pixels belonging to them.

# Approach

U-nets https://arxiv.org/abs/1505.04597 are a powerful tool for this kind of biological segmentation. Combining the scalability of convolutional neural networks with deconvolutional layers enables for robust, accurate pixel to pixel segmentation.

![u-net-architecture](https://user-images.githubusercontent.com/3740610/110843422-20469c80-825d-11eb-865b-8027fe1fe15c.png)

Unet code and generators largely based on: https://github.com/zhixuhao/unet

# What I changed from the paper

The paper uses binary cross entropy as its metric and a weight matrix that highly exaggerates space between objects to force object segmentation (note the above code does not do that). As the goal of this project is just to capture the majority of pixels correctly, I used a different strategy- the dice coefficient. This works by taking the union of the test mask and the predicted mask multiplying it by 2 and dividing by the sum of the masks. For instance if you had 10 pixels in each and 8 overlapped your dice score would be 2 x 8 div 20 or 0.8 accuracy. I also modified the above network to take 3 color images.

# Gotchas 

The objects were quite big vs. 256 pixels so I shrunk the images by half this allowed for more accurate mask identification as the masks are more findable in situ. I also tested 1/4 shrunk images but found them to end with lower accuracies.

Many of the images were of wildly intensities so I added a normalization step to my generator. Conceptually it probably would make more sense to normalize the whole images, but that would have required a lot of finagling to realign the images to 0-255 so I took the easy way out.
![u-net-architecture](https://user-images.githubusercontent.com/3740610/110843422-20469c80-825d-11eb-865b-8027fe1fe15c.png)

# Outcome

On my validation set I was hitting 75%. There are a few reasons for this; the first that accounts for probably about 5% of the issue is that my mask is more accurate. This makes sense as humans are going to be somewhat limited in their ability to label images. The other, perhaps larger issue is that masks are often clipped on the edges; this makes it so that I often am not seeing the full glomeruli in situ. This could possibly be fixed by using some kind of sliding window and consensus system.

![u-net-architecture](https://user-images.githubusercontent.com/3740610/110843422-20469c80-825d-11eb-865b-8027fe1fe15c.png)

