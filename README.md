# Image-Classification-using-CNN

Convolutional neural networks (CNN) is a special architecture of artificial neural networks, proposed by Yann LeCun in 1988. CNN uses some features of the visual cortex. One of the most popular uses of this architecture is image classification.

The main task of image classification is acceptance of the input image and the following definition of its class. This is a skill that people learn from their birth and are able to easily determine that the image in the picture is an elephant. 

Instead of the image, the computer sees an array of pixels. For example, if image size is 300 x 300. In this case, the size of the array will be 300x300x3(for colored image). Where 300 is width, next 300 is height and 3 is RGB channel values. The computer is assigned a value from 0 to 255 to each of these numbers. Тhis value describes the intensity of the pixel at each point.

It includes various steps:

Step 1 – Convolution Operation

The three elements that enter into the convolution operations are Input image,Feature detector and Feature map.

How exactly does the Convolution Operation work?
You can think of the feature detector as a window consisting of 9 (3×3) cells. Here is what you do with it:

-You place it over the input image beginning from the top-left corner within the borders you see demarcated above, and then you count the number of cells in which the feature detector matches the input image.

-The number of matching cells is then inserted in the top-left cell of the feature map.

-You then move the feature detector one cell to the right and do the same thing. This movement is called a and since we are moving the feature detector one cell at time, that would be called a stride of one pixel.

-What you will find in this example is that the feature detector's middle-left cell with the number 1 inside it matches the cell that it is standing over inside the input image. That's the only matching cell, and so you write “1” in the next cell in the feature map, and so on and so forth.

-After you have gone through the whole first row, you can then move it over to the next row and go through the same process.

Step 2 – Max Pooling

The purpose of max pooling is enabling the convolutional neural network to detect the object's image when presented with the image in any manner.

This time you'll place a 2×2 box at the top-left corner, and move along the row. For every 4 cells your box stands on, you'll find the maximum numerical value and insert it into the pooled feature map.

Step 3: Flattening

After finishing the previous two steps, we're supposed to have a pooled feature map by now. As the name of this step implies, we are literally going to flatten our pooled feature map into a single column.

Step 4: Full Connection

As written above, the input layer contains the vector of data that was created in the flattening step. The features that we distilled throughout the previous steps are encoded in this vector.
