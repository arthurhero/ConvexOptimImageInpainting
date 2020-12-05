# ConvexOptimImageInpainting
Image inpainting using matrix completion and compressive sensing in fourier domain.

## Files
*main.py*
:  driver code for running experiments

*image_process.py*
:  methods for reading, saving, displaying images, and creating masks

*co_utils.py*
:  algorithms shared by various convex optimization algorithms, e.g., fista

*fourier.py*
:  compressive sensing algorithm for image inpainting using DCT

*matrix_completion.py*
:  matrix completion algorithm for image inpainting

## Demo
Image 4 with p:0.1, nh:2 (masked, mc, fo)
<img src="demo/3_0_0masked.jpg" alt="3_0_0masked" width="200" height="200" /> <img src="demo/3_0_0mc.jpg" alt="3_0_0mc" width="200" height="200" /> <img src="demo/3_0_0fo.jpg" alt="3_0_0fo" width="200" height="200" />

Image 4 with p:0.9, nh:10000 (masked, mc, fo)
<img src="demo/3_1_1masked.jpg" alt="3_1_1masked" width="200" height="200" /> <img src="demo/3_1_1mc.jpg" alt="3_1_1mc" width="200" height="200" /> <img src="demo/3_1_1fo.jpg" alt="3_1_1fo" width="200" height="200" />
