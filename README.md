# Water Detect Sensitivity Test

## Synopsis

Mapping water over large areas and multiple periods can be time-consuming if using supervised classifications; hence unsupervised methods that automatically map water features are preferred. If the rivers’ spatial characteristics change markedly over time, e.g., intermittent rivers, the frequency of cloud-free imagery is also important to capture these changes precisely as possible. Moreover, the image spatial resolution determines the size of the rivers that can be analyzed. Cordeiro et al. (2021) proposed an unsupervised automatic algorithm based on multidimensional agglomerative clustering and a naive Bayesian classifier to quickly identify open water over large multispectral satellite datasets. 

a decision-making algorithm to test all possible combinations of input parameters and assess accuracy, determining the most accurate inputs for the water detect algorithm, given a specific image type, is provided


In addition to choosing the main water spectral indices and the combination of bands, two other parameters can significantly influence the results of the Water Detect algorithm: the maximum clustering and the regularization of the normalized spectral indices. 

For that, we developed a decision-making algorithm to test all possible combinations within a specified range of input parameters (i.e., spectral indices, maximum clustering, and regularization) and assess accuracy, determining the most accurate inputs for the Water Detect for each specific case, and producing a most-to-less accuracy ranking. 

The entry data for our brute force sensitivity algorithm are the range of each parameter given by lowest, highest and step values, the ground truth raster to be used in the accuracy assessment, and the images to be classified.

We then executed the Water Detect using each parameter combination for the considered images 

## How to Cite
Tayer et al 2022......

## Tutorial


## Supported Formats


## Dependencies

```


```

## References
Cordeiro, M. C. R.; Martinez, J.-M.; Peña-Luque, S. Automatic Water Detection from Multidimensional Hierarchical Clustering for Sentinel-2 Images and a Comparison with Level 2A Processors. Remote Sensing of Environment 2021, 253, 112209. https://doi.org/10.1016/j.rse.2020.112209.
