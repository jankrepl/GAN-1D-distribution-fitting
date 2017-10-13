# General Description
Approximating 1-D distributions through Generative Adversarial Networks.


## Code

### Running
```
python Main.py
```

### Dependencies
*  numpy
*  scipy
*  tensorflow
---
## Description
If our noise follows a uniform distribution on (0,1) we can think of the discriminators
network as an approximator of the inverse cdf.
Let U ~ Unifrom(0,1) and F be increasing cdf, then it holds F^{-1}(U) ~ F. Note that it can also learn a different
mapping than inverse cdf.


## Visualizations
Evolution of G approximator, density (histogram) and decision boundary where the true distribution is N(0,1)
* G (approximation of inverse cdf of standard normal)

![alt text](https://github.com/jankrepl/GAN-1D-distribution-fitting/blob/master/pics/G.gif "G")

* Decision boundary (probability of discriminator classifying a given point as real (=coming from true distribution))

![alt text](https://github.com/jankrepl/GAN-1D-distribution-fitting/blob/master/pics/db.gif "Decision Boundary")

* Histogram (pdf)

![alt text](https://github.com/jankrepl/GAN-1D-distribution-fitting/blob/master/pics/hist.gif "Evolution of G")

## Conclusion and results
GANs are really sensitive to input parameters and the convergence is not
guaranteed.

**Problems**
* convergence not guaranteed
* mode collapse (generator learns to fool the discriminator without approximating the true distirbution)

**ToDos**
* Minibatch discrimination (discriminator classifies batches rather than single numbers)


# Sources

[https://github.com/ericjang/genadv_tutorial]()

[https://github.com/emsansone/GAN]()

[http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/]()
