# General Description
Approximating 1-D distributions through Generative Adversarial Networks.


# Description
If our noise follows a uniform distribution on (0,1) we can think of the discriminators
network as an approximator of the inverse cdf.
Let U ~ Unifrom(0,1) and F be increasing cdf, then it holds F^{-1}(U) ~ F.



# Parameters

# Visualizations
Evolution of G approximator, G density (histogram) and decision boundary


# Conclusion and results
GANs are really sensitive to input parameters and the convergence is not
guaranteed.

# Sources

[https://github.com/ericjang/genadv_tutorial]()

[https://github.com/emsansone/GAN]()

[http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/]()
