This package contains an implementation of the experimental results in the following two papers:

* \[1\] Yoshua Bengio, Eric Thibodeau-Laufer, Jason
  Yosinski. [Deep Generative Stochastic Networks Trainable by Backprop](http://arxiv.org/abs/1306.1091). _arXiv
  preprint arXiv:1306.1091._ ([PDF](http://arxiv.org/pdf/1306.1091v3),
  [BibTeX](https://raw.github.com/yaoli/GSN/master/doc/gsn.bib))

* \[2\] Yoshua Bengio, Li Yao, Guillaume Alain, Pascal
  Vincent. [Generalized Denoising Auto-Encoders as Generative Models](http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models). _NIPS,
  2013._ ([PDF](http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips26/491.pdf),
  [BibTeX](https://raw.github.com/yaoli/GSN/master/doc/dae.bib))


Setup
---------------------

#### Install TensorFlow
Information for setting up TensorFlow can be found in the [official documentation](https://www.tensorflow.org/versions/r0.7/get_started/index.html).  
The implementation has been tested with version r0.7 of TensorFlow running on Python 2.7.

Running the Experiments
---------------------

1. To run a one layer Generalized Denoising Autoencoder with a walkback procedure (paper \[2\])

        python dae_walkback.py

2. To run a one layer Generalized Denoising Autoencoder without a walkback procedure (paper \[2\])

        python dae_no_walkback.py

3. To run a Generative Stochastic Network (paper \[1\])

        python gsn.py
