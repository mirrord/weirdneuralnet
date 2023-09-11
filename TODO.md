# TODO
### Refactoring
 * parameterize priming with number of far points [UNTESTED]
 * test & trial infrastructures (high priority)
 * change timing tests to use perf_counter
 * generalize time trial code
 * add in type hints
 * review for speedups
 * add more unit tests
 * reorganize speed tests for better reproduceability
 * add numpy fallback for cupy operations
 * Cython-ize code for speed

### New Features
 * implement more dataset usage (high priority)
 * implement subset selection methods
   * random pick [DONE]
   * core subset
   * LTS - maybe
 * implement model structure "DNA" encoding
 * impl transfer training
 * impl network compilation [INCOMPLETE]
   * ...actually, this may not be reasonable
 * clean up CLI
 * implement more initialization types

### New Network Features
 * implement convolutions
   * implement max/min/avg-pooling
 * implement batch normalization
 * implement auto-regressive connections/training
 * feature parity with Transformer
   * self-attention
   * vector append
 * feature parity with diffusion models
   * diffusion operation (gaussian noise chain)
 * transfer learning QOL functions
 * autoencoder QOL functions
 * reinforcement learning
 * unsupervised learning
 * recurrence
   * unrolled backprop

### Experiments
 * write up experimental procedure and results [DONE]
   * more work needed: images and more experiments
 * fix pretraining experiment functions [DONE]
 * re-run experiments using convergence target
   * create graphics for experimental accuracy results
   * compare to other subset selection methods

### Documentation
 * document existing code [REDO]
 * document how to use WNN class
 * document how to subclass Nodes