# Algorithm Roadmap

Legend: ✅ implemented · 🟡 in progress · ⬜ planned

## Supervised learning

### Linear / generalised linear models
- ✅ Linear Regression (`supervised/linear_regression`)
- ✅ Logistic Regression (binary + multinomial) (`supervised/logistic_regression`)
- ✅ Ridge Regression (`supervised/ridge`)
- ✅ Lasso Regression — coordinate descent (`supervised/lasso`)
- ✅ Softmax Classifier (`supervised/softmax_classifier`)
- ✅ Perceptron (`supervised/perceptron`)
- ✅ Linear Discriminant Analysis (`supervised/lda`)

### Instance-based
- ✅ K-Nearest Neighbours — classification + regression; uniform & distance weights (`supervised/knn`)
- ⬜ KD-tree / Ball-tree acceleration for KNN

### Tree-based
- ✅ Decision Tree — CART (`supervised/decision_tree`)
- ✅ Random Forest (`supervised/random_forest`)
- ✅ Gradient Boosted Decision Trees (`supervised/gradient_boosting`)

### Probabilistic
- ✅ Gaussian Naive Bayes (`supervised/gaussian_nb`)
- ✅ Multinomial Naive Bayes (`supervised/multinomial_nb`)
- ✅ Bayesian Linear Regression (`supervised/bayesian_linear_regression`)

### Kernel methods
- ✅ Support Vector Machine — SMO SVC / Kernel-Ridge-style SVR (`supervised/svm`)
- ✅ Kernel Ridge Regression (`supervised/kernel_ridge`)

### Neural networks
- ✅ Perceptron (`supervised/perceptron`)
- ✅ Softmax Classifier (`supervised/softmax_classifier`)
- ✅ Multi-Layer Perceptron (`supervised/mlp`)
- ✅ Convolutional Neural Network — toy (`supervised/cnn`)
- ✅ Vanilla RNN — toy (`supervised/rnn`)
- ✅ LSTM / GRU (`supervised/lstm`)
- ✅ Autoencoder (`unsupervised/autoencoder`)
- ✅ Denoising Autoencoder (`unsupervised/denoising_autoencoder`)
- ✅ Variational Autoencoder (`unsupervised/vae`)
- ✅ Convolutional Autoencoder (`unsupervised/conv_autoencoder`)

## Unsupervised learning

### Clustering
- ✅ K-Means — Lloyd + k-means++ default (`unsupervised/kmeans`)
- ✅ Mini-batch K-Means (`unsupervised/minibatch_kmeans`)
- ✅ DBSCAN (`unsupervised/dbscan`)
- ✅ Gaussian Mixture Model — EM (`unsupervised/gmm`)
- ✅ Hierarchical / Agglomerative (`unsupervised/hierarchical`)

### Dimensionality reduction
- ✅ PCA — SVD-based (`unsupervised/pca`)
- ✅ Kernel PCA (`unsupervised/kernel_pca`)
- ✅ t-SNE (`unsupervised/tsne`)
- ✅ UMAP — simplified didactic (`unsupervised/umap`)

### Density estimation
- ✅ Kernel Density Estimation (`unsupervised/kde`)

## Preprocessing
- ✅ StandardScaler (`preprocessing/standard_scaler`; util retained in `utils`)
- ✅ MinMaxScaler (`preprocessing/minmax_scaler`)
- ✅ One-Hot Encoder (`preprocessing/one_hot_encoder`)
- ✅ Polynomial Features (`preprocessing/polynomial_features`)

## Metrics
- ✅ Curated metric module (`algo_stack/metrics`; core scores still in `utils`)

## Shared primitives
- ✅ SGD / Momentum / Adam (`utils/optim.py`)
- ✅ Linear / RBF / Polynomial kernels (`utils/kernels.py`)
- ✅ Activations (`utils/activations.py`)
- ⬜ Line search / L-BFGS-style optimiser
- ⬜ KD-tree / Ball-tree neighbour index

---

When adding an item, move it to ✅ and link to its module in this file *and*
in the top-level `README.md` table.
