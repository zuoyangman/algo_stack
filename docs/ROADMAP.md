# Algorithm Roadmap

Legend: ✅ implemented · 🟡 in progress · ⬜ planned

## Supervised learning

### Linear / generalised linear models
- ✅ Linear Regression (`supervised/linear_regression`)
- ✅ Logistic Regression (binary + multinomial) (`supervised/logistic_regression`)
- ⬜ Ridge Regression
- ⬜ Lasso Regression (coordinate descent)
- ✅ Softmax Classifier (`supervised/softmax_classifier`)
- ✅ Perceptron (`supervised/perceptron`)
- ⬜ Linear Discriminant Analysis

### Instance-based
- ✅ K-Nearest Neighbours (classification + regression) (`supervised/knn`)
- ⬜ Weighted KNN / KD-tree acceleration

### Tree-based
- ⬜ Decision Tree (CART)
- ⬜ Random Forest
- ⬜ Gradient Boosted Decision Trees

### Probabilistic
- ⬜ Gaussian Naive Bayes
- ⬜ Multinomial Naive Bayes
- ⬜ Bayesian Linear Regression

### Kernel methods
- ⬜ Support Vector Machine (SMO)
- ⬜ Kernel Ridge Regression

### Neural networks
- ✅ Multi-Layer Perceptron (`supervised/mlp`)
- ✅ Perceptron (`supervised/perceptron`)
- ✅ Softmax Classifier (`supervised/softmax_classifier`)
- ✅ Convolutional Neural Network — toy (`supervised/cnn`)
- ✅ Vanilla RNN — toy (`supervised/rnn`)
- ✅ Autoencoder (`unsupervised/autoencoder`)
- ⬜ LSTM / GRU
- ⬜ Denoising / Variational Autoencoder
- ⬜ Convolutional Autoencoder

## Unsupervised learning

### Clustering
- ✅ K-Means (`unsupervised/kmeans`)
- ⬜ K-Means++ initialisation as default
- ⬜ Mini-batch K-Means
- ⬜ DBSCAN
- ⬜ Gaussian Mixture Model (EM)
- ⬜ Hierarchical / Agglomerative

### Dimensionality reduction
- ⬜ PCA (SVD-based)
- ⬜ Kernel PCA
- ⬜ t-SNE
- ⬜ UMAP

### Density estimation
- ⬜ Kernel Density Estimation

## Preprocessing
- ⬜ StandardScaler (currently a utility, promote to module)
- ⬜ MinMaxScaler
- ⬜ One-Hot Encoder
- ⬜ Polynomial Features

## Metrics
- ⬜ Curated metric module (currently a utility)

## Optimisation primitives (shared)
- ✅ SGD / Momentum / Adam (`utils/optim.py`)
- ⬜ Line search / L-BFGS-style optimiser

---

When adding an item, move it to ✅ and link to its module in this file *and*
in the top-level `README.md` table.
