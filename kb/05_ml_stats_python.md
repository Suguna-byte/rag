# Machine Learning, Statistics and Python Essentials

## Bias-variance and overfitting
Overfitting means the model learns noise in the training data and performs worse on new data. Signs: training score much higher than validation score. Fixes: more data, regularization (L1, L2, dropout), simpler models, early stopping, data augmentation, cross-validation for honest estimates. Underfitting is the opposite: the model is too simple to capture the pattern.

## Classification metrics
- Accuracy is misleading on imbalanced data.
- Precision = TP / (TP + FP): of predicted positives, how many were right. Matters when false alarms are costly.
- Recall = TP / (TP + FN): of real positives, how many were found. Matters when misses are costly, such as disease detection.
- F1 is the harmonic mean of precision and recall.
- ROC-AUC measures ranking quality across thresholds; PR-AUC is better under heavy class imbalance.

## Cross-validation
k-fold cross-validation splits data into k parts, trains on k-1 and validates on the remaining part, rotating. Use stratified folds for classification. Essential for small datasets such as DNA promoter data.

## Neural network basics (MLP)
A multilayer perceptron is layers of linear transformations followed by non-linear activations such as ReLU. Training uses backpropagation to compute gradients of the loss and an optimizer (SGD, Adam) to update weights. Cross-entropy loss for classification, mean squared error for regression. Softmax turns logits into probabilities.

## Gradient descent
Weights move opposite to the gradient of the loss. The learning rate controls step size: too high diverges, too low is slow. Adam adapts the step per parameter using running averages of gradients.

## Probability and statistics quick list
- Bayes' theorem: P(A|B) = P(B|A) P(A) / P(B).
- Expected value, variance, standard deviation.
- Normal distribution; central limit theorem: means of samples tend to a normal distribution.
- p-value: probability of seeing data at least this extreme if the null hypothesis is true. Not the probability the hypothesis is true.
- Correlation is not causation.

## Cosine similarity
cos(a, b) = (a · b) / (|a| |b|). Ranges from -1 to 1. Used to compare embeddings. If vectors are normalized to length 1, cosine similarity equals the dot product.

## Pandas and NumPy questions
- Vectorize instead of looping over rows.
- groupby, merge (join types), pivot_table, handling missing values (dropna, fillna).
- Broadcasting in NumPy applies operations across arrays of different shapes.

## Python questions that come up
- List vs tuple vs set vs dict, and their time complexities.
- Generators and yield for memory-efficient streaming of large files.
- Decorators, context managers (with statements).
- async/await and when it helps (I/O-bound work, like API calls) vs multiprocessing (CPU-bound work).
- The GIL limits CPU-bound threading in CPython.
- Type hints and Pydantic for validation.

## SQL questions
- Joins (inner, left), GROUP BY with HAVING, window functions (ROW_NUMBER, RANK, running totals).
- Indexes speed up reads but slow writes; composite index column order matters.
- EXPLAIN ANALYZE to inspect query plans in PostgreSQL.
- pgvector adds a vector column type and similarity operators with HNSW or IVFFlat indexes.
