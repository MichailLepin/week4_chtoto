/*
 * Two simple recommendation models built with TensorFlow.js.
 *
 * The first model (TwoTowerModel) implements a classic two‑tower
 * retrieval architecture: users and items are represented by learned
 * embeddings of fixed dimension. During training, positive user–item
 * pairs are contrasted against in‑batch negatives via a sampled
 * softmax loss (implemented as a cross‑entropy over the similarity
 * matrix).  This model does not consume any side information.
 *
 * The second model (DeepTwoTowerModel) extends the baseline by
 * incorporating rich user and item features.  Each tower combines an
 * ID embedding with a projection of the feature vector, followed by
 * at least one hidden layer (a fully‑connected layer with a ReLU
 * activation).  The two towers then produce fixed‑dimensional
 * vectors that are compared via dot product.  Because features are
 * constant, they are passed into the constructor and stored as
 * tf.tensors.  As with the baseline, the loss is the in‑batch
 * sampled softmax.
 */

class TwoTowerModel {
    /**
     * Construct a baseline two‑tower model.
     * @param {number} numUsers Number of unique users.
     * @param {number} numItems Number of unique items.
     * @param {number} embeddingDim Dimension of the latent embeddings.
     */
    constructor(numUsers, numItems, embeddingDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;

        // Embedding tables for users and items.  Entries are trainable.
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05),
            true,
            'user_embeddings'
        );
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05),
            true,
            'item_embeddings'
        );

        // Optimizer for both towers.  The learning rate will be
        // provided externally when the model is used; by default we
        // initialise with 0.001.
        this.optimizer = tf.train.adam(0.001);
    }

    /**
     * Look up user embeddings for a batch of user indices.
     * @param {tf.Tensor|number[]} userIndices Array of user indices.
     * @returns {tf.Tensor} [batch, embeddingDim]
     */
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }

    /**
     * Look up item embeddings for a batch of item indices.
     * @param {tf.Tensor|number[]} itemIndices Array of item indices.
     * @returns {tf.Tensor} [batch, embeddingDim]
     */
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }

    /**
     * Compute dot products between user and item embeddings.
     * @param {tf.Tensor} userEmbeddings [batch, embeddingDim]
     * @param {tf.Tensor} itemEmbeddings [batch, embeddingDim]
     * @returns {tf.Tensor} [batch] scores
     */
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }

    /**
     * Perform one optimisation step using in‑batch sampled softmax.
     * Each batch uses its own items as negatives for all other users.
     * @param {number[]} userIndices Array of user indices for the batch.
     * @param {number[]} itemIndices Array of item indices for the batch.
     * @returns {Promise<number>} The loss value for this batch.
     */
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');

            // Define the loss function inside tidy so gradients are
            // computed correctly.  We compute a matrix of logits
            // between each user and each item in the batch.  The
            // diagonal corresponds to positive pairs.
            const lossFn = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor);
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                const labels = tf.oneHot(tf.range(0, userIndices.length, 1, 'int32'), userIndices.length);
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };

            const { value, grads } = this.optimizer.computeGradients(lossFn);
            this.optimizer.applyGradients(grads);
            return value.dataSync()[0];
        });
    }

    /**
     * Get a single user embedding.
     * @param {number} userIndex 0‑based user index.
     * @returns {tf.Tensor1D} [embeddingDim]
     */
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }

    /**
     * Compute scores for all items given a user embedding.
     * @param {tf.Tensor1D} userEmbedding [embeddingDim]
     * @returns {Promise<Float32Array>} Scores for each item.
     */
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            const scores = tf.dot(this.itemEmbeddings, userEmbedding);
            return scores.dataSync();
        });
    }

    /**
     * Return the raw item embedding table.  Used for PCA visualisation.
     * @returns {tf.Tensor} [numItems, embeddingDim]
     */
    getItemEmbeddings() {
        return this.itemEmbeddings;
    }
}

class DeepTwoTowerModel {
    /**
     * Construct a deep two‑tower model with side information.
     * @param {number} numUsers Number of users.
     * @param {number} numItems Number of items.
     * @param {number} userFeatureDim Dimensionality of the user feature vector.
     * @param {number} itemFeatureDim Dimensionality of the item feature vector.
     * @param {number} embeddingDim Base embedding dimension for ID
     *   lookups and final output.  Hidden layers project to this size.
     * @param {number} hiddenDim Size of the hidden layer in the MLP.
     * @param {Array<Array<number>>} userFeatures2D Array of user feature
     *   vectors aligned with 0‑based user indices.
     * @param {Array<Array<number>>} itemFeatures2D Array of item feature
     *   vectors aligned with 0‑based item indices.
     */
    constructor(numUsers, numItems, userFeatureDim, itemFeatureDim, embeddingDim, hiddenDim = 64, userFeatures2D, itemFeatures2D) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.userFeatureDim = userFeatureDim;
        this.itemFeatureDim = itemFeatureDim;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;

        // Embeddings for IDs
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embeddingDim], 0, 0.05),
            true,
            'deep_user_embeddings'
        );
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embeddingDim], 0, 0.05),
            true,
            'deep_item_embeddings'
        );

        // Project raw features into the embedding space
        this.userFeatWeight = tf.variable(
            tf.randomNormal([userFeatureDim, embeddingDim], 0, 0.05),
            true,
            'user_feat_weight'
        );
        this.userFeatBias = tf.variable(tf.zeros([embeddingDim]));

        this.itemFeatWeight = tf.variable(
            tf.randomNormal([itemFeatureDim, embeddingDim], 0, 0.05),
            true,
            'item_feat_weight'
        );
        this.itemFeatBias = tf.variable(tf.zeros([embeddingDim]));

        // Hidden layer weights (concatenated embed + projected features)
        const concatDimUser = embeddingDim * 2;
        this.userHiddenWeight = tf.variable(
            tf.randomNormal([concatDimUser, embeddingDim], 0, 0.05),
            true,
            'user_hidden_weight'
        );
        this.userHiddenBias = tf.variable(tf.zeros([embeddingDim]));

        const concatDimItem = embeddingDim * 2;
        this.itemHiddenWeight = tf.variable(
            tf.randomNormal([concatDimItem, embeddingDim], 0, 0.05),
            true,
            'item_hidden_weight'
        );
        this.itemHiddenBias = tf.variable(tf.zeros([embeddingDim]));

        // Feature tensors are constants and not trainable
        this.userFeatures = tf.tensor2d(userFeatures2D, [numUsers, userFeatureDim], 'float32');
        this.itemFeatures = tf.tensor2d(itemFeatures2D, [numItems, itemFeatureDim], 'float32');

        // Optimiser
        this.optimizer = tf.train.adam(0.001);
    }

    /**
     * Compute user tower output for given indices.  Combines the ID
     * embedding with a projection of the user features and applies an
     * MLP with one hidden layer.
     * @param {tf.Tensor|number[]} userIndices
     * @returns {tf.Tensor} [batch, embeddingDim]
     */
    userForward(userIndices) {
        return tf.tidy(() => {
            const idEmb = tf.gather(this.userEmbeddings, userIndices);
            const feat = tf.gather(this.userFeatures, userIndices);
            let featProj = tf.add(tf.matMul(feat, this.userFeatWeight), this.userFeatBias);
            featProj = tf.relu(featProj);
            const concat = tf.concat([idEmb, featProj], 1);
            let hidden = tf.add(tf.matMul(concat, this.userHiddenWeight), this.userHiddenBias);
            hidden = tf.relu(hidden);
            return hidden;
        });
    }

    /**
     * Compute item tower output for given indices.  Combines the ID
     * embedding with projected genre features and applies an MLP.
     * @param {tf.Tensor|number[]} itemIndices
     * @returns {tf.Tensor} [batch, embeddingDim]
     */
    itemForward(itemIndices) {
        return tf.tidy(() => {
            const idEmb = tf.gather(this.itemEmbeddings, itemIndices);
            const feat = tf.gather(this.itemFeatures, itemIndices);
            let featProj = tf.add(tf.matMul(feat, this.itemFeatWeight), this.itemFeatBias);
            featProj = tf.relu(featProj);
            const concat = tf.concat([idEmb, featProj], 1);
            let hidden = tf.add(tf.matMul(concat, this.itemHiddenWeight), this.itemHiddenBias);
            hidden = tf.relu(hidden);
            return hidden;
        });
    }

    /**
     * Compute dot product scores between user and item tower outputs.
     * @param {tf.Tensor} userEmbeddings [batch, embeddingDim]
     * @param {tf.Tensor} itemEmbeddings [batch, embeddingDim]
     * @returns {tf.Tensor} [batch] scores
     */
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }

    /**
     * In‑batch sampled softmax training step.  This function mirrors
     * the baseline implementation but runs the forward passes through
     * the MLP towers.  Features are looked up internally.
     * @param {number[]} userIndices
     * @param {number[]} itemIndices
     * @returns {Promise<number>} Loss value
     */
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            const lossFn = () => {
                const userEmbs = this.userForward(userTensor);
                const itemEmbs = this.itemForward(itemTensor);
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                const labels = tf.oneHot(tf.range(0, userIndices.length, 1, 'int32'), userIndices.length);
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            const { value, grads } = this.optimizer.computeGradients(lossFn);
            this.optimizer.applyGradients(grads);
            return value.dataSync()[0];
        });
    }

    /**
     * Get the final user representation for a single user index.
     * @param {number} userIndex
     * @returns {tf.Tensor1D} [embeddingDim]
     */
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }

    /**
     * Compute scores for all items given a user embedding.  This
     * method computes the item tower outputs for all items before
     * taking the dot product.  For the MovieLens 100K dataset the
     * number of items (1682) is modest, so a full forward pass is
     * tractable.  If memory becomes a concern, consider batching
     * across items.
     * @param {tf.Tensor1D} userEmbedding
     * @returns {Promise<Float32Array>} Scores for each item.
     */
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Expand user embedding to [embeddingDim, 1] for matmul
            const userCol = userEmbedding.reshape([this.embeddingDim, 1]);
            // Compute all item tower outputs
            const allItemEmb = this.itemForward(tf.range(0, this.numItems));
            const scores = tf.matMul(allItemEmb, userCol).squeeze();
            return scores.dataSync();
        });
    }
}

// Export the classes to the global scope so they can be used from app.js
if (typeof window !== 'undefined') {
    window.TwoTowerModel = TwoTowerModel;
    window.DeepTwoTowerModel = DeepTwoTowerModel;
}