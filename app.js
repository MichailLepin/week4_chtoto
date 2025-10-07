class MovieLensApp {
    constructor() {
        // Raw interactions (userId, itemId, rating, timestamp)
        this.interactions = [];
        // Map itemId -> { title, year }
        this.items = new Map();
        // Raw user features (userId -> { age, gender, occupation })
        this.userRaw = new Map();
        // Map itemId -> genre flags
        this.itemFeatureMap = new Map();
        // Data structures for indexing
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();
        // Pre‑computed top‑rated movies per user
        this.userTopRated = new Map();
        // Qualified users (≥20 ratings)
        this.qualifiedUsers = [];
        // Feature arrays aligned with indices
        this.userFeaturesList = [];
        this.itemFeaturesList = [];
        this.userFeatureDim = 0;
        this.itemFeatureDim = 0;
        // Models
        this.baselineModel = null;
        this.deepModel = null;
        // Training configuration
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 20,
            learningRate: 0.001,
            hiddenDim: 64
        };
        // Loss history for baseline model plotting
        this.lossHistory = [];
        this.isTraining = false;
        // Bind UI
        this.initializeUI();
    }

    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        this.updateStatus('Click “Load Data” to start.');
    }

    /**
     * Fetch and parse dataset files.  This function reads the rating
     * interactions (u.data), the movie metadata and genres (u.item), and
     * the user demographics (u.user).  It builds internal maps for
     * faster lookup and prepares feature matrices for the deep model.
     */
    async loadData() {
        this.updateStatus('Loading data…');
        try {
            // Load interactions
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            const interactionsLines = interactionsText.trim().split(/\r?\n/);
            // Respect maxInteractions to limit memory usage
            this.interactions = interactionsLines.slice(0, this.config.maxInteractions).map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });

            // Load items (metadata + genre flags)
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            const itemsLines = itemsText.trim().split(/\r?\n/);
            itemsLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                // Parse release year from the title
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                // Store movie title without year
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year
                });
                // The last 19 fields after the first 5 columns are genre flags
                const genreFlags = parts.slice(5).map(v => parseInt(v));
                this.itemFeatureMap.set(itemId, genreFlags);
            });

            // Attempt to load user demographics from u.user
            try {
                const usersResponse = await fetch('data/u.user');
                const usersText = await usersResponse.text();
                const userLines = usersText.trim().split(/\r?\n/);
                userLines.forEach(line => {
                    const parts = line.split('|');
                    if (parts.length >= 5) {
                        const uid = parseInt(parts[0]);
                        const age = parseInt(parts[1]);
                        const gender = parts[2];
                        const occupation = parts[3];
                        this.userRaw.set(uid, { age, gender, occupation });
                    }
                });
            } catch (err) {
                // If demographic file is missing, leave userRaw empty.  Deep
                // model will still construct zero features.
                console.warn('Could not load u.user:', err);
            }

            // Build mappings and top‑rated lists
            this.createMappings();
            // Prepare feature matrices for the deep model
            this.prepareFeatures();

            this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} movies. ${this.qualifiedUsers.length} users have ≥20 ratings.`);
            document.getElementById('train').disabled = false;
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    /**
     * Create mappings from external IDs to 0‑based indices.  Also
     * compute each user’s historically top‑rated list and identify
     * users with at least 20 ratings for testing.
     */
    createMappings() {
        // Unique user and item IDs from the interactions
        const userSet = new Set(this.interactions.map(i => i.userId));
        const itemSet = new Set(this.interactions.map(i => i.itemId));
        // Create index mappings
        Array.from(userSet).forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        Array.from(itemSet).forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
        // Group interactions by user to compute top‑rated lists
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const u = interaction.userId;
            if (!userInteractions.has(u)) userInteractions.set(u, []);
            userInteractions.get(u).push(interaction);
        });
        // Sort each user’s interactions by rating desc then recency desc
        userInteractions.forEach((arr, uid) => {
            arr.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
            this.userTopRated.set(uid, arr);
        });
        // Identify qualified users (≥20 ratings)
        this.qualifiedUsers = Array.from(userInteractions.entries()).filter(([uid, arr]) => arr.length >= 20).map(([uid]) => uid);
    }

    /**
     * Prepare numeric feature matrices for the deep model.  User
     * features include age (normalised), gender and occupation.  Item
     * features are the 19 genre flags from u.item.  If demographic
     * data is missing, user vectors fall back to zeros.
     */
    prepareFeatures() {
        // Collect unique occupations and maximum age for normalisation
        const occupations = new Set();
        let maxAge = 1;
        this.userRaw.forEach(({ age, occupation }) => {
            if (!Number.isNaN(age) && age > maxAge) maxAge = age;
            occupations.add(occupation);
        });
        const occupationList = Array.from(occupations);
        const occIndexMap = new Map();
        occupationList.forEach((occ, idx) => occIndexMap.set(occ, idx));
        const occCount = occupationList.length;
        // Set feature dimensions
        this.userFeatureDim = 2 + occCount; // age, gender, occupation one‑hot
        this.itemFeatureDim = this.itemFeatureMap.size > 0 ? this.itemFeatureMap.values().next().value.length : 0;
        // Build user feature list aligned to index ordering
        const numUsers = this.userMap.size;
        this.userFeaturesList = new Array(numUsers);
        for (let i = 0; i < numUsers; i++) {
            const uid = this.reverseUserMap.get(i);
            const raw = this.userRaw.get(uid);
            const feat = new Array(this.userFeatureDim).fill(0);
            if (raw) {
                // Age normalised to [0,1]
                const ageNorm = raw.age / maxAge;
                const genderVal = raw.gender === 'M' ? 1 : 0;
                feat[0] = ageNorm;
                feat[1] = genderVal;
                const occIdx = occIndexMap.get(raw.occupation);
                if (occIdx !== undefined) {
                    feat[2 + occIdx] = 1;
                }
            }
            this.userFeaturesList[i] = feat;
        }
        // Build item feature list aligned to index ordering
        const numItems = this.itemMap.size;
        this.itemFeaturesList = new Array(numItems);
        for (let i = 0; i < numItems; i++) {
            const itemId = this.reverseItemMap.get(i);
            const flags = this.itemFeatureMap.get(itemId);
            // If item features not found (should not happen), fill zeros
            if (flags) {
                this.itemFeaturesList[i] = flags.map(v => parseFloat(v));
            } else {
                this.itemFeaturesList[i] = new Array(this.itemFeatureDim).fill(0);
            }
        }
    }

    /**
     * Train both the baseline and the deep models using the stored
     * interactions.  Loss values for the baseline model are tracked
     * and plotted live.  The deep model is trained sequentially after
     * the baseline using the same batches to allow a fair comparison.
     */
    async train() {
        if (this.isTraining) return;
        if (this.interactions.length === 0) {
            this.updateStatus('No data loaded. Click “Load Data” first.');
            return;
        }
        this.isTraining = true;
        document.getElementById('train').disabled = true;
        this.lossHistory = [];
        // Create models
        const numUsers = this.userMap.size;
        const numItems = this.itemMap.size;
        this.baselineModel = new TwoTowerModel(numUsers, numItems, this.config.embeddingDim);
        this.baselineModel.optimizer = tf.train.adam(this.config.learningRate);
        this.deepModel = new DeepTwoTowerModel(
            numUsers,
            numItems,
            this.userFeatureDim,
            this.itemFeatureDim,
            this.config.embeddingDim,
            this.config.hiddenDim,
            this.userFeaturesList,
            this.itemFeaturesList
        );
        this.deepModel.optimizer = tf.train.adam(this.config.learningRate);
        // Prepare index arrays for training
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
        // Train baseline model
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                const batchUsers = userIndices.slice(start, end);
                const batchItems = itemIndices.slice(start, end);
                const loss = await this.baselineModel.trainStep(batchUsers, batchItems);
                epochLoss += loss;
                this.lossHistory.push(loss);
                this.updateLossChart();
                if (batch % 10 === 0) {
                    this.updateStatus(`Baseline: epoch ${epoch + 1}/${this.config.epochs}, batch ${batch + 1}/${numBatches}, loss ${loss.toFixed(4)}`);
                }
                // Allow UI updates
                await new Promise(res => setTimeout(res, 0));
            }
            epochLoss /= numBatches;
            this.updateStatus(`Baseline model epoch ${epoch + 1} complete (avg loss ${epochLoss.toFixed(4)})`);
        }
        // Train deep model
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                const batchUsers = userIndices.slice(start, end);
                const batchItems = itemIndices.slice(start, end);
                const loss = await this.deepModel.trainStep(batchUsers, batchItems);
                epochLoss += loss;
                if (batch % 10 === 0) {
                    this.updateStatus(`Deep: epoch ${epoch + 1}/${this.config.epochs}, batch ${batch + 1}/${numBatches}, loss ${loss.toFixed(4)}`);
                }
                await new Promise(res => setTimeout(res, 0));
            }
            epochLoss /= numBatches;
            this.updateStatus(`Deep model epoch ${epoch + 1} complete (avg loss ${epochLoss.toFixed(4)})`);
        }
        this.isTraining = false;
        document.getElementById('train').disabled = false;
        document.getElementById('test').disabled = false;
        this.updateStatus('Training finished. Click “Test” to see recommendations.');
        // Visualise baseline item embeddings using PCA
        this.visualizeEmbeddings();
    }

    /**
     * Plot the baseline model’s loss history on a canvas.  Losses
     * accumulate per batch.
     */
    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (this.lossHistory.length === 0) return;
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;
        ctx.strokeStyle = '#007acc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        this.lossHistory.forEach((loss, idx) => {
            const x = (idx / this.lossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
            if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 8, canvas.height - 8);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 8, 16);
    }

    /**
     * Visualise a PCA projection of the baseline item embeddings.  The
     * deep model’s embeddings are not shown to keep the UI simple.
     */
    async visualizeEmbeddings() {
        if (!this.baselineModel) return;
        this.updateStatus('Computing embedding visualisation…');
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        try {
            const sampleSize = Math.min(500, this.itemMap.size);
            const sampleIndices = Array.from({ length: sampleSize }, (_, i) => Math.floor(i * this.itemMap.size / sampleSize));
            const embeddingsTensor = this.baselineModel.getItemEmbeddings();
            const embeddings = embeddingsTensor.arraySync();
            const sampleEmbeddings = sampleIndices.map(idx => embeddings[idx]);
            const projected = this.computePCA(sampleEmbeddings, 2);
            const xs = projected.map(p => p[0]);
            const ys = projected.map(p => p[1]);
            const xMin = Math.min(...xs), xMax = Math.max(...xs);
            const yMin = Math.min(...ys), yMax = Math.max(...ys);
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            ctx.fillStyle = 'rgba(0, 122, 204, 0.6)';
            sampleIndices.forEach((itemIdx, i) => {
                const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 40) + 20;
                const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 40) + 20;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();
            });
            ctx.fillStyle = '#000';
            ctx.font = '14px Arial';
            ctx.fillText('Baseline Item Embeddings Projection (PCA)', 10, 20);
            ctx.font = '12px Arial';
            ctx.fillText(`Showing ${sampleSize} items`, 10, 38);
            this.updateStatus('Embedding visualisation complete.');
        } catch (err) {
            this.updateStatus(`Error computing embeddings: ${err.message}`);
        }
    }

    /**
     * Compute a simple PCA of high‑dimensional vectors down to
     * `dimensions` using power iteration.  Returns an array of
     * projected points.
     * @param {Array<Array<number>>} embeddings
     * @param {number} dimensions
     */
    computePCA(embeddings, dimensions) {
        const n = embeddings.length;
        const dim = embeddings[0].length;
        // Centre data
        const mean = Array(dim).fill(0);
        embeddings.forEach(vec => {
            vec.forEach((v, i) => { mean[i] += v; });
        });
        mean.forEach((v, i) => { mean[i] = v / n; });
        const centred = embeddings.map(vec => vec.map((v, i) => v - mean[i]));
        // Covariance matrix
        const cov = Array(dim).fill(0).map(() => Array(dim).fill(0));
        centred.forEach(vec => {
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    cov[i][j] += vec[i] * vec[j];
                }
            }
        });
        cov.forEach(row => row.forEach((v, j) => { row[j] = v / n; }));
        // Power iteration for each principal component
        const components = [];
        const covMatrix = cov.map(row => row.slice());
        for (let d = 0; d < dimensions; d++) {
            let v = Array(dim).fill(1 / Math.sqrt(dim));
            for (let it = 0; it < 10; it++) {
                const newV = Array(dim).fill(0);
                for (let i = 0; i < dim; i++) {
                    for (let j = 0; j < dim; j++) {
                        newV[i] += covMatrix[i][j] * v[j];
                    }
                }
                const norm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
                v = newV.map(x => x / norm);
            }
            components.push(v);
            // Deflate
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covMatrix[i][j] -= v[i] * v[j];
                }
            }
        }
        // Project
        return embeddings.map(vec => components.map(comp => comp.reduce((sum, c, i) => sum + c * vec[i], 0)));
    }

    /**
     * Generate and display recommendations for a random qualified user.
     * The historical top‑10 movies, the baseline recommendations and
     * the deep recommendations are presented side‑by‑side.
     */
    async test() {
        if (!this.baselineModel || !this.deepModel || this.qualifiedUsers.length === 0) {
            this.updateStatus('Models not trained or no qualified users.');
            return;
        }
        this.updateStatus('Generating recommendations…');
        try {
            const randomUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
            const interactions = this.userTopRated.get(randomUser);
            const userIndex = this.userMap.get(randomUser);
            // Baseline recommendations
            const userEmb = this.baselineModel.getUserEmbedding(userIndex);
            const scores = await this.baselineModel.getScoresForAllItems(userEmb);
            // Deep recommendations
            const userEmbDeep = this.deepModel.getUserEmbedding(userIndex);
            const scoresDeep = await this.deepModel.getScoresForAllItems(userEmbDeep);
            // Exclude already rated items
            const ratedIds = new Set(interactions.map(i => i.itemId));
            const candidateScores = [];
            const candidateScoresDeep = [];
            scores.forEach((score, idx) => {
                const itemId = this.reverseItemMap.get(idx);
                if (!ratedIds.has(itemId)) {
                    candidateScores.push({ itemId, score });
                }
            });
            scoresDeep.forEach((score, idx) => {
                const itemId = this.reverseItemMap.get(idx);
                if (!ratedIds.has(itemId)) {
                    candidateScoresDeep.push({ itemId, score });
                }
            });
            // Sort and select top 10
            candidateScores.sort((a, b) => b.score - a.score);
            const baselineTop = candidateScores.slice(0, 10).map(({ itemId, score }) => ({ itemId, score }));
            candidateScoresDeep.sort((a, b) => b.score - a.score);
            const deepTop = candidateScoresDeep.slice(0, 10).map(({ itemId, score }) => ({ itemId, score }));
            // Display results
            this.displayResults(randomUser, interactions, baselineTop, deepTop);
        } catch (err) {
            this.updateStatus(`Error generating recommendations: ${err.message}`);
            return;
        }
        this.updateStatus('Recommendations generated.');
    }

    /**
     * Render the three lists (historical, baseline, deep) side‑by‑side
     * in the results section.
     * @param {number} userId
     * @param {Array} userInteractions
     * @param {Array} baselineRecs
     * @param {Array} deepRecs
     */
    displayResults(userId, userInteractions, baselineRecs, deepRecs) {
        const div = document.getElementById('results');
        const topRated = userInteractions.slice(0, 10);
        let html = `<h2>Recommendations for User ${userId}</h2>`;
        html += '<div class="side-by-side">';
        // Historical
        html += '<div><h3>Top 10 Rated Movies</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Rating</th><th>Year</th></tr></thead><tbody>';
        topRated.forEach((interaction, idx) => {
            const item = this.items.get(interaction.itemId);
            html += `<tr><td>${idx + 1}</td><td>${item.title}</td><td>${interaction.rating.toFixed(1)}</td><td>${item.year || 'N/A'}</td></tr>`;
        });
        html += '</tbody></table></div>';
        // Baseline recommendations
        html += '<div><h3>Top 10 Baseline Recommendations</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Score</th><th>Year</th></tr></thead><tbody>';
        baselineRecs.forEach((rec, idx) => {
            const item = this.items.get(rec.itemId);
            html += `<tr><td>${idx + 1}</td><td>${item.title}</td><td>${rec.score.toFixed(4)}</td><td>${item.year || 'N/A'}</td></tr>`;
        });
        html += '</tbody></table></div>';
        // Deep recommendations
        html += '<div><h3>Top 10 Deep Recommendations</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Score</th><th>Year</th></tr></thead><tbody>';
        deepRecs.forEach((rec, idx) => {
            const item = this.items.get(rec.itemId);
            html += `<tr><td>${idx + 1}</td><td>${item.title}</td><td>${rec.score.toFixed(4)}</td><td>${item.year || 'N/A'}</td></tr>`;
        });
        html += '</tbody></table></div>';
        html += '</div>';
        div.innerHTML = html;
    }

    /**
     * Update the status message in the UI.
     * @param {string} message
     */
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

// Initialise the application once the DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});