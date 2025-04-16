# Farcaster Topic Analysis

This project analyzes Farcaster data to identify popular users and the topics they discuss. The main goal is to cluster users based on their content and identify dominant topics in each cluster.

## Project Architecture

```
fc-topics-analysis/
├── src/
│   ├── data_preprocessing.py    - Initial data cleaning and preprocessing
│   └── user_clustering.py       - User-based embedding clustering pipeline
├── output/                      - Generated outputs
│   ├── cache/                   - Cached intermediate results
│   ├── embeddings/              - User and post embeddings
│   ├── clusters/                - Clustering results
│   └── topics/                  - Topic analysis results
├── LICENSE                      - MIT License
└── README.md                    - This file
```

## Approach

The project follows a multi-step approach:

1. **Data Preprocessing**:
   - Clean and filter Farcaster data
   - Remove duplicates and noise
   - Calculate engagement metrics for posts

2. **User Popularity Analysis**:
   - Identify top users based on:
     - Number of casts (posts)
     - Engagement (likes, recasts)
     - Reply count on their casts
   - Select top 1000 users by popularity

3. **Embedding Generation**:
   - Process all casts by popular users
   - Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
   - Compute average embeddings for each user

4. **Clustering**:
   - Apply clustering to user embeddings
   - Identify clusters of users who discuss similar topics
   - Extract key posts from each cluster

5. **Topic Identification**:
   - Send representative posts to Gemini API
   - Generate topic labels for each cluster
   - Analyze topic distribution across users

## Technologies Used

- **DuckDB**: Fast in-memory SQL analytics database for efficient data processing
- **HuggingFace**: Sentence transformer models for generating embeddings
- **Scikit-learn**: Clustering algorithms and dimensionality reduction
- **Gemini API**: Large language model for topic identification

## Getting Started

1. Install dependencies:
   ```
   pip install duckdb pandas numpy sentence-transformers scikit-learn google-generativeai matplotlib seaborn
   ```

2. Run the user clustering pipeline:
   ```
   python src/user_clustering.py
   ```

3. To skip specific steps (using cached results):
   ```
   python src/user_clustering.py --skip-user-identification --skip-embedding-generation
   ```

## Output

The pipeline produces several outputs:
- List of top users by popularity
- User embedding clusters
- Topic labels for each cluster
- Visualizations of user clusters

## Caching

The pipeline implements caching for each major step, allowing you to:
- Resume from interruptions
- Skip time-consuming steps when rerunning
- Experiment with different parameters for later steps

Cached data is stored in the `output/cache` directory.

## License

MIT License - See LICENSE file for details