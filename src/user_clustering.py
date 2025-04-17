import duckdb
import pandas as pd
import numpy as np
import time
import os
import re
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import random
import google.generativeai as genai
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Start measuring execution time
start_time = time.time()

# Constants
OUTPUT_DIR = Path('output')
CACHE_DIR = OUTPUT_DIR / 'cache'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOP_USERS_COUNT = 1000
MIN_CASTS_PER_USER = 5
NUM_CLUSTERS = 10  # Default, will be optimized
NUM_SAMPLE_POSTS = 5  # Number of posts to sample per cluster for LLM analysis
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Define cache files
CACHE_TOP_USERS = CACHE_DIR / 'top_users.pkl'
CACHE_USER_EMBEDDINGS = CACHE_DIR / 'user_embeddings.pkl'
CACHE_CLUSTERS = CACHE_DIR / 'user_clusters.pkl'
CACHE_REPRESENTATIVE_POSTS = CACHE_DIR / 'representative_posts.pkl'
CACHE_TOPICS = CACHE_DIR / 'cluster_topics.pkl'

def save_to_cache(data: Any, cache_file: Path) -> None:
    """Save data to cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Cached data saved to {cache_file}")

def load_from_cache(cache_file: Path) -> Optional[Any]:
    """Load data from cache file if it exists."""
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded cached data from {cache_file}")
            return data
        except Exception as e:
            print(f"Error loading cache from {cache_file}: {e}")
    return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Farcaster User Clustering Pipeline")
    
    # Add arguments to skip specific steps
    parser.add_argument('--skip-user-identification', action='store_true', 
                        help='Skip user identification step and use cached results')
    parser.add_argument('--skip-embedding-generation', action='store_true',
                        help='Skip embedding generation step and use cached results')
    parser.add_argument('--skip-clustering', action='store_true',
                        help='Skip clustering step and use cached results')
    parser.add_argument('--skip-representative-posts', action='store_true',
                        help='Skip finding representative posts and use cached results')
    parser.add_argument('--skip-topic-analysis', action='store_true',
                        help='Skip topic analysis with Gemini API and use cached results')
    
    # Add configuration parameters
    parser.add_argument('--top-users', type=int, default=TOP_USERS_COUNT,
                        help=f'Number of top users to analyze (default: {TOP_USERS_COUNT})')
    parser.add_argument('--min-casts', type=int, default=MIN_CASTS_PER_USER,
                        help=f'Minimum number of casts per user (default: {MIN_CASTS_PER_USER})')
    parser.add_argument('--clusters', type=int, default=None,
                        help='Number of clusters (default: automatically optimized)')
    parser.add_argument('--sample-posts', type=int, default=NUM_SAMPLE_POSTS,
                        help=f'Number of sample posts per cluster for topic analysis (default: {NUM_SAMPLE_POSTS})')
    parser.add_argument('--gemini-key', type=str, default=None,
                        help='Gemini API key (default: read from GEMINI_API_KEY environment variable)')
    
    return parser.parse_args()

def setup_environment():
    """Set up the environment and create necessary directories."""
    # Create all required directories
    for dir_path in [
        OUTPUT_DIR,
        OUTPUT_DIR / 'embeddings',
        OUTPUT_DIR / 'clusters',
        OUTPUT_DIR / 'topics',
        CACHE_DIR
    ]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Check for GPU availability for faster embedding generation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} with {gpu_info.total_memory / 1e9:.2f} GB memory")
        # Optimize CUDA settings for maximum performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    
    return device

def connect_to_duckdb():
    """Create and configure a DuckDB connection."""
    conn = duckdb.connect(database=':memory:')
    
    # Use absolute value instead of percentage (80% of 98GB is ~78GB)
    conn.execute("SET memory_limit='78GB'")  # Use fixed size for memory limit
    
    conn.execute("SET temp_directory='/tmp'")  # Set temp directory for spilling
    conn.execute("PRAGMA threads=12")  # Use 12 threads (leaving 2 cores for system)
    return conn

def load_farcaster_data(conn):
    """Load Farcaster data from parquet files."""
    print("Loading Farcaster data...")
    try:
        conn.execute("CREATE VIEW casts AS SELECT * FROM read_parquet('casts.parquet')")
        conn.execute("CREATE VIEW reactions AS SELECT * FROM read_parquet('farcaster_reactions.parquet')")
        
        # Convert timestamps to datetime
        conn.execute("""
        CREATE VIEW casts_with_datetime AS 
        SELECT *, 
               TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
        FROM casts
        """)
        
        conn.execute("""
        CREATE VIEW reactions_with_datetime AS 
        SELECT *, 
               TIMESTAMP '2021-01-01 00:00:00' + (CAST("Timestamp" AS BIGINT) * INTERVAL '1 second') AS datetime
        FROM reactions
        """)
        
        # Count the data and report statistics
        cast_count = conn.execute("SELECT COUNT(*) FROM casts").fetchone()[0]
        reaction_count = conn.execute("SELECT COUNT(*) FROM reactions").fetchone()[0]
        user_count = conn.execute("SELECT COUNT(DISTINCT Fid) FROM casts").fetchone()[0]
        
        print(f"Loaded {cast_count:,} casts from {user_count:,} users")
        print(f"Loaded {reaction_count:,} reactions")
        
        # Check timestamp range
        time_range = conn.execute("""
        SELECT MIN(datetime), MAX(datetime) FROM casts_with_datetime
        """).fetchone()
        print(f"Data spans from {time_range[0]} to {time_range[1]}")
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def clean_text(text):
    """Clean text by removing URLs, mentions, and special characters."""
    if pd.isna(text) or text == "":
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,?!]', '', text)
    # Convert to lowercase and strip whitespace
    return text.lower().strip()

def identify_popular_users(conn, top_n=TOP_USERS_COUNT, min_casts=MIN_CASTS_PER_USER):
    """Identify the top users based on engagement and activity metrics."""
    print(f"Identifying top {top_n} users by popularity (min {min_casts} casts per user)...")
    
    # Create normalized hash fields for reliable joining
    conn.execute("""
    CREATE OR REPLACE VIEW casts_normalized AS
    SELECT *, 
        LOWER(TRIM(Hash)) as hash_normalized
    FROM casts_with_datetime
    """)
    
    # Extract reaction target hash from TargetCastId
    conn.execute("""
    CREATE OR REPLACE VIEW reactions_normalized AS
    SELECT 
        *,
        CASE 
            WHEN POSITION(':' IN TargetCastId) > 0 
            THEN SUBSTRING(TargetCastId, POSITION(':' IN TargetCastId) + 1)
            ELSE TargetCastId 
        END AS target_hash,
        LOWER(TRIM(
            CASE 
                WHEN POSITION(':' IN TargetCastId) > 0 
                THEN SUBSTRING(TargetCastId, POSITION(':' IN TargetCastId) + 1)
                ELSE TargetCastId 
            END
        )) AS reaction_target_normalized
    FROM reactions_with_datetime
    """)
    
    # Compute reactions per cast
    conn.execute("""
    CREATE OR REPLACE TABLE cast_reactions AS
    SELECT 
        reaction_target_normalized,
        COUNT(*) AS total_reactions,
        SUM(CASE WHEN ReactionType = 'Like' THEN 1 ELSE 0 END) AS likes_count,
        SUM(CASE WHEN ReactionType = 'Recast' THEN 1 ELSE 0 END) AS recasts_count
    FROM reactions_normalized
    GROUP BY reaction_target_normalized
    """)
    
    # Compute replies per cast
    conn.execute("""
    CREATE OR REPLACE TABLE cast_replies AS
    SELECT 
        LOWER(TRIM(
            CASE 
                WHEN POSITION(':' IN ParentCastId) > 0 
                THEN SUBSTRING(ParentCastId, POSITION(':' IN ParentCastId) + 1)
                ELSE ParentCastId 
            END
        )) AS parent_hash_normalized,
        COUNT(*) AS reply_count
    FROM casts_normalized
    WHERE ParentCastId IS NOT NULL AND ParentCastId != ''
    GROUP BY parent_hash_normalized
    """)
    
    # Compute user-level metrics for popularity ranking
    conn.execute(f"""
    CREATE OR REPLACE TABLE user_metrics AS
    SELECT 
        c.Fid,
        COUNT(DISTINCT c.Hash) AS cast_count,
        SUM(COALESCE(r.likes_count, 0)) AS total_likes,
        SUM(COALESCE(r.recasts_count, 0)) AS total_recasts,
        SUM(COALESCE(r.total_reactions, 0)) AS total_reactions,
        SUM(COALESCE(p.reply_count, 0)) AS total_replies,
        AVG(COALESCE(r.likes_count, 0)) AS avg_likes_per_cast,
        AVG(COALESCE(r.recasts_count, 0)) AS avg_recasts_per_cast,
        AVG(COALESCE(p.reply_count, 0)) AS avg_replies_per_cast,
        (
            SUM(COALESCE(r.likes_count, 0)) + 
            3 * SUM(COALESCE(r.recasts_count, 0)) + 
            5 * SUM(COALESCE(p.reply_count, 0))
        ) AS popularity_score
    FROM casts_normalized c
    LEFT JOIN cast_reactions r ON c.hash_normalized = r.reaction_target_normalized
    LEFT JOIN cast_replies p ON c.hash_normalized = p.parent_hash_normalized
    WHERE c.Text IS NOT NULL AND LENGTH(c.Text) > 0
    GROUP BY c.Fid
    HAVING COUNT(DISTINCT c.Hash) >= {min_casts}
    """)
    
    # Get the top users by popularity score
    top_users = conn.execute(f"""
    SELECT 
        Fid,
        cast_count,
        total_likes,
        total_recasts,
        total_replies,
        total_reactions,
        popularity_score
    FROM user_metrics
    ORDER BY popularity_score DESC
    LIMIT {top_n}
    """).df()
    
    print(f"Found top {len(top_users)} users")
    print(f"  - Min popularity score: {top_users['popularity_score'].min():.1f}")
    print(f"  - Max popularity score: {top_users['popularity_score'].max():.1f}")
    print(f"  - Mean casts per user: {top_users['cast_count'].mean():.1f}")
    print(f"  - Min casts per user: {top_users['cast_count'].min()}")
    print(f"  - Max casts per user: {top_users['cast_count'].max()}")
    
    # Save top users to file
    top_users.to_csv(OUTPUT_DIR / 'top_users.csv', index=False)
    
    return top_users

def get_user_casts(conn, fid):
    """Get all casts by a specific user."""
    casts = conn.execute(f"""
    SELECT 
        Hash, 
        Text, 
        datetime, 
        COALESCE(r.likes_count, 0) AS likes_count,
        COALESCE(r.recasts_count, 0) AS recasts_count,
        COALESCE(p.reply_count, 0) AS reply_count
    FROM casts_normalized c
    LEFT JOIN cast_reactions r ON c.hash_normalized = r.reaction_target_normalized
    LEFT JOIN cast_replies p ON c.hash_normalized = p.parent_hash_normalized
    WHERE c.Fid = {fid}
    AND c.Text IS NOT NULL 
    AND LENGTH(c.Text) > 0
    ORDER BY datetime DESC
    """).df()
    
    # Clean text
    casts['cleaned_text'] = casts['Text'].apply(clean_text)
    
    # Filter out empty texts after cleaning
    casts = casts[casts['cleaned_text'].str.len() > 10]
    
    return casts

def generate_embeddings(texts, model, batch_size=64, device='cpu'):
    """Generate embeddings for a list of texts."""
    # Handle empty input
    if not texts or len(texts) == 0:
        return np.array([])
    
    # Filter out None values and replace empty strings
    filtered_texts = []
    for t in texts:
        if t is None or len(str(t).strip()) == 0:
            filtered_texts.append("empty_text")
        else:
            filtered_texts.append(str(t))
    
    # Efficiently generate embeddings in batches
    with torch.no_grad():  # Disable gradient computation for inference
        all_embeddings = []
        for i in range(0, len(filtered_texts), batch_size):
            batch = filtered_texts[i:i+batch_size]
            
            # Show progress for large batches
            if i % (batch_size * 10) == 0 and len(filtered_texts) > batch_size * 10:
                print(f"  Processing batch {i//batch_size + 1}/{(len(filtered_texts) + batch_size - 1) // batch_size}...")
            
            # Generate embeddings with optimizations for GPU
            if device == 'cuda':
                try:
                    # Try with half precision (much faster)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        batch_embeddings = model.encode(
                            batch, 
                            convert_to_tensor=True, 
                            show_progress_bar=False,
                            batch_size=batch_size,
                            normalize_embeddings=True
                        )
                except:
                    # Fall back to regular precision
                    batch_embeddings = model.encode(
                        batch, 
                        convert_to_tensor=True, 
                        show_progress_bar=False,
                        batch_size=batch_size,
                        normalize_embeddings=True
                    )
            else:
                # On CPU, use regular precision
                batch_embeddings = model.encode(
                    batch, 
                    convert_to_tensor=True, 
                    show_progress_bar=False,
                    batch_size=batch_size,
                    normalize_embeddings=True
                )
            
            # Move to CPU and convert to numpy to save memory
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)
            
            # Force CUDA to clean up memory after each batch
            if device == 'cuda':
                del batch_embeddings
                torch.cuda.empty_cache()
    
    # Combine all batches
    return np.vstack(all_embeddings) if all_embeddings else np.array([])

def process_user_embeddings(conn, top_users, device):
    """Generate and process embeddings for all posts by top users."""
    print(f"Processing embeddings for {len(top_users)} users...")
    
    # Load the embedding model
    print(f"Loading SentenceTransformer model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    model.eval()  # Set to evaluation mode for inference

    # Dictionary to store user embeddings
    user_embeddings = {}
    user_cast_counts = {}
    
    # Adapter batch size based on available device
    batch_size = 128 if device == 'cuda' else 32
    
    # Track timing for benchmarking
    embedding_start_time = time.time()
    processed_users = 0
    total_casts = 0
    
    # Process each user
    for idx, (_, user) in enumerate(top_users.iterrows()):
        fid = user['Fid']
        
        # Get all casts by this user
        user_casts = get_user_casts(conn, fid)
        
        if len(user_casts) == 0:
            print(f"  User {fid} has no valid casts, skipping...")
            continue
            
        # Generate embeddings for user's casts
        cast_texts = user_casts['cleaned_text'].tolist()
        
        if idx % 10 == 0:
            print(f"Generating embeddings for user {fid} ({len(cast_texts)} casts)...")
        
        # Generate embeddings
        cast_embeddings = generate_embeddings(
            cast_texts, 
            model, 
            batch_size=batch_size,
            device=device
        )
        
        if len(cast_embeddings) == 0:
            print(f"  Warning: No valid embeddings for user {fid}")
            continue
        
        # Average the embeddings to get a user-level representation
        user_embedding = np.mean(cast_embeddings, axis=0)
        
        # Store results
        user_embeddings[fid] = user_embedding
        user_cast_counts[fid] = len(cast_embeddings)
        
        # Update counters
        processed_users += 1
        total_casts += len(cast_embeddings)
        
        # Report progress every 100 users
        if (idx + 1) % 100 == 0 or idx == len(top_users) - 1:
            elapsed = time.time() - embedding_start_time
            print(f"Processed {idx+1}/{len(top_users)} users, {total_casts} total casts in {elapsed:.1f}s")
            print(f"  Average processing time: {elapsed/(idx+1):.2f}s per user")
    
    print(f"Completed embedding generation for {processed_users} users ({total_casts} total casts)")
    
    # Prepare user embeddings for clustering
    user_ids = list(user_embeddings.keys())
    X = np.array([user_embeddings[uid] for uid in user_ids])
    
    # Save embeddings
    print("Saving embeddings...")
    np.save(OUTPUT_DIR / 'embeddings' / 'user_embeddings.npy', X)
    with open(OUTPUT_DIR / 'embeddings' / 'user_ids.json', 'w') as f:
        json.dump(user_ids, f)
    
    return X, user_ids, user_cast_counts, model

def optimize_clusters(X, min_clusters=5, max_clusters=20):
    """Find the optimal number of clusters using silhouette score."""
    print("Optimizing number of clusters...")
    
    silhouette_scores = []
    k_values = range(min_clusters, max_clusters + 1)
    
    for k in k_values:
        print(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"  Silhouette score for k={k}: {silhouette_avg:.4f}")
    
    # Find the best number of clusters
    best_k = k_values[np.argmax(silhouette_scores)]
    best_score = silhouette_scores[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {best_k} (silhouette score: {best_score:.4f})")
    
    # Save silhouette analysis results
    silhouette_df = pd.DataFrame({
        'k': k_values,
        'silhouette_score': silhouette_scores
    })
    silhouette_df.to_csv(OUTPUT_DIR / 'clusters' / 'silhouette_scores.csv', index=False)
    
    return best_k

def cluster_users(X, user_ids, user_cast_counts, n_clusters=None):
    """Cluster users based on their embeddings."""
    if n_clusters is None:
        # Find optimal number of clusters
        n_clusters = optimize_clusters(X)
    
    print(f"Clustering {len(user_ids)} users into {n_clusters} clusters...")
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Create dataframe with cluster assignments
    cluster_df = pd.DataFrame({
        'user_id': user_ids,
        'cluster': cluster_labels,
        'cast_count': [user_cast_counts.get(uid, 0) for uid in user_ids]
    })
    
    # Count users per cluster
    cluster_counts = Counter(cluster_labels)
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cluster_id}: {count} users")
    
    # Save cluster assignments
    cluster_df.to_csv(OUTPUT_DIR / 'clusters' / 'user_clusters.csv', index=False)
    np.save(OUTPUT_DIR / 'clusters' / 'cluster_centers.npy', kmeans.cluster_centers_)
    
    return cluster_df, kmeans.cluster_centers_

def visualize_clusters(X, cluster_df):
    """Visualize the user clusters using t-SNE."""
    print("Generating t-SNE visualization of user clusters...")
    
    # Apply t-SNE for dimensionality reduction to 2D
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Create DataFrame for plotting
    viz_df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'cluster': cluster_df['cluster'],
        'user_id': cluster_df['user_id'],
        'cast_count': cluster_df['cast_count']
    })
    
    # Save t-SNE coordinates for future use
    viz_df.to_csv(OUTPUT_DIR / 'clusters' / 'tsne_coordinates.csv', index=False)
    
    # Plot the clusters
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='x', y='y', 
        hue='cluster', 
        size='cast_count',
        sizes=(20, 200),
        palette='tab10',
        data=viz_df,
        alpha=0.7
    )
    plt.title('User Clusters Visualization (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clusters' / 'user_clusters_tsne.png', dpi=300)
    plt.close()
    
    return viz_df

def find_representative_posts(conn, cluster_df, model, device):
    """Find representative posts from each cluster for topic analysis."""
    print("Finding representative posts for each cluster...")
    
    # Cluster center embeddings
    centers = np.load(OUTPUT_DIR / 'clusters' / 'cluster_centers.npy')
    
    # Dictionary to store representative posts for each cluster
    cluster_posts = {i: [] for i in range(len(centers))}
    
    # Process each cluster
    for cluster_id in range(len(centers)):
        print(f"Finding representative posts for cluster {cluster_id}...")
        
        # Get users in this cluster
        cluster_users = cluster_df[cluster_df['cluster'] == cluster_id]['user_id'].tolist()
        
        if not cluster_users:
            print(f"  Warning: No users found in cluster {cluster_id}")
            continue
        
        # Collect posts from random users in the cluster
        sampled_users = random.sample(cluster_users, min(5, len(cluster_users)))
        all_cluster_posts = []
        
        for fid in sampled_users:
            user_casts = get_user_casts(conn, fid)
            
            if len(user_casts) == 0:
                continue
                
            # Sort by engagement score
            user_casts['engagement_score'] = (
                user_casts['likes_count'] + 
                3 * user_casts['recasts_count'] + 
                5 * user_casts['reply_count']
            )
            user_casts = user_casts.sort_values('engagement_score', ascending=False)
            
            # Take top posts
            top_posts = user_casts.head(10)
            all_cluster_posts.append(top_posts)
            
        if not all_cluster_posts:
            print(f"  Warning: No posts found for cluster {cluster_id}")
            continue
            
        # Combine posts from all sampled users
        combined_posts = pd.concat(all_cluster_posts)
        
        # Generate embeddings for these posts
        post_texts = combined_posts['cleaned_text'].tolist()
        post_embeddings = generate_embeddings(
            post_texts, 
            model, 
            batch_size=32,
            device=device
        )
        
        if len(post_embeddings) == 0:
            print(f"  Warning: No valid embeddings for cluster {cluster_id} posts")
            continue
            
        # Calculate similarity to cluster center
        similarities = np.dot(post_embeddings, centers[cluster_id])
        
        # Add similarities to dataframe
        combined_posts['similarity'] = similarities
        
        # Sort by similarity to cluster center
        combined_posts = combined_posts.sort_values('similarity', ascending=False)
        
        # Take the top representative posts
        representative_posts = combined_posts.head(NUM_SAMPLE_POSTS)
        cluster_posts[cluster_id] = representative_posts.to_dict('records')
        
        print(f"  Selected {len(representative_posts)} posts for cluster {cluster_id}")
    
    # Save representative posts - convert timestamps to strings to make JSON serializable
    serializable_posts = {}
    for cluster_id, posts in cluster_posts.items():
        serializable_posts[cluster_id] = []
        for post in posts:
            # Create a copy of the post dict
            serialized_post = {}
            for key, value in post.items():
                # Convert pandas Timestamp objects to strings
                if isinstance(value, pd.Timestamp):
                    serialized_post[key] = value.isoformat()
                else:
                    serialized_post[key] = value
            serializable_posts[cluster_id].append(serialized_post)
    
    # Save JSON-serializable posts
    with open(OUTPUT_DIR / 'clusters' / 'representative_posts.json', 'w') as f:
        json.dump(serializable_posts, f, indent=2)
    
    # Return the original posts (with Timestamp objects) for further processing
    return cluster_posts

def analyze_topics_with_gemini(cluster_posts, api_key=None):
    """Analyze cluster topics using Gemini API."""
    if api_key is None:
        # Check for API key in environment variable
        api_key = os.environ.get('GEMINI_API_KEY')
        
    if not api_key:
        print("Warning: Gemini API key not provided. Skipping topic analysis.")
        return None
        
    print("Analyzing topics with Gemini API...")
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')
    
    # Dictionary to store topic analysis for each cluster
    cluster_topics = {}
    
    for cluster_id, posts in cluster_posts.items():
        if not posts:
            print(f"  Skipping cluster {cluster_id} - no posts available")
            continue
            
        print(f"Analyzing topics for cluster {cluster_id}...")
        
        # Prepare posts for analysis
        post_texts = [f"Post {i+1}: {post['Text']}" for i, post in enumerate(posts)]
        post_content = "\n\n".join(post_texts)
        
        # Create prompt for Gemini
        prompt = f"""
        You are an expert at analyzing social media conversations and identifying discussion topics.
        
        Below are {len(posts)} representative posts from a cluster of users on the Farcaster social network.
        
        {post_content}
        
        Based only on these posts:
        1. Identify the 3-5 main topics being discussed in this cluster
        2. Provide a short label (2-5 words) for each topic
        3. Give a brief explanation for each topic (1-2 sentences)
        4. Assign an overall theme/label for this user cluster (maximum 5 words)
        
        Format your response as valid JSON with the following structure:
        {{
            "cluster_theme": "Overall theme of this cluster",
            "topics": [
                {{
                    "label": "Topic 1 label",
                    "explanation": "Brief explanation of Topic 1"
                }},
                ...
            ]
        }}
        
        Return ONLY the JSON with no additional text.
        """
        
        # Call Gemini API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                response_text = response.text
                
                # Extract JSON from response (handling potential text wrappers)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    topics_data = json.loads(json_str)
                    cluster_topics[cluster_id] = topics_data
                    break
                else:
                    print(f"  Attempt {attempt+1}: Failed to parse JSON from response")
                    if attempt == max_retries - 1:
                        print(f"  Warning: Could not extract valid JSON for cluster {cluster_id}")
            except Exception as e:
                print(f"  Attempt {attempt+1}: Error calling Gemini API: {e}")
                if attempt == max_retries - 1:
                    print(f"  Warning: Failed to analyze cluster {cluster_id} after {max_retries} attempts")
                time.sleep(2)  # Wait before retrying
    
    # Save topic analysis results
    with open(OUTPUT_DIR / 'topics' / 'cluster_topics.json', 'w') as f:
        json.dump(cluster_topics, f, indent=2)
    
    # Create a summary of cluster themes
    theme_summary = pd.DataFrame([
        {
            'cluster': cluster_id,
            'theme': data.get('cluster_theme', 'Unknown'),
            'topics': ', '.join([t.get('label', '') for t in data.get('topics', [])])
        }
        for cluster_id, data in cluster_topics.items()
    ])
    
    # Save theme summary
    theme_summary.to_csv(OUTPUT_DIR / 'topics' / 'cluster_themes.csv', index=False)
    
    print(f"Completed topic analysis for {len(cluster_topics)} clusters")
    return cluster_topics

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Update constants based on arguments
    top_n = args.top_users
    min_casts = args.min_casts
    num_clusters = args.clusters
    num_sample_posts = args.sample_posts
    gemini_key = args.gemini_key
    
    # Setup environment
    device = setup_environment()
    
    # Connect to DuckDB
    conn = connect_to_duckdb()
    
    # Load Farcaster data
    if not load_farcaster_data(conn):
        print("Failed to load data. Exiting.")
        return
    
    # Step 1: Identify popular users (with caching)
    if args.skip_user_identification:
        print("Skipping user identification step (using cached results)...")
        top_users = load_from_cache(CACHE_TOP_USERS)
        if top_users is None:
            print("No cached user data found! Running user identification...")
            top_users = identify_popular_users(conn, top_n=top_n, min_casts=min_casts)
            save_to_cache(top_users, CACHE_TOP_USERS)
    else:
        top_users = identify_popular_users(conn, top_n=top_n, min_casts=min_casts)
        save_to_cache(top_users, CACHE_TOP_USERS)
    
    # Step 2: Process user embeddings (with caching)
    embedding_cache = load_from_cache(CACHE_USER_EMBEDDINGS) if args.skip_embedding_generation else None
    
    if embedding_cache is not None:
        print("Using cached user embeddings...")
        user_embeddings = embedding_cache['embeddings']
        user_ids = embedding_cache['user_ids']
        user_cast_counts = embedding_cache['cast_counts']
        
        # Load model regardless since we might need it later
        print(f"Loading SentenceTransformer model {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        model.eval()
    else:
        user_embeddings, user_ids, user_cast_counts, model = process_user_embeddings(conn, top_users, device)
        
        # Cache the results
        embedding_data = {
            'embeddings': user_embeddings,
            'user_ids': user_ids,
            'cast_counts': user_cast_counts
        }
        save_to_cache(embedding_data, CACHE_USER_EMBEDDINGS)
    
    # Step 3: Cluster users (with caching)
    clusters_cache = load_from_cache(CACHE_CLUSTERS) if args.skip_clustering else None
    
    if clusters_cache is not None:
        print("Using cached clustering results...")
        cluster_df = clusters_cache['cluster_df']
        cluster_centers = clusters_cache['centers']
    else:
        cluster_df, cluster_centers = cluster_users(user_embeddings, user_ids, user_cast_counts, n_clusters=num_clusters)
        
        # Visualize clusters (this doesn't need caching as it's based on cluster results)
        visualize_clusters(user_embeddings, cluster_df)
        
        # Cache the results
        clusters_data = {
            'cluster_df': cluster_df,
            'centers': cluster_centers
        }
        save_to_cache(clusters_data, CACHE_CLUSTERS)
    
    # Step 4: Find representative posts for each cluster (with caching)
    posts_cache = load_from_cache(CACHE_REPRESENTATIVE_POSTS) if args.skip_representative_posts else None
    
    if posts_cache is not None:
        print("Using cached representative posts...")
        cluster_posts = posts_cache
    else:
        cluster_posts = find_representative_posts(conn, cluster_df, model, device)
        save_to_cache(cluster_posts, CACHE_REPRESENTATIVE_POSTS)
    
    # Step 5: Analyze topics with Gemini (with caching)
    topics_cache = load_from_cache(CACHE_TOPICS) if args.skip_topic_analysis else None
    
    if topics_cache is not None:
        print("Using cached topic analysis results...")
        cluster_topics = topics_cache
    else:
        # Use provided API key or environment variable
        cluster_topics = analyze_topics_with_gemini(cluster_posts, api_key=gemini_key)
        if cluster_topics:  # Only cache if we got results
            save_to_cache(cluster_topics, CACHE_TOPICS)
    
    # Calculate execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nAnalysis complete! Results are saved in the 'output' directory.")
    print("To use Gemini API for topic analysis, set the GEMINI_API_KEY environment variable.")
    print("\nTo skip steps in future runs, use command line arguments:")
    print("  --skip-user-identification: Skip user identification")
    print("  --skip-embedding-generation: Skip embedding generation")
    print("  --skip-clustering: Skip clustering")
    print("  --skip-representative-posts: Skip finding representative posts")
    print("  --skip-topic-analysis: Skip topic analysis")

if __name__ == "__main__":
    main()