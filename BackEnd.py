from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np


import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import FileResponse, Response


# ---- GLOBAL CONFIG ----
CSV_FILENAME = "cleaned_steam_games.csv"
dataset_context = {}

# Use one current_dir everywhere
current_dir = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Looking for local file: {CSV_FILENAME}...")
    
    csv_path = os.path.join(current_dir, CSV_FILENAME)

    if os.path.exists(csv_path):
        try:
            # Load Data
            print("Loading CSV...")
            df = pd.read_csv(csv_path)
            df = df.head(40000)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Prepare Data for ML
            print("Preparing Machine Learning Model...")
            df['genres'] = df['genres'].fillna('')
            df['about_the_game'] = df['about_the_game'].fillna('')
            
            # Create text soup
            df['combined_features'] = df['genres'] + " " + df['about_the_game'].astype(str).str.slice(0, 500)

            # Create Matrix
            # Create Matrix
            tfidf = TfidfVectorizer(
                stop_words='english',
                max_features=3000,      # was 5000
                dtype=np.float32        # <-- add this
            )
            feature_matrix = tfidf.fit_transform(df['combined_features'])


            
            # Save to Context
            dataset_context["df"] = df
            dataset_context["feature_matrix"] = feature_matrix
            
            print(f"Success! API is ready with {len(df)} games.")
        except Exception as e:
            print(f"Error loading data: {e}")
            dataset_context["df"] = pd.DataFrame()
            dataset_context["feature_matrix"] = None
    else:
        print(f"File not found at: {csv_path}")
        dataset_context["df"] = pd.DataFrame()
        dataset_context["feature_matrix"] = None
        
    yield
    dataset_context.clear()


app = FastAPI(lifespan=lifespan)

# Serve static files (images, etc.)
app.mount("/static", StaticFiles(directory=current_dir), name="static")

# CORS (safe to leave open for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- ENDPOINTS ----

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_index(request: Request):
    # Let Render's health check HEAD / succeed
    if request.method == "HEAD":
        return Response(status_code=200)
    return FileResponse(os.path.join(current_dir, "index.html"))


@app.get("/games")
def get_games(limit: int = 10, search: str | None = None):
    df = dataset_context.get("df")
    if df is None or df.empty:
        return []

    if search:
        filtered_df = df[df['name'].astype(str).str.contains(search, case=False, na=False)]
    else:
        filtered_df = df

    subset = filtered_df.head(limit).where(pd.notnull(filtered_df), None)
    return subset[['name']].to_dict(orient="records")


@app.get("/recommend")
def get_recommendation(game_name: str):
    df = dataset_context.get("df")
    feature_matrix = dataset_context.get("feature_matrix")
    
    if df is None or df.empty or feature_matrix is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Try exact match first
    matches = df[df['name'].astype(str).str.lower() == game_name.lower()]
    
    # If no exact match, try partial match ("contains")
    if matches.empty:
        matches = df[
            df['name'].astype(str).str.lower().str.contains(game_name.lower(), regex=False)
        ]
        
    if matches.empty:
        raise HTTPException(status_code=404, detail="Game not found")
        
    # Take the first match found
    idx = matches.index[0]
    matched_name = df.iloc[idx]['name']

    # ML logic
    target_vec = feature_matrix.getrow(idx)
    sim_scores = cosine_similarity(target_vec, feature_matrix).flatten()
    
    # Get top 5 similar (skip the game itself)
    similar_idx = sim_scores.argsort()[-6:-1][::-1]
    results = df.iloc[similar_idx]
    
    results = results.where(pd.notnull(results), None)
    
    response_data = results[['name', 'genres', 'about_the_game', 'price']].to_dict(orient="records")
    
    return {
        "source_game": matched_name,
        "recommendations": response_data
    }
