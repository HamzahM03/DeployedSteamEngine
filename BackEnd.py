from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- GLOBAL CONFIG ----
CSV_FILENAME = "cleaned_steam_games.csv"
dataset_context: dict = {}

# Base directory for index.html + static files
current_dir = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Looking for local file: {CSV_FILENAME}...")
    csv_path = os.path.join(current_dir, CSV_FILENAME)

    if os.path.exists(csv_path):
        try:
            print("Loading CSV...")
            df = pd.read_csv(csv_path)

            # Normalize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # OPTIONAL: downsample for memory safety on free tier
            # Comment this out if you want all rows and your host has enough RAM
            # df = df.head(40000)

            print("Preparing TF-IDF features...")
            df["genres"] = df["genres"].fillna("")
            df["about_the_game"] = df["about_the_game"].fillna("")

            # Text feature: genres + truncated description
            df["combined_features"] = (
                df["genres"] + " " + df["about_the_game"].astype(str).str.slice(0, 500)
            )

            # Lighter TF-IDF: fewer features + float32
            tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=3000,   # was higher before; this saves memory
                dtype=np.float32     # 4 bytes instead of 8
            )
            feature_matrix = tfidf.fit_transform(df["combined_features"])

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

    # Run the app
    yield

    # Cleanup on shutdown
    dataset_context.clear()


app = FastAPI(lifespan=lifespan)

# Serve static files (logo, favicon, etc.)
app.mount("/static", StaticFiles(directory=current_dir), name="static")

# CORS - open for now (fine for this project)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- ROUTES ----

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_index(request: Request):
    # Let Render do HEAD / for health checks
    if request.method == "HEAD":
        return Response(status_code=200)
    return FileResponse(os.path.join(current_dir, "index.html"))


@app.get("/games")
def get_games(limit: int = 10, search: str | None = None):
    df = dataset_context.get("df")
    if df is None or df.empty:
        return []

    if search:
        mask = df["name"].astype(str).str.contains(search, case=False, na=False)
        filtered_df = df[mask]
    else:
        filtered_df = df

    subset = filtered_df.head(limit).where(pd.notnull(filtered_df), None)
    return subset[["name"]].to_dict(orient="records")


@app.get("/recommend")
def get_recommendation(game_name: str):
    df = dataset_context.get("df")
    feature_matrix = dataset_context.get("feature_matrix")

    if df is None or df.empty or feature_matrix is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Exact match first
    names = df["name"].astype(str)
    matches = df[names.str.lower() == game_name.lower()]

    # Partial match fallback
    if matches.empty:
        matches = df[names.str.lower().str.contains(game_name.lower(), regex=False)]

    if matches.empty:
        raise HTTPException(status_code=404, detail="Game not found")

    # Use first match
    idx = matches.index[0]
    matched_name = df.iloc[idx]["name"]

    # Similarity via TF-IDF
    target_vec = feature_matrix.getrow(idx)
    sim_scores = cosine_similarity(target_vec, feature_matrix).flatten()

    # Get top 5 similar (skip the game itself)
    similar_idx = sim_scores.argsort()[-6:-1][::-1]
    results = df.iloc[similar_idx]
    results = results.where(pd.notnull(results), None)

    # Include header_image if the column exists
    cols = ["name", "genres", "about_the_game", "price"]
    if "header_image" in df.columns:
        cols.append("header_image")

    response_data = results[cols].to_dict(orient="records")

    return {
        "source_game": matched_name,
        "recommendations": response_data,
    }
