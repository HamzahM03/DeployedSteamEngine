from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from contextlib import asynccontextmanager
import re

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CSV_FILENAME = "cleaned_steam_games.csv"
dataset_context: dict = {}

current_dir = os.path.dirname(os.path.abspath(__file__))


def clean_price(price_val):
    try:
        if pd.isna(price_val):
            return 0.0
        s = str(price_val).lower().strip()
        if "free" in s:
            return 0.0
        # keep digits + dot
        s = "".join(c for c in s if c.isdigit() or c == ".")
        return float(s) if s else 0.0
    except Exception:
        return 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Looking for local file: {CSV_FILENAME}...")

    csv_path = os.path.join(current_dir, CSV_FILENAME)

    if os.path.exists(csv_path):
        try:
            print("Loading CSV...")
            df = pd.read_csv(csv_path)

            # standardize col names
            df.columns = (
                df.columns.str.strip().str.lower().str.replace(" ", "_")
            )

            # normalized name for fuzzy-ish search: remove non-alphanumerics
            df["search_name"] = (
                df["name"]
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9]", "", regex=True)
            )

            print("Preparing TF-IDF model...")
            df["genres"] = df.get("genres", "").fillna("")
            df["about_the_game"] = df.get("about_the_game", "").fillna("")

            # simple content-based features (genres + about text)
            df["combined_features"] = (
                df["genres"] + " " + df["about_the_game"].astype(str).str.slice(0, 500)
            )

            tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=3000,
                dtype=np.float32,  # reduce memory
            )
            feature_matrix = tfidf.fit_transform(df["combined_features"])

            dataset_context["df"] = df
            dataset_context["feature_matrix"] = feature_matrix

            print(f"Success! API is ready with {len(df)} games.")
        except Exception as e:
            print("Error loading data:", e)
            dataset_context["df"] = pd.DataFrame()
            dataset_context["feature_matrix"] = None
    else:
        print(f"File not found at: {csv_path}")
        dataset_context["df"] = pd.DataFrame()
        dataset_context["feature_matrix"] = None

    yield
    dataset_context.clear()


app = FastAPI(lifespan=lifespan)

# static files (logo, etc.)
app.mount("/static", StaticFiles(directory=current_dir), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_index(request: Request):
    # allow Render health checks
    if request.method == "HEAD":
        return Response(status_code=200)
    return FileResponse(os.path.join(current_dir, "index.html"))


@app.get("/games")
def get_games(limit: int = 10, search: Optional[str] = None):
    df = dataset_context.get("df")
    if df is None or df.empty:
        return []

    if search:
        # original loose match on raw name
        base_matches = df[
            df["name"].astype(str).str.contains(search, case=False, na=False)
        ]

        # normalized "pub g" -> "pubg" match
        norm_query = re.sub(r"[^a-z0-9]", "", search.lower())
        norm_matches = df[
            df["search_name"].str.contains(norm_query, na=False)
        ]

        # combine and drop duplicates
        filtered_df = pd.concat([base_matches, norm_matches]).drop_duplicates()
    else:
        filtered_df = df

    subset = filtered_df.head(limit).where(pd.notnull(filtered_df), None)
    return subset[["name"]].to_dict(orient="records")


@app.get("/recommend")
def get_recommendation(
    game_name: str,
    price_filter: Optional[str] = Query(None),
):
    df = dataset_context.get("df")
    feature_matrix = dataset_context.get("feature_matrix")

    if df is None or df.empty or feature_matrix is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # exact match on full name
    matches = df[df["name"].astype(str).str.lower() == game_name.lower()]

    # partial match on raw name
    if matches.empty:
        matches = df[
            df["name"]
            .astype(str)
            .str.lower()
            .str.contains(game_name.lower(), regex=False)
        ]

    # normalized match "pub g" -> "pubg"
    if matches.empty and "search_name" in df.columns:
        norm_query = re.sub(r"[^a-z0-9]", "", game_name.lower())
        matches = df[df["search_name"].str.contains(norm_query, na=False)]

    if matches.empty:
        raise HTTPException(status_code=404, detail="Game not found")

    idx = matches.index[0]
    matched_name = df.iloc[idx]["name"]

    # similarity
    target_vec = feature_matrix.getrow(idx)
    sim_scores = cosine_similarity(target_vec, feature_matrix).flatten()

    # take a pool of most similar to allow filtering
    top_idx = sim_scores.argsort()[-100:-1][::-1]
    candidate_df = df.iloc[top_idx].copy()

    # ---- apply price filter (if any) ----
    if price_filter and price_filter != "any":
        candidate_df["numeric_price"] = candidate_df["price"].apply(clean_price)

        if price_filter == "free":
            candidate_df = candidate_df[candidate_df["numeric_price"] == 0]
        elif price_filter == "under_5":
            candidate_df = candidate_df[candidate_df["numeric_price"] < 5]
        elif price_filter == "under_10":
            candidate_df = candidate_df[candidate_df["numeric_price"] < 10]
        elif price_filter == "under_30":
            candidate_df = candidate_df[candidate_df["numeric_price"] < 30]
        elif price_filter == "under_50":
            candidate_df = candidate_df[candidate_df["numeric_price"] < 50]
        elif price_filter == "above_50":
            candidate_df = candidate_df[candidate_df["numeric_price"] >= 50]

    # final top 5
    results = candidate_df.head(5)
    results = results.where(pd.notnull(results), None)

    # detect id column for Steam link
    possible_id_cols = ["appid", "app_id", "steam_appid", "id"]
    app_id_col = next((c for c in possible_id_cols if c in df.columns), None)

    cols_to_return = ["name", "genres", "about_the_game", "price", "header_image"]

    # add optional review columns if present
    for col in ["pct_pos_total", "num_reviews_total"]:
        if col in df.columns:
            cols_to_return.append(col)

    if app_id_col:
        cols_to_return.append(app_id_col)

    response_data = results[cols_to_return].to_dict(orient="records")

    # normalize key name to 'app_id'
    if app_id_col:
        for item in response_data:
            if app_id_col in item:
                item["app_id"] = item.pop(app_id_col)

    return {
        "source_game": matched_name,
        "recommendations": response_data,
    }
