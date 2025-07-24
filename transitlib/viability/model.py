import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
from tqdm import tqdm
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler

"""
§ 3.1 Evaluating Roads for Transit Viability — Modified semi‑supervised self‑training 
"""

def initialize_seed_labels(
    segments_gdf: gpd.GeoDataFrame,
    feature_matrix: pd.DataFrame,
    poi_gdf: gpd.GeoDataFrame,
    buffer_dist: float,
    neg_percentile: float = 15.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Build initial seed set:
      - positives: segments within 100 m of any POI (Fig 2B) :contentReference[oaicite:25]{index=25}
      - negatives: segments in bottom 15th percentile of **any** attribute (Table 2) :contentReference[oaicite:26]{index=26}
    """
    seg_buf = segments_gdf.copy()
    seg_buf['buffer'] = seg_buf.geometry.buffer(buffer_dist)
    pos = gpd.sjoin(poi_gdf, seg_buf.set_geometry('buffer'), predicate='within', how='inner')
    pos_ids = pos.segment_id.unique()

    thresh = feature_matrix.quantile(neg_percentile/100.0)
    neg_mask = (feature_matrix <= thresh).any(axis=1)
    neg_ids = feature_matrix.index[neg_mask]

    seed_ids = list(set(pos_ids) | set(neg_ids))
    seeds = feature_matrix.loc[seed_ids].copy()
    seeds['label'] = 0
    seeds.loc[pos_ids, 'label'] = 1

    # balance
    p_df = seeds[seeds.label==1]; n_df = seeds[seeds.label==0]
    n = min(len(p_df), len(n_df))
    return pd.concat([
        p_df.sample(n, random_state=random_state),
        n_df.sample(n, random_state=random_state)
    ]).sort_index()


def train_initial_model(
    seeds_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """
    Train RF on seeds; report metrics.
    """
    X = seeds_df.drop(columns='label'); y = seeds_df['label']
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                random_state=random_state, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:,1]
    print(f"Acc: {accuracy_score(y_val,y_pred):.3f}, "
          f"AUC: {roc_auc_score(y_val,y_proba):.3f}, "
          f"LogLoss: {log_loss(y_val, rf.predict_proba(X_val)):.3f}")
    print(classification_report(y_val,y_pred))

    return rf, X_val, y_val


def select_pseudo_labels(
    model: RandomForestClassifier,
    feature_matrix: pd.DataFrame,
    seeds_df: pd.DataFrame,
    K_pos: int = 5,
    K_neg: int = 5,
    pos_thresh: float = 0.95,
    neg_thresh: float = 0.80
) -> Tuple[List[int], List[int]]:
    """
    Select high‑confidence unlabeled:
      - pos ≥0.95, up to K_pos
      - neg ≥0.80, up to K_neg  
    """
    unl = feature_matrix.index.difference(seeds_df.index)
    probs = model.predict_proba(feature_matrix.loc[unl])
    dfp = pd.DataFrame(probs, index=unl, columns=[0,1])
    pos = dfp[dfp[1]>=pos_thresh].nlargest(K_pos,1).index.tolist()
    neg = dfp[dfp[0]>=neg_thresh].nlargest(K_neg,0).index.tolist()
    return pos, neg


def inject_noise_labels(
    seeds_df: pd.DataFrame,
    new_pos: List[int],
    new_neg: List[int],
    segments_gdf: gpd.GeoDataFrame
) -> Tuple[List[int], List[int]]:
    """
    Flip labels if neighbors unanimously oppose:
      - pos with all neg neighbors → neg
      - neg with ≥2 pos neighbors → pos  
    """
    segs = segments_gdf.reset_index(drop=True)
    segs['left'] = segs['segment_id']
    neigh = gpd.sjoin(segs[['left','geometry']], segs[['segment_id','geometry']],
                      predicate='intersects', how='inner')
    map_n = neigh.groupby('left')['segment_id'].apply(set).to_dict()

    final_p, final_n = [], []
    for sid in new_pos:
        nbrs = map_n.get(sid, set()) & set(seeds_df.index)
        final_n.append(sid) if nbrs and all(seeds_df.loc[list(nbrs),'label']==0) else final_p.append(sid)
    for sid in new_neg:
        nbrs = map_n.get(sid, set()) & set(seeds_df.index)
        final_p.append(sid) if nbrs and (seeds_df.loc[list(nbrs),'label']==1).sum()>=2 else final_n.append(sid)
    return final_p, final_n


def run_self_training(
    segments_gdf: gpd.GeoDataFrame,
    feature_matrix: pd.DataFrame,
    poi_gdf: gpd.GeoDataFrame,
    buffer_dist: float = 100.0,
    max_iters: int = 200,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterate self‑training per Fig 3 until convergence or 200 iters .
    """
    seeds = initialize_seed_labels(segments_gdf, feature_matrix, poi_gdf, buffer_dist, **kwargs)
    history = []
    for it in tqdm(range(1, max_iters+1), desc="Self‑Training"):
        model, _, _ = train_initial_model(seeds, random_state=kwargs.get('random_state',42)+it)
        pos_c, neg_c = select_pseudo_labels(model, feature_matrix, seeds, **kwargs)
        final_p, final_n = inject_noise_labels(seeds, pos_c, neg_c, segments_gdf)
        final_p = [i for i in final_p if i not in seeds.index]
        final_n = [i for i in final_n if i not in seeds.index]
        history.append({'iter':it,'new_pos':len(final_p),'new_neg':len(final_n),'total':len(seeds)})
        if not final_p and not final_n:
            print(f"Converged at iter {it}")
            break
        dp = feature_matrix.loc[final_p].copy(); dn = feature_matrix.loc[final_n].copy()
        dp['label'], dn['label'] = 1, 0
        seeds = pd.concat([seeds, dp, dn]).sort_index()

    return seeds, pd.DataFrame(history)
