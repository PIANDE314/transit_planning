import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, List, Dict
from joblib import Parallel, delayed
from transitlib.config import Config

cfg = Config()

poi_buf         = cfg.get("buffer_poi")
neg_pct         = cfg.get("neg_percentile")
K_pos           = cfg.get("K_pos")
K_neg           = cfg.get("K_neg")
pos_th          = cfg.get("pos_thresh")
neg_th          = cfg.get("neg_thresh")
logreg_neg_th   = cfg.get("logreg_neg_thresh")
noise_th_frac   = cfg.get("noise_thresh_frac")
max_iters       = cfg.get("self_max_iters")
test_size       = cfg.get("self_test_size")
runs            = cfg.get("self_runs")
rs              = cfg.get("random_state")
trees_per_iter  = cfg.get("rf_trees_per_iter", 5)
max_trees       = cfg.get("rf_max_trees", 100)

def initialize_seed_labels(
    segments_gdf: gpd.GeoDataFrame,
    feature_matrix: pd.DataFrame,
    poi_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    # 1) POI-based positives
    seg_buf = segments_gdf.copy()
    seg_buf['buffer'] = seg_buf.geometry.buffer(poi_buf)
    pos = gpd.sjoin(
        poi_gdf,
        seg_buf.set_geometry('buffer'),
        predicate='within', how='inner'
    )
    pos_ids = pos.segment_id.unique()
    # 2) Strict “all‑features” negatives
    thresh = feature_matrix.quantile(neg_pct / 100.0)
    neg_mask = (feature_matrix <= thresh).all(axis=1)
    neg_ids = feature_matrix.index[neg_mask]
    # 3) Combine
    seed_ids = list(set(pos_ids) | set(neg_ids))
    seeds = feature_matrix.loc[seed_ids].copy()
    seeds['label'] = 0
    seeds.loc[pos_ids, 'label'] = 1
    return seeds

def expand_negatives_with_logreg(
    raw_seeds: pd.DataFrame,
    feature_matrix: pd.DataFrame
) -> pd.DataFrame:
    # Train LR on the full initial seeds (all pos + all neg)
    X_train = raw_seeds.drop(columns="label")
    y_train = raw_seeds["label"]
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    # High‑confidence negatives from the unlabeled pool
    unlabeled = feature_matrix.index.difference(raw_seeds.index)
    probs = model.predict_proba(feature_matrix.loc[unlabeled])[:, 0]
    high_conf_neg = feature_matrix.loc[unlabeled][probs >= logreg_neg_th].copy()
    high_conf_neg['label'] = 0

    # Return the expanded seed set
    return pd.concat([raw_seeds, high_conf_neg]).sort_index()

def select_pseudo_labels(
    model: RandomForestClassifier,
    feature_matrix: pd.DataFrame,
    seeds_df: pd.DataFrame
) -> Tuple[List[int], List[int]]:
    unlabeled = feature_matrix.index.difference(seeds_df.index)
    probs = model.predict_proba(feature_matrix.loc[unlabeled])
    dfp = pd.DataFrame(probs, index=unlabeled, columns=[0, 1])

    pos = dfp[dfp[1] >= pos_th].nlargest(K_pos, 1).index.tolist()
    neg = dfp[dfp[0] >= neg_th].nlargest(K_neg, 0).index.tolist()
    return pos, neg

def inject_noise_labels(
    seeds_df: pd.DataFrame,
    new_pos: List[int],
    new_neg: List[int],
    segments_gdf: gpd.GeoDataFrame
) -> Tuple[List[int], List[int]]:
    segs = segments_gdf.reset_index(drop=True)
    segs['left'] = segs['segment_id']
    neigh = gpd.sjoin(
        segs[['segment_id','geometry']],
        segs[['segment_id','geometry']],
        predicate='intersects', how='inner',
        lsuffix='left', rsuffix='right'
    )
    map_n = neigh.groupby('segment_id_left')['segment_id_right'].apply(set).to_dict()

    final_p, final_n = [], []
    for sid in new_pos:
        nbrs = map_n.get(sid, set()) & set(seeds_df.index)
        if nbrs and all(seeds_df.loc[list(nbrs), 'label'] == 0):
            final_n.append(sid)
        else:
            final_p.append(sid)
    for sid in new_neg:
        nbrs = map_n.get(sid, set()) & set(seeds_df.index)
        if nbrs and (seeds_df.loc[list(nbrs), 'label'] == 1).sum() >= 2:
            final_p.append(sid)
        else:
            final_n.append(sid)
    return final_p, final_n

def run_self_training_single_pass(
    segments_gdf: gpd.GeoDataFrame,
    feature_matrix: pd.DataFrame,
    poi_gdf: gpd.GeoDataFrame,
    map_n: Dict[int, set] = None,
    warm_start: bool = False
) -> pd.Series:
    # 0) Precompute neighbor map if needed
    if map_n is None:
        segs = segments_gdf.reset_index(drop=True)
        segs['left'] = segs['segment_id']
        neigh = gpd.sjoin(
            segs[['left','geometry']],
            segs[['segment_id','geometry']],
            predicate='intersects', how='inner'
        )
        map_n = neigh.groupby('left')['segment_id'].apply(set).to_dict()

    # 1) Raw seeds + LR expansion
    raw_seeds = initialize_seed_labels(segments_gdf, feature_matrix, poi_gdf)
    seeds = expand_negatives_with_logreg(raw_seeds, feature_matrix)

    # 2) Balance before first RF fit
    p_df = seeds[seeds.label == 1]
    n_df = seeds[seeds.label == 0]
    n_bal = min(len(p_df), len(n_df))
    seeds = pd.concat([
        p_df.sample(n_bal, random_state=rs),
        n_df.sample(n_bal, random_state=rs)
    ]).sort_index()

    # 3) One train/val split, one RF instantiation
    X_all, y_all = seeds.drop(columns='label'), seeds['label']
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=test_size,
        stratify=y_all, random_state=rs
    )
    classes = np.array([0,1])
    weights = compute_class_weight('balanced', classes=classes, y=y_tr)
    class_weights = dict(zip(classes, weights))

    rf = RandomForestClassifier(
        n_estimators = 0 if warm_start else max_trees,
        warm_start  = warm_start,
        class_weight= class_weights,
        random_state = rs,
        n_jobs       = -1
    )

    # 4) Iterative self‑training
    for it in range(1, max_iters+1):
        if warm_start:
            rf.n_estimators = min(rf.n_estimators + trees_per_iter, max_trees)
        rf.fit(X_tr, y_tr)

        # eval
        y_pred  = rf.predict(X_val)
        y_proba = rf.predict_proba(X_val)[:,1]
        print(f"[Iter {it}] "
              f"{'warm' if warm_start else 'cold'} RF trees={rf.n_estimators}  "
              f"AUC={roc_auc_score(y_val, y_proba):.3f}")

        # pseudo‑label selection + noise injection
        pos_c, neg_c = select_pseudo_labels(rf, feature_matrix, seeds)
        if (len(pos_c) < K_pos*noise_th_frac or
            len(neg_c) < K_neg*noise_th_frac):
            final_p, final_n = inject_noise_labels(seeds, pos_c, neg_c, segments_gdf)
        else:
            final_p, final_n = pos_c, neg_c

        # remove already‑seen seeds
        final_p = [i for i in final_p if i not in seeds.index]
        final_n = [i for i in final_n if i not in seeds.index]
        if not final_p and not final_n:
            print(f"Converged at iter {it}.")
            break

        # add new labels and rebalance for next iter
        dp = feature_matrix.loc[final_p].copy(); dp['label'] = 1
        dn = feature_matrix.loc[final_n].copy(); dn['label'] = 0
        seeds = pd.concat([seeds, dp, dn]).sort_index()

        # rebalance + new split
        p_df = seeds[seeds.label == 1]; n_df = seeds[seeds.label == 0]
        n_bal = min(len(p_df), len(n_df))
        seeds = pd.concat([
            p_df.sample(n_bal, random_state=rs),
            n_df.sample(n_bal, random_state=rs)
        ]).sort_index()
        X_all, y_all = seeds.drop(columns='label'), seeds['label']
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=test_size,
            stratify=y_all, random_state=rs
        )

    # 5) Build full label series
    full_labels = pd.Series(index=feature_matrix.index, dtype=int)
    full_labels.update(seeds['label'])
    return full_labels

def run_self_training(
    segments_gdf: gpd.GeoDataFrame,
    feature_matrix: pd.DataFrame,
    poi_gdf: gpd.GeoDataFrame,
    warm_start: bool
) -> pd.Series:
    # Precompute neighbor map once
    segs = segments_gdf.reset_index(drop=True)
    neigh = gpd.sjoin(
        segs[['segment_id','geometry']],
        segs[['segment_id','geometry']],
        predicate='intersects', how='inner',
        lsuffix='left', rsuffix='right'
    )
    map_n = neigh.groupby('segment_id_left')['segment_id_right'].apply(set).to_dict()

    # Parallel runs
    def _run(_):
        return run_self_training_single_pass(
            segments_gdf, feature_matrix, poi_gdf,
            map_n=map_n, warm_start=warm_start
        )

    label_matrix = Parallel(n_jobs=cfg.get("n_jobs", 4))(
        delayed(_run)(i) for i in range(runs)
    )

    # Majority vote
    label_df    = pd.DataFrame(label_matrix)
    final_labels = label_df.mode().iloc[0]
    return final_labels
