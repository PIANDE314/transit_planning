import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
from typing import Tuple, List
from transitlib.config import Config

cfg = Config()

poi_buf = cfg.get("buffer_poi")
neg_pct = cfg.get("neg_percentile")
K_pos = cfg.get("K_pos")
K_neg = cfg.get("K_neg")
pos_th = cfg.get("pos_thresh")
neg_th = cfg.get("neg_thresh")
logreg_neg_th = cfg.get("logreg_neg_thresh")
noise_th_frac = cfg.get("noise_thresh_frac")
max_iters = cfg.get("self_max_iters")
test_size = cfg.get("self_test_size")
runs = cfg.get("self_runs")
rs = cfg.get("random_state")

def initialize_seed_labels(
    segments_gdf: gpd.GeoDataFrame,
    feature_matrix: pd.DataFrame,
    poi_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:

    seg_buf = segments_gdf.copy()
    seg_buf['buffer'] = seg_buf.geometry.buffer(poi_buf)
    pos = gpd.sjoin(
        poi_gdf,
        seg_buf.set_geometry('buffer'),
        predicate='within', how='inner'
    )
    pos_ids = pos.segment_id.unique()

    thresh = feature_matrix.quantile(neg_pct / 100.0)
    neg_mask = (feature_matrix <= thresh).any(axis=1)
    neg_ids = feature_matrix.index[neg_mask]

    seed_ids = list(set(pos_ids) | set(neg_ids))
    seeds = feature_matrix.loc[seed_ids].copy()
    seeds['label'] = 0
    seeds.loc[pos_ids, 'label'] = 1

    p_df = seeds[seeds.label == 1]
    n_df = seeds[seeds.label == 0]
    n = min(len(p_df), len(n_df))
    balanced = pd.concat([
        p_df.sample(n, random_state=rs),
        n_df.sample(n, random_state=rs)
    ]).sort_index()

    return balanced

def expand_negatives_with_logreg(
    seeds_df: pd.DataFrame,
    feature_matrix: pd.DataFrame
) -> pd.DataFrame:
    pos_df = seeds_df[seeds_df.label == 1]
    neg_df = seeds_df[seeds_df.label == 0]
    X_train = pd.concat([pos_df, neg_df]).drop(columns="label")
    y_train = pd.concat([pos_df, neg_df])["label"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    unlabeled = feature_matrix.index.difference(seeds_df.index)
    probs = model.predict_proba(feature_matrix.loc[unlabeled])[:, 0]  # P(label=0)
    high_conf_neg = feature_matrix.loc[unlabeled][probs >= logreg_neg_th]
    high_conf_neg["label"] = 0

    return pd.concat([seeds_df, high_conf_neg])

def train_initial_model(seeds_df: pd.DataFrame) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:

    X = seeds_df.drop(columns='label')
    y = seeds_df['label']
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=rs,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]
    print(
        f"Acc: {accuracy_score(y_val, y_pred):.3f}, "
        f"AUC: {roc_auc_score(y_val, y_proba):.3f}, "
        f"LogLoss: {log_loss(y_val, rf.predict_proba(X_val)):.3f}"
    )
    print(classification_report(y_val, y_pred))

    return rf, X_val, y_val

def select_pseudo_labels(
    model: RandomForestClassifier,
    feature_matrix: pd.DataFrame,
    seeds_df: pd.DataFrame
) -> Tuple[List[int], List[int]]:

    unl = feature_matrix.index.difference(seeds_df.index)
    probs = model.predict_proba(feature_matrix.loc[unl])
    dfp = pd.DataFrame(probs, index=unl, columns=[0, 1])

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
        segs[['left','geometry']],
        segs[['segment_id','geometry']],
        predicate='intersects', how='inner'
    )
    map_n = neigh.groupby('left')['segment_id'].apply(set).to_dict()

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
    poi_gdf: gpd.GeoDataFrame
) -> pd.Series:

    seeds = initialize_seed_labels(segments_gdf, feature_matrix, poi_gdf)
    seeds = expand_negatives_with_logreg(seeds, feature_matrix)

    for it in range(1, max_iters + 1):
        model, _, _ = train_initial_model(seeds)
        pos_c, neg_c = select_pseudo_labels(model, feature_matrix, seeds)

        # noise injection only if pseudo-labels sparse
        if len(pos_c) < K_pos * noise_th_frac or len(neg_c) < K_neg * noise_th_frac:
            final_p, final_n = inject_noise_labels(seeds, pos_c, neg_c, segments_gdf)
        else:
            final_p, final_n = pos_c, neg_c

        final_p = [i for i in final_p if i not in seeds.index]
        final_n = [i for i in final_n if i not in seeds.index]

        if not final_p and not final_n:
            print(f"Converged at iter {it}")
            break

        dp = feature_matrix.loc[final_p].copy()
        dn = feature_matrix.loc[final_n].copy()
        dp['label'], dn['label'] = 1, 0
        seeds = pd.concat([seeds, dp, dn]).sort_index()

    full_labels = pd.Series(index=feature_matrix.index, dtype=int)
    full_labels.update(seeds['label'])
    return full_labels

def run_self_training(
    segments_gdf: gpd.GeoDataFrame,
    feature_matrix: pd.DataFrame,
    poi_gdf: gpd.GeoDataFrame,
) -> pd.Series:
    label_matrix = []

    for i in range(runs):
        print(f"\n--- Run {i+1} ---")
        labels = run_self_training_single_pass(segments_gdf, feature_matrix, poi_gdf)
        label_matrix.append(labels)

    # majority vote
    label_df = pd.DataFrame(label_matrix)
    final_labels = label_df.mode().iloc[0]
    return final_labels
