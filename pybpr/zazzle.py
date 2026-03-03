"""Zazzle e-commerce dataset loader."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Default data path relative to project root
_DEFAULT_DATA = (
    Path(__file__).parent.parent / 'zazzle' / 'ZAZZLE_1GB'
)

# Supported event type modes
# 'clicks' : clicked → pos, viewed-not-clicked → neg
# 'orders' : ordered → pos, clicked-not-ordered → neg
EVENT_TYPES = ['clicks', 'orders']


def _load_shards(
    data_dir: Path,
    prefix: str,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Concatenate all parquet shards for a given prefix."""
    files = sorted(
        data_dir.glob(f'{prefix}_*_part_00.parquet')
    )
    if not files:
        raise FileNotFoundError(
            f"No {prefix} shards in {data_dir}"
        )
    return pd.concat(
        [pd.read_parquet(f, columns=columns) for f in files],
        ignore_index=True
    )


def load_zazzle(
    data_dir: Union[str, Path, None] = None,
    event_type: str = 'clicks',
) -> Dict[str, pd.DataFrame]:
    """Load Zazzle dataset; returns positives, negatives, features.

    event_type='clicks' (default):
      Positive: viewed AND clicked (is_click=True)
      Negative: viewed but NOT clicked (is_click=False)

    event_type='orders':
      Positive: purchased (OrderItems)
      Negative: clicked but NOT purchased (is_click=True, not ordered)

    Features: product metadata with multi-hot encoding.
      Three feature types are stacked with ID offsets so they
      occupy non-overlapping ranges in the feature space:
        final_department_id → IDs 0 .. N_dept
        product_type        → IDs N_dept+1 .. N_dept+N_ptype
        vision_style_id_1   → IDs N_dept+N_ptype+1 .. (end)

    Returns dict with keys 'positives', 'negatives', 'features'.
    """
    data_dir = Path(data_dir or _DEFAULT_DATA)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )
    if event_type not in EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {EVENT_TYPES}"
        )

    # Always load Clicks (needed for both modes)
    print("Loading Clicks...")
    clicks = _load_shards(
        data_dir, 'Clicks',
        ['user_id', 'product_id', 'is_click']
    )
    clicked = (
        clicks[clicks['is_click']][['user_id', 'product_id']]
        .drop_duplicates()
    )
    viewed_only = (
        clicks[~clicks['is_click']][['user_id', 'product_id']]
        .drop_duplicates()
    )
    del clicks

    if event_type == 'clicks':
        # Positive: clicked; negative: viewed but not clicked
        pos_raw = clicked
        neg_raw = viewed_only

    else:  # 'orders'
        # Positive: purchased products (OrderItems)
        print("Loading OrderItems...")
        orders = _load_shards(
            data_dir, 'OrderItems', ['user_id', 'product_id']
        ).drop_duplicates()

        # Negative: clicked products that were NOT purchased
        # Merge clicks with orders on (user_id, product_id);
        # keep only clicks that have no matching order row
        merged = clicked.merge(
            orders,
            on=['user_id', 'product_id'],
            how='left',
            indicator=True
        )
        clicked_not_ordered = merged[
            merged['_merge'] == 'left_only'
        ][['user_id', 'product_id']].drop_duplicates()

        pos_raw = orders
        neg_raw = clicked_not_ordered

    # Factorize UUID user_ids to integers (consistent across pos/neg)
    all_user_ids = pd.concat(
        [pos_raw['user_id'], neg_raw['user_id']],
        ignore_index=True
    ).unique()
    user_map = {uid: i for i, uid in enumerate(all_user_ids)}

    # Build standardized positives DataFrame
    pos = pos_raw.copy()
    pos['UserID'] = pos['user_id'].map(user_map).astype(int)
    pos = (
        pos[['UserID', 'product_id']]
        .rename(columns={'product_id': 'ProductID'})
        .drop_duplicates()
    )

    # Build standardized negatives DataFrame
    neg = neg_raw.copy()
    neg['UserID'] = neg['user_id'].map(user_map).astype(int)
    neg = (
        neg[['UserID', 'product_id']]
        .rename(columns={'product_id': 'ProductID'})
        .drop_duplicates()
    )

    # Load product metadata for item features
    print("Loading Products...")
    products = _load_shards(
        data_dir, 'Products',
        [
            'product_id', 'final_department_id',
            'product_type', 'vision_style_id_1'
        ]
    ).drop_duplicates(subset='product_id').copy()

    # Factorize product_type string to integer
    products['ptype_id'] = pd.factorize(
        products['product_type']
    )[0]

    # Compute offsets so each feature type occupies a distinct ID range
    ptype_offset = int(products['final_department_id'].max()) + 1
    style_offset = (
        ptype_offset + int(products['ptype_id'].max()) + 1
    )

    # Stack department, type, and style feature rows
    dept_feat = products[
        ['product_id', 'final_department_id']
    ].copy()
    dept_feat.columns = ['ProductID', 'FeatureID']

    ptype_feat = products[['product_id', 'ptype_id']].copy()
    ptype_feat.columns = ['ProductID', 'FeatureID']
    ptype_feat['FeatureID'] += ptype_offset

    style_feat = products[
        ['product_id', 'vision_style_id_1']
    ].copy()
    style_feat.columns = ['ProductID', 'FeatureID']
    style_feat['FeatureID'] += style_offset

    features = pd.concat(
        [dept_feat, ptype_feat, style_feat], ignore_index=True
    )

    print(
        f"Loaded {len(pos):,} positives, "
        f"{len(neg):,} negatives, "
        f"{len(features):,} item features"
    )
    return {
        'positives': pos,
        'negatives': neg,
        'features': features,
    }
