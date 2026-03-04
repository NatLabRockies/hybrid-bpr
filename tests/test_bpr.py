"""
Script for performing unit test on BPR class

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""
import os
import numpy as np
from hybridbpr import UserItemInteractions, BPR, bpr_fit
from hybridbpr.utils import load_movielens_data

print(os.path.dirname(os.path.abspath(__file__)))


def test_loading_of_movielens_data():
    """Check if movielens data is loaded properly"""

    df = load_movielens_data(
        flag='ml-100k',
        data_dir=os.path.join(os.path.curdir, 'examples', 'data')
    )
    assert df.shape[0] == 100000


def test_bpr_for_synthetic_data_8user_10item():
    """Pytest"""
    test_data_1 = UserItemInteractions(
        users_index=np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                              5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])-1,
        items_index=np.array([1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 2, 3, 4,
                              6, 7, 8, 9, 10, 6, 7, 8, 8, 9, 10, 7, 8, 9])-1,
        num_items=10,
        num_users=8
    )
    bpr1 = BPR(
        num_features=2,
        reg_lambda=0.0,
        num_iters=200,
        learning_rate=0.2,
        batch_size=8,
        initial_std=0.01,
    )

    bpr1.initiate(uimat=test_data_1.mat)
    bpr_fit(bpr_obj=bpr1, iumat=test_data_1.mat, ncores=1)
    q1, q2, q3 = test_data_1.get_metric_v1(
        umat=bpr1.umat,
        imat=bpr1.imat,
        perc_active_users=1.0,
        perc_active_items=1.0,
        num_recs=3
    )
    assert q1 <= q2 <= q3
    assert q2 == 1
    assert q1 == 1
    assert q3 == 1
    assert bpr1.get_recomendations_for_this_user(0, 1)[0] == 2
    assert bpr1.get_recomendations_for_this_user(4, 1)[0] == 7
