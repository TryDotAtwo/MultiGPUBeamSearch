from multigpubeamsearch.publication import kaggle_run_url


def test_kaggle_run_url_points_to_the_published_notebook():
    assert kaggle_run_url("trydotatwo", "cube555-native-2xt4-blend") == (
        "https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend"
    )


def test_kaggle_run_url_escapes_path_segments():
    assert kaggle_run_url("team name", "cube 555") == (
        "https://www.kaggle.com/code/team%20name/cube%20555"
    )
