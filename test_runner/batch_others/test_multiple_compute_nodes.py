from contextlib import closing
from fixtures.zenith_fixtures import DEFAULT_BRANCH_NAME, ZenithEnv


def test_multiple_compute_nodes(zenith_simple_env: ZenithEnv):
    env = zenith_simple_env
    pg1 = env.postgres.create_start(DEFAULT_BRANCH_NAME, "test_multiple_compute_nodes_1")
    pg2 = env.postgres.create_start(DEFAULT_BRANCH_NAME, "test_multiple_compute_nodes_2")

    n_rows = 10000

    with closing(pg1.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE t1(key int primary key, value text)")
            cur.execute(f"INSERT INTO t1 SELECT generate_series(1,{n_rows}), 'payload'")
            cur.execute("SELECT COUNT(*) FROM t1")
            assert cur.fetchone()[0] == n_rows

    with closing(pg2.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM t1")
            assert cur.fetchone()[0] == n_rows

            cur.execute("CREATE TABLE t2(key int primary key, value text)")
            cur.execute(f"INSERT INTO t2 SELECT generate_series(1,{n_rows}), 'payload'")
            cur.execute("SELECT COUNT(*) FROM t2")
            assert cur.fetchone()[0] == n_rows
