import sqlite3
import pandas as pd

# con = sqlite3.connect("/mnt/c/Users/ophth/Downloads/reddit_db/database.sqlite")

con = sqlite3.connect("/path/to/database.sqlite")

cur = con.cursor()

linked_query = """SELECT post.subreddit, parent.body AS parent_body, post.gilded, post.score,
                            parent.score AS parent_score, post.edited,
                            (post.created_utc - parent.created_utc) AS delay, post.body
                    FROM May2015 post
                    INNER JOIN May2015 parent ON substring(post.parent_id,4) = parent.id
                    WHERE post.body NOT LIKE '[deleted]'
                    AND parent.body NOT LIKE '[deleted]'
                    LIMIT 25000"""

# vertical_query = """SELECT subreddit, body, id, substring(parent_id,4) AS parent_id, score
#                     FROM May2015
#                     WHERE parent_id LIKE 't1%'
#                     AND score > 10
#                     LIMIT 1000"""

# horizontal_query = """SELECT subreddit, body, id, substring(parent_id,4) AS post_id, score
#                       FROM May2015
#                       WHERE parent_id LIKE 't3%'
#                       AND score > 10
#                       LIMIT 1000"""

linked_df = pd.read_sql_query(linked_query, con)

linked_df.to_feather('reddit_data/reddit_df.ft')

print(linked_df)

con.close()
